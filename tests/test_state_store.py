from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from state_store import Snapshot, SQLiteStateStore


def _store(tmp_path: Path, name: str = "state.sqlite3") -> SQLiteStateStore:
    store = SQLiteStateStore(
        tmp_path / name,
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )
    store.initialize()
    return store


def test_legacy_json_is_migrated_once_and_preserved_as_migrated_file(tmp_path: Path) -> None:
    traffic_path = tmp_path / "traffic_anchors.json"
    history_path = tmp_path / "torrent_history.json"
    traffic_path.write_text(
        json.dumps(
            {
                "day": {"key": "2026-07-10", "downloaded": 100, "uploaded": 40},
                "history": {
                    "days": [
                        {"date": "2026-07-09", "downloaded": 75, "uploaded": 30},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps({"items": {"hash:abc": {"name": "Ubuntu", "status": "seeding"}}}),
        encoding="utf-8",
    )

    db_path = tmp_path / "state.sqlite3"
    store = SQLiteStateStore(
        db_path,
        legacy_traffic_path=traffic_path,
        legacy_torrent_history_path=history_path,
    )
    store.initialize()

    anchors, traffic_history = store.load_traffic_state()
    assert anchors["day"] == {"key": "2026-07-10", "downloaded": 100, "uploaded": 40}
    assert traffic_history == [{"date": "2026-07-09", "downloaded": 75, "uploaded": 30}]
    assert store.load_torrent_history() == {"hash:abc": {"name": "Ubuntu", "status": "seeding"}}
    assert not traffic_path.exists()
    assert not history_path.exists()
    assert len(list(tmp_path.glob("traffic_anchors.json*.migrated"))) == 1
    assert len(list(tmp_path.glob("torrent_history.json*.migrated"))) == 1

    reopened = SQLiteStateStore(
        db_path,
        legacy_traffic_path=traffic_path,
        legacy_torrent_history_path=history_path,
    )
    reopened.initialize()
    assert reopened.load_traffic_state() == (anchors, traffic_history)
    assert reopened.load_torrent_history() == store.load_torrent_history()
    assert len(list(tmp_path.glob("traffic_anchors.json*.migrated"))) == 1
    assert len(list(tmp_path.glob("torrent_history.json*.migrated"))) == 1


@pytest.mark.parametrize(
    ("filename", "other_filename", "loader"),
    [
        ("traffic_anchors.json", "missing-history.json", "load_traffic_state"),
        ("torrent_history.json", "missing-traffic.json", "load_torrent_history"),
    ],
)
def test_corrupt_legacy_json_is_quarantined(
    tmp_path: Path,
    filename: str,
    other_filename: str,
    loader: str,
) -> None:
    corrupt_path = tmp_path / filename
    corrupt_path.write_text("{ definitely-not-json", encoding="utf-8")
    traffic_path = corrupt_path if filename.startswith("traffic") else tmp_path / other_filename
    history_path = corrupt_path if filename.startswith("torrent") else tmp_path / other_filename
    store = SQLiteStateStore(
        tmp_path / "state.sqlite3",
        legacy_traffic_path=traffic_path,
        legacy_torrent_history_path=history_path,
    )

    store.initialize()

    assert not corrupt_path.exists()
    quarantined = list(tmp_path.glob(f"{filename}.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{ definitely-not-json"
    assert getattr(store, loader)() in (({}, []), {})


def test_notification_preferences_persist_and_toggle_across_reopen(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.ensure_chat(101) is True
    assert store.toggle_chat(101) is False
    assert store.notifications_enabled(101) is False

    reopened = _store(tmp_path)
    assert reopened.notifications_enabled(101) is False
    assert reopened.toggle_chat(101) is True
    assert reopened.list_enabled_chats() == {101}
    assert reopened.ensure_chat(202, default_enabled=False) is False
    assert reopened.list_enabled_chats() == {101}


def test_outbox_is_idempotent_and_tracks_failure_then_delivery(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.enqueue_outbox("completion:hash:abc:1", 101, "completion", "Done") is True
    assert store.enqueue_outbox("completion:hash:abc:1", 101, "completion", "Duplicate") is False
    item = store.list_due_outbox(now_ts=10**20)[0]
    assert item.text == "Done"
    assert item.attempts == 0

    assert store.mark_outbox_failed(item.id, 1, 500.0, error="temporary failure") is True
    assert store.list_due_outbox(now_ts=499.9) == []
    failed = store.list_due_outbox(now_ts=500.0)[0]
    assert failed.status == "pending"
    assert failed.attempts == 1
    assert failed.last_error == "temporary failure"

    assert store.mark_outbox_delivered(failed.id, delivered_at=600.0) is True
    assert store.mark_outbox_delivered(failed.id, delivered_at=601.0) is False
    assert store.list_due_outbox(now_ts=10**20) == []


def test_outbox_claim_is_atomic_across_parallel_drainers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.enqueue_outbox("completion:hash:abc:1", 101, "completion", "Done")

    claim_time = 10_000_000_000.0
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: store.claim_due_outbox(now_ts=claim_time), range(2)))

    claimed_items = [item for batch in claims for item in batch]
    assert len(claimed_items) == 1
    assert claimed_items[0].status == "sending"

    # An abandoned lease is recovered without creating a second row.
    recovered = store.claim_due_outbox(now_ts=claim_time + 121.0)
    assert [item.id for item in recovered] == [claimed_items[0].id]


def test_initialize_upgrades_v1_outbox_for_atomic_claims(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE notification_outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'delivered', 'cancelled')),
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                delivered_at REAL,
                last_error TEXT,
                UNIQUE (event_key, chat_id)
            );
            INSERT INTO notification_outbox
                (event_key, chat_id, kind, text, next_attempt_at, created_at, updated_at)
            VALUES ('legacy-event', 101, 'completion', 'Done', 1, 1, 1);
            PRAGMA user_version=1;
            """
        )

    store = SQLiteStateStore(
        db_path,
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )
    store.initialize()

    claimed = store.claim_due_outbox(now_ts=10.0)
    assert len(claimed) == 1
    assert claimed[0].event_key == "legacy-event"
    assert claimed[0].status == "sending"


def test_snapshots_and_start_watches_survive_reopen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_monitor_snapshot(
        "hash:abc",
        completed=False,
        started=True,
        present=True,
        generation=3,
        updated_at=100.0,
    )
    first_watch = store.add_start_watch("hash:abc", 101, 7, "Initial name", added_at=50.0)
    assert first_watch.added_at == 50.0
    updated_watch = store.add_start_watch("hash:abc", 101, 8, "Updated name", added_at=75.0)
    assert updated_watch.torrent_id == 8
    assert updated_watch.name == "Updated name"
    assert updated_watch.added_at == 50.0
    store.update_start_watch("hash:abc", 101, start_notified=True, no_peers_notified=True)

    reopened = _store(tmp_path)
    snapshot = reopened.get_monitor_snapshot("hash:abc")
    assert snapshot == Snapshot(
        torrent_key="hash:abc",
        completed=False,
        started=True,
        present=True,
        generation=3,
        updated_at=100.0,
    )
    watches = reopened.list_start_watches(torrent_key="hash:abc", chat_id=101)
    assert len(watches) == 1
    assert watches[0].start_notified is True
    assert watches[0].no_peers_notified is True
    assert watches[0].completion_notified is False

    reopened.replace_monitor_snapshots(
        [Snapshot("hash:def", completed=True, started=True, present=True, generation=1, updated_at=200.0)]
    )
    assert set(reopened.load_monitor_snapshots()) == {"hash:def"}
    assert reopened.delete_start_watch("hash:abc", 101) is True
    assert reopened.list_start_watches() == []


def test_connection_per_operation_supports_concurrent_writers(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        inserted = list(
            pool.map(
                lambda _: store.enqueue_outbox("same-event", 101, "completion", "Done"),
                range(32),
            )
        )
        list(
            pool.map(
                lambda index: store.save_monitor_snapshot(
                    f"hash:{index}",
                    completed=bool(index % 2),
                    started=True,
                    present=True,
                    generation=index,
                ),
                range(32),
            )
        )

    assert sum(inserted) == 1
    assert len(store.list_due_outbox(now_ts=10**20)) == 1
    assert set(store.load_monitor_snapshots()) == {f"hash:{index}" for index in range(32)}
