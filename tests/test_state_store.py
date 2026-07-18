from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from state_store import Snapshot, SQLiteStateStore, StateStoreError


def _store(tmp_path: Path, name: str = "state.sqlite3") -> SQLiteStateStore:
    store = SQLiteStateStore(
        tmp_path / name,
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )
    store.initialize()
    return store


def _emulate_v2_rollback_initialize(db_path: Path) -> None:
    """Run the previous release's outbox migration probe and version update."""

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notification_outbox'"
        ).fetchone()
        if row is not None and "'sending'" not in str(row[0]):
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                DROP INDEX IF EXISTS notification_outbox_due_idx;
                DROP INDEX IF EXISTS notification_outbox_chat_idx;
                ALTER TABLE notification_outbox RENAME TO notification_outbox_v1;
                CREATE TABLE notification_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL,
                    chat_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'sending', 'delivered', 'cancelled')),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    next_attempt_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    delivered_at REAL,
                    last_error TEXT,
                    UNIQUE (event_key, chat_id)
                );
                INSERT INTO notification_outbox
                    (id, event_key, chat_id, kind, text, status, attempts,
                     next_attempt_at, created_at, updated_at, delivered_at, last_error)
                SELECT id, event_key, chat_id, kind, text, status, attempts,
                       next_attempt_at, created_at, updated_at, delivered_at, last_error
                  FROM notification_outbox_v1;
                DROP TABLE notification_outbox_v1;
                CREATE INDEX notification_outbox_due_idx
                    ON notification_outbox (status, next_attempt_at, id);
                CREATE INDEX notification_outbox_chat_idx
                    ON notification_outbox (chat_id, status);
                COMMIT;
                """
            )
        # Every remaining v2 schema statement is CREATE IF NOT EXISTS and the
        # v3 database already has those objects; its only further effect is the
        # exact version downgrade below.
        connection.executescript("BEGIN IMMEDIATE; PRAGMA user_version=2; COMMIT;")


def _emulate_v2_claim(connection: sqlite3.Connection, now_ts: float, limit: int = 100) -> list[int]:
    """Claim with the previous release's exact unfenced SQL shape."""

    lease_until = now_ts + 120.0
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            UPDATE notification_outbox
               SET status = 'pending', updated_at = ?
             WHERE status = 'sending' AND next_attempt_at <= ?
            """,
            (now_ts, now_ts),
        )
        rows = connection.execute(
            """
            SELECT id
              FROM notification_outbox
             WHERE status = 'pending' AND next_attempt_at <= ?
             ORDER BY next_attempt_at, id
             LIMIT ?
            """,
            (now_ts, limit),
        ).fetchall()
        item_ids = [int(row[0]) for row in rows]
        if item_ids:
            placeholders = ",".join("?" for _ in item_ids)
            # Keep the dynamic placeholder layout identical to the previous release.
            connection.execute(
                """
                UPDATE notification_outbox
                   SET status = 'sending', next_attempt_at = ?, updated_at = ?
                 WHERE id IN ({}) AND status = 'pending'
                """.format(placeholders),  # nosec B608
                [lease_until, now_ts, *item_ids],
            )
        connection.commit()
        return item_ids
    except BaseException:
        connection.rollback()
        raise


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
    for migrated_path in tmp_path.glob("*.migrated"):
        assert migrated_path.stat().st_mode & 0o777 == 0o600

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


def test_disabling_chat_cancels_active_claim_and_start_watches(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_chat(101)
    store.enqueue_outbox("completion:hash:abc:1", 101, "completion", "Done")
    store.add_start_watch("hash:abc", 101, 7, "Ubuntu")
    claimed = store.claim_due_outbox(now_ts=10_000_000_000.0)
    claim_token = claimed[0].claim_token
    assert claim_token is not None

    assert store.set_notification_enabled(101, False) is False
    assert store.list_start_watches(chat_id=101) == []
    assert store.mark_outbox_delivered(claimed[0].id, claim_token) is False
    with sqlite3.connect(store.db_path) as connection:
        status_and_token = connection.execute(
            "SELECT status, claim_token FROM notification_outbox WHERE id = ?",
            (claimed[0].id,),
        ).fetchone()
    assert status_and_token == ("cancelled", None)


def test_outbox_is_idempotent_and_tracks_failure_then_delivery(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.enqueue_outbox("completion:hash:abc:1", 101, "completion", "Done") is True
    assert store.enqueue_outbox("completion:hash:abc:1", 101, "completion", "Duplicate") is False
    pending = store.list_due_outbox(now_ts=10**20)[0]
    assert pending.claim_token is None
    item = store.claim_due_outbox(now_ts=10_000_000_000.0)[0]
    assert item.text == "Done"
    assert item.attempts == 0
    assert item.claim_token is not None

    assert store.mark_outbox_failed(item.id, item.claim_token, 1, 500.0, error="temporary failure") is True
    assert store.list_due_outbox(now_ts=499.9) == []
    failed = store.list_due_outbox(now_ts=500.0)[0]
    assert failed.status == "pending"
    assert failed.claim_token is None
    assert failed.attempts == 1
    assert failed.last_error == "temporary failure"

    retry = store.claim_due_outbox(now_ts=500.0)[0]
    assert retry.claim_token is not None
    assert store.mark_outbox_delivered(retry.id, retry.claim_token, delivered_at=600.0) is True
    assert store.mark_outbox_delivered(retry.id, retry.claim_token, delivered_at=601.0) is False
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
    original_token = claimed_items[0].claim_token
    assert original_token is not None

    # An abandoned lease is recovered without creating a second row.
    recovered = store.claim_due_outbox(now_ts=claim_time + 121.0)
    assert [item.id for item in recovered] == [claimed_items[0].id]
    recovered_token = recovered[0].claim_token
    assert recovered_token is not None
    assert recovered_token != original_token

    # The expired worker cannot release or complete the new worker's lease.
    assert store.mark_outbox_failed(recovered[0].id, original_token, 1, claim_time + 122.0) is False
    assert store.mark_outbox_delivered(recovered[0].id, original_token) is False
    assert store.claim_due_outbox(now_ts=claim_time + 122.0) == []


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
    assert claimed[0].claim_token is not None
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(notification_outbox)")}
    assert "claim_token" in columns


def test_initialize_upgrades_v2_outbox_and_recovers_active_leases(tmp_path: Path) -> None:
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
                    CHECK (status IN ('pending', 'sending', 'delivered', 'cancelled')),
                attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                next_attempt_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                delivered_at REAL,
                last_error TEXT,
                UNIQUE (event_key, chat_id)
            );
            INSERT INTO notification_outbox
                (id, event_key, chat_id, kind, text, status, attempts,
                 next_attempt_at, created_at, updated_at, delivered_at, last_error)
            VALUES
                (10, 'pending-event', 101, 'completion', 'Pending', 'pending', 2, 1, 1, 1, NULL, 'retry'),
                (11, 'leased-event', 101, 'completion', 'Sending', 'sending', 3, 2, 1, 1, NULL, NULL),
                (12, 'delivered-event', 101, 'completion', 'Delivered', 'delivered', 1, 1, 1, 1, 1, NULL);
            PRAGMA user_version=2;
            """
        )

    store = SQLiteStateStore(
        db_path,
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )
    store.initialize()

    with sqlite3.connect(db_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
        rows = connection.execute(
            "SELECT id, status, claim_token, attempts, last_error FROM notification_outbox ORDER BY id"
        ).fetchall()
    assert version == (3,)
    assert rows == [
        (10, "pending", None, 2, "retry"),
        (11, "pending", None, 3, None),
        (12, "delivered", None, 1, None),
    ]

    claimed = store.claim_due_outbox(now_ts=10.0, limit=10)
    assert [item.id for item in claimed] == [10, 11]
    assert claimed[0].claim_token is not None
    assert {item.claim_token for item in claimed} == {claimed[0].claim_token}
    assert store.enqueue_outbox("new-event", 101, "completion", "New") is True
    with sqlite3.connect(db_path) as connection:
        new_id = connection.execute("SELECT id FROM notification_outbox WHERE event_key = 'new-event'").fetchone()
    assert new_id is not None
    assert int(new_id[0]) > 12


def test_initialize_rejects_unfinished_outbox_migration_without_data_loss(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE notification_outbox_pre_v3 (
                id INTEGER PRIMARY KEY,
                event_key TEXT NOT NULL
            );
            INSERT INTO notification_outbox_pre_v3 (id, event_key)
            VALUES (7, 'preserve-me');
            PRAGMA user_version=2;
            """
        )

    store = SQLiteStateStore(
        db_path,
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )
    with pytest.raises(StateStoreError, match="notification_outbox_pre_v3 already exists"):
        store.initialize()

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert connection.execute("SELECT id, event_key FROM notification_outbox_pre_v3").fetchone() == (
            7,
            "preserve-me",
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'notification_outbox'"
            ).fetchone()
            is None
        )


def test_v3_outbox_survives_v2_rollback_and_upgrade_cycle(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim_time = 10_000_000_000.0
    store.enqueue_outbox("active-event", 101, "completion", "Active")
    active = store.claim_due_outbox(now_ts=claim_time, limit=1)[0]
    assert active.claim_token is not None
    store.enqueue_outbox("pending-event", 101, "completion", "Pending")
    store.enqueue_outbox("cancel-event", 202, "completion", "Cancel")

    with sqlite3.connect(store.db_path) as connection:
        table_sql = str(
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notification_outbox'"
            ).fetchone()[0]
        )
        assert "'sending'" not in table_sql
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE notification_outbox SET status = 'sending' WHERE event_key = 'pending-event'")

    _emulate_v2_rollback_initialize(store.db_path)

    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(notification_outbox)")}
        assert "claim_token" not in columns
        indexes = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'notification_outbox'"
            )
        }
        assert "notification_outbox_claim_idx" not in indexes
        rows = connection.execute(
            "SELECT id, event_key, status, attempts, text FROM notification_outbox ORDER BY id"
        ).fetchall()
        assert rows == [
            (active.id, "active-event", "sending", 0, "Active"),
            (active.id + 1, "pending-event", "pending", 0, "Pending"),
            (active.id + 2, "cancel-event", "pending", 0, "Cancel"),
        ]

        cancelled = connection.execute(
            """
            UPDATE notification_outbox
               SET status = 'cancelled', updated_at = ?
             WHERE chat_id = ? AND status IN ('pending', 'sending')
            """,
            (claim_time + 1.0, 202),
        )
        assert cancelled.rowcount == 1
        connection.commit()

        claimed_ids = _emulate_v2_claim(connection, claim_time + 121.0)
        assert set(claimed_ids) == {active.id, active.id + 1}
        failed = connection.execute(
            """
            UPDATE notification_outbox
               SET status = 'pending',
                   attempts = MAX(attempts, ?),
                   next_attempt_at = ?,
                   updated_at = ?,
                   last_error = ?
             WHERE id = ? AND status IN ('pending', 'sending')
            """,
            (4, claim_time + 500.0, claim_time + 121.0, "old failure", active.id),
        )
        delivered = connection.execute(
            """
            UPDATE notification_outbox
               SET status = 'delivered', delivered_at = ?, updated_at = ?, last_error = NULL
             WHERE id = ? AND status IN ('pending', 'sending')
            """,
            (claim_time + 121.0, claim_time + 121.0, active.id + 1),
        )
        assert failed.rowcount == 1
        assert delivered.rowcount == 1
        connection.commit()

    upgraded = _store(tmp_path)
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(notification_outbox)")}
        assert "claim_token" in columns
        rows = connection.execute(
            "SELECT id, status, claim_token, attempts, last_error FROM notification_outbox ORDER BY id"
        ).fetchall()
    assert rows == [
        (active.id, "pending", None, 4, "old failure"),
        (active.id + 1, "delivered", None, 0, None),
        (active.id + 2, "cancelled", None, 0, None),
    ]

    retry = upgraded.claim_due_outbox(now_ts=claim_time + 500.0, limit=1)[0]
    retry_token = retry.claim_token
    assert retry.id == active.id
    assert retry_token is not None
    assert upgraded.mark_outbox_delivered(retry.id, "stale-token") is False
    assert upgraded.mark_outbox_delivered(retry.id, retry_token) is True


def test_initialize_rejects_future_schema_without_downgrading(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE future_sentinel (value TEXT NOT NULL);
            INSERT INTO future_sentinel (value) VALUES ('preserve-me');
            PRAGMA user_version=4;
            """
        )

    store = SQLiteStateStore(
        db_path,
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )
    with pytest.raises(StateStoreError, match="newer than supported"):
        store.initialize()

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (4,)
        assert connection.execute("SELECT value FROM future_sentinel").fetchone() == ("preserve-me",)
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'metadata'").fetchone()
            is None
        )


def test_database_sidecars_are_private_and_connections_use_full_sync(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    db_path.write_bytes(b"")
    db_path.chmod(0o644)
    store = SQLiteStateStore(
        db_path,
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )
    store.initialize()

    connection = store._connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert db_path.stat().st_mode & 0o777 == 0o600
        assert Path("{}-wal".format(db_path)).stat().st_mode & 0o777 == 0o600
        assert Path("{}-shm".format(db_path)).stat().st_mode & 0o777 == 0o600
    finally:
        connection.rollback()
        connection.close()


def test_database_path_must_not_be_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "unrelated.txt"
    target.write_text("preserve me", encoding="utf-8")
    target.chmod(0o644)
    db_path = tmp_path / "state.sqlite3"
    db_path.symlink_to(target)
    store = SQLiteStateStore(
        db_path,
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )

    with pytest.raises(StateStoreError, match="regular file"):
        store.initialize()

    assert target.read_text(encoding="utf-8") == "preserve me"
    assert target.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_timestamps_and_json_numbers_are_rejected(tmp_path: Path, value: float) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="finite number"):
        store.list_due_outbox(now_ts=value)
    with pytest.raises(ValueError, match="finite number"):
        store.claim_due_outbox(lease_seconds=value)
    with pytest.raises(ValueError, match="finite number"):
        store.save_monitor_snapshot(
            "hash:abc",
            completed=False,
            started=False,
            present=True,
            generation=0,
            updated_at=value,
        )
    with pytest.raises(ValueError, match="finite number"):
        store.add_start_watch("hash:abc", 101, 7, "Ubuntu", added_at=value)
    with pytest.raises(ValueError, match="Out of range float values"):
        store.save_torrent_history({"hash:abc": {"ratio": value}})


def test_large_outbox_batch_does_not_depend_on_sqlite_variable_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batch_size = 1_001
    with store._connection(write=True) as connection:
        connection.executemany(
            """
            INSERT INTO notification_outbox
                (event_key, chat_id, kind, text, status, attempts,
                 next_attempt_at, created_at, updated_at)
            VALUES (?, ?, 'completion', 'Done', 'pending', 0, 1, 1, 1)
            """,
            [("event:{}".format(index), index) for index in range(batch_size)],
        )

    claimed = store.claim_due_outbox(now_ts=2.0, limit=batch_size)
    assert len(claimed) == batch_size
    assert claimed[0].claim_token is not None
    assert {item.claim_token for item in claimed} == {claimed[0].claim_token}


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
