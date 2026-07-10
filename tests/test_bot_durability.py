from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram.error import TelegramError
from transmission_rpc.error import TransmissionConnectError

import bot
from state_store import SQLiteStateStore


def _store(tmp_path: Path) -> SQLiteStateStore:
    store = SQLiteStateStore(
        tmp_path / "state.sqlite3",
        legacy_traffic_path=tmp_path / "missing-traffic.json",
        legacy_torrent_history_path=tmp_path / "missing-history.json",
    )
    store.initialize()
    return store


def test_logical_traffic_counters_survive_transmission_reset(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(bot, "STATE_STORE", store)
    monkeypatch.setattr(bot, "TRAFFIC_STATE_LOCK", asyncio.Lock())
    now = datetime(2026, 7, 10, 12, 0, tzinfo=bot.ZoneInfo("Europe/Moscow"))

    async def exercise() -> None:
        await bot.update_traffic_state(now, 100, 40)
        await bot.update_traffic_state(now, 150, 70)
        await bot.update_traffic_state(now, 10, 5)
        await bot.update_traffic_state(now, 20, 8)

    asyncio.run(exercise())

    reopened = _store(tmp_path)
    anchors, history = reopened.load_traffic_state()
    assert anchors["_counter"] == {
        "key": "logical-v1",
        "last_downloaded": 20,
        "last_uploaded": 8,
        "logical_downloaded": 170,
        "logical_uploaded": 78,
    }
    assert anchors["day"] == {"key": "2026-07-10", "downloaded": 100, "uploaded": 40}
    assert history == [{"date": "2026-07-10", "downloaded": 100, "uploaded": 40}]
    assert bot._effective_traffic_totals(anchors, 20, 8) == (170, 78)


def test_tr_call_serializes_concurrent_rpc_calls(monkeypatch) -> None:
    monkeypatch.setattr(bot, "_TR_CALL_LOCK", None)
    client = object()
    monkeypatch.setattr(bot, "get_client", lambda: client)
    guard = threading.Lock()
    active = 0
    peak_active = 0

    def rpc_call(actual_client: object) -> str:
        nonlocal active, peak_active
        assert actual_client is client
        with guard:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return "ok"

    async def exercise() -> list[str]:
        return await asyncio.gather(*(bot.tr_call(rpc_call) for _ in range(6)))

    assert asyncio.run(exercise()) == ["ok"] * 6
    assert peak_active == 1


def test_tr_call_does_not_retry_non_retryable_connection_error(monkeypatch) -> None:
    monkeypatch.setattr(bot, "_TR_CALL_LOCK", None)
    monkeypatch.setattr(bot, "get_client", lambda: object())
    reset_calls = 0
    rpc_calls = 0

    def reset_client() -> None:
        nonlocal reset_calls
        reset_calls += 1

    def failing_call(_: object) -> None:
        nonlocal rpc_calls
        rpc_calls += 1
        raise TransmissionConnectError("offline")

    monkeypatch.setattr(bot, "_reset_client", reset_client)

    with pytest.raises(bot.TRCallError, match="connection failed"):
        asyncio.run(bot.tr_call(failing_call, retry_on_connection=False, operation="write"))

    assert rpc_calls == 1
    assert reset_calls == 1


def test_tr_call_retries_read_once_after_connection_error(monkeypatch) -> None:
    monkeypatch.setattr(bot, "_TR_CALL_LOCK", None)
    clients = iter((object(), object()))
    client_calls = 0
    reset_calls = 0
    rpc_calls = 0

    def get_client() -> object:
        nonlocal client_calls
        client_calls += 1
        return next(clients)

    def reset_client() -> None:
        nonlocal reset_calls
        reset_calls += 1

    def flaky_call(_: object) -> str:
        nonlocal rpc_calls
        rpc_calls += 1
        if rpc_calls == 1:
            raise TransmissionConnectError("offline")
        return "recovered"

    monkeypatch.setattr(bot, "get_client", get_client)
    monkeypatch.setattr(bot, "_reset_client", reset_client)

    result = asyncio.run(bot.tr_call(flaky_call, operation="read"))

    assert result == "recovered"
    assert rpc_calls == 2
    assert client_calls == 2
    assert reset_calls == 1


def test_completion_is_enqueued_once_and_failed_delivery_remains_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    store.save_monitor_snapshot(
        "hash:abcdef",
        completed=False,
        started=True,
        present=True,
        generation=0,
        updated_at=1.0,
    )
    monkeypatch.setattr(bot, "STATE_STORE", store)
    torrent = SimpleNamespace(
        id=7,
        hash_string="ABCDEF",
        name="Linux image",
        status="seeding",
        percent_done=1.0,
        downloaded_ever=1_000,
        rate_download=0,
    )

    class FailingBot:
        async def send_message(self, **_: object) -> None:
            raise TelegramError("temporary network failure")

    class SuccessfulBot:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_message(self, **kwargs: object) -> None:
            self.messages.append(kwargs)

    application = SimpleNamespace(bot_data={bot.NOTIFY_ENABLED_CHATS_KEY: {101}})
    processing_context = SimpleNamespace(application=application, bot=FailingBot())

    asyncio.run(bot._process_torrent_notifications(processing_context, [torrent]))

    reopened = _store(tmp_path)
    monkeypatch.setattr(bot, "STATE_STORE", reopened)
    asyncio.run(bot._process_torrent_notifications(processing_context, [torrent]))
    queued = reopened.list_due_outbox(now_ts=10**20)
    assert len(queued) == 1
    assert queued[0].event_key == "completion:hash:abcdef:1"
    assert queued[0].kind == "completion"

    asyncio.run(bot._deliver_outbox_item(processing_context, queued[0]))
    pending = reopened.list_due_outbox(now_ts=10**20)
    assert len(pending) == 1
    assert pending[0].status == "pending"
    assert pending[0].attempts == 1
    assert pending[0].last_error == "temporary network failure"

    successful_bot = SuccessfulBot()
    success_context = SimpleNamespace(application=application, bot=successful_bot)
    asyncio.run(bot._deliver_outbox_item(success_context, pending[0]))
    assert reopened.list_due_outbox(now_ts=10**20) == []
    assert successful_bot.messages[0]["chat_id"] == 101


def test_timezone_config_prefers_bot_timezone_and_rejects_unknown_zone(monkeypatch) -> None:
    monkeypatch.setenv("TZ", "America/New_York")
    monkeypatch.setenv("BOT_TIMEZONE", "Europe/Moscow")

    name, timezone = bot._parse_timezone_env()

    assert name == "Europe/Moscow"
    assert timezone.key == "Europe/Moscow"
    assert datetime(2026, 1, 1, tzinfo=timezone).utcoffset().total_seconds() == 3 * 60 * 60

    monkeypatch.setenv("BOT_TIMEZONE", "Invalid/Definitely-Unknown")
    with pytest.raises(RuntimeError, match="BOT_TIMEZONE has unknown timezone"):
        bot._parse_timezone_env()
