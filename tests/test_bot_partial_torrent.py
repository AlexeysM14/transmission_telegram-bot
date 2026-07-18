from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from transmission_rpc.torrent import Torrent

import bot
from state_store import SQLiteStateStore


def _partial_torrent(
    *,
    status: int = 4,
    downloaded_ever: int = 0,
    percent_done: float = 0.0,
    rate_download: int | None = 0,
    hash_string: str = "ABCDEF0123456789ABCDEF0123456789ABCDEF01",
) -> Torrent:
    fields: dict[str, Any] = {
        "id": 7,
        "hashString": hash_string,
        "name": "Linux image",
        "status": status,
        "totalSize": 1_000,
        "sizeWhenDone": 1_000,
        "downloadedEver": downloaded_ever,
        "uploadedEver": 0,
        "uploadRatio": 0.0,
        "percentDone": percent_done,
        "leftUntilDone": max(0, int(1_000 * (1.0 - percent_done))),
        "addedDate": 1,
        "doneDate": 0,
    }
    if rate_download is not None:
        fields["rateDownload"] = rate_download
    return Torrent(fields=fields)


def _minimal_add_response() -> Torrent:
    return Torrent(
        fields={
            "id": 7,
            "hashString": "ABCDEF0123456789ABCDEF0123456789ABCDEF01",
            "name": "Linux image",
        }
    )


@pytest.mark.parametrize(
    ("status", "downloaded_ever", "percent_done", "rate_download", "expected"),
    [
        (4, 0, 0.0, None, False),
        (4, 0, 0.0, 0, False),
        (3, 0, 0.0, 0, False),
        (0, 1, 0.0, 0, True),
        (2, 0, 0.01, 0, True),
        (0, 0, 0.0, 512, True),
    ],
)
def test_start_detection_uses_activity_not_status(
    status: int,
    downloaded_ever: int,
    percent_done: float,
    rate_download: int | None,
    expected: bool,
) -> None:
    torrent = _partial_torrent(
        status=status,
        downloaded_ever=downloaded_ever,
        percent_done=percent_done,
        rate_download=rate_download,
    )

    assert bot._torrent_start_detected(torrent) is expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (4, True),
        (3, False),
        (0, False),
        (2, False),
        (6, False),
    ],
)
def test_attempting_download_requires_downloading_status(status: int, expected: bool) -> None:
    assert bot._torrent_is_attempting_download(_partial_torrent(status=status)) is expected


@pytest.mark.parametrize(
    ("status", "expect_no_peers"),
    [
        (4, True),
        (3, False),
        (0, False),
        (2, False),
    ],
)
def test_no_peers_notification_only_for_attempting_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expect_no_peers: bool,
) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    store.initialize()
    torrent = _partial_torrent(status=status, rate_download=None)
    torrent_key = bot._notification_torrent_key(torrent)
    assert torrent_key is not None
    store.add_start_watch(
        torrent_key,
        chat_id=101,
        torrent_id=torrent.id,
        name=torrent.name,
        added_at=time.time() - bot.NOTIFY_NO_PEERS_DELAY_SEC - 1,
    )
    monkeypatch.setattr(bot, "STATE_STORE", store)
    context: Any = SimpleNamespace(
        application=SimpleNamespace(bot_data={bot.NOTIFY_ENABLED_CHATS_KEY: {101}}),
        bot=SimpleNamespace(),
    )

    asyncio.run(bot._process_torrent_notifications(context, [torrent]))

    queued_kinds = [item.kind for item in store.list_due_outbox(now_ts=10**20)]
    assert ("no_peers" in queued_kinds) is expect_no_peers


def test_minimal_add_response_is_safe_for_partial_field_helpers() -> None:
    torrent = _minimal_add_response()

    assert bot.torrent_progress_percent(torrent) == 0.0
    assert bot.torrent_total_size(torrent) == 0
    assert bot._torrent_left_until_done(torrent) is None
    assert bot._torrent_start_detected(torrent) is False
    assert "не удалось рассчитать" in bot._build_projected_free_space_text(10_000, torrent)


@pytest.mark.parametrize("source", ["url", "file"])
def test_add_reports_success_when_details_refresh_fails(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    added_torrent = _minimal_add_response()
    operations: list[str] = []
    registered: list[Torrent] = []
    synchronized: list[Torrent] = []
    scheduled: list[Torrent] = []
    replies: list[str] = []

    async def fake_tr_call(
        _: Any,
        *,
        retry_on_connection: bool = True,
        operation: str = "rpc",
    ) -> Any:
        del retry_on_connection
        operations.append(operation)
        if operation in {"add_torrent_url", "add_torrent_file"}:
            return added_torrent
        if source == "url":
            raise KeyError("torrent disappeared")
        raise bot.TRCallError("details unavailable")

    async def fake_free_space() -> int:
        return 10_000

    async def fake_register(_: Any, __: int | None, torrent: Torrent) -> None:
        registered.append(torrent)

    async def fake_sync(torrents: list[Torrent], *, mark_missing: bool = True) -> list[dict[str, Any]]:
        del mark_missing
        synchronized.extend(torrents)
        return []

    async def fake_reply(_: Any, text: str, **__: Any) -> None:
        replies.append(text)

    def fake_schedule(_: Any, torrent: Torrent) -> None:
        scheduled.append(torrent)

    class FakeTelegramFile:
        async def download_to_drive(self, *, custom_path: str) -> None:
            Path(custom_path).write_bytes(b"d4:test4:datae")

    class FakeDocument:
        file_name = "sample.torrent"
        file_size = 14

        async def get_file(self) -> FakeTelegramFile:
            return FakeTelegramFile()

    monkeypatch.setattr(bot, "tr_call", fake_tr_call)
    monkeypatch.setattr(bot, "_get_download_dir_free_space", fake_free_space)
    monkeypatch.setattr(bot, "_register_torrent_start_watch", fake_register)
    monkeypatch.setattr(bot, "sync_torrent_history", fake_sync)
    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    monkeypatch.setattr(bot, "_schedule_torrent_start_watch", fake_schedule)

    update: Any = SimpleNamespace(
        effective_chat=SimpleNamespace(id=101),
        effective_message=SimpleNamespace(document=FakeDocument()),
    )
    context: Any = SimpleNamespace()
    if source == "url":
        asyncio.run(bot.add_magnet_or_url(update, context, "magnet:?xt=urn:btih:abcdef"))
    else:
        asyncio.run(bot.add_torrent_file(update, context))

    assert registered == [added_torrent]
    assert synchronized == []
    assert scheduled == [added_torrent]
    assert operations[0] == f"add_torrent_{source}"
    assert operations[1] == f"hydrate_added_torrent_{source}"
    assert len(replies) == 1
    assert replies[0].startswith("✅ Торрент добавлен")
    assert "свежие детали пока не удалось получить" in replies[0]
    assert "не удалось рассчитать" in replies[0]


def test_add_reports_success_when_ancillary_state_operations_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    added_torrent = _minimal_add_response()
    hydrated_torrent = _partial_torrent()
    replies: list[str] = []

    async def fake_tr_call(_: Any, **kwargs: Any) -> Torrent:
        if kwargs.get("operation") == "add_torrent_url":
            return added_torrent
        return hydrated_torrent

    async def fake_free_space() -> int:
        return 10_000

    async def fail_register(_: Any, __: int | None, ___: Torrent) -> None:
        raise RuntimeError("state unavailable")

    async def fail_sync(_: list[Torrent], *, mark_missing: bool = True) -> list[dict[str, Any]]:
        del mark_missing
        raise RuntimeError("history unavailable")

    async def fake_reply(_: Any, text: str, **__: Any) -> None:
        replies.append(text)

    def fail_schedule(_: Any, __: Torrent) -> None:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(bot, "tr_call", fake_tr_call)
    monkeypatch.setattr(bot, "_get_download_dir_free_space", fake_free_space)
    monkeypatch.setattr(bot, "_register_torrent_start_watch", fail_register)
    monkeypatch.setattr(bot, "sync_torrent_history", fail_sync)
    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    monkeypatch.setattr(bot, "_schedule_torrent_start_watch", fail_schedule)

    update: Any = SimpleNamespace(
        effective_chat=SimpleNamespace(id=101),
        effective_message=SimpleNamespace(),
    )
    context: Any = SimpleNamespace()
    asyncio.run(bot.add_magnet_or_url(update, context, "magnet:?xt=urn:btih:abcdef"))

    assert len(replies) == 1
    assert replies[0].startswith("✅ Торрент добавлен")
    assert "Linux image" in replies[0]


def test_hydration_uses_stable_hash_and_returns_full_torrent(monkeypatch: pytest.MonkeyPatch) -> None:
    added_torrent = _minimal_add_response()
    hydrated_torrent = _partial_torrent(rate_download=0)
    selectors: list[str | int] = []

    class FakeClient:
        def get_torrent(self, selector: str | int) -> Torrent:
            selectors.append(selector)
            return hydrated_torrent

    async def fake_tr_call(fn: Any, **_: Any) -> Torrent:
        return fn(FakeClient())

    monkeypatch.setattr(bot, "tr_call", fake_tr_call)

    result, refreshed = asyncio.run(bot._hydrate_added_torrent(added_torrent, operation="hydrate_test"))

    assert refreshed is True
    assert result is hydrated_torrent
    assert selectors == ["abcdef0123456789abcdef0123456789abcdef01"]
    assert "rateDownload" in bot.TORRENT_HISTORY_FIELDS


def test_torrent_file_declared_size_is_rejected_before_download(monkeypatch: pytest.MonkeyPatch) -> None:
    replies: list[str] = []
    get_file_called = False

    class OversizedDocument:
        file_name = "oversized.torrent"
        file_size = bot.TORRENT_FILE_MAX_BYTES + 1

        async def get_file(self) -> None:
            nonlocal get_file_called
            get_file_called = True

    async def fake_reply(_: Any, text: str, **__: Any) -> None:
        replies.append(text)

    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    update: Any = SimpleNamespace(
        effective_chat=SimpleNamespace(id=101),
        effective_message=SimpleNamespace(document=OversizedDocument()),
    )
    context: Any = SimpleNamespace()

    asyncio.run(bot.add_torrent_file(update, context))

    assert get_file_called is False
    assert len(replies) == 1
    assert "слишком большой" in replies[0]
    assert "10 MiB" in replies[0]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (b"", "пустой"),
        (b"12345", "превышает лимит"),
    ],
)
def test_torrent_file_actual_size_is_validated_after_download(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    expected_error: str,
) -> None:
    replies: list[str] = []
    downloaded_paths: list[Path] = []
    rpc_called = False

    class FakeTelegramFile:
        async def download_to_drive(self, *, custom_path: str) -> None:
            path = Path(custom_path)
            path.write_bytes(payload)
            downloaded_paths.append(path)

    class UnknownSizeDocument:
        file_name = "sample.torrent"
        file_size = None

        async def get_file(self) -> FakeTelegramFile:
            return FakeTelegramFile()

    async def fake_free_space() -> None:
        return None

    async def fake_tr_call(_: Any, **__: Any) -> None:
        nonlocal rpc_called
        rpc_called = True

    async def fake_reply(_: Any, text: str, **__: Any) -> None:
        replies.append(text)

    monkeypatch.setattr(bot, "TORRENT_FILE_MAX_BYTES", 4)
    monkeypatch.setattr(bot, "_get_download_dir_free_space", fake_free_space)
    monkeypatch.setattr(bot, "tr_call", fake_tr_call)
    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    update: Any = SimpleNamespace(
        effective_chat=SimpleNamespace(id=101),
        effective_message=SimpleNamespace(document=UnknownSizeDocument()),
    )
    context: Any = SimpleNamespace()

    asyncio.run(bot.add_torrent_file(update, context))

    assert rpc_called is False
    assert len(replies) == 1
    assert expected_error in replies[0]
    assert len(downloaded_paths) == 1
    assert downloaded_paths[0].exists() is False


def test_torrent_file_get_file_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    replies: list[str] = []

    class BrokenDocument:
        file_name = "sample.torrent"
        file_size = None

        async def get_file(self) -> None:
            raise bot.TelegramError("download unavailable")

    async def fake_free_space() -> None:
        return None

    async def fake_reply(_: Any, text: str, **__: Any) -> None:
        replies.append(text)

    monkeypatch.setattr(bot, "_get_download_dir_free_space", fake_free_space)
    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    update: Any = SimpleNamespace(
        effective_chat=SimpleNamespace(id=101),
        effective_message=SimpleNamespace(document=BrokenDocument()),
    )
    context: Any = SimpleNamespace()

    asyncio.run(bot.add_torrent_file(update, context))

    assert len(replies) == 1
    assert "download unavailable" in replies[0]


def test_temporary_file_cleanup_error_does_not_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torrent_path = tmp_path / "sample.torrent"
    torrent_path.write_bytes(b"content")

    def fail_unlink(_: Path, *, missing_ok: bool = False) -> None:
        del missing_ok
        raise OSError("cleanup failed")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    bot._delete_temporary_file_safe(torrent_path)


class _DeleteClient:
    def __init__(self, torrent: Torrent) -> None:
        self.torrent = torrent
        self.removals: list[tuple[str | int, bool]] = []

    def get_torrent(self, _: str | int, arguments: Any = None) -> Torrent:
        del arguments
        return self.torrent

    def remove_torrent(self, selector: str | int, delete_data: bool = False) -> None:
        self.removals.append((selector, delete_data))


class _DeleteQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.answers: list[tuple[str | None, bool]] = []
        self.reply_markup_removed = False

    async def answer(self, text: str | None = None, *, show_alert: bool = False) -> None:
        self.answers.append((text, show_alert))

    async def edit_message_reply_markup(self, *, reply_markup: Any) -> None:
        self.reply_markup_removed = reply_markup is None


@pytest.mark.parametrize("replace_before_confirm", [False, True])
def test_confirmed_delete_is_bound_to_original_hash(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    replace_before_confirm: bool,
) -> None:
    original = _partial_torrent(hash_string="A" * 40)
    replacement = _partial_torrent(hash_string="B" * 40)
    client = _DeleteClient(original)
    replies: list[str] = []
    synchronized: list[Torrent] = []
    marked: list[tuple[Torrent, bool]] = []

    async def fake_tr_call(fn: Any, **_: Any) -> Any:
        return fn(client)

    async def fake_sync(torrents: list[Torrent], *, mark_missing: bool = True) -> list[dict[str, Any]]:
        del mark_missing
        synchronized.extend(torrents)
        return []

    async def fake_mark(torrent: Torrent, *, with_data: bool) -> None:
        marked.append((torrent, with_data))

    async def fake_reply(_: Any, text: str, **__: Any) -> None:
        replies.append(text)

    async def allow_callback(_: Any) -> bool:
        return True

    monkeypatch.setattr(bot, "tr_call", fake_tr_call)
    monkeypatch.setattr(bot, "sync_torrent_history", fake_sync)
    monkeypatch.setattr(bot, "mark_torrent_history_removed", fake_mark)
    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    monkeypatch.setattr(bot, "callback_user_allowed", allow_callback)
    context: Any = SimpleNamespace(
        user_data={},
        application=SimpleNamespace(bot_data={bot.NOTIFY_ENABLED_CHATS_KEY: set()}),
    )
    request_update: Any = SimpleNamespace(effective_chat=SimpleNamespace(id=101))

    asyncio.run(
        bot._request_delete_confirmation(
            request_update,
            context,
            action="del_data",
            torrent_id=original.id,
        )
    )

    pending = context.user_data[bot.PENDING_CTRL_ACTION_KEY]
    assert pending["torrent_hash"] == "a" * 40
    if replace_before_confirm:
        client.torrent = replacement

    query = _DeleteQuery(f"{bot.CONFIRM_DEL_DATA_CB_PREFIX}{original.id}")
    confirm_update: Any = SimpleNamespace(
        effective_chat=SimpleNamespace(id=101),
        callback_query=query,
    )
    asyncio.run(bot.on_delete_confirmation(confirm_update, context))

    assert bot.PENDING_CTRL_ACTION_KEY not in context.user_data
    assert query.reply_markup_removed is True
    if replace_before_confirm:
        assert client.removals == []
        assert synchronized == []
        assert marked == []
        assert "под этим ID уже другой торрент" in replies[-1]
    else:
        assert client.removals == [("a" * 40, True)]
        assert synchronized == [original]
        assert marked == [(original, True)]


def test_direct_delete_keep_uses_hash_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    torrent = _partial_torrent(hash_string="C" * 40)
    client = _DeleteClient(torrent)
    marked: list[tuple[Torrent, bool]] = []

    async def fake_tr_call(fn: Any, **_: Any) -> Any:
        return fn(client)

    async def fake_sync(_: list[Torrent], *, mark_missing: bool = True) -> list[dict[str, Any]]:
        del mark_missing
        return []

    async def fake_mark(actual: Torrent, *, with_data: bool) -> None:
        marked.append((actual, with_data))

    async def fake_reply(_: Any, __: str, **___: Any) -> None:
        return None

    monkeypatch.setattr(bot, "tr_call", fake_tr_call)
    monkeypatch.setattr(bot, "sync_torrent_history", fake_sync)
    monkeypatch.setattr(bot, "mark_torrent_history_removed", fake_mark)
    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    context: Any = SimpleNamespace(application=SimpleNamespace(bot_data={bot.NOTIFY_ENABLED_CHATS_KEY: set()}))
    update: Any = SimpleNamespace(effective_chat=SimpleNamespace(id=101))

    asyncio.run(bot.ctrl_action(update, context, "del_keep", torrent_id=torrent.id))

    assert client.removals == [("c" * 40, False)]
    assert marked == [(torrent, False)]


def test_successful_delete_is_reported_when_history_storage_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    torrent = _partial_torrent(hash_string="D" * 40)
    client = _DeleteClient(torrent)
    replies: list[str] = []

    async def fake_tr_call(fn: Any, **_: Any) -> Any:
        return fn(client)

    async def fail_sync(_: list[Torrent], *, mark_missing: bool = True) -> list[dict[str, Any]]:
        del mark_missing
        raise RuntimeError("history unavailable")

    async def fail_mark(_: Torrent, *, with_data: bool) -> None:
        del with_data
        raise RuntimeError("history unavailable")

    async def fake_reply(_: Any, text: str, **__: Any) -> None:
        replies.append(text)

    monkeypatch.setattr(bot, "tr_call", fake_tr_call)
    monkeypatch.setattr(bot, "sync_torrent_history", fail_sync)
    monkeypatch.setattr(bot, "mark_torrent_history_removed", fail_mark)
    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    context: Any = SimpleNamespace(application=SimpleNamespace(bot_data={bot.NOTIFY_ENABLED_CHATS_KEY: set()}))
    update: Any = SimpleNamespace(effective_chat=SimpleNamespace(id=101))

    asyncio.run(bot.ctrl_action(update, context, "del_keep", torrent_id=torrent.id))

    assert client.removals == [("d" * 40, False)]
    assert len(replies) == 1
    assert replies[0].startswith("🗑️ Удалено")


def test_missing_torrent_during_delete_is_reported_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    replies: list[str] = []

    async def missing_torrent(*_: Any, **__: Any) -> None:
        raise KeyError("Torrent not found")

    async def fake_reply(_: Any, text: str, **__: Any) -> None:
        replies.append(text)

    monkeypatch.setattr(bot, "tr_call", missing_torrent)
    monkeypatch.setattr(bot, "reply_chunks", fake_reply)
    context: Any = SimpleNamespace(application=SimpleNamespace(bot_data={bot.NOTIFY_ENABLED_CHATS_KEY: set()}))
    update: Any = SimpleNamespace(effective_chat=SimpleNamespace(id=101))

    asyncio.run(bot.ctrl_action(update, context, "del_keep", torrent_id=7))

    assert replies == ["❌ Торрент с ID 7 больше не найден."]
