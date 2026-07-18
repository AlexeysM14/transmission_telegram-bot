from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from telegram.error import BadRequest

import bot


class _Query:
    def __init__(self, data: str = "") -> None:
        self.data = data
        self.answers: list[tuple[str | None, bool]] = []
        self.edits: list[dict[str, Any]] = []

    async def answer(self, text: str | None = None, *, show_alert: bool = False) -> None:
        self.answers.append((text, show_alert))

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edits.append(kwargs)


def _torrent(torrent_id: int, *, name: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=torrent_id,
        name=name or f"Torrent {torrent_id:02d}",
        status="stopped",
        progress=0.0,
        total_size=1_024,
        left_until_done=1_024,
        rate_download=0,
        rate_upload=0,
        upload_ratio=0.0,
    )


def _install_torrent_source(
    monkeypatch: pytest.MonkeyPatch,
    torrents: list[SimpleNamespace],
    *,
    list_limit: int = 8,
) -> None:
    async def fake_tr_call(_: Any, **__: Any) -> list[SimpleNamespace]:
        return torrents

    async def fake_sync(_: list[SimpleNamespace], *, mark_missing: bool = True) -> list[dict[str, Any]]:
        del mark_missing
        return []

    monkeypatch.setattr(bot, "CFG", SimpleNamespace(list_limit=list_limit))
    monkeypatch.setattr(bot, "tr_call", fake_tr_call)
    monkeypatch.setattr(bot, "sync_torrent_history", fake_sync)


async def _render_page(ctx: Any, *, page: int, query: str | None = None) -> dict[str, Any]:
    callback_query = _Query()
    update: Any = SimpleNamespace(callback_query=callback_query)
    await bot.send_torrent_list(
        update,
        ctx,
        mode="all",
        query=query,
        page=page,
        edit_existing=True,
    )
    assert len(callback_query.edits) == 1
    return callback_query.edits[0]


def _action_ids(markup: Any) -> list[int]:
    ids: list[int] = []
    for row in markup.inline_keyboard:
        for button in row:
            data = button.callback_data or ""
            if not data.startswith(f"{bot.TORRENT_ACTION_CB_PREFIX}start:"):
                continue
            _, torrent_id, _, _ = bot._parse_torrent_action_payload(data[len(bot.TORRENT_ACTION_CB_PREFIX) :])
            ids.append(torrent_id)
    return ids


def _navigation_callbacks(markup: Any) -> list[str]:
    for row in markup.inline_keyboard:
        if any("/" in button.text for button in row):
            return [str(button.callback_data) for button in row]
    raise AssertionError("pagination row not found")


def _button_callback(markup: Any, text: str) -> str:
    for row in markup.inline_keyboard:
        for button in row:
            if button.text == text:
                assert button.callback_data is not None
                return button.callback_data
    raise AssertionError(f"button {text!r} not found")


def test_torrent_list_paginates_17_items_and_builds_page_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torrent_source(monkeypatch, [_torrent(index) for index in range(1, 18)])
    ctx: Any = SimpleNamespace(user_data={})

    async def exercise() -> list[dict[str, Any]]:
        return [await _render_page(ctx, page=page) for page in range(3)]

    first, middle, last = asyncio.run(exercise())

    assert _action_ids(first["reply_markup"]) == list(range(1, 9))
    assert _action_ids(middle["reply_markup"]) == list(range(9, 17))
    assert _action_ids(last["reply_markup"]) == [17]
    assert "Страница <b>1</b> из <b>3</b>" in first["text"]
    assert "Страница <b>2</b> из <b>3</b>" in middle["text"]
    assert "Страница <b>3</b> из <b>3</b>" in last["text"]

    prefix = bot.LIST_REFRESH_CB_PREFIX
    assert _navigation_callbacks(first["reply_markup"]) == [f"{prefix}all:0", f"{prefix}all:1"]
    assert _navigation_callbacks(middle["reply_markup"]) == [
        f"{prefix}all:0",
        f"{prefix}all:1",
        f"{prefix}all:2",
    ]
    assert _navigation_callbacks(last["reply_markup"]) == [f"{prefix}all:1", f"{prefix}all:2"]


def test_torrent_list_clamps_page_past_last(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torrent_source(monkeypatch, [_torrent(index) for index in range(1, 18)])
    ctx: Any = SimpleNamespace(user_data={})

    rendered = asyncio.run(_render_page(ctx, page=10_000))
    markup = rendered["reply_markup"]

    assert _action_ids(markup) == [17]
    assert "Страница <b>3</b> из <b>3</b>" in rendered["text"]
    assert _button_callback(markup, "🔄 Обновить") == f"{bot.LIST_REFRESH_CB_PREFIX}all:2"
    assert _navigation_callbacks(markup) == [
        f"{bot.LIST_REFRESH_CB_PREFIX}all:1",
        f"{bot.LIST_REFRESH_CB_PREFIX}all:2",
    ]


def test_legacy_list_and_action_payloads_default_to_first_page() -> None:
    assert bot._parse_list_view_payload("all") == ("all", 0)
    assert bot._parse_list_view_payload("downloading") == ("downloading", 0)
    assert bot._parse_list_view_payload("search.abcdef12:2") == ("search.abcdef12", 2)
    assert bot._parse_torrent_action_payload("pause:7:all") == ("pause", 7, "all", 0)
    assert bot._parse_torrent_action_payload("start:12:search") == ("start", 12, "search", 0)


def test_search_context_is_preserved_by_refresh_and_action_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torrents = [_torrent(index, name=f"Needle {index:02d}") for index in range(1, 11)]
    _install_torrent_source(monkeypatch, torrents)
    ctx: Any = SimpleNamespace(user_data={})
    rendered = asyncio.run(_render_page(ctx, page=1, query="  Needle  "))
    markup = rendered["reply_markup"]

    refresh_payload = _button_callback(markup, "🔄 Обновить")
    action_payload = next(
        str(button.callback_data)
        for row in markup.inline_keyboard
        for button in row
        if str(button.callback_data).startswith(f"{bot.TORRENT_ACTION_CB_PREFIX}start:")
    )
    search_revision = ctx.user_data[bot.TORRENT_LIST_SEARCH_REVISION_KEY]
    search_view = f"{bot.TORRENT_LIST_SEARCH_VIEW_PREFIX}{search_revision}"
    assert refresh_payload == f"{bot.LIST_REFRESH_CB_PREFIX}{search_view}:1"
    assert action_payload == f"{bot.TORRENT_ACTION_CB_PREFIX}start:9:{search_view}:1"
    assert ctx.user_data[bot.TORRENT_LIST_LAST_MODE_KEY] == "all"
    assert ctx.user_data[bot.TORRENT_LIST_LAST_QUERY_KEY] == "Needle"
    assert isinstance(search_revision, str)

    list_calls: list[dict[str, Any]] = []
    running_state_calls: list[tuple[str, int]] = []

    async def allow_callback(_: Any) -> bool:
        return True

    async def fake_send_torrent_list(_: Any, __: Any, **kwargs: Any) -> None:
        list_calls.append(kwargs)

    async def fake_set_running_state(action: str, torrent_id: int) -> str:
        running_state_calls.append((action, torrent_id))
        return "done"

    monkeypatch.setattr(bot, "callback_user_allowed", allow_callback)
    monkeypatch.setattr(bot, "send_torrent_list", fake_send_torrent_list)
    monkeypatch.setattr(bot, "_set_torrent_running_state", fake_set_running_state)

    refresh_query = _Query(refresh_payload)
    refresh_update: Any = SimpleNamespace(callback_query=refresh_query)
    asyncio.run(bot.on_list_refresh(refresh_update, ctx))

    assert list_calls == [
        {
            "mode": "all",
            "query": "Needle",
            "search_revision": search_revision,
            "page": 1,
            "edit_existing": True,
        }
    ]

    list_calls.clear()
    action_query = _Query(action_payload)
    action_update: Any = SimpleNamespace(callback_query=action_query)
    asyncio.run(bot.on_torrent_action(action_update, ctx))

    assert running_state_calls == [("start", 9)]
    assert list_calls == [
        {
            "mode": "all",
            "query": "Needle",
            "search_revision": search_revision,
            "page": 1,
            "edit_existing": True,
        }
    ]


def test_unchanged_torrent_list_edit_is_ignored() -> None:
    class UnchangedQuery:
        async def edit_message_text(self, **_: Any) -> None:
            raise BadRequest("Message is not modified")

    asyncio.run(
        bot._edit_torrent_list_message(
            UnchangedQuery(),
            text="same",
            reply_markup=bot.TORRENT_LIST_KEYBOARD,
        )
    )


def test_previous_search_callbacks_expire_after_a_new_search(monkeypatch: pytest.MonkeyPatch) -> None:
    torrents = [
        _torrent(1, name="Alpha release"),
        _torrent(2, name="Beta release"),
    ]
    _install_torrent_source(monkeypatch, torrents)
    ctx: Any = SimpleNamespace(user_data={})

    first_render = asyncio.run(_render_page(ctx, page=0, query="Alpha"))
    old_refresh_payload = _button_callback(first_render["reply_markup"], "🔄 Обновить")
    asyncio.run(_render_page(ctx, page=0, query="Beta"))

    list_calls: list[dict[str, Any]] = []

    async def allow_callback(_: Any) -> bool:
        return True

    async def fake_send_torrent_list(_: Any, __: Any, **kwargs: Any) -> None:
        list_calls.append(kwargs)

    monkeypatch.setattr(bot, "callback_user_allowed", allow_callback)
    monkeypatch.setattr(bot, "send_torrent_list", fake_send_torrent_list)
    old_query = _Query(old_refresh_payload)
    update: Any = SimpleNamespace(callback_query=old_query)

    asyncio.run(bot.on_list_refresh(update, ctx))

    assert list_calls == []
    assert old_query.answers[-1] == ("Поиск устарел. Запусти его заново.", True)


def test_long_search_query_is_shortened_before_html_rendering() -> None:
    raw_query = "&<" * 2_048

    rendered_query = bot._format_search_query_for_display(raw_query)
    empty_text = bot._build_empty_torrent_list_text("all", raw_query)

    assert rendered_query.endswith("…")
    assert "&amp;" in rendered_query
    assert "&lt;" in rendered_query
    assert len(empty_text) < bot.TG_MAX_MESSAGE


def test_inline_action_answers_callback_before_transmission_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class OrderedQuery(_Query):
        async def answer(self, text: str | None = None, *, show_alert: bool = False) -> None:
            del text, show_alert
            events.append("answer")

    async def allow_callback(_: Any) -> bool:
        return True

    async def fake_set_running_state(_: str, __: int) -> str:
        events.append("rpc")
        return "done"

    async def fake_send_torrent_list(_: Any, __: Any, **___: Any) -> None:
        events.append("refresh")

    monkeypatch.setattr(bot, "callback_user_allowed", allow_callback)
    monkeypatch.setattr(bot, "_set_torrent_running_state", fake_set_running_state)
    monkeypatch.setattr(bot, "send_torrent_list", fake_send_torrent_list)
    update: Any = SimpleNamespace(callback_query=OrderedQuery(f"{bot.TORRENT_ACTION_CB_PREFIX}pause:7:all:0"))
    ctx: Any = SimpleNamespace(user_data={})

    asyncio.run(bot.on_torrent_action(update, ctx))

    assert events == ["answer", "rpc", "refresh"]


def test_cancel_text_clears_pending_delete_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    replies: list[str] = []
    control_calls: list[tuple[str, int]] = []

    async def fake_initialize(_: Any, __: int | None) -> None:
        return None

    async def fake_ephemeral(_: Any, __: Any, text: str, reply_markup: Any) -> None:
        del reply_markup
        replies.append(text)

    async def fake_delete_user_message(_: Any, __: Any) -> None:
        return None

    async def allow_callback(_: Any) -> bool:
        return True

    async def fake_ctrl_action(_: Any, __: Any, action: str, torrent_id: int, **___: Any) -> None:
        control_calls.append((action, torrent_id))

    monkeypatch.setattr(bot, "user_allowed", lambda _: True)
    monkeypatch.setattr(bot, "_ensure_chat_notifications_initialized", fake_initialize)
    monkeypatch.setattr(bot, "send_ephemeral", fake_ephemeral)
    monkeypatch.setattr(bot, "_delete_user_message", fake_delete_user_message)
    monkeypatch.setattr(bot, "callback_user_allowed", allow_callback)
    monkeypatch.setattr(bot, "ctrl_action", fake_ctrl_action)

    ctx: Any = SimpleNamespace(
        user_data={
            "menu": bot.MENU_CTRL,
            "wait": bot.WAIT_NONE,
            bot.PENDING_CTRL_ACTION_KEY: {"action": "del_data", "torrent_id": 7},
        }
    )
    update: Any = SimpleNamespace(
        effective_chat=SimpleNamespace(id=101, type="private"),
        effective_message=SimpleNamespace(text=bot.CANCEL_INPUT_BUTTON),
    )

    asyncio.run(bot.on_text(update, ctx))

    assert bot.PENDING_CTRL_ACTION_KEY not in ctx.user_data
    assert ctx.user_data["menu"] == bot.MENU_MAIN
    assert "Удаление отменено" in replies[-1]

    confirm_query = _Query(f"{bot.CONFIRM_DEL_DATA_CB_PREFIX}7")
    confirm_update: Any = SimpleNamespace(
        effective_chat=SimpleNamespace(id=101, type="private"),
        callback_query=confirm_query,
    )
    asyncio.run(bot.on_delete_confirmation(confirm_update, ctx))

    assert control_calls == []
    assert confirm_query.answers[-1] == ("Запрос подтверждения устарел", True)
