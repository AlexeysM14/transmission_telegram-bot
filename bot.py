#!/usr/bin/env python3
"""Telegram bot for managing Transmission RPC via menu buttons."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from transmission_rpc import Client, from_url
from transmission_rpc.error import TransmissionError


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("tg-transmission-bot")



def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


load_dotenv_file(Path(__file__).resolve().with_name(".env"))

TG_MAX_MESSAGE = 4096
TORRENT_ID_RE = re.compile(r"\b(\d{1,9})\b")
_TR_CLIENT: Optional[Client] = None
_TR_CLIENT_LOCK = threading.Lock()


class TRCallError(Exception):
    """Wrapper for errors during Transmission RPC calls."""


@dataclass(frozen=True)
class Config:
    tg_token: str
    allowed_user_ids: Optional[set[int]]

    tr_url: Optional[str]
    tr_protocol: str
    tr_host: str
    tr_port: int
    tr_path: str
    tr_user: Optional[str]
    tr_pass: Optional[str]
    tr_timeout: float

    list_limit: int


def _parse_allowed_ids(raw: str) -> Optional[set[int]]:
    raw = raw.strip()
    if not raw:
        return None

    values: set[int] = set()
    ignored: list[str] = []

    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            values.add(int(token))
        else:
            ignored.append(token)

    if ignored:
        log.warning("Ignored invalid ALLOWED_USER_IDS entries: %s", ", ".join(ignored))

    if not values:
        log.warning("No valid ALLOWED_USER_IDS values found; access restriction disabled")
        return None

    return values


def _parse_int_env(name: str, default: str, *, min_value: int = 1, max_value: Optional[int] = None) -> int:
    raw = os.environ.get(name, default).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc

    if value < min_value or (max_value is not None and value > max_value):
        if max_value is None:
            raise RuntimeError(f"{name} must be >= {min_value}")
        raise RuntimeError(f"{name} must be in {min_value}..{max_value}")
    return value


def _parse_float_env(name: str, default: str, *, min_exclusive: float = 0.0) -> float:
    raw = os.environ.get(name, default).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc

    if value <= min_exclusive:
        raise RuntimeError(f"{name} must be > {min_exclusive:g}")
    return value


def load_config() -> Config:
    tg_token = os.environ.get("TG_TOKEN", "").strip()
    if not tg_token:
        raise RuntimeError("ENV TG_TOKEN is required")

    tr_protocol = os.environ.get("TR_PROTOCOL", "http").strip().lower()
    if tr_protocol not in {"http", "https"}:
        raise RuntimeError("TR_PROTOCOL must be 'http' or 'https'")

    tr_port = _parse_int_env("TR_PORT", "9091", min_value=1, max_value=65535)
    tr_timeout = _parse_float_env("TR_TIMEOUT", "10", min_exclusive=0.0)
    list_limit = _parse_int_env("LIST_LIMIT", "25", min_value=1)

    return Config(
        tg_token=tg_token,
        allowed_user_ids=_parse_allowed_ids(os.environ.get("ALLOWED_USER_IDS", "")),
        tr_url=os.environ.get("TR_URL", "").strip() or None,
        tr_protocol=tr_protocol,
        tr_host=os.environ.get("TR_HOST", "127.0.0.1").strip(),
        tr_port=tr_port,
        tr_path=os.environ.get("TR_PATH", "/transmission/rpc").strip(),
        tr_user=os.environ.get("TR_USER", "").strip() or None,
        tr_pass=os.environ.get("TR_PASS", "").strip() or None,
        tr_timeout=tr_timeout,
        list_limit=list_limit,
    )


CFG = load_config()

MENU_MAIN = "MAIN"
MENU_TORRENTS = "TORRENTS"
MENU_ADD = "ADD"
MENU_CTRL = "CTRL"

WAIT_NONE = None
WAIT_SEARCH = "WAIT_SEARCH"
WAIT_ADD_MAGNET = "WAIT_ADD_MAGNET"
WAIT_ADD_TORRENT_FILE = "WAIT_ADD_TORRENT_FILE"
WAIT_CTRL_PAUSE = "WAIT_CTRL_PAUSE"
WAIT_CTRL_START = "WAIT_CTRL_START"
WAIT_CTRL_DEL_KEEP = "WAIT_CTRL_DEL_KEEP"
WAIT_CTRL_DEL_DATA = "WAIT_CTRL_DEL_DATA"


def kb(rows: Sequence[Sequence[str]]) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        rows,
        resize_keyboard=True,
        one_time_keyboard=False,
        is_persistent=True,
    )


def kb_main() -> ReplyKeyboardMarkup:
    return kb(
        [
            ["📊 Статус", "📋 Торренты"],
            ["➕ Добавить", "⚙️ Управление"],
            ["ℹ️ Помощь"],
        ]
    )


def kb_torrents() -> ReplyKeyboardMarkup:
    return kb(
        [
            ["📋 Все", "▶️ Активные"],
            ["⏹️ Остановл.", "✅ Завершённые"],
            ["🔎 Поиск", "⬅️ Назад"],
        ]
    )


def kb_add() -> ReplyKeyboardMarkup:
    return kb([["🧲 Магнет/URL", "📄 .torrent файл"], ["⬅️ Назад"]])


def kb_ctrl(notify_enabled: bool = False) -> ReplyKeyboardMarkup:
    notify_label = "🔔 Уведомления: ВКЛ" if notify_enabled else "🔕 Уведомления: ВЫКЛ"
    return kb(
        [
            ["⏸️ Пауза", "▶️ Старт"],
            ["🗑️ Удалить (оставить данные)"],
            ["💥 Удалить (с данными)"],
            [notify_label],
            ["⬅️ Назад"],
        ]
    )


KB_MAIN = kb_main()
KB_TORRENTS = kb_torrents()
KB_ADD = kb_add()
KB_CTRL = kb_ctrl(False)

STATUS_REFRESH_CB = "status_refresh"
LIST_REFRESH_CB_PREFIX = "list_refresh:"
LAST_EPHEMERAL_MESSAGE_KEY = "last_ephemeral_message_id"
NOTIFY_ENABLED_CHATS_KEY = "notify_enabled_chat_ids"
NOTIFY_COMPLETED_CACHE_KEY = "notify_completed_cache"
NOTIFY_INITIALIZED_KEY = "notify_initialized"

NOTIFY_POLL_INTERVAL_SEC = 60


def set_menu(ctx: ContextTypes.DEFAULT_TYPE, menu: str) -> None:
    ctx.user_data["menu"] = menu


def get_menu(ctx: ContextTypes.DEFAULT_TYPE) -> str:
    return str(ctx.user_data.get("menu", MENU_MAIN))


def set_wait(ctx: ContextTypes.DEFAULT_TYPE, wait: Optional[str]) -> None:
    ctx.user_data["wait"] = wait


def get_wait(ctx: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    value = ctx.user_data.get("wait", WAIT_NONE)
    return str(value) if isinstance(value, str) else None


def user_allowed(update: Update) -> bool:
    chat = update.effective_chat
    if chat and chat.type != "private":
        return False

    if CFG.allowed_user_ids is None:
        return True
    uid = update.effective_user.id if update.effective_user else None
    return uid in CFG.allowed_user_ids


def _sort_torrents(items: Sequence[Any]) -> list[Any]:
    return sorted(
        items,
        key=lambda t: (
            0 if _is_active(str(t.status)) else 1,
            -float(getattr(t, "progress", 0.0)),
            (t.name or "").lower(),
        ),
    )


def fmt_bytes(n: int | float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    x = float(max(0, n))
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{int(x)} {units[i]}" if i == 0 else f"{x:.2f} {units[i]}"


def fmt_rate(bps: int | float) -> str:
    return f"{fmt_bytes(bps)}/s"


def status_icon(status: str) -> str:
    if status in ("downloading", "download pending"):
        return "⬇️"
    if status in ("seeding", "seed pending"):
        return "⬆️"
    if status in ("checking", "check pending"):
        return "🧪"
    if status == "stopped":
        return "⏸️"
    return "❔"


def parse_id(text: str) -> Optional[int]:
    match = TORRENT_ID_RE.search(text.strip())
    return int(match.group(1)) if match else None


def build_client() -> Client:
    if CFG.tr_url:
        return from_url(CFG.tr_url, timeout=CFG.tr_timeout)
    return Client(
        protocol=CFG.tr_protocol,
        host=CFG.tr_host,
        port=CFG.tr_port,
        path=CFG.tr_path,
        username=CFG.tr_user,
        password=CFG.tr_pass,
        timeout=CFG.tr_timeout,
    )


def get_client() -> Client:
    global _TR_CLIENT
    if _TR_CLIENT is None:
        with _TR_CLIENT_LOCK:
            if _TR_CLIENT is None:
                _TR_CLIENT = build_client()
    return _TR_CLIENT


async def tr_call(fn: Callable[[Client], Any]) -> Any:
    def _run() -> Any:
        client = get_client()
        return fn(client)

    def _reset_client() -> None:
        global _TR_CLIENT
        with _TR_CLIENT_LOCK:
            _TR_CLIENT = None

    def _call() -> Any:
        try:
            return _run()
        except TransmissionError:
            # Recreate client once on Transmission RPC failure and retry.
            _reset_client()
            try:
                return _run()
            except Exception as exc:
                raise TRCallError("Transmission RPC request failed") from exc
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise TRCallError("Transmission RPC connection failed") from exc

    return await asyncio.to_thread(_call)


def _chunk_text(text: str, *, max_len: int) -> list[str]:
    return [text[i : i + max_len] for i in range(0, len(text), max_len)] or [""]


def _build_torrent_messages(header: str, lines: Sequence[str], tail: str) -> list[str]:
    messages: list[str] = []
    current = f"{header}\n\n"

    for line in lines:
        for part in _chunk_text(line, max_len=TG_MAX_MESSAGE - len(header) - 2):
            separator = "" if current.endswith("\n\n") else "\n\n"
            candidate = f"{current}{separator}{part}"
            if len(candidate) <= TG_MAX_MESSAGE:
                current = candidate
            else:
                messages.append(current)
                current = f"{header}\n\n{part}"

    if tail:
        candidate = f"{current}{tail}"
        if len(candidate) <= TG_MAX_MESSAGE:
            current = candidate
        else:
            messages.append(current)
            current = f"{header}\n\n{tail.strip()}"

    messages.append(current)
    return messages

async def reply_chunks(
    update: Update,
    text: str,
    *,
    parse_mode: Optional[str] = None,
    reply_markup: Optional[Any] = None,
) -> None:
    message = update.effective_message
    if message is None:
        return

    chunks = _chunk_text(text, max_len=TG_MAX_MESSAGE)
    for idx, part in enumerate(chunks):
        kwargs: dict[str, Any] = {"text": part, "parse_mode": parse_mode}
        if idx == len(chunks) - 1:
            kwargs["reply_markup"] = reply_markup
        await message.reply_text(**kwargs)


def _status_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Обновить статус", callback_data=STATUS_REFRESH_CB)]])


def _torrent_list_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("▶️ Активные", callback_data=f"{LIST_REFRESH_CB_PREFIX}active"),
            InlineKeyboardButton("📋 Все", callback_data=f"{LIST_REFRESH_CB_PREFIX}all"),
            InlineKeyboardButton("✅ Завершённые", callback_data=f"{LIST_REFRESH_CB_PREFIX}done"),
        ]]
    )


def _notifications_enabled(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int) -> bool:
    enabled_chats = ctx.application.bot_data.setdefault(NOTIFY_ENABLED_CHATS_KEY, set())
    return chat_id in enabled_chats


def _ctrl_keyboard_for_chat(ctx: ContextTypes.DEFAULT_TYPE, chat_id: Optional[int]) -> ReplyKeyboardMarkup:
    if chat_id is None:
        return KB_CTRL
    return kb_ctrl(_notifications_enabled(ctx, chat_id))


def _format_session_duration(seconds: int | float) -> str:
    total_seconds = max(0, int(seconds))
    days, rem = divmod(total_seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)

    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


async def _delete_message_safe(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int) -> None:
    try:
        await ctx.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except TelegramError:
        return


async def _cleanup_previous_ephemeral(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is None:
        return
    old_message_id = ctx.user_data.get(LAST_EPHEMERAL_MESSAGE_KEY)
    if isinstance(old_message_id, int):
        await _delete_message_safe(ctx, chat.id, old_message_id)


async def send_ephemeral(update: Update, ctx: ContextTypes.DEFAULT_TYPE, text: str, reply_markup: ReplyKeyboardMarkup) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return

    await _cleanup_previous_ephemeral(update, ctx)
    sent = await message.reply_text(text=text, reply_markup=reply_markup)
    ctx.user_data[LAST_EPHEMERAL_MESSAGE_KEY] = sent.message_id


async def _delete_user_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return
    await _delete_message_safe(ctx, chat.id, message.message_id)


def _build_status_text(stats: Any) -> str:
    cur = stats.current_stats
    cum = stats.cumulative_stats
    session_duration = _format_session_duration(getattr(cur, "seconds_active", 0))
    return (
        "📊 <b>Transmission — статус</b>\n"
        f"Скорость: ⇣ <b>{fmt_rate(stats.download_speed)}</b> | ⇡ <b>{fmt_rate(stats.upload_speed)}</b>\n"
        f"Торренты: активных <b>{stats.active_torrent_count}</b>, на паузе <b>{stats.paused_torrent_count}</b>, всего <b>{stats.torrent_count}</b>\n\n"
        f"Трафик (сессия - {session_duration}): ⇣ <b>{fmt_bytes(cur.downloaded_bytes)}</b> | ⇡ <b>{fmt_bytes(cur.uploaded_bytes)}</b>\n"
        f"Трафик (всего): ⇣ <b>{fmt_bytes(cum.downloaded_bytes)}</b> | ⇡ <b>{fmt_bytes(cum.uploaded_bytes)}</b>\n"
        f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


async def send_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        stats = await tr_call(lambda c: c.session_stats())
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Ошибка Transmission: {html.escape(str(exc))}", reply_markup=KB_MAIN)
        return

    text = _build_status_text(stats)
    await reply_chunks(update, text, parse_mode=ParseMode.HTML, reply_markup=_status_keyboard())


async def on_status_refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    await query.answer()

    try:
        stats = await tr_call(lambda c: c.session_stats())
    except (TransmissionError, TRCallError) as exc:
        await query.edit_message_text(
            text=f"❌ Ошибка Transmission: {html.escape(str(exc))}",
            reply_markup=_status_keyboard(),
        )
        return

    await query.edit_message_text(
        text=_build_status_text(stats),
        parse_mode=ParseMode.HTML,
        reply_markup=_status_keyboard(),
    )


async def on_list_refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    data = query.data or ""
    if not data.startswith(LIST_REFRESH_CB_PREFIX):
        await query.answer()
        return

    mode = data[len(LIST_REFRESH_CB_PREFIX) :]
    if mode not in {"all", "active", "done"}:
        await query.answer("Неизвестный тип списка", show_alert=True)
        return

    await query.answer("Обновляю список…")
    await send_torrent_list(update, ctx, mode=mode)


def _is_active(status: str) -> bool:
    return status in (
        "downloading",
        "download pending",
        "seeding",
        "seed pending",
        "checking",
        "check pending",
    )


async def send_torrent_list(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    mode: str,
    query: Optional[str] = None,
) -> None:
    try:
        torrents = await tr_call(lambda c: c.get_torrents())
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Ошибка Transmission: {html.escape(str(exc))}", reply_markup=_torrent_list_keyboard())
        return

    items = torrents
    if mode == "active":
        items = [t for t in items if _is_active(str(t.status))]
    elif mode == "stopped":
        items = [t for t in items if str(t.status) == "stopped"]
    elif mode == "done":
        items = [t for t in items if float(t.percent_done) >= 1.0]

    if query:
        q = query.strip().lower()
        items = [t for t in items if q in (t.name or "").lower()]

    items = _sort_torrents(items)

    ctx.user_data["last_list_mode"] = mode
    ctx.user_data["last_list_query"] = query

    total = len(items)
    items = items[: CFG.list_limit]

    if total == 0:
        await reply_chunks(update, "Пусто.", reply_markup=_torrent_list_keyboard())
        return

    lines = []
    for t in items:
        st = str(t.status)
        safe_name = html.escape(t.name or "<без названия>")
        lines.append(
            f"<b>{t.id}</b> {status_icon(st)} {safe_name} — <b>{t.progress:.2f}%</b>\n"
            f"   ⇣ {fmt_rate(t.rate_download)} | ⇡ {fmt_rate(t.rate_upload)} | Ratio {t.upload_ratio:.2f} | {html.escape(st)}"
        )

    header = {
        "all": "📋 <b>Все торренты</b>",
        "active": "▶️ <b>Активные</b>",
        "stopped": "⏹️ <b>Остановленные</b>",
        "done": "✅ <b>Завершённые</b>",
    }.get(mode, "📋 <b>Список</b>")

    tail = ""
    if total > CFG.list_limit:
        tail = f"\n\nПоказано: {len(items)} из {total}."

    messages = _build_torrent_messages(header, lines, tail)
    for idx, text in enumerate(messages):
        await reply_chunks(
            update,
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=_torrent_list_keyboard() if idx == len(messages) - 1 else None,
        )


async def add_magnet_or_url(update: Update, text: str) -> None:
    link = text.strip()
    if not (link.startswith("magnet:") or link.startswith("http://") or link.startswith("https://")):
        await reply_chunks(update, "❌ Нужна magnet-ссылка или http(s) URL на .torrent.", reply_markup=KB_ADD)
        return

    try:
        torrent = await tr_call(lambda c: c.add_torrent(link))
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Не удалось добавить: {html.escape(str(exc))}", reply_markup=KB_ADD)
        return

    await reply_chunks(
        update,
        f"✅ Добавлено: <b>{html.escape(torrent.name)}</b>\nID: <b>{torrent.id}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=KB_ADD,
    )


async def add_torrent_file(update: Update) -> None:
    message = update.effective_message
    if message is None or message.document is None:
        await reply_chunks(update, "Пришли .torrent файлом.", reply_markup=KB_ADD)
        return

    doc = message.document
    if not (doc.file_name or "").lower().endswith(".torrent"):
        await reply_chunks(update, "Это не .torrent файл.", reply_markup=KB_ADD)
        return

    tg_file = await doc.get_file()
    tmp_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(prefix="tg_", suffix=".torrent", delete=False) as temp_file:
            tmp_path = Path(temp_file.name)

        await tg_file.download_to_drive(custom_path=str(tmp_path))

        def _add(client: Client):
            assert tmp_path is not None
            with tmp_path.open("rb") as rf:
                return client.add_torrent(rf)

        torrent = await tr_call(_add)

    except (TelegramError, OSError, TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Не удалось добавить .torrent: {html.escape(str(exc))}", reply_markup=KB_ADD)
        return
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    await reply_chunks(
        update,
        f"✅ Добавлено из файла: <b>{html.escape(torrent.name)}</b>\nID: <b>{torrent.id}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=KB_ADD,
    )


async def ctrl_action(update: Update, ctx: ContextTypes.DEFAULT_TYPE, action: str, torrent_id: int) -> None:
    try:
        if action == "pause":
            await tr_call(lambda c: c.stop_torrent(torrent_id))
            msg = f"⏸️ Остановлено: ID {torrent_id}"
        elif action == "start":
            await tr_call(lambda c: c.start_torrent(torrent_id))
            msg = f"▶️ Запущено: ID {torrent_id}"
        elif action == "del_keep":
            await tr_call(lambda c: c.remove_torrent(torrent_id, delete_data=False))
            msg = f"🗑️ Удалено (данные сохранены): ID {torrent_id}"
        elif action == "del_data":
            await tr_call(lambda c: c.remove_torrent(torrent_id, delete_data=True))
            msg = f"💥 Удалено вместе с данными: ID {torrent_id}"
        else:
            msg = "❌ Неизвестное действие"
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Ошибка Transmission: {html.escape(str(exc))}", reply_markup=_ctrl_keyboard_for_chat(ctx, update.effective_chat.id if update.effective_chat else None))
        return

    await reply_chunks(update, msg, reply_markup=_ctrl_keyboard_for_chat(ctx, update.effective_chat.id if update.effective_chat else None))


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.type != "private":
        return

    if not user_allowed(update):
        await reply_chunks(update, "⛔️ Доступ запрещён.")
        return

    set_menu(ctx, MENU_MAIN)
    set_wait(ctx, WAIT_NONE)
    await reply_chunks(update, "Привет! Я бот для управления Transmission.\nВыбирай пункт меню ниже 👇", reply_markup=KB_MAIN)


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.type != "private":
        return

    if not user_allowed(update):
        await reply_chunks(update, "⛔️ Доступ запрещён.")
        return

    text = (
        "ℹ️ <b>Команды</b>\n"
        "/start — показать меню\n"
        "/help — помощь\n\n"
        "<b>Как пользоваться</b>\n"
        "• 📊 Статус — скорость и трафик\n"
        "• 📋 Торренты — списки + поиск\n"
        "• ➕ Добавить — magnet/URL или .torrent файл\n"
        "• ⚙️ Управление — пауза/старт/удаление по ID\n\n"
        "Подсказка: ID виден в списках торрентов."
    )
    await reply_chunks(update, text, parse_mode=ParseMode.HTML, reply_markup=KB_MAIN)


async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.type != "private":
        return

    if not user_allowed(update):
        return

    wait = get_wait(ctx)
    menu = get_menu(ctx)
    if wait == WAIT_ADD_TORRENT_FILE or menu == MENU_ADD:
        await add_torrent_file(update)
        set_menu(ctx, MENU_ADD)
        set_wait(ctx, WAIT_NONE)
        await _delete_user_message(update, ctx)
        return

    await send_ephemeral(update, ctx, "Я жду команду из меню 🙂", reply_markup=KB_MAIN)
    await _delete_user_message(update, ctx)


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.type != "private":
        return

    if not user_allowed(update):
        return

    message = update.effective_message
    text = (message.text if message else "") or ""
    text = text.strip()
    if not text:
        return

    chat = update.effective_chat
    chat_id = chat.id if chat else None

    try:
        if text == "⬅️ Назад":
            set_menu(ctx, MENU_MAIN)
            set_wait(ctx, WAIT_NONE)
            await send_ephemeral(update, ctx, "Ок, назад в главное меню.", reply_markup=KB_MAIN)
            return

        menu = get_menu(ctx)
        wait = get_wait(ctx)

        if wait == WAIT_SEARCH:
            set_wait(ctx, WAIT_NONE)
            set_menu(ctx, MENU_TORRENTS)
            await send_torrent_list(update, ctx, mode="all", query=text)
            return

        if wait == WAIT_ADD_MAGNET:
            set_wait(ctx, WAIT_NONE)
            set_menu(ctx, MENU_ADD)
            await add_magnet_or_url(update, text)
            return

        if wait in {WAIT_CTRL_PAUSE, WAIT_CTRL_START, WAIT_CTRL_DEL_KEEP, WAIT_CTRL_DEL_DATA}:
            torrent_id = parse_id(text)
            if torrent_id is None:
                await send_ephemeral(update, ctx, "Пришли числовой ID торрента (например: 12).", reply_markup=KB_CTRL)
                return

            set_wait(ctx, WAIT_NONE)
            set_menu(ctx, MENU_CTRL)
            action_map = {
                WAIT_CTRL_PAUSE: "pause",
                WAIT_CTRL_START: "start",
                WAIT_CTRL_DEL_KEEP: "del_keep",
                WAIT_CTRL_DEL_DATA: "del_data",
            }
            await ctrl_action(update, ctx, action_map[wait], torrent_id=torrent_id)
            return

        if text == "ℹ️ Помощь":
            await cmd_help(update, ctx)
            return
        if text == "📊 Статус":
            set_menu(ctx, MENU_MAIN)
            set_wait(ctx, WAIT_NONE)
            await send_status(update, ctx)
            return
        if text == "📋 Торренты":
            set_menu(ctx, MENU_TORRENTS)
            set_wait(ctx, WAIT_NONE)
            await send_ephemeral(update, ctx, "Меню торрентов:", reply_markup=KB_TORRENTS)
            return
        if text == "➕ Добавить":
            set_menu(ctx, MENU_ADD)
            set_wait(ctx, WAIT_NONE)
            await send_ephemeral(update, ctx, "Как будем добавлять?", reply_markup=KB_ADD)
            return
        if text == "⚙️ Управление":
            set_menu(ctx, MENU_CTRL)
            set_wait(ctx, WAIT_NONE)
            await send_ephemeral(update, ctx, "Выбери действие:", reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id))
            return

        if menu == MENU_TORRENTS:
            if text == "📋 Все":
                await send_torrent_list(update, ctx, mode="all")
                return
            if text == "▶️ Активные":
                await send_torrent_list(update, ctx, mode="active")
                return
            if text == "⏹️ Остановл.":
                await send_torrent_list(update, ctx, mode="stopped")
                return
            if text == "✅ Завершённые":
                await send_torrent_list(update, ctx, mode="done")
                return
            if text == "🔎 Поиск":
                set_wait(ctx, WAIT_SEARCH)
                await send_ephemeral(update, ctx, "Введи часть названия для поиска:", reply_markup=KB_TORRENTS)
                return

        if menu == MENU_ADD:
            if text == "🧲 Магнет/URL":
                set_wait(ctx, WAIT_ADD_MAGNET)
                await send_ephemeral(update, ctx, "Пришли magnet-ссылку или URL на .torrent:", reply_markup=KB_ADD)
                return
            if text == "📄 .torrent файл":
                set_wait(ctx, WAIT_ADD_TORRENT_FILE)
                await send_ephemeral(update, ctx, "Ок, пришли .torrent файлом сюда в чат.", reply_markup=KB_ADD)
                return

        if menu == MENU_CTRL:
            if text == "⏸️ Пауза":
                set_wait(ctx, WAIT_CTRL_PAUSE)
                await send_ephemeral(update, ctx, "Пришли ID торрента для остановки:", reply_markup=KB_CTRL)
                return
            if text == "▶️ Старт":
                set_wait(ctx, WAIT_CTRL_START)
                await send_ephemeral(update, ctx, "Пришли ID торрента для запуска:", reply_markup=KB_CTRL)
                return
            if text == "🗑️ Удалить (оставить данные)":
                set_wait(ctx, WAIT_CTRL_DEL_KEEP)
                await send_ephemeral(update, ctx, "Пришли ID торрента для удаления (данные останутся на диске):", reply_markup=KB_CTRL)
                return
            if text == "💥 Удалить (с данными)":
                set_wait(ctx, WAIT_CTRL_DEL_DATA)
                await send_ephemeral(update, ctx, "⚠️ Пришли ID торрента для удаления вместе с данными:", reply_markup=KB_CTRL)
                return
            if text.startswith("🔔 Уведомления:") or text.startswith("🔕 Уведомления:"):
                if chat_id is None:
                    await send_ephemeral(update, ctx, "❌ Не удалось определить чат для настройки уведомлений.", reply_markup=KB_CTRL)
                    return

                enabled_chats = ctx.application.bot_data.setdefault(NOTIFY_ENABLED_CHATS_KEY, set())
                if chat_id in enabled_chats:
                    enabled_chats.remove(chat_id)
                    status_text = "🔕 Уведомления о завершении торрентов выключены."
                else:
                    enabled_chats.add(chat_id)
                    status_text = "🔔 Уведомления о завершении торрентов включены."

                await send_ephemeral(update, ctx, status_text, reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id))
                return

        await send_ephemeral(update, ctx, "Не понял. Выбери пункт меню 🙂", reply_markup=KB_MAIN)
    finally:
        await _delete_user_message(update, ctx)


def main() -> None:
    app: Application = ApplicationBuilder().token(CFG.tg_token).build()

    async def notify_completed_torrents(ctx: ContextTypes.DEFAULT_TYPE) -> None:
        enabled_chats = ctx.application.bot_data.get(NOTIFY_ENABLED_CHATS_KEY)
        if not isinstance(enabled_chats, set) or not enabled_chats:
            return

        try:
            torrents = await tr_call(lambda c: c.get_torrents())
        except (TransmissionError, TRCallError):
            log.warning("Skipping completion notifications due to Transmission error", exc_info=True)
            return

        completed_now = {
            int(t.id): (t.name or "<без названия>")
            for t in torrents
            if float(getattr(t, "percent_done", 0.0)) >= 1.0
        }

        initialized = bool(ctx.application.bot_data.get(NOTIFY_INITIALIZED_KEY))
        prev_completed = ctx.application.bot_data.get(NOTIFY_COMPLETED_CACHE_KEY, {})
        if not isinstance(prev_completed, dict):
            prev_completed = {}

        if not initialized:
            ctx.application.bot_data[NOTIFY_COMPLETED_CACHE_KEY] = completed_now
            ctx.application.bot_data[NOTIFY_INITIALIZED_KEY] = True
            return

        new_ids = sorted(set(completed_now) - set(prev_completed))
        ctx.application.bot_data[NOTIFY_COMPLETED_CACHE_KEY] = completed_now

        if not new_ids:
            return

        for torrent_id in new_ids:
            name = html.escape(completed_now.get(torrent_id, "<без названия>"))
            text = f"✅ Торрент завершён: <b>{name}</b>\nID: <b>{torrent_id}</b>"
            for chat_id in list(enabled_chats):
                try:
                    await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
                except TelegramError:
                    log.warning("Failed to send completion notification to chat %s", chat_id, exc_info=True)

    app.job_queue.run_repeating(notify_completed_torrents, interval=NOTIFY_POLL_INTERVAL_SEC, first=NOTIFY_POLL_INTERVAL_SEC)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))

    app.add_handler(CallbackQueryHandler(on_status_refresh, pattern=f"^{STATUS_REFRESH_CB}$"))
    app.add_handler(CallbackQueryHandler(on_list_refresh, pattern=f"^{LIST_REFRESH_CB_PREFIX}(all|active|done)$"))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    log.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
