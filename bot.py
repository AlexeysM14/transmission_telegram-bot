#!/usr/bin/env python3
"""Telegram bot for managing Transmission RPC via menu buttons."""

from __future__ import annotations

import asyncio
import calendar
import contextlib
import html
import heapq
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, ReplyKeyboardMarkup, Update
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


def configure_logging() -> logging.Logger:
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    log_file_path = Path(os.environ.get("LOG_FILE", "bot-errors.log")).expanduser()
    if not log_file_path.is_absolute():
        log_file_path = Path(__file__).resolve().parent / log_file_path

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=1_048_576,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("tg-transmission-bot")
    logger.info("Error logs will be written to %s", log_file_path)
    return logger


log = configure_logging()


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
            ["📈 Статистика"],
        ]
    )


def kb_torrents() -> ReplyKeyboardMarkup:
    return kb(
        [
            ["📋 Все", "⬇️ Скачиваются"],
            ["⏹️ Остановл.", "✅ Завершённые"],
            ["🔎 Поиск", "⬅️ Назад"],
        ]
    )


def kb_add() -> ReplyKeyboardMarkup:
    return kb([["🧲 Магнет/URL", "📄 .torrent файл"], ["⬅️ Назад"]])


def kb_ctrl(notify_enabled: bool = True) -> ReplyKeyboardMarkup:
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
KB_CTRL = kb_ctrl(True)

STATUS_REFRESH_CB = "status_refresh"
LIST_REFRESH_CB_PREFIX = "list_refresh:"
TRAFFIC_VIEW_CB_PREFIX = "traffic_view:"
LAST_EPHEMERAL_MESSAGE_KEY = "last_ephemeral_message_id"
NOTIFY_ENABLED_CHATS_KEY = "notify_enabled_chat_ids"
NOTIFY_KNOWN_CHATS_KEY = "notify_known_chat_ids"
NOTIFY_COMPLETED_CACHE_KEY = "notify_completed_cache"
NOTIFY_INITIALIZED_KEY = "notify_initialized"
NOTIFY_START_PENDING_KEY = "notify_start_pending"
TRAFFIC_LAST_SNAPSHOT_DAY_KEY = "traffic_last_snapshot_day"

NOTIFY_POLL_INTERVAL_SEC = 60
NOTIFY_NO_PEERS_DELAY_SEC = 10 * 60
TRAFFIC_ANCHORS_PATH = Path(__file__).resolve().with_name("traffic_anchors.json")
ACTIVE_STATUSES = frozenset(
    {
        "downloading",
        "download pending",
        "seeding",
        "seed pending",
        "checking",
        "check pending",
    }
)
STATUS_ICONS = {
    "downloading": "⬇️",
    "download pending": "⬇️",
    "seeding": "⬆️",
    "seed pending": "⬆️",
    "checking": "🧪",
    "check pending": "🧪",
    "stopped": "⏸️",
}


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
    return f"{int(x)} {units[i]}" if i == 0 else f"{x:.3f} {units[i]}"


def fmt_rate(bps: int | float) -> str:
    return f"{fmt_bytes(bps)}/s"


def torrent_total_size(torrent: Any) -> int:
    for attr in ("total_size", "size_when_done"):
        value = getattr(torrent, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return 0


def status_icon(status: str) -> str:
    return STATUS_ICONS.get(status, "❔")


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


def _build_single_torrent_message(header: str, lines: Sequence[str], tail: str) -> str:
    message = f"{header}\n\n"
    rendered_count = 0

    for line in lines:
        separator = "" if message.endswith("\n\n") else "\n\n"
        candidate = f"{message}{separator}{line}"
        if len(candidate) > TG_MAX_MESSAGE:
            break
        message = candidate
        rendered_count += 1

    suffix = tail
    if rendered_count < len(lines):
        hidden_count = len(lines) - rendered_count
        suffix = f"\n\n⚠️ Список не поместился в одно сообщение. Скрыто элементов: {hidden_count}."
        if tail:
            suffix = f"{suffix}{tail}"

    if suffix and len(f"{message}{suffix}") <= TG_MAX_MESSAGE:
        return f"{message}{suffix}"
    return message

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


STATUS_KEYBOARD = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Обновить статус", callback_data=STATUS_REFRESH_CB)]])
TORRENT_LIST_KEYBOARD = InlineKeyboardMarkup(
    [[
        InlineKeyboardButton("⬇️ Скачиваются", callback_data=f"{LIST_REFRESH_CB_PREFIX}downloading"),
        InlineKeyboardButton("📋 Все", callback_data=f"{LIST_REFRESH_CB_PREFIX}all"),
        InlineKeyboardButton("✅ Завершённые", callback_data=f"{LIST_REFRESH_CB_PREFIX}done"),
    ]]
)

TRAFFIC_OVERVIEW_KEYBOARD = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("🔄 Обновить статистику", callback_data=f"{TRAFFIC_VIEW_CB_PREFIX}refresh")],
        [InlineKeyboardButton("📅 Последние 7 дней", callback_data=f"{TRAFFIC_VIEW_CB_PREFIX}7d")],
        [InlineKeyboardButton("🗓️ По дням (месяц)", callback_data=f"{TRAFFIC_VIEW_CB_PREFIX}4w")],
    ]
)


def _notifications_enabled(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int) -> bool:
    enabled_chats = ctx.application.bot_data.setdefault(NOTIFY_ENABLED_CHATS_KEY, set())
    return chat_id in enabled_chats


def _register_torrent_start_watch(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: Optional[int],
    torrent: Any,
    *,
    now_ts: Optional[float] = None,
) -> None:
    if chat_id is None:
        return

    pending = ctx.application.bot_data.setdefault(NOTIFY_START_PENDING_KEY, {})
    if not isinstance(pending, dict):
        pending = {}
        ctx.application.bot_data[NOTIFY_START_PENDING_KEY] = pending

    torrent_id = int(getattr(torrent, "id", 0))
    if torrent_id <= 0:
        return

    now_value = now_ts if now_ts is not None else asyncio.get_running_loop().time()
    state = pending.get(torrent_id)
    if not isinstance(state, dict):
        pending[torrent_id] = {
            "added_at": now_value,
            "name": str(getattr(torrent, "name", "") or "<без названия>"),
            "chat_ids": {chat_id},
        }
        return

    chat_ids = state.get("chat_ids")
    if not isinstance(chat_ids, set):
        chat_ids = set()
        state["chat_ids"] = chat_ids
    chat_ids.add(chat_id)


def _ensure_chat_notifications_initialized(ctx: ContextTypes.DEFAULT_TYPE, chat_id: Optional[int]) -> None:
    if chat_id is None:
        return

    known_chats = ctx.application.bot_data.setdefault(NOTIFY_KNOWN_CHATS_KEY, set())
    enabled_chats = ctx.application.bot_data.setdefault(NOTIFY_ENABLED_CHATS_KEY, set())
    if chat_id in known_chats:
        return

    known_chats.add(chat_id)
    enabled_chats.add(chat_id)


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


async def _get_download_dir_free_space() -> Optional[int]:
    try:
        session = await tr_call(lambda c: c.get_session())
    except (TransmissionError, TRCallError):
        return None

    free_space = getattr(session, "download_dir_free_space", None)
    if isinstance(free_space, (int, float)):
        return int(max(0, free_space))
    return None


def _build_free_space_text(free_space: Optional[int]) -> str:
    if free_space is None:
        return "💾 Свободно на диске: <i>не удалось получить</i>."
    return f"💾 Свободно на диске: <b>{fmt_bytes(free_space)}</b>."


def _build_projected_free_space_text(free_space_before: Optional[int], torrent: Any) -> str:
    if free_space_before is None:
        return "💾 После полной скачки: <i>не удалось рассчитать</i>."

    required_space = getattr(torrent, "left_until_done", None)
    if not isinstance(required_space, (int, float)):
        required_space = getattr(torrent, "total_size", None)
    if not isinstance(required_space, (int, float)):
        return "💾 После полной скачки: <i>не удалось рассчитать</i>."

    projected_free_space = max(0, int(free_space_before - max(0, int(required_space))))
    return f"💾 После полной скачки: <b>{fmt_bytes(projected_free_space)}</b>."


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


def _build_status_text(stats: Any, free_space: Optional[int]) -> str:
    cur = stats.current_stats
    cum = stats.cumulative_stats
    session_duration = _format_session_duration(getattr(cur, "seconds_active", 0))
    free_space_text = _build_free_space_text(free_space)
    return (
        "📊 <b>Transmission — статус</b>\n"
        f"Скорость: ⇣ <b>{fmt_rate(stats.download_speed)}</b> | ⇡ <b>{fmt_rate(stats.upload_speed)}</b>\n"
        f"Торренты: активных <b>{stats.active_torrent_count}</b>, на паузе <b>{stats.paused_torrent_count}</b>, всего <b>{stats.torrent_count}</b>\n\n"
        f"{free_space_text}\n"
        f"Трафик (сессия - {session_duration}): ⇣ <b>{fmt_bytes(cur.downloaded_bytes)}</b> | ⇡ <b>{fmt_bytes(cur.uploaded_bytes)}</b>\n"
        f"Трафик (всего): ⇣ <b>{fmt_bytes(cum.downloaded_bytes)}</b> | ⇡ <b>{fmt_bytes(cum.uploaded_bytes)}</b>\n"
        f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def _read_traffic_state() -> tuple[dict[str, dict[str, int | str]], list[dict[str, int | str]]]:
    try:
        data = json.loads(TRAFFIC_ANCHORS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, []
    except (OSError, ValueError, TypeError):
        return {}, []

    if not isinstance(data, dict):
        return {}, []

    anchors: dict[str, dict[str, int | str]] = {}
    for period in ("day", "week", "month"):
        row = data.get(period)
        if not isinstance(row, dict):
            continue
        key = row.get("key")
        downloaded = row.get("downloaded")
        uploaded = row.get("uploaded")
        if isinstance(key, str) and isinstance(downloaded, int) and isinstance(uploaded, int):
            anchors[period] = {
                "key": key,
                "downloaded": max(0, downloaded),
                "uploaded": max(0, uploaded),
            }

    raw_history = data.get("history") if isinstance(data, dict) else None
    days = raw_history.get("days") if isinstance(raw_history, dict) else None
    if not isinstance(days, list):
        return anchors, []

    history: list[dict[str, int | str]] = []
    for item in days:
        if not isinstance(item, dict):
            continue
        date = item.get("date")
        downloaded = item.get("downloaded")
        uploaded = item.get("uploaded")
        if isinstance(date, str) and isinstance(downloaded, int) and isinstance(uploaded, int):
            history.append({
                "date": date,
                "downloaded": max(0, downloaded),
                "uploaded": max(0, uploaded),
            })
    return anchors, history


def _persist_traffic_state(anchors: dict[str, dict[str, int | str]], history: list[dict[str, int | str]]) -> None:
    payload = json.dumps({**anchors, "history": {"days": history}}, ensure_ascii=False, separators=(",", ":"))
    tmp_path = TRAFFIC_ANCHORS_PATH.with_suffix(".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(TRAFFIC_ANCHORS_PATH)


def _period_keys(now: datetime) -> dict[str, str]:
    iso_year, iso_week, _ = now.isocalendar()
    return {
        "day": now.strftime("%Y-%m-%d"),
        "week": f"{iso_year}-W{iso_week:02d}",
        "month": now.strftime("%Y-%m"),
    }


def _ensure_traffic_anchors(
    anchors: dict[str, dict[str, int | str]],
    now: datetime,
    downloaded: int,
    uploaded: int,
) -> bool:
    keys = _period_keys(now)
    changed = False

    for period, period_key in keys.items():
        current = anchors.get(period)
        if not isinstance(current, dict):
            current = None

        base_downloaded = int(current["downloaded"]) if current and isinstance(current.get("downloaded"), int) else 0
        base_uploaded = int(current["uploaded"]) if current and isinstance(current.get("uploaded"), int) else 0
        key_changed = not current or current.get("key") != period_key
        counter_reset = downloaded < base_downloaded or uploaded < base_uploaded

        if key_changed or counter_reset:
            anchors[period] = {
                "key": period_key,
                "downloaded": downloaded,
                "uploaded": uploaded,
            }
            changed = True
    return changed


def _ensure_daily_traffic_history(
    history: list[dict[str, int | str]], now: datetime, downloaded: int, uploaded: int
) -> bool:
    day_key = now.strftime("%Y-%m-%d")

    if history and history[-1].get("date") == day_key:
        return False

    history.append({"date": day_key, "downloaded": downloaded, "uploaded": uploaded})
    # Храним небольшой хвост: достаточно для текущего месяца + запас.
    if len(history) > 45:
        del history[:-45]
    return True


def _weekday_short_ru(date_value: datetime) -> str:
    labels = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
    return labels[date_value.weekday()]


def _build_last_7_days_text(now: datetime, downloaded: int, uploaded: int, history: list[dict[str, int | str]]) -> str:
    lines = ["📅 <b>Трафик за последние 7 дней</b>"]
    points = history[-8:]

    if len(points) < 2:
        lines.append("Недостаточно данных. История начнёт заполняться автоматически раз в день.")
        lines.append(f"🕒 {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    for idx in range(1, len(points)):
        prev_point = points[idx - 1]
        current_point = points[idx]
        date_raw = str(prev_point.get("date", ""))
        try:
            date_value = datetime.strptime(date_raw, "%Y-%m-%d")
        except ValueError:
            continue

        prev_downloaded = int(prev_point.get("downloaded", 0))
        current_downloaded = int(current_point.get("downloaded", 0))
        prev_uploaded = int(prev_point.get("uploaded", 0))
        current_uploaded = int(current_point.get("uploaded", 0))
        day_downloaded = max(0, current_downloaded - prev_downloaded)
        day_uploaded = max(0, current_uploaded - prev_uploaded)
        lines.append(
            f"{_weekday_short_ru(date_value)} {date_value.strftime('%d.%m')}: "
            f"⇣ <b>{fmt_bytes(day_downloaded)}</b> | ⇡ <b>{fmt_bytes(day_uploaded)}</b>"
        )

    today_anchor = points[-1]
    today_downloaded = max(0, downloaded - int(today_anchor.get("downloaded", downloaded)))
    today_uploaded = max(0, uploaded - int(today_anchor.get("uploaded", uploaded)))
    lines.append(
        f"Сегодня {now.strftime('%d.%m')}: "
        f"⇣ <b>{fmt_bytes(today_downloaded)}</b> | ⇡ <b>{fmt_bytes(today_uploaded)}</b>"
    )
    lines.append(f"🕒 {now.strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def _traffic_points_last_7_days(
    now: datetime,
    downloaded: int,
    uploaded: int,
    history: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    points = history[-8:]
    if len(points) < 2:
        return []

    result: list[dict[str, int | str]] = []
    for idx in range(1, len(points)):
        prev_point = points[idx - 1]
        current_point = points[idx]
        date_raw = str(prev_point.get("date", ""))
        try:
            date_value = datetime.strptime(date_raw, "%Y-%m-%d")
        except ValueError:
            continue

        day_downloaded = max(0, int(current_point.get("downloaded", 0)) - int(prev_point.get("downloaded", 0)))
        day_uploaded = max(0, int(current_point.get("uploaded", 0)) - int(prev_point.get("uploaded", 0)))
        result.append(
            {
                "date": date_value.strftime("%d.%m"),
                "downloaded": day_downloaded,
                "uploaded": day_uploaded,
            }
        )

    latest_day = points[-1]
    if str(latest_day.get("date", "")) == now.strftime("%Y-%m-%d"):
        result.append(
            {
                "date": now.strftime("%d.%m"),
                "downloaded": max(0, downloaded - int(latest_day.get("downloaded", downloaded))),
                "uploaded": max(0, uploaded - int(latest_day.get("uploaded", uploaded))),
            }
        )

    return result[-7:]


def _build_traffic_chart_last_7_days(
    now: datetime,
    downloaded: int,
    uploaded: int,
    history: list[dict[str, int | str]],
) -> tuple[Optional[bytes], Optional[str]]:
    points = _traffic_points_last_7_days(now, downloaded, uploaded, history)
    if len(points) < 2:
        return None, "Недостаточно данных для графика. История заполняется раз в день."

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return None, "Графики недоступны: установите optional-зависимость matplotlib."

    labels = [str(item["date"]) for item in points]
    down_values = [float(item["downloaded"]) / (1024 * 1024 * 1024) for item in points]
    up_values = [float(item["uploaded"]) / (1024 * 1024 * 1024) for item in points]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=120)
    try:
        _draw_traffic_chart(
            ax=ax,
            labels=labels,
            down_values=down_values,
            up_values=up_values,
            title="Трафик за последние 7 дней",
            y_label="GiB / день",
        )
        fig.tight_layout()
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format="png")
        image_buffer.seek(0)
    finally:
        plt.close(fig)

    return image_buffer.getvalue(), None


def _daily_totals_current_month(
    now: datetime,
    downloaded: int,
    uploaded: int,
    history: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    points = history[-45:]
    daily_totals: dict[str, dict[str, int]] = {}

    for idx in range(1, len(points)):
        prev_point = points[idx - 1]
        current_point = points[idx]
        try:
            date_value = datetime.strptime(str(prev_point.get("date", "")), "%Y-%m-%d")
        except ValueError:
            continue

        if date_value.year != now.year or date_value.month != now.month:
            continue

        day_key = date_value.strftime("%Y-%m-%d")
        delta_downloaded = max(0, int(current_point.get("downloaded", 0)) - int(prev_point.get("downloaded", 0)))
        delta_uploaded = max(0, int(current_point.get("uploaded", 0)) - int(prev_point.get("uploaded", 0)))
        daily_totals[day_key] = {"downloaded": delta_downloaded, "uploaded": delta_uploaded}

    today_date = now.strftime("%Y-%m-%d")
    if points and points[-1].get("date") == today_date:
        today_delta_downloaded = max(0, downloaded - int(points[-1].get("downloaded", downloaded)))
        today_delta_uploaded = max(0, uploaded - int(points[-1].get("uploaded", uploaded)))
        daily_totals[today_date] = {"downloaded": today_delta_downloaded, "uploaded": today_delta_uploaded}

    month_days = calendar.monthrange(now.year, now.month)[1]
    result: list[dict[str, int | str]] = []
    for day in range(1, month_days + 1):
        date_value = now.replace(day=day)
        day_key = date_value.strftime("%Y-%m-%d")
        totals = daily_totals.get(day_key)
        result.append(
            {
                "date": date_value.strftime("%d.%m"),
                "downloaded": int(totals["downloaded"]) if totals else 0,
                "uploaded": int(totals["uploaded"]) if totals else 0,
            }
        )

    return result


def _build_traffic_chart_current_month(
    now: datetime,
    downloaded: int,
    uploaded: int,
    history: list[dict[str, int | str]],
) -> tuple[Optional[list[dict[str, int | str]]], Optional[bytes], Optional[str]]:
    points = _daily_totals_current_month(now, downloaded, uploaded, history)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return None, None, "Графики недоступны: установите optional-зависимость matplotlib."

    labels = [str(item["date"]) for item in points]
    down_values = [float(item["downloaded"]) / (1024 * 1024 * 1024) for item in points]
    up_values = [float(item["uploaded"]) / (1024 * 1024 * 1024) for item in points]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=120)
    try:
        _draw_traffic_chart(
            ax=ax,
            labels=labels,
            down_values=down_values,
            up_values=up_values,
            title="Трафик по дням (текущий месяц)",
            y_label="GiB / день",
        )
        fig.tight_layout()
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format="png")
        image_buffer.seek(0)
    finally:
        plt.close(fig)

    return points, image_buffer.getvalue(), None


def _build_last_4_weeks_text(now: datetime, downloaded: int, uploaded: int, history: list[dict[str, int | str]]) -> str:
    lines = ["🗓️ <b>Трафик по дням (текущий месяц)</b>"]
    points = _daily_totals_current_month(now, downloaded, uploaded, history)

    for day in points:
        lines.append(
            f"{day['date']}: ⇣ <b>{fmt_bytes(int(day['downloaded']))}</b> "
            f"| ⇡ <b>{fmt_bytes(int(day['uploaded']))}</b>"
        )

    lines.append(f"🕒 {now.strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def _draw_traffic_chart(
    ax: Any,
    labels: list[str],
    down_values: list[float],
    up_values: list[float],
    title: str,
    y_label: str,
) -> None:
    down_color = "#2B7DE9"
    up_color = "#FF8A33"

    ax.set_facecolor("#F7FAFF")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CAD7E6")
    ax.spines["bottom"].set_color("#CAD7E6")

    ax.plot(
        labels,
        down_values,
        marker="o",
        linewidth=2.3,
        markersize=5,
        color=down_color,
        label="Скачано",
    )
    ax.plot(
        labels,
        up_values,
        marker="o",
        linewidth=2.3,
        markersize=5,
        color=up_color,
        label="Отдано",
    )

    ax.fill_between(labels, down_values, alpha=0.16, color=down_color)
    ax.fill_between(labels, up_values, alpha=0.16, color=up_color)

    if labels:
        ax.annotate(
            f"{down_values[-1]:.2f}",
            xy=(labels[-1], down_values[-1]),
            xytext=(8, 8),
            textcoords="offset points",
            color=down_color,
            fontsize=9,
            weight="bold",
        )
        ax.annotate(
            f"{up_values[-1]:.2f}",
            xy=(labels[-1], up_values[-1]),
            xytext=(8, -12),
            textcoords="offset points",
            color=up_color,
            fontsize=9,
            weight="bold",
        )

    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.grid(False, axis="x")
    ax.legend(loc="upper left", frameon=False)


def _traffic_delta(current: int, anchor: dict[str, int | str], field: str) -> int:
    base = anchor.get(field)
    if not isinstance(base, int):
        return 0
    return max(0, current - base)


def _traffic_last_7_days_delta(
    now: datetime,
    downloaded: int,
    uploaded: int,
    history: list[dict[str, int | str]],
) -> tuple[int, int]:
    points = _traffic_points_last_7_days(now, downloaded, uploaded, history)
    if not points:
        return 0, 0
    total_downloaded = sum(int(item.get("downloaded", 0)) for item in points)
    total_uploaded = sum(int(item.get("uploaded", 0)) for item in points)
    return max(0, total_downloaded), max(0, total_uploaded)


def _build_traffic_stats_text(
    now: datetime,
    downloaded: int,
    uploaded: int,
    anchors: dict[str, dict[str, int | str]],
    history: list[dict[str, int | str]],
) -> str:
    labels = (("day", "За день"), ("month", "За месяц"))
    lines = ["📈 <b>Статистика трафика</b>"]

    for period, label in labels:
        anchor = anchors.get(period, {"downloaded": downloaded, "uploaded": uploaded})
        down = _traffic_delta(downloaded, anchor, "downloaded")
        up = _traffic_delta(uploaded, anchor, "uploaded")
        lines.append(f"{label}: ⇣ <b>{fmt_bytes(down)}</b> | ⇡ <b>{fmt_bytes(up)}</b>")

    last_7d_down, last_7d_up = _traffic_last_7_days_delta(now, downloaded, uploaded, history)
    lines.insert(2, f"За последние 7 дней: ⇣ <b>{fmt_bytes(last_7d_down)}</b> | ⇡ <b>{fmt_bytes(last_7d_up)}</b>")

    lines.append(f"🕒 {now.strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


async def send_traffic_stats(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        stats = await tr_call(lambda c: c.session_stats())
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Ошибка Transmission: {html.escape(str(exc))}", reply_markup=KB_MAIN)
        return

    now = datetime.now()
    downloaded = int(max(0, getattr(stats.cumulative_stats, "downloaded_bytes", 0)))
    uploaded = int(max(0, getattr(stats.cumulative_stats, "uploaded_bytes", 0)))

    anchors, history = _read_traffic_state()
    anchors_changed = _ensure_traffic_anchors(anchors, now, downloaded, uploaded)
    history_changed = _ensure_daily_traffic_history(history, now, downloaded, uploaded)
    if anchors_changed or history_changed:
        try:
            _persist_traffic_state(anchors, history)
        except OSError:
            log.warning("Failed to persist traffic state", exc_info=True)

    text = _build_traffic_stats_text(now, downloaded, uploaded, anchors, history)
    await reply_chunks(update, text, parse_mode=ParseMode.HTML, reply_markup=TRAFFIC_OVERVIEW_KEYBOARD)


async def send_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        stats, free_space = await asyncio.gather(
            tr_call(lambda c: c.session_stats()),
            _get_download_dir_free_space(),
        )
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Ошибка Transmission: {html.escape(str(exc))}", reply_markup=KB_MAIN)
        return

    text = _build_status_text(stats, free_space)
    await reply_chunks(update, text, parse_mode=ParseMode.HTML, reply_markup=STATUS_KEYBOARD)


async def on_status_refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    await query.answer()

    try:
        stats, free_space = await asyncio.gather(
            tr_call(lambda c: c.session_stats()),
            _get_download_dir_free_space(),
        )
    except (TransmissionError, TRCallError) as exc:
        await query.edit_message_text(
            text=f"❌ Ошибка Transmission: {html.escape(str(exc))}",
            reply_markup=STATUS_KEYBOARD,
        )
        return

    await query.edit_message_text(
        text=_build_status_text(stats, free_space),
        parse_mode=ParseMode.HTML,
        reply_markup=STATUS_KEYBOARD,
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
    if mode not in {"all", "downloading", "done"}:
        await query.answer("Неизвестный тип списка", show_alert=True)
        return

    await query.answer("Обновляю список…")
    await send_torrent_list(update, ctx, mode=mode, edit_existing=True)


async def _edit_traffic_message(query: Any, text: str) -> None:
    """Edit callback message text/caption depending on message type."""
    message = query.message
    if message is not None and message.text is None and message.caption is not None:
        await query.edit_message_caption(text, parse_mode=ParseMode.HTML, reply_markup=TRAFFIC_OVERVIEW_KEYBOARD)
        return

    await query.edit_message_text(text=text, parse_mode=ParseMode.HTML, reply_markup=TRAFFIC_OVERVIEW_KEYBOARD)


async def on_traffic_view(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    data = query.data or ""
    if not data.startswith(TRAFFIC_VIEW_CB_PREFIX):
        await query.answer()
        return

    mode = data[len(TRAFFIC_VIEW_CB_PREFIX) :]
    if mode not in {"refresh", "7d", "4w"}:
        await query.answer("Неизвестный режим", show_alert=True)
        return

    await query.answer("Обновляю статистику…")

    try:
        stats = await tr_call(lambda c: c.session_stats())
    except (TransmissionError, TRCallError) as exc:
        await _edit_traffic_message(query, f"❌ Ошибка Transmission: {html.escape(str(exc))}")
        return

    now = datetime.now()
    downloaded = int(max(0, getattr(stats.cumulative_stats, "downloaded_bytes", 0)))
    uploaded = int(max(0, getattr(stats.cumulative_stats, "uploaded_bytes", 0)))
    anchors, history = _read_traffic_state()
    anchors_changed = _ensure_traffic_anchors(anchors, now, downloaded, uploaded)
    history_changed = _ensure_daily_traffic_history(history, now, downloaded, uploaded)
    if anchors_changed or history_changed:
        try:
            _persist_traffic_state(anchors, history)
        except OSError:
            log.warning("Failed to persist traffic state", exc_info=True)

    if mode == "refresh":
        text = _build_traffic_stats_text(now, downloaded, uploaded, anchors, history)
        await _edit_traffic_message(query, text)
        return

    if mode == "4w":
        day_points, chart_payload, chart_error = await asyncio.to_thread(
            _build_traffic_chart_current_month,
            now,
            downloaded,
            uploaded,
            history,
        )
        if chart_payload is None or day_points is None:
            text = _build_last_4_weeks_text(now, downloaded, uploaded, history)
            if chart_error:
                text = f"{text}\n\n⚠️ {chart_error}"
            await _edit_traffic_message(query, text)
            return

        caption = (
            "🗓️ <b>Трафик по дням (текущий месяц)</b>\n"
            f"Сумма: ⇣ <b>{fmt_bytes(sum(int(item['downloaded']) for item in day_points))}</b> "
            f"| ⇡ <b>{fmt_bytes(sum(int(item['uploaded']) for item in day_points))}</b>"
        )

        if query.message is None:
            await query.answer("Не удалось отправить график", show_alert=True)
            return

        image_file = InputFile(io.BytesIO(chart_payload), filename="traffic_month.png")
        await query.message.reply_photo(
            photo=image_file,
            caption=caption,
            parse_mode=ParseMode.HTML,
            reply_markup=TRAFFIC_OVERVIEW_KEYBOARD,
        )
        await query.answer("График отправлен")
        return

    chart_points = _traffic_points_last_7_days(now, downloaded, uploaded, history)
    chart_payload, chart_error = await asyncio.to_thread(_build_traffic_chart_last_7_days, now, downloaded, uploaded, history)
    if chart_payload is None:
        text = _build_last_7_days_text(now, downloaded, uploaded, history)
        if chart_error:
            text = f"{text}\n\n⚠️ {chart_error}"
        await _edit_traffic_message(query, text)
        return

    caption = (
        "📅 <b>Трафик за последние 7 дней</b>\n"
        f"Сумма: ⇣ <b>{fmt_bytes(sum(int(item['downloaded']) for item in chart_points))}</b> "
        f"| ⇡ <b>{fmt_bytes(sum(int(item['uploaded']) for item in chart_points))}</b>"
    )

    if query.message is None:
        await query.answer("Не удалось отправить график", show_alert=True)
        return

    image_file = InputFile(io.BytesIO(chart_payload), filename="traffic_7d.png")
    await query.message.reply_photo(photo=image_file, caption=caption, parse_mode=ParseMode.HTML, reply_markup=TRAFFIC_OVERVIEW_KEYBOARD)
    await query.answer("График отправлен")


def _is_active(status: str) -> bool:
    return status in ACTIVE_STATUSES


async def send_torrent_list(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    mode: str,
    query: Optional[str] = None,
    edit_existing: bool = False,
) -> None:
    try:
        torrents = await tr_call(lambda c: c.get_torrents())
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Ошибка Transmission: {html.escape(str(exc))}", reply_markup=TORRENT_LIST_KEYBOARD)
        return

    items = torrents
    if mode == "downloading":
        items = [t for t in items if float(t.percent_done) < 1.0]
    elif mode == "stopped":
        items = [t for t in items if str(t.status) == "stopped"]
    elif mode == "done":
        items = [t for t in items if float(t.percent_done) >= 1.0]

    if query:
        q = query.strip().lower()
        items = [t for t in items if q in (t.name or "").lower()]

    total = len(items)
    max_items = CFG.list_limit
    if total > max_items:
        items = heapq.nsmallest(max_items, items, key=lambda t: (0 if _is_active(str(t.status)) else 1, -float(getattr(t, "progress", 0.0)), (t.name or "").lower()))
    else:
        items = _sort_torrents(items)

    ctx.user_data["last_list_mode"] = mode
    ctx.user_data["last_list_query"] = query

    if total == 0:
        await reply_chunks(update, "Пусто.", reply_markup=TORRENT_LIST_KEYBOARD)
        return

    lines = []
    for t in items:
        st = str(t.status)
        safe_name = html.escape(t.name or "<без названия>")
        size_text = fmt_bytes(torrent_total_size(t))
        lines.append(
            f"<b>{t.id}</b> {status_icon(st)} {safe_name} — <b>{t.progress:.2f}%</b> • <b>{size_text}</b>\n"
            f"   ⇣ {fmt_rate(t.rate_download)} | ⇡ {fmt_rate(t.rate_upload)} | Ratio {t.upload_ratio:.2f} | {html.escape(st)}"
        )

    header = {
        "all": "📋 <b>Все торренты</b>",
        "downloading": "⬇️ <b>Скачиваются</b>",
        "stopped": "⏹️ <b>Остановленные</b>",
        "done": "✅ <b>Завершённые</b>",
    }.get(mode, "📋 <b>Список</b>")

    tail = ""
    if total > max_items:
        tail = f"\n\nПоказано: {len(items)} из {total}."

    if edit_existing and update.callback_query is not None:
        await update.callback_query.edit_message_text(
            text=_build_single_torrent_message(header, lines, tail),
            parse_mode=ParseMode.HTML,
            reply_markup=TORRENT_LIST_KEYBOARD,
        )
        return

    messages = _build_torrent_messages(header, lines, tail)
    for idx, text in enumerate(messages):
        await reply_chunks(
            update,
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=TORRENT_LIST_KEYBOARD if idx == len(messages) - 1 else None,
        )


async def add_magnet_or_url(update: Update, ctx: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    link = text.strip()
    if not (link.startswith("magnet:") or link.startswith("http://") or link.startswith("https://")):
        await reply_chunks(update, "❌ Нужна magnet-ссылка или http(s) URL на .torrent.", reply_markup=KB_ADD)
        return

    free_space_before = await _get_download_dir_free_space()

    try:
        torrent = await tr_call(lambda c: c.add_torrent(link))
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Не удалось добавить: {html.escape(str(exc))}", reply_markup=KB_ADD)
        return

    _register_torrent_start_watch(ctx, update.effective_chat.id if update.effective_chat else None, torrent)

    await reply_chunks(
        update,
        (
            f"✅ Добавлено: <b>{html.escape(torrent.name)}</b>\n"
            f"ID: <b>{torrent.id}</b>\n"
            f"{_build_projected_free_space_text(free_space_before, torrent)}"
        ),
        parse_mode=ParseMode.HTML,
        reply_markup=KB_ADD,
    )


async def add_torrent_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or message.document is None:
        await reply_chunks(update, "Пришли .torrent файлом.", reply_markup=KB_ADD)
        return

    doc = message.document
    if not (doc.file_name or "").lower().endswith(".torrent"):
        await reply_chunks(update, "Это не .torrent файл.", reply_markup=KB_ADD)
        return

    tg_file = await doc.get_file()
    free_space_before = await _get_download_dir_free_space()
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

    _register_torrent_start_watch(ctx, update.effective_chat.id if update.effective_chat else None, torrent)

    await reply_chunks(
        update,
        (
            f"✅ Добавлено из файла: <b>{html.escape(torrent.name)}</b>\n"
            f"ID: <b>{torrent.id}</b>\n"
            f"{_build_projected_free_space_text(free_space_before, torrent)}"
        ),
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
            torrent = await tr_call(lambda c: c.get_torrent(torrent_id))
            await tr_call(lambda c: c.remove_torrent(torrent_id, delete_data=False))
            msg = f"🗑️ Удалено (данные сохранены): ID {torrent_id} | {torrent.name}"
        elif action == "del_data":
            torrent = await tr_call(lambda c: c.get_torrent(torrent_id))
            await tr_call(lambda c: c.remove_torrent(torrent_id, delete_data=True))
            msg = f"💥 Удалено вместе с данными: ID {torrent_id} | {torrent.name}"
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

    _ensure_chat_notifications_initialized(ctx, update.effective_chat.id if update.effective_chat else None)

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
        "• 📊 Статус — скорость и текущая активность\n"
        "• 📈 Статистика — сводка + график/детально за 7 дней и по дням текущего месяца\n"
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
        await add_torrent_file(update, ctx)
        set_menu(ctx, MENU_ADD)
        set_wait(ctx, WAIT_NONE)
        await _delete_user_message(update, ctx)
        return

    await send_ephemeral(update, ctx, "Я жду команду из меню 🙂", reply_markup=KB_MAIN)
    await _delete_user_message(update, ctx)


async def _handle_wait_state(update: Update, ctx: ContextTypes.DEFAULT_TYPE, wait: Optional[str], text: str) -> bool:
    if wait == WAIT_SEARCH:
        set_wait(ctx, WAIT_NONE)
        set_menu(ctx, MENU_TORRENTS)
        await send_torrent_list(update, ctx, mode="all", query=text)
        return True

    if wait == WAIT_ADD_MAGNET:
        set_wait(ctx, WAIT_NONE)
        set_menu(ctx, MENU_ADD)
        await add_magnet_or_url(update, ctx, text)
        return True

    if wait in {WAIT_CTRL_PAUSE, WAIT_CTRL_START, WAIT_CTRL_DEL_KEEP, WAIT_CTRL_DEL_DATA}:
        torrent_id = parse_id(text)
        if torrent_id is None:
            await send_ephemeral(update, ctx, "Пришли числовой ID торрента (например: 12).", reply_markup=KB_CTRL)
            return True

        set_wait(ctx, WAIT_NONE)
        set_menu(ctx, MENU_CTRL)
        action_map = {
            WAIT_CTRL_PAUSE: "pause",
            WAIT_CTRL_START: "start",
            WAIT_CTRL_DEL_KEEP: "del_keep",
            WAIT_CTRL_DEL_DATA: "del_data",
        }
        await ctrl_action(update, ctx, action_map[wait], torrent_id=torrent_id)
        return True

    return False


async def _toggle_notifications(update: Update, ctx: ContextTypes.DEFAULT_TYPE, chat_id: Optional[int]) -> None:
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


async def _handle_global_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE, text: str, chat_id: Optional[int]) -> bool:
    async def _open_main_status() -> None:
        set_menu(ctx, MENU_MAIN)
        set_wait(ctx, WAIT_NONE)
        await send_status(update, ctx)

    async def _open_torrents() -> None:
        set_menu(ctx, MENU_TORRENTS)
        set_wait(ctx, WAIT_NONE)
        await send_ephemeral(update, ctx, "Меню торрентов:", reply_markup=KB_TORRENTS)

    async def _open_add() -> None:
        set_menu(ctx, MENU_ADD)
        set_wait(ctx, WAIT_NONE)
        free_space = await _get_download_dir_free_space()
        await send_ephemeral(
            update,
            ctx,
            f"Как будем добавлять?\n{_build_free_space_text(free_space)}",
            reply_markup=KB_ADD,
        )

    async def _open_ctrl() -> None:
        set_menu(ctx, MENU_CTRL)
        set_wait(ctx, WAIT_NONE)
        await send_ephemeral(update, ctx, "Выбери действие:", reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id))

    handlers: dict[str, Callable[[], Awaitable[None]]] = {
        "📊 Статус": _open_main_status,
        "📈 Статистика": lambda: send_traffic_stats(update, ctx),
        "📋 Торренты": _open_torrents,
        "➕ Добавить": _open_add,
        "⚙️ Управление": _open_ctrl,
    }

    handler = handlers.get(text)
    if handler is None:
        return False

    await handler()
    return True


async def _handle_menu_command(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    menu: str,
    text: str,
    chat_id: Optional[int],
) -> bool:
    async def _list_all() -> None:
        await send_torrent_list(update, ctx, mode="all")

    async def _list_downloading() -> None:
        await send_torrent_list(update, ctx, mode="downloading")

    async def _list_stopped() -> None:
        await send_torrent_list(update, ctx, mode="stopped")

    async def _list_done() -> None:
        await send_torrent_list(update, ctx, mode="done")

    async def _ask_search() -> None:
        set_wait(ctx, WAIT_SEARCH)
        await send_ephemeral(update, ctx, "Введи часть названия для поиска:", reply_markup=KB_TORRENTS)

    async def _ask_add_magnet() -> None:
        set_wait(ctx, WAIT_ADD_MAGNET)
        free_space = await _get_download_dir_free_space()
        await send_ephemeral(
            update,
            ctx,
            f"Пришли magnet-ссылку или URL на .torrent:\n{_build_free_space_text(free_space)}",
            reply_markup=KB_ADD,
        )

    async def _ask_add_file() -> None:
        set_wait(ctx, WAIT_ADD_TORRENT_FILE)
        free_space = await _get_download_dir_free_space()
        await send_ephemeral(
            update,
            ctx,
            f"Ок, пришли .torrent файлом сюда в чат.\n{_build_free_space_text(free_space)}",
            reply_markup=KB_ADD,
        )

    async def _ask_pause() -> None:
        set_wait(ctx, WAIT_CTRL_PAUSE)
        await send_ephemeral(update, ctx, "Пришли ID торрента для остановки:", reply_markup=KB_CTRL)

    async def _ask_start() -> None:
        set_wait(ctx, WAIT_CTRL_START)
        await send_ephemeral(update, ctx, "Пришли ID торрента для запуска:", reply_markup=KB_CTRL)

    async def _ask_del_keep() -> None:
        set_wait(ctx, WAIT_CTRL_DEL_KEEP)
        await send_ephemeral(update, ctx, "Пришли ID торрента для удаления (данные останутся на диске):", reply_markup=KB_CTRL)

    async def _ask_del_data() -> None:
        set_wait(ctx, WAIT_CTRL_DEL_DATA)
        await send_ephemeral(update, ctx, "⚠️ Пришли ID торрента для удаления вместе с данными:", reply_markup=KB_CTRL)

    menu_handlers: dict[str, dict[str, Callable[[], Awaitable[None]]]] = {
        MENU_TORRENTS: {
            "📋 Все": _list_all,
            "⬇️ Скачиваются": _list_downloading,
            "⏹️ Остановл.": _list_stopped,
            "✅ Завершённые": _list_done,
            "🔎 Поиск": _ask_search,
        },
        MENU_ADD: {
            "🧲 Магнет/URL": _ask_add_magnet,
            "📄 .torrent файл": _ask_add_file,
        },
        MENU_CTRL: {
            "⏸️ Пауза": _ask_pause,
            "▶️ Старт": _ask_start,
            "🗑️ Удалить (оставить данные)": _ask_del_keep,
            "💥 Удалить (с данными)": _ask_del_data,
            "🔔 Уведомления: ВКЛ": lambda: _toggle_notifications(update, ctx, chat_id),
            "🔕 Уведомления: ВЫКЛ": lambda: _toggle_notifications(update, ctx, chat_id),
        },
    }

    handler = menu_handlers.get(menu, {}).get(text)
    if handler is None:
        return False

    await handler()
    return True


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
    _ensure_chat_notifications_initialized(ctx, chat_id)

    try:
        if text == "⬅️ Назад":
            set_menu(ctx, MENU_MAIN)
            set_wait(ctx, WAIT_NONE)
            await send_ephemeral(update, ctx, "Ок, назад в главное меню.", reply_markup=KB_MAIN)
            return

        menu = get_menu(ctx)
        wait = get_wait(ctx)

        if await _handle_wait_state(update, ctx, wait, text):
            return

        if await _handle_global_command(update, ctx, text, chat_id):
            return

        if await _handle_menu_command(update, ctx, menu, text, chat_id):
            return

        await send_ephemeral(update, ctx, "Не понял. Выбери пункт меню 🙂", reply_markup=KB_MAIN)
    finally:
        await _delete_user_message(update, ctx)


def main() -> None:
    async def notify_completed_torrents(ctx: ContextTypes.DEFAULT_TYPE) -> None:
        enabled_chats = ctx.application.bot_data.get(NOTIFY_ENABLED_CHATS_KEY)
        if not isinstance(enabled_chats, set) or not enabled_chats:
            return

        now_ts = asyncio.get_running_loop().time()

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
            pass
        else:
            for torrent_id in new_ids:
                name = html.escape(completed_now.get(torrent_id, "<без названия>"))
                text = f"✅ Торрент завершён: <b>{name}</b>\nID: <b>{torrent_id}</b>"
                for chat_id in list(enabled_chats):
                    try:
                        await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
                    except TelegramError:
                        log.warning("Failed to send completion notification to chat %s", chat_id, exc_info=True)

        pending_start = ctx.application.bot_data.get(NOTIFY_START_PENDING_KEY)
        if not isinstance(pending_start, dict) or not pending_start:
            return

        torrents_by_id = {int(t.id): t for t in torrents}
        expired_ids: list[int] = []

        for torrent_id, state in list(pending_start.items()):
            if not isinstance(state, dict):
                expired_ids.append(torrent_id)
                continue

            torrent = torrents_by_id.get(torrent_id)
            if torrent is None:
                expired_ids.append(torrent_id)
                continue

            downloaded_ever = float(max(0.0, getattr(torrent, "downloaded_ever", 0.0)))
            percent_done = float(max(0.0, getattr(torrent, "percent_done", 0.0)))
            if downloaded_ever > 0.0 or percent_done > 0.0:
                expired_ids.append(torrent_id)
                continue

            added_at = state.get("added_at")
            if not isinstance(added_at, (int, float)):
                expired_ids.append(torrent_id)
                continue

            if now_ts - float(added_at) < NOTIFY_NO_PEERS_DELAY_SEC:
                continue

            chat_ids = state.get("chat_ids")
            if not isinstance(chat_ids, set) or not chat_ids:
                expired_ids.append(torrent_id)
                continue

            safe_name = html.escape(str(state.get("name") or getattr(torrent, "name", "<без названия>")))
            text = (
                "⚠️ <b>Торрент не начал скачиваться за 10 минут</b>\n"
                f"Возможно, сейчас нет раздающих.\n"
                f"Торрент: <b>{safe_name}</b>\n"
                f"ID: <b>{torrent_id}</b>"
            )

            for chat_id in list(chat_ids):
                if chat_id not in enabled_chats:
                    continue
                try:
                    await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
                except TelegramError:
                    log.warning("Failed to send no-peers notification to chat %s", chat_id, exc_info=True)

            expired_ids.append(torrent_id)

        for torrent_id in expired_ids:
            pending_start.pop(torrent_id, None)

    async def snapshot_traffic_anchors(ctx: ContextTypes.DEFAULT_TYPE) -> None:
        now = datetime.now()
        day_key = now.strftime("%Y-%m-%d")
        last_snapshot_day = ctx.application.bot_data.get(TRAFFIC_LAST_SNAPSHOT_DAY_KEY)
        if last_snapshot_day == day_key:
            return

        try:
            stats = await tr_call(lambda c: c.session_stats())
        except (TransmissionError, TRCallError):
            log.warning("Skipping traffic anchor snapshot due to Transmission error", exc_info=True)
            return

        downloaded = int(max(0, getattr(stats.cumulative_stats, "downloaded_bytes", 0)))
        uploaded = int(max(0, getattr(stats.cumulative_stats, "uploaded_bytes", 0)))
        anchors, history = _read_traffic_state()
        anchors_changed = _ensure_traffic_anchors(anchors, now, downloaded, uploaded)
        history_changed = _ensure_daily_traffic_history(history, now, downloaded, uploaded)
        if anchors_changed or history_changed:
            try:
                _persist_traffic_state(anchors, history)
            except OSError:
                log.warning("Failed to persist traffic state", exc_info=True)
        ctx.application.bot_data[TRAFFIC_LAST_SNAPSHOT_DAY_KEY] = day_key

    async def notify_completed_torrents_fallback(app: Application) -> None:
        while True:
            fake_ctx = SimpleNamespace(application=app, bot=app.bot)
            await notify_completed_torrents(fake_ctx)
            await snapshot_traffic_anchors(fake_ctx)
            await asyncio.sleep(NOTIFY_POLL_INTERVAL_SEC)

    async def on_post_init(app: Application) -> None:
        if app.job_queue is None:
            app.bot_data["notify_poll_task"] = asyncio.create_task(notify_completed_torrents_fallback(app))
            log.warning(
                "python-telegram-bot job queue is unavailable; using fallback polling task "
                "for completion notifications and traffic snapshots."
            )

    async def on_post_shutdown(app: Application) -> None:
        task = app.bot_data.pop("notify_poll_task", None)
        if isinstance(task, asyncio.Task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app: Application = (
        ApplicationBuilder()
        .token(CFG.tg_token)
        .post_init(on_post_init)
        .post_shutdown(on_post_shutdown)
        .build()
    )

    if app.job_queue is None:
        log.info("Job queue is unavailable; fallback polling task will be used.")
    else:
        app.job_queue.run_repeating(
            notify_completed_torrents,
            interval=NOTIFY_POLL_INTERVAL_SEC,
            first=NOTIFY_POLL_INTERVAL_SEC,
        )
        app.job_queue.run_daily(snapshot_traffic_anchors, time=time(hour=0, minute=0, second=0))

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))

    app.add_handler(CallbackQueryHandler(on_status_refresh, pattern=f"^{STATUS_REFRESH_CB}$"))
    app.add_handler(CallbackQueryHandler(on_list_refresh, pattern=f"^{LIST_REFRESH_CB_PREFIX}(all|downloading|done)$"))
    app.add_handler(CallbackQueryHandler(on_traffic_view, pattern=f"^{TRAFFIC_VIEW_CB_PREFIX}(refresh|7d|4w)$"))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    log.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
