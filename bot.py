#!/usr/bin/env python3
"""Telegram bot for managing Transmission RPC via menu buttons."""

from __future__ import annotations

import asyncio
import contextlib
import html
import io
import logging
import os
import re
import secrets
import sqlite3
import struct
import tempfile
import threading
import time as time_module
import zlib
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from math import ceil, isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Coroutine, Literal, Optional, Sequence, cast
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message, ReplyKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError, TimedOut
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from transmission_rpc import Client, from_url
from transmission_rpc.error import TransmissionAuthError, TransmissionConnectError, TransmissionError

from state_store import OutboxItem, Snapshot, SQLiteStateStore, StartWatch

_BOT_TOKEN_RE = re.compile(r"(?i)(/bot)(\d{6,}:[A-Za-z0-9_-]{20,})")
_URL_CREDENTIALS_RE = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)([^/@\s:]+):([^/@\s]+)@")
_LOG_SECRETS: tuple[str, ...] = ()


def _redact_log_text(value: str) -> str:
    redacted = _BOT_TOKEN_RE.sub(r"\1<redacted>", value)
    redacted = _URL_CREDENTIALS_RE.sub(r"\1***:***@", redacted)
    for secret in _LOG_SECRETS:
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


class EventTimeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        formatted = _redact_log_text(super().format(record))
        if "\n" not in formatted:
            return formatted

        prefix = self._line_prefix(record)
        return "\n".join(line if index == 0 else f"{prefix}{line}" for index, line in enumerate(formatted.splitlines()))

    def _line_prefix(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        return f"{timestamp}.{int(record.msecs):03d}Z | {record.levelname} | {record.name} | "


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


def configure_logging() -> logging.Logger:
    global _LOG_SECRETS

    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    log_format = "%(asctime)s.%(msecs)03dZ | %(levelname)s | %(name)s | %(message)s"
    log_date_format = "%Y-%m-%dT%H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        with contextlib.suppress(Exception):
            handler.close()
    root_logger.handlers.clear()

    secrets = {
        os.environ.get("TG_TOKEN", "").strip(),
        os.environ.get("TR_PASS", "").strip(),
    }
    for env_name in ("TG_PROXY", "TG_GET_UPDATES_PROXY", "TR_URL"):
        raw_url = os.environ.get(env_name, "").strip()
        if not raw_url:
            continue
        with contextlib.suppress(ValueError):
            password = urlsplit(raw_url).password
            if password:
                secrets.add(password)
    _LOG_SECRETS = tuple(sorted((secret for secret in secrets if len(secret) >= 4), key=len, reverse=True))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = EventTimeFormatter(log_format, datefmt=log_date_format)
    console_formatter.converter = time_module.gmtime
    console_handler.setFormatter(console_formatter)
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
    file_formatter = EventTimeFormatter(log_format, datefmt=log_date_format)
    file_formatter.converter = time_module.gmtime
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # HTTPX logs the full Telegram Bot API URL (including the token) at INFO.
    # Keep transport loggers quiet even when application debug logging is enabled.
    for logger_name in ("httpx", "httpcore", "transmission-rpc"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger = logging.getLogger("tg-transmission-bot")
    logger.info("Error logs will be written to %s", log_file_path)
    return logger


log = logging.getLogger("tg-transmission-bot")


TG_MAX_MESSAGE = 4096
TORRENT_FILE_MAX_BYTES = 10 * 1024 * 1024
TORRENT_ID_RE = re.compile(r"\b(\d{1,9})\b")
_TR_CLIENT: Optional[Client] = None
_TR_CLIENT_LOCK = threading.Lock()
_TR_CALL_LOCK: Optional[asyncio.Lock] = None
_OUTBOX_DRAIN_LOCK: Optional[asyncio.Lock] = None
SUPPORTED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}
TransmissionProtocol = Literal["http", "https"]


class TRCallError(Exception):
    """Wrapper for errors during Transmission RPC calls."""


@dataclass(frozen=True)
class Config:
    tg_token: str = dataclass_field(repr=False)
    allowed_user_ids: Optional[set[int]]
    allow_all_users: bool
    tg_proxy: Optional[str] = dataclass_field(repr=False)
    tg_get_updates_proxy: Optional[str] = dataclass_field(repr=False)
    hysteria2_socks5_proxy: Optional[str] = dataclass_field(repr=False)

    tr_url: Optional[str] = dataclass_field(repr=False)
    tr_protocol: TransmissionProtocol
    tr_host: str
    tr_port: int
    tr_path: str
    tr_user: Optional[str] = dataclass_field(repr=False)
    tr_pass: Optional[str] = dataclass_field(repr=False)
    tr_timeout: float

    list_limit: int
    state_dir: Path
    timezone: ZoneInfo
    timezone_name: str


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
        log.warning("No valid ALLOWED_USER_IDS values found")
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

    if not isfinite(value) or value <= min_exclusive:
        raise RuntimeError(f"{name} must be a finite number > {min_exclusive:g}")
    return value


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be one of: 1/0, true/false, yes/no, on/off")


def _parse_timezone_env() -> tuple[str, ZoneInfo]:
    timezone_name = os.environ.get("BOT_TIMEZONE", "UTC").strip() or "UTC"
    try:
        return timezone_name, ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"BOT_TIMEZONE has unknown timezone: {timezone_name}") from exc


def _normalize_proxy_url(raw_url: Optional[str], *, env_name: str) -> Optional[str]:
    if not raw_url:
        return None

    if any(character.isspace() for character in raw_url):
        raise RuntimeError(f"{env_name} must not contain whitespace")

    try:
        parts = urlsplit(raw_url)
        proxy_port = parts.port
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be a valid proxy URL") from exc
    scheme = parts.scheme.lower()
    if not scheme or not parts.netloc or parts.hostname is None:
        raise RuntimeError(f"{env_name} must be a valid proxy URL")

    if scheme == "mtproto":
        raise RuntimeError(f"{env_name} does not support mtproto:// for Telegram Bot API; use http(s):// or socks5://")

    if scheme not in SUPPORTED_PROXY_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_PROXY_SCHEMES))
        raise RuntimeError(f"{env_name} has unsupported proxy scheme '{scheme}', supported: {supported}")

    if proxy_port is not None and not 1 <= proxy_port <= 65535:
        raise RuntimeError(f"{env_name} proxy port must be in 1..65535")

    return raw_url


def _normalize_hysteria2_proxy_url(raw_url: Optional[str]) -> Optional[str]:
    proxy_url = _normalize_proxy_url(raw_url, env_name="HYSTERIA2_SOCKS5_PROXY")
    if proxy_url and urlsplit(proxy_url).scheme.lower() not in {"socks5", "socks5h"}:
        raise RuntimeError("HYSTERIA2_SOCKS5_PROXY must use socks5:// or socks5h://")
    return proxy_url


def _resolve_telegram_proxy_urls(
    tg_proxy: Optional[str],
    tg_get_updates_proxy: Optional[str],
    hysteria2_socks5_proxy: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve explicit Telegram proxies before the Hysteria 2 fallback."""
    bot_proxy = tg_proxy or hysteria2_socks5_proxy
    get_updates_proxy = tg_get_updates_proxy or tg_proxy or hysteria2_socks5_proxy
    return bot_proxy, get_updates_proxy


def _validate_transmission_url(raw_url: Optional[str]) -> Optional[str]:
    if not raw_url:
        return None
    if any(character.isspace() for character in raw_url):
        raise RuntimeError("TR_URL must not contain whitespace")

    try:
        parts = urlsplit(raw_url)
        port = parts.port
    except ValueError as exc:
        raise RuntimeError("TR_URL must be a valid HTTP(S) URL") from exc

    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc or parts.hostname is None:
        raise RuntimeError("TR_URL must be a valid HTTP(S) URL")
    if port is not None and not 1 <= port <= 65535:
        raise RuntimeError("TR_URL port must be in 1..65535")
    if parts.query or parts.fragment:
        raise RuntimeError("TR_URL must not contain a query string or fragment")
    return raw_url


def _mask_proxy_url(raw_url: Optional[str]) -> str:
    if not raw_url:
        return "direct"

    try:
        parts = urlsplit(raw_url)
        _ = parts.port
    except ValueError:
        return "<invalid URL>"
    if not parts.scheme or not parts.netloc or not parts.hostname:
        return "<invalid URL>"

    safe_netloc = parts.netloc
    if "@" in safe_netloc:
        _, _, hostinfo = safe_netloc.rpartition("@")
        safe_netloc = f"***:***@{hostinfo}"
    return urlunsplit((parts.scheme, safe_netloc, parts.path, "", ""))


def _parse_transmission_fallback_endpoint() -> tuple[TransmissionProtocol, str, int, str]:
    tr_protocol_raw = os.environ.get("TR_PROTOCOL", "http").strip().lower()
    if tr_protocol_raw == "http":
        tr_protocol: TransmissionProtocol = "http"
    elif tr_protocol_raw == "https":
        tr_protocol = "https"
    else:
        raise RuntimeError("TR_PROTOCOL must be 'http' or 'https'")

    tr_port = _parse_int_env("TR_PORT", "9091", min_value=1, max_value=65535)
    tr_host = os.environ.get("TR_HOST", "127.0.0.1").strip()
    if not tr_host or any(character.isspace() for character in tr_host) or "://" in tr_host or "/" in tr_host:
        raise RuntimeError("TR_HOST must be a hostname or IP address without a URL scheme or path")
    tr_path = os.environ.get("TR_PATH", "/transmission/rpc").strip()
    if (
        not tr_path.startswith("/")
        or any(character.isspace() for character in tr_path)
        or "?" in tr_path
        or "#" in tr_path
    ):
        raise RuntimeError("TR_PATH must be an absolute URL path without whitespace, query string, or fragment")
    return tr_protocol, tr_host, tr_port, tr_path


def load_config() -> Config:
    tg_token = os.environ.get("TG_TOKEN", "").strip()
    if not tg_token:
        raise RuntimeError("ENV TG_TOKEN is required")

    tg_proxy = _normalize_proxy_url(os.environ.get("TG_PROXY", "").strip() or None, env_name="TG_PROXY")
    tg_get_updates_proxy = _normalize_proxy_url(
        os.environ.get("TG_GET_UPDATES_PROXY", "").strip() or None,
        env_name="TG_GET_UPDATES_PROXY",
    )
    hysteria2_socks5_proxy = _normalize_hysteria2_proxy_url(
        os.environ.get("HYSTERIA2_SOCKS5_PROXY", "").strip() or None
    )
    tr_url = _validate_transmission_url(os.environ.get("TR_URL", "").strip() or None)

    if tr_url is None:
        tr_protocol, tr_host, tr_port, tr_path = _parse_transmission_fallback_endpoint()
    else:
        # These fields are unused when TR_URL is configured. Ignore stale
        # fallback values instead of preventing startup with a valid URL.
        tr_protocol, tr_host, tr_port, tr_path = "http", "127.0.0.1", 9091, "/transmission/rpc"

    tr_timeout = _parse_float_env("TR_TIMEOUT", "10", min_exclusive=0.0)
    list_limit = _parse_int_env("LIST_LIMIT", "25", min_value=1)
    allowed_user_ids = _parse_allowed_ids(os.environ.get("ALLOWED_USER_IDS", ""))
    allow_all_users = _parse_bool_env("ALLOW_ALL_USERS", default=False)
    timezone_name, bot_timezone = _parse_timezone_env()
    state_dir_raw = os.environ.get("STATE_DIR", "").strip()
    state_dir = Path(state_dir_raw).expanduser() if state_dir_raw else Path(__file__).resolve().parent
    if allowed_user_ids is None:
        if allow_all_users:
            log.warning(
                "ALLOWED_USER_IDS is empty and ALLOW_ALL_USERS is enabled; every private Telegram user is allowed"
            )
        else:
            log.warning("ALLOWED_USER_IDS is empty; access is denied until at least one Telegram user id is configured")

    return Config(
        tg_token=tg_token,
        allowed_user_ids=allowed_user_ids,
        allow_all_users=allow_all_users,
        tg_proxy=tg_proxy,
        tg_get_updates_proxy=tg_get_updates_proxy,
        hysteria2_socks5_proxy=hysteria2_socks5_proxy,
        tr_url=tr_url,
        tr_protocol=tr_protocol,
        tr_host=tr_host,
        tr_port=tr_port,
        tr_path=tr_path,
        tr_user=os.environ.get("TR_USER", "").strip() or None,
        tr_pass=os.environ.get("TR_PASS", "").strip() or None,
        tr_timeout=tr_timeout,
        list_limit=list_limit,
        state_dir=state_dir,
        timezone=bot_timezone,
        timezone_name=timezone_name,
    )


CFG: Optional[Config] = None
STATE_STORE: Optional[SQLiteStateStore] = None


def get_config() -> Config:
    if CFG is None:
        raise RuntimeError("Configuration is not initialized")
    return CFG


def get_state_store() -> SQLiteStateStore:
    if STATE_STORE is None:
        raise RuntimeError("State store is not initialized")
    return STATE_STORE


def bot_now() -> datetime:
    return datetime.now(get_config().timezone)


def initialize_runtime() -> None:
    global CFG, CONFIRM_DEL_KEEP_FLOW, STATE_STORE
    global TORRENT_HISTORY_LOCK, TRAFFIC_STATE_LOCK, _OUTBOX_DRAIN_LOCK, _TR_CALL_LOCK, _TR_CLIENT, log

    load_dotenv_file(Path(__file__).resolve().with_name(".env"))
    log = configure_logging()
    CFG = load_config()
    CFG.state_dir.mkdir(parents=True, exist_ok=True)
    STATE_STORE = SQLiteStateStore(
        CFG.state_dir / "bot-state.sqlite3",
        legacy_traffic_path=CFG.state_dir / "traffic_anchors.json",
        legacy_torrent_history_path=CFG.state_dir / "torrent_history.json",
        logger=log,
    )
    STATE_STORE.initialize()
    CONFIRM_DEL_KEEP_FLOW = _parse_bool_env("CONFIRM_DEL_KEEP", default=False)
    _TR_CALL_LOCK = None
    _OUTBOX_DRAIN_LOCK = None
    TRAFFIC_STATE_LOCK = None
    TORRENT_HISTORY_LOCK = None
    with _TR_CLIENT_LOCK:
        _TR_CLIENT = None


MENU_MAIN = "MAIN"
MENU_TORRENTS = "TORRENTS"
MENU_ADD = "ADD"
MENU_CTRL = "CTRL"
MENU_HISTORY = "HISTORY"

WAIT_NONE = None
WAIT_SEARCH = "WAIT_SEARCH"
WAIT_ADD_MAGNET = "WAIT_ADD_MAGNET"
WAIT_ADD_TORRENT_FILE = "WAIT_ADD_TORRENT_FILE"
WAIT_CTRL_PAUSE = "WAIT_CTRL_PAUSE"
WAIT_CTRL_START = "WAIT_CTRL_START"
WAIT_CTRL_DEL_KEEP = "WAIT_CTRL_DEL_KEEP"
WAIT_CTRL_DEL_DATA = "WAIT_CTRL_DEL_DATA"
CONFIRM_DEL_KEEP_FLOW = False

CANCEL_INPUT_BUTTON = "❌ Отменить ввод"


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
            ["📈 Статистика", "📚 История раздач"],
        ]
    )


def kb_torrents() -> ReplyKeyboardMarkup:
    return kb(
        [
            ["📋 Все", "⬇️ Скачиваются"],
            ["⏹️ Остановл.", "✅ Завершённые"],
            ["🔎 Поиск"],
            ["⬅️ Назад"],
        ]
    )


def kb_add(*, input_active: bool = False) -> ReplyKeyboardMarkup:
    rows = [["🧲 Магнет/URL", "📄 .torrent файл"]]
    if input_active:
        rows.append([CANCEL_INPUT_BUTTON])
    rows.append(["⬅️ Назад"])
    return kb(rows)


def kb_ctrl(notify_enabled: bool = True, *, input_active: bool = False) -> ReplyKeyboardMarkup:
    notify_label = "🔔 Уведомления: ВКЛ" if notify_enabled else "🔕 Уведомления: ВЫКЛ"
    rows = [
        ["⏸️ Пауза", "▶️ Старт"],
        ["🗑️ Удалить (оставить данные)"],
        ["💥 Удалить (с данными)"],
        [notify_label],
    ]
    if input_active:
        rows.append([CANCEL_INPUT_BUTTON])
    rows.append(["⬅️ Назад"])
    return kb(rows)


KB_MAIN = kb_main()
KB_TORRENTS = kb_torrents()
KB_ADD = kb_add()
KB_ADD_INPUT = kb_add(input_active=True)
KB_CTRL = kb_ctrl(True)
KB_CTRL_INPUT = kb_ctrl(True, input_active=True)

STATUS_REFRESH_CB = "status_refresh"
LIST_REFRESH_CB_PREFIX = "list_refresh:"
TRAFFIC_VIEW_CB_PREFIX = "traffic_view:"
TORRENT_HISTORY_CB_PREFIX = "torrent_history:"
TORRENT_ACTION_CB_PREFIX = "torrent_action:"
CONFIRM_DEL_DATA_CB_PREFIX = "confirm_del_data:"
CANCEL_DEL_DATA_CB = "cancel_del_data"
CONFIRM_DEL_KEEP_CB_PREFIX = "confirm_del_keep:"
CANCEL_DEL_KEEP_CB = "cancel_del_keep"
LAST_EPHEMERAL_MESSAGE_KEY = "last_ephemeral_message_id"
PENDING_CTRL_ACTION_KEY = "pending_ctrl_action"
NOTIFY_ENABLED_CHATS_KEY = "notify_enabled_chat_ids"
NOTIFY_KNOWN_CHATS_KEY = "notify_known_chat_ids"
NOTIFY_START_PENDING_KEY = "notify_start_pending"
NOTIFY_START_TASKS_KEY = "notify_start_tasks"
TORRENT_HISTORY_LAST_PAGE_KEY = "torrent_history_last_page"
TORRENT_LIST_LAST_MODE_KEY = "last_list_mode"
TORRENT_LIST_LAST_QUERY_KEY = "last_list_query"
TORRENT_LIST_SEARCH_REVISION_KEY = "last_list_search_revision"

TORRENT_LIST_PAGE_SIZE_MAX = 8
TORRENT_LIST_MODES = frozenset({"all", "downloading", "stopped", "done"})
TORRENT_LIST_VIEW_MODES = frozenset({*TORRENT_LIST_MODES, "search"})
TORRENT_LIST_SEARCH_VIEW_PREFIX = "search."
TORRENT_LIST_SEARCH_REVISION_RE = re.compile(r"^[0-9a-f]{8}$")

NOTIFY_POLL_INTERVAL_SEC = 60
NOTIFY_START_QUICK_DELAYS_SEC = (0.0, 2.0, 5.0, 15.0, 30.0)
NOTIFY_NO_PEERS_DELAY_SEC = 10 * 60
TRAFFIC_STATE_LOCK: Optional[asyncio.Lock] = None
TORRENT_HISTORY_LOCK: Optional[asyncio.Lock] = None
TORRENT_HISTORY_FIELDS = (
    "id",
    "hashString",
    "name",
    "status",
    "totalSize",
    "sizeWhenDone",
    "downloadedEver",
    "uploadedEver",
    "uploadRatio",
    "percentDone",
    "rateDownload",
    "leftUntilDone",
    "addedDate",
    "doneDate",
)
DOWNLOADING_STATUSES = frozenset(
    {
        "downloading",
        "download pending",
    }
)
ACTIVE_STATUSES = frozenset(
    {
        *DOWNLOADING_STATUSES,
        "seeding",
        "seed pending",
        "checking",
        "check pending",
    }
)
COMPLETED_STATUSES = frozenset({"seeding", "seed pending"})
STATUS_ICONS = {
    "downloading": "⬇️",
    "download pending": "⬇️",
    "seeding": "⬆️",
    "seed pending": "⬆️",
    "checking": "🧪",
    "check pending": "🧪",
    "stopped": "⏸️",
}


def _require_user_data(ctx: ContextTypes.DEFAULT_TYPE) -> dict[Any, Any]:
    user_data = ctx.user_data
    if user_data is None:
        raise RuntimeError("User data is unavailable for this context")
    return user_data


def set_menu(ctx: ContextTypes.DEFAULT_TYPE, menu: str) -> None:
    _require_user_data(ctx)["menu"] = menu


def get_menu(ctx: ContextTypes.DEFAULT_TYPE) -> str:
    return str(_require_user_data(ctx).get("menu", MENU_MAIN))


def set_wait(ctx: ContextTypes.DEFAULT_TYPE, wait: Optional[str]) -> None:
    _require_user_data(ctx)["wait"] = wait


def get_wait(ctx: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    value = _require_user_data(ctx).get("wait", WAIT_NONE)
    return str(value) if isinstance(value, str) else None


def user_allowed(update: Update) -> bool:
    chat = update.effective_chat
    if chat and chat.type != "private":
        return False

    cfg = get_config()
    if cfg.allowed_user_ids is None:
        return cfg.allow_all_users
    uid = update.effective_user.id if update.effective_user else None
    return uid in cfg.allowed_user_ids


async def callback_user_allowed(update: Update) -> bool:
    if user_allowed(update):
        return True

    query = update.callback_query
    if query is not None:
        await query.answer("Доступ запрещён", show_alert=True)
    return False


def _sort_torrents(items: Sequence[Any]) -> list[Any]:
    return sorted(
        items,
        key=lambda t: (
            0 if _is_active(str(t.status)) else 1,
            -torrent_progress_percent(t),
            (t.name or "").lower(),
        ),
    )


def fmt_bytes(n: int | float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    try:
        x = float(n)
    except (OverflowError, TypeError, ValueError):
        x = 0.0
    if not isfinite(x):
        x = 0.0
    x = max(0.0, x)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    if i == 0:
        value = str(int(x))
    elif x >= 100:
        value = f"{x:.0f}"
    elif x >= 10:
        value = f"{x:.1f}"
    else:
        value = f"{x:.2f}"
    return f"{value} {units[i]}"


def fmt_rate(bps: int | float) -> str:
    return f"{fmt_bytes(bps)}/s"


def _clamp_progress(value: int | float) -> float:
    normalized = float(value)
    if not isfinite(normalized):
        return 0.0
    return min(100.0, max(0.0, normalized))


def torrent_progress_percent(torrent: Any) -> float:
    progress = _get_mapping_or_attr_value(torrent, ("progress",))
    if isinstance(progress, (int, float)):
        return _clamp_progress(progress)

    percent_done = _get_mapping_or_attr_value(
        torrent,
        ("percent_done", "percentDone", "percent-done"),
    )
    if isinstance(percent_done, (int, float)):
        return _clamp_progress(percent_done * 100.0)

    return 0.0


def _progress_segment_symbol(index: int, width: int) -> str:
    position = (index + 1) / max(1, width)
    if position <= 1 / 3:
        return "🟥"
    if position <= 2 / 3:
        return "🟨"
    return "🟩"


def _format_progress_bar(progress: int | float, *, width: int = 9) -> str:
    normalized = _clamp_progress(progress)
    filled = int((normalized * width / 100.0) + 0.5)
    if normalized > 0 and filled == 0:
        filled = 1
    if normalized >= 100:
        filled = width
    filled_bar = "".join(_progress_segment_symbol(index, width) for index in range(filled))
    return f"{filled_bar}{'⬜' * (width - filled)}"


def torrent_total_size(torrent: Any) -> int:
    for names in (
        ("total_size", "totalSize", "total-size"),
        ("size_when_done", "sizeWhenDone", "size-when-done"),
    ):
        value = _get_mapping_or_attr_value(torrent, names)
        normalized = _non_negative_int(value)
        if normalized is not None and normalized > 0:
            return normalized
    return 0


def _torrent_left_until_done(torrent: Any) -> Optional[int]:
    value = _get_mapping_or_attr_value(
        torrent,
        ("left_until_done", "leftUntilDone", "left-until-done"),
    )
    return _non_negative_int(value)


def _torrent_eta_seconds(torrent: Any) -> Optional[int]:
    eta = _get_mapping_or_attr_value(torrent, ("eta",))
    if isinstance(eta, (int, float)):
        return _non_negative_int(eta)

    total_seconds = getattr(eta, "total_seconds", None)
    if callable(total_seconds):
        raw_seconds = total_seconds()
        normalized_seconds = _non_negative_int(raw_seconds)
        if normalized_seconds is not None:
            return normalized_seconds

    left_until_done = _torrent_left_until_done(torrent)
    rate_download = _get_mapping_or_attr_value(
        torrent,
        ("rate_download", "rateDownload", "rate-download"),
    )
    if (
        left_until_done is not None
        and left_until_done > 0
        and (normalized_rate := _non_negative_float(rate_download)) is not None
        and normalized_rate > 0
    ):
        estimated_seconds = left_until_done / normalized_rate
        return int(ceil(estimated_seconds)) if isfinite(estimated_seconds) else None

    if torrent_progress_percent(torrent) >= 100:
        return 0
    return None


def _format_eta(torrent: Any) -> str:
    eta_seconds = _torrent_eta_seconds(torrent)
    if eta_seconds is None:
        return "осталось: неизвестно"
    if eta_seconds <= 0:
        return "готово"
    return f"осталось: {_format_download_duration(eta_seconds)}"


def _format_progress_summary(torrent: Any, *, hide_completed_bar: bool = False) -> str:
    progress = torrent_progress_percent(torrent)
    progress_text = f"<b>{progress:.1f}%</b> · {_format_eta(torrent)}"
    if hide_completed_bar and _is_torrent_completed(torrent):
        return progress_text
    return f"{_format_progress_bar(progress)} {progress_text}"


def status_icon(status: str) -> str:
    return STATUS_ICONS.get(status, "❔")


def parse_id(text: str) -> Optional[int]:
    match = TORRENT_ID_RE.search(text.strip())
    return int(match.group(1)) if match else None


def build_client() -> Client:
    cfg = get_config()
    if cfg.tr_url:
        return from_url(cfg.tr_url, timeout=cfg.tr_timeout)
    return Client(
        protocol=cfg.tr_protocol,
        host=cfg.tr_host,
        port=cfg.tr_port,
        path=cfg.tr_path,
        username=cfg.tr_user,
        password=cfg.tr_pass,
        timeout=cfg.tr_timeout,
    )


def get_client() -> Client:
    global _TR_CLIENT
    if _TR_CLIENT is None:
        with _TR_CLIENT_LOCK:
            if _TR_CLIENT is None:
                _TR_CLIENT = build_client()
    return _TR_CLIENT


def _reset_client() -> None:
    global _TR_CLIENT
    with _TR_CLIENT_LOCK:
        _TR_CLIENT = None


def _get_tr_call_lock() -> asyncio.Lock:
    global _TR_CALL_LOCK
    if _TR_CALL_LOCK is None:
        _TR_CALL_LOCK = asyncio.Lock()
    return _TR_CALL_LOCK


def _get_traffic_state_lock() -> asyncio.Lock:
    global TRAFFIC_STATE_LOCK
    if TRAFFIC_STATE_LOCK is None:
        TRAFFIC_STATE_LOCK = asyncio.Lock()
    return TRAFFIC_STATE_LOCK


def _get_torrent_history_lock() -> asyncio.Lock:
    global TORRENT_HISTORY_LOCK
    if TORRENT_HISTORY_LOCK is None:
        TORRENT_HISTORY_LOCK = asyncio.Lock()
    return TORRENT_HISTORY_LOCK


async def tr_call(
    fn: Callable[[Client], Any],
    *,
    retry_on_connection: bool = True,
    operation: str = "rpc",
) -> Any:
    def _run() -> Any:
        client = get_client()
        return fn(client)

    def _call() -> Any:
        try:
            return _run()
        except TransmissionAuthError as exc:
            log.warning("Transmission authentication failed during %s", operation)
            raise TRCallError("Transmission authentication failed") from exc
        except TransmissionConnectError as exc:
            _reset_client()
            if not retry_on_connection:
                log.warning("Transmission connection failed during non-retryable %s", operation)
                raise TRCallError("Transmission RPC connection failed") from exc

            log.warning("Transmission connection failed during %s; retrying once", operation)
            try:
                return _run()
            except TransmissionAuthError as retry_exc:
                raise TRCallError("Transmission authentication failed") from retry_exc
            except TransmissionError as retry_exc:
                raise TRCallError("Transmission RPC connection failed after retry") from retry_exc
        except TransmissionError as exc:
            log.warning("Transmission RPC rejected %s: %s", operation, exc)
            raise TRCallError("Transmission RPC request failed") from exc

    async with _get_tr_call_lock():
        return await asyncio.to_thread(_call)


def build_telegram_application(
    *,
    post_init: Callable[[Application], Coroutine[Any, Any, None]],
    post_shutdown: Callable[[Application], Coroutine[Any, Any, None]],
) -> Application:
    cfg = get_config()
    builder = ApplicationBuilder().token(cfg.tg_token).post_init(post_init).post_shutdown(post_shutdown)

    configured_tg_proxy = _normalize_proxy_url(cfg.tg_proxy, env_name="TG_PROXY")
    configured_updates_proxy = _normalize_proxy_url(
        cfg.tg_get_updates_proxy,
        env_name="TG_GET_UPDATES_PROXY",
    )
    hysteria2_proxy = _normalize_hysteria2_proxy_url(cfg.hysteria2_socks5_proxy)
    tg_proxy, tg_get_updates_proxy = _resolve_telegram_proxy_urls(
        configured_tg_proxy,
        configured_updates_proxy,
        hysteria2_proxy,
    )

    if not tg_proxy and not tg_get_updates_proxy:
        log.info("Telegram proxy is not configured; using direct connection")
        return builder.build()

    if tg_proxy:
        builder = builder.proxy(tg_proxy)
    if tg_get_updates_proxy:
        builder = builder.get_updates_proxy(tg_get_updates_proxy)

    source = "Hysteria 2 SOCKS5 fallback" if hysteria2_proxy and not configured_tg_proxy else "configured proxy"
    log.info(
        "Telegram proxy enabled via %s (bot=%s, get_updates=%s)",
        source,
        _mask_proxy_url(tg_proxy),
        _mask_proxy_url(tg_get_updates_proxy),
    )

    return builder.build()


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
        send_kwargs = kwargs.copy()
        await _send_with_timeout_retry(
            lambda send_kwargs=send_kwargs: message.reply_text(**send_kwargs),
            op_name="reply_text",
        )


async def _send_with_timeout_retry(
    sender: Callable[[], Awaitable[Any]],
    *,
    op_name: str,
    attempts: int = 3,
    base_delay_sec: float = 1.0,
) -> Any:
    last_error: Optional[TimedOut] = None

    for attempt in range(1, max(1, attempts) + 1):
        try:
            return await sender()
        except TimedOut as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = base_delay_sec * attempt
            log.warning(
                "Telegram request timed out during %s (attempt %d/%d), retrying in %.1fs",
                op_name,
                attempt,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)

    raise last_error if last_error is not None else RuntimeError(f"{op_name} failed without TimedOut exception")


STATUS_KEYBOARD = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Обновить статус", callback_data=STATUS_REFRESH_CB)]])
TORRENT_LIST_KEYBOARD = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("📋 Все", callback_data=f"{LIST_REFRESH_CB_PREFIX}all:0"),
            InlineKeyboardButton("⬇️ Скачиваются", callback_data=f"{LIST_REFRESH_CB_PREFIX}downloading:0"),
        ],
        [
            InlineKeyboardButton("⏸️ Остановлены", callback_data=f"{LIST_REFRESH_CB_PREFIX}stopped:0"),
            InlineKeyboardButton("✅ Завершённые", callback_data=f"{LIST_REFRESH_CB_PREFIX}done:0"),
        ],
    ]
)

TRAFFIC_OVERVIEW_KEYBOARD = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("🔄 Обновить статистику", callback_data=f"{TRAFFIC_VIEW_CB_PREFIX}refresh")],
        [InlineKeyboardButton("📅 Последние 7 дней", callback_data=f"{TRAFFIC_VIEW_CB_PREFIX}7d")],
        [InlineKeyboardButton("🗓️ По дням (месяц)", callback_data=f"{TRAFFIC_VIEW_CB_PREFIX}4w")],
    ]
)


def _list_view_callback_data(view_mode: str, page: int) -> str:
    return f"{LIST_REFRESH_CB_PREFIX}{view_mode}:{max(0, page)}"


def _torrent_actions_keyboard(
    items: Sequence[Any],
    mode: str,
    *,
    page: int = 0,
    total_pages: int = 1,
    view_mode: Optional[str] = None,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    callback_mode = view_mode or mode
    current_page = max(0, min(page, max(1, total_pages) - 1))

    for torrent in items[:TORRENT_LIST_PAGE_SIZE_MAX]:
        status = str(getattr(torrent, "status", ""))
        action = "pause" if _is_active(status) else "start"
        action_icon = "⏸️" if action == "pause" else "▶️"
        label_name = " ".join(str(torrent.name or "<без названия>").split())
        short_name = _shorten_text(label_name, 23)
        rows.append(
            [
                InlineKeyboardButton(
                    f"{action_icon} {torrent.id} · {short_name}",
                    callback_data=(f"{TORRENT_ACTION_CB_PREFIX}{action}:{torrent.id}:{callback_mode}:{current_page}"),
                ),
                InlineKeyboardButton(
                    "💥",
                    callback_data=(f"{TORRENT_ACTION_CB_PREFIX}del_data:{torrent.id}:{callback_mode}:{current_page}"),
                ),
            ]
        )

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if current_page > 0:
            nav.append(
                InlineKeyboardButton(
                    "◀️",
                    callback_data=_list_view_callback_data(callback_mode, current_page - 1),
                )
            )
        nav.append(
            InlineKeyboardButton(
                f"{current_page + 1}/{total_pages}",
                callback_data=_list_view_callback_data(callback_mode, current_page),
            )
        )
        if current_page + 1 < total_pages:
            nav.append(
                InlineKeyboardButton(
                    "▶️",
                    callback_data=_list_view_callback_data(callback_mode, current_page + 1),
                )
            )
        rows.append(nav)

    rows.extend(
        [
            [
                InlineKeyboardButton(
                    "📋 Все" if callback_mode != "all" else "• 📋 Все",
                    callback_data=_list_view_callback_data("all", 0),
                ),
                InlineKeyboardButton(
                    "⬇️ Скачиваются" if callback_mode != "downloading" else "• ⬇️ Скачиваются",
                    callback_data=_list_view_callback_data("downloading", 0),
                ),
            ],
            [
                InlineKeyboardButton(
                    "⏸️ Остановлены" if callback_mode != "stopped" else "• ⏸️ Остановлены",
                    callback_data=_list_view_callback_data("stopped", 0),
                ),
                InlineKeyboardButton(
                    "✅ Завершённые" if callback_mode != "done" else "• ✅ Завершённые",
                    callback_data=_list_view_callback_data("done", 0),
                ),
            ],
            [
                InlineKeyboardButton(
                    "🔄 Обновить",
                    callback_data=_list_view_callback_data(callback_mode, current_page),
                )
            ],
        ]
    )
    return InlineKeyboardMarkup(rows)


def _notifications_enabled(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int) -> bool:
    enabled_chats = ctx.application.bot_data.setdefault(NOTIFY_ENABLED_CHATS_KEY, set())
    return chat_id in enabled_chats


def _notification_torrent_key(torrent: Any) -> Optional[str]:
    return _torrent_history_key(torrent)


async def _enqueue_notification(
    *,
    event_key: str,
    chat_id: int,
    kind: str,
    text: str,
) -> bool:
    return await asyncio.to_thread(
        get_state_store().enqueue_outbox,
        event_key,
        chat_id,
        kind,
        text,
    )


def _notification_retry_delay(attempts: int) -> float:
    return float(min(60 * 60, 5 * (2 ** min(max(0, attempts - 1), 9))))


async def _deliver_outbox_item(ctx: ContextTypes.DEFAULT_TYPE, item: OutboxItem) -> None:
    enabled_chats = ctx.application.bot_data.get(NOTIFY_ENABLED_CHATS_KEY)
    if not isinstance(enabled_chats, set) or item.chat_id not in enabled_chats:
        await asyncio.to_thread(get_state_store().cancel_pending_outbox, item.chat_id)
        return

    try:
        # A send timeout is ambiguous: do not immediately send a duplicate. The
        # durable outbox retries later with backoff until Telegram confirms it.
        await ctx.bot.send_message(chat_id=item.chat_id, text=item.text, parse_mode=ParseMode.HTML)
    except Forbidden:
        enabled_chats.discard(item.chat_id)
        await asyncio.to_thread(get_state_store().set_notification_enabled, item.chat_id, False)
        log.warning("Notifications disabled because chat %s rejected the bot", item.chat_id)
    except TelegramError as exc:
        attempts = item.attempts + 1
        next_attempt_at = time_module.time() + _notification_retry_delay(attempts)
        await asyncio.to_thread(
            get_state_store().mark_outbox_failed,
            item.id,
            cast(str, item.claim_token),
            attempts,
            next_attempt_at,
            error=_redact_log_text(str(exc)),
        )
        log.warning(
            "Notification delivery failed for chat %s; retry %d scheduled",
            item.chat_id,
            attempts,
        )
    else:
        await asyncio.to_thread(
            get_state_store().mark_outbox_delivered,
            item.id,
            cast(str, item.claim_token),
        )


async def _drain_notification_outbox(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global _OUTBOX_DRAIN_LOCK
    if _OUTBOX_DRAIN_LOCK is None:
        _OUTBOX_DRAIN_LOCK = asyncio.Lock()

    async with _OUTBOX_DRAIN_LOCK:
        while True:
            items = await asyncio.to_thread(get_state_store().claim_due_outbox, time_module.time(), 1)
            if not items:
                return
            await _deliver_outbox_item(ctx, items[0])


async def _register_torrent_start_watch(
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
    torrent_key = _notification_torrent_key(torrent)
    if torrent_id <= 0 or torrent_key is None:
        return

    now_value = now_ts if now_ts is not None else time_module.time()
    name = str(getattr(torrent, "name", "") or "<без названия>")
    await asyncio.to_thread(
        get_state_store().add_start_watch,
        torrent_key,
        chat_id,
        torrent_id,
        name,
        added_at=now_value,
    )
    state = pending.get(torrent_id)
    if not isinstance(state, dict):
        pending[torrent_id] = {
            "added_at": now_value,
            "name": name,
            "torrent_key": torrent_key,
            "chat_ids": {chat_id},
        }
        return

    chat_ids = state.get("chat_ids")
    if not isinstance(chat_ids, set):
        chat_ids = set()
        state["chat_ids"] = chat_ids
    chat_ids.add(chat_id)


def _torrent_start_detected(torrent: Any) -> bool:
    downloaded_ever = _non_negative_float(
        _get_mapping_or_attr_value(
            torrent,
            ("downloaded_ever", "downloadedEver", "downloaded-ever"),
        )
    )
    percent_done = _non_negative_float(
        _get_mapping_or_attr_value(
            torrent,
            ("percent_done", "percentDone", "percent-done"),
        )
    )
    rate_download = _non_negative_float(
        _get_mapping_or_attr_value(
            torrent,
            ("rate_download", "rateDownload", "rate-download"),
        )
    )
    return any(value is not None and value > 0.0 for value in (downloaded_ever, percent_done, rate_download))


def _torrent_is_attempting_download(torrent: Any) -> bool:
    status_raw = _get_mapping_or_attr_value(torrent, ("status",))
    return str(status_raw or "").strip().lower() == "downloading"


def _pop_pending_torrent_start(ctx: ContextTypes.DEFAULT_TYPE, torrent_id: int) -> Optional[dict[Any, Any]]:
    pending = ctx.application.bot_data.get(NOTIFY_START_PENDING_KEY)
    if not isinstance(pending, dict):
        return None

    state = pending.pop(torrent_id, None)
    return state if isinstance(state, dict) else None


def _build_torrent_start_notification_text(torrent_id: int, name: str) -> str:
    safe_name = html.escape(name or "<без названия>")
    return f"▶️ <b>Скачивание началось</b>\nТоррент: <b>{safe_name}</b>\nID: <b>{torrent_id}</b>"


async def _send_torrent_start_notification(
    ctx: ContextTypes.DEFAULT_TYPE,
    torrent_id: int,
    state: dict[Any, Any],
    torrent: Any,
) -> None:
    enabled_chats = ctx.application.bot_data.get(NOTIFY_ENABLED_CHATS_KEY)
    if not isinstance(enabled_chats, set):
        enabled_chats = set()

    chat_ids = state.get("chat_ids")
    if not isinstance(chat_ids, set) or not chat_ids:
        return

    name = str(state.get("name") or getattr(torrent, "name", "<без названия>"))
    text = _build_torrent_start_notification_text(torrent_id, name)
    torrent_key = str(state.get("torrent_key") or _notification_torrent_key(torrent) or f"id:{torrent_id}")
    added_at = float(state.get("added_at", time_module.time()))

    for chat_id in list(chat_ids):
        if chat_id in enabled_chats:
            await _enqueue_notification(
                event_key=f"start:{torrent_key}:{int(added_at * 1000)}",
                chat_id=chat_id,
                kind="start",
                text=text,
            )
        await asyncio.to_thread(
            get_state_store().update_start_watch,
            torrent_key,
            chat_id,
            start_notified=True,
        )

    await _drain_notification_outbox(ctx)


async def _notify_torrent_start_soon(
    ctx: ContextTypes.DEFAULT_TYPE,
    torrent_id: int,
    initial_torrent: Any,
) -> None:
    torrent = initial_torrent
    for delay_sec in NOTIFY_START_QUICK_DELAYS_SEC:
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)

        pending = ctx.application.bot_data.get(NOTIFY_START_PENDING_KEY)
        if not isinstance(pending, dict) or torrent_id not in pending:
            return

        if torrent is None:
            try:
                torrent = await tr_call(lambda c: c.get_torrent(torrent_id))
            except (KeyError, TransmissionError, TRCallError):
                log.warning("Skipping quick start notification check due to Transmission error", exc_info=True)
                return

            pending = ctx.application.bot_data.get(NOTIFY_START_PENDING_KEY)
            if not isinstance(pending, dict) or torrent_id not in pending:
                return

        if _torrent_start_detected(torrent):
            state = _pop_pending_torrent_start(ctx, torrent_id)
            if state is not None:
                await _send_torrent_start_notification(ctx, torrent_id, state, torrent)
            return

        torrent = None


def _finish_torrent_start_task(app: Application, torrent_id: int, task: asyncio.Task[None]) -> None:
    tasks = app.bot_data.get(NOTIFY_START_TASKS_KEY)
    if isinstance(tasks, set):
        tasks.discard(task)
    pending = app.bot_data.get(NOTIFY_START_PENDING_KEY)
    if isinstance(pending, dict):
        pending.pop(torrent_id, None)

    with contextlib.suppress(asyncio.CancelledError):
        exc = task.exception()
        if exc is not None:
            log.warning("Quick start notification task failed", exc_info=(type(exc), exc, exc.__traceback__))


def _schedule_torrent_start_watch(ctx: ContextTypes.DEFAULT_TYPE, torrent: Any) -> None:
    torrent_id = int(getattr(torrent, "id", 0))
    if torrent_id <= 0:
        return

    tasks = ctx.application.bot_data.setdefault(NOTIFY_START_TASKS_KEY, set())
    if not isinstance(tasks, set):
        tasks = set()
        ctx.application.bot_data[NOTIFY_START_TASKS_KEY] = tasks

    task_ctx = cast(ContextTypes.DEFAULT_TYPE, SimpleNamespace(application=ctx.application, bot=ctx.bot))
    task = ctx.application.create_task(
        _notify_torrent_start_soon(task_ctx, torrent_id, torrent),
        name=f"torrent-start-watch-{torrent_id}",
    )
    tasks.add(task)
    task.add_done_callback(lambda done_task: _finish_torrent_start_task(ctx.application, torrent_id, done_task))


async def _ensure_chat_notifications_initialized(ctx: ContextTypes.DEFAULT_TYPE, chat_id: Optional[int]) -> None:
    if chat_id is None:
        return

    known_chats = ctx.application.bot_data.setdefault(NOTIFY_KNOWN_CHATS_KEY, set())
    enabled_chats = ctx.application.bot_data.setdefault(NOTIFY_ENABLED_CHATS_KEY, set())
    if chat_id in known_chats:
        return

    enabled = await asyncio.to_thread(get_state_store().ensure_chat, chat_id, default_enabled=True)
    known_chats.add(chat_id)
    if enabled:
        enabled_chats.add(chat_id)
    else:
        enabled_chats.discard(chat_id)


def _ctrl_keyboard_for_chat(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: Optional[int],
    *,
    input_active: bool = False,
) -> ReplyKeyboardMarkup:
    if chat_id is None:
        return KB_CTRL_INPUT if input_active else KB_CTRL
    return kb_ctrl(_notifications_enabled(ctx, chat_id), input_active=input_active)


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


def _format_download_duration(seconds: int | float) -> str:
    total_seconds = max(0, int(seconds))
    days, rem = divmod(total_seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    if days > 0:
        return f"{days}д {hours}ч {minutes}м"
    if hours > 0:
        return f"{hours}ч {minutes}м"
    if minutes > 0:
        return f"{minutes}м {secs}с"
    return f"{secs}с"


def _torrent_timepoint_to_ts(value: Any) -> Optional[int]:
    if isinstance(value, datetime):
        try:
            timestamp = value.timestamp()
        except (OSError, OverflowError, ValueError):
            return None
        return int(timestamp) if isfinite(timestamp) else None
    if isinstance(value, int) and not isinstance(value, bool):
        as_int = value
        return as_int if as_int > 0 else None
    if isinstance(value, float):
        if not isfinite(value):
            return None
        as_int = int(value)
        return as_int if as_int > 0 else None
    return None


def _extract_download_duration_seconds(torrent: Any) -> Optional[int]:
    added_raw = getattr(torrent, "added_date", None)
    if added_raw is None:
        added_raw = getattr(torrent, "date_added", None)

    done_raw = getattr(torrent, "done_date", None)
    if done_raw is None:
        done_raw = getattr(torrent, "date_done", None)

    added_ts = _torrent_timepoint_to_ts(added_raw)
    done_ts = _torrent_timepoint_to_ts(done_raw)
    if added_ts is None or done_ts is None or done_ts < added_ts:
        return None

    return done_ts - added_ts


def _build_completion_notification_text(torrent_id: int, name: str, torrent: Optional[Any]) -> str:
    duration_text = ""
    if torrent is not None:
        duration_seconds = _extract_download_duration_seconds(torrent)
        if duration_seconds is not None:
            duration_text = f"\n⏱️ Время скачивания: <b>{_format_download_duration(duration_seconds)}</b>"

    safe_name = html.escape(name or "<без названия>")
    return f"✅ Торрент завершён: <b>{safe_name}</b>\nID: <b>{torrent_id}</b>{duration_text}"


def _non_negative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if not isfinite(value):
            return None
        return int(max(0, value))
    return None


def _get_mapping_or_attr_value(source: Any, names: Sequence[str]) -> Any:
    try:
        source_fields = getattr(source, "fields", None)
    except (AttributeError, KeyError):
        source_fields = None

    for name in names:
        if isinstance(source, dict) and name in source:
            return source[name]
        try:
            value = getattr(source, name, None)
        except (AttributeError, KeyError):
            value = None
        if value is not None:
            return value
        if isinstance(source_fields, dict) and name in source_fields:
            return source_fields[name]
    return None


def _extract_free_space_value(source: Any) -> Optional[int]:
    value = _non_negative_int(source)
    if value is not None:
        return value

    if isinstance(source, dict) and isinstance(source.get("arguments"), dict):
        value = _extract_free_space_value(source["arguments"])
        if value is not None:
            return value

    raw_value = _get_mapping_or_attr_value(
        source,
        (
            "size_bytes",
            "size-bytes",
            "download_dir_free_space",
            "download-dir-free-space",
        ),
    )
    return _non_negative_int(raw_value)


async def _get_download_dir_free_space() -> Optional[int]:
    try:
        session = await tr_call(lambda c: c.get_session())
    except (KeyError, TransmissionError, TRCallError):
        return None

    return _extract_free_space_value(session)


def _build_free_space_text(free_space: Optional[int]) -> str:
    if free_space is None:
        return "💾 Свободно на диске: <i>не удалось получить</i>."
    return f"💾 Свободно на диске: <b>{fmt_bytes(free_space)}</b>."


def _build_projected_free_space_text(free_space_before: Optional[int], torrent: Any) -> str:
    if free_space_before is None:
        return "💾 После полной скачки: <i>не удалось рассчитать</i>."

    required_space = _get_mapping_or_attr_value(
        torrent,
        ("left_until_done", "leftUntilDone", "left-until-done"),
    )
    if not isinstance(required_space, (int, float)):
        required_space = _get_mapping_or_attr_value(
            torrent,
            ("total_size", "totalSize", "total-size"),
        )
    normalized_required_space = _non_negative_int(required_space)
    if normalized_required_space is None:
        return "💾 После полной скачки: <i>не удалось рассчитать</i>."

    projected_free_space = max(0, free_space_before - normalized_required_space)
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
    old_message_id = _require_user_data(ctx).get(LAST_EPHEMERAL_MESSAGE_KEY)
    if isinstance(old_message_id, int):
        await _delete_message_safe(ctx, chat.id, old_message_id)


async def send_ephemeral(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup: ReplyKeyboardMarkup,
) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return

    await _cleanup_previous_ephemeral(update, ctx)
    sent = await _send_with_timeout_retry(
        lambda: message.reply_text(text=text, reply_markup=reply_markup),
        op_name="send_ephemeral.reply_text",
    )
    _require_user_data(ctx)[LAST_EPHEMERAL_MESSAGE_KEY] = sent.message_id


async def _delete_user_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return
    await _delete_message_safe(ctx, chat.id, message.message_id)


def _build_active_torrents_text(torrents: Sequence[Any]) -> str:
    active = [t for t in torrents if _is_downloading(str(getattr(t, "status", "")))]
    if not active:
        return "Сейчас активных скачиваний нет"

    active_sorted = sorted(active, key=torrent_progress_percent, reverse=True)
    lines: list[str] = []
    for torrent in active_sorted[:5]:
        safe_name = html.escape(str(getattr(torrent, "name", "") or "<без названия>"))
        lines.append(f"• {safe_name}\n  {_format_progress_summary(torrent)}")

    hidden_count = len(active_sorted) - len(lines)
    if hidden_count > 0:
        lines.append(f"• …и ещё <b>{hidden_count}</b>")

    return "\n".join(lines)


def _build_status_text(stats: Any, free_space: Optional[int], torrents: Sequence[Any]) -> str:
    cur = stats.current_stats
    cum = stats.cumulative_stats
    session_duration = _format_session_duration(getattr(cur, "seconds_active", 0))
    free_space_text = _build_free_space_text(free_space)
    active_torrents_text = _build_active_torrents_text(torrents)
    torrent_counts_text = (
        f"Торренты: активных <b>{stats.active_torrent_count}</b>, "
        f"на паузе <b>{stats.paused_torrent_count}</b>, всего <b>{stats.torrent_count}</b>"
    )
    session_traffic_text = (
        f"Трафик (сессия - {session_duration}): "
        f"⇣ <b>{fmt_bytes(cur.downloaded_bytes)}</b> | ⇡ <b>{fmt_bytes(cur.uploaded_bytes)}</b>"
    )
    total_traffic_text = (
        f"Трафик (всего): ⇣ <b>{fmt_bytes(cum.downloaded_bytes)}</b> | ⇡ <b>{fmt_bytes(cum.uploaded_bytes)}</b>"
    )
    return (
        "📊 <b>Transmission — статус</b>\n"
        f"Скорость: ⇣ <b>{fmt_rate(stats.download_speed)}</b> | ⇡ <b>{fmt_rate(stats.upload_speed)}</b>\n"
        f"{torrent_counts_text}\n"
        f"Активные сейчас:\n{active_torrents_text}\n\n"
        f"{free_space_text}\n"
        f"{session_traffic_text}\n"
        f"{total_traffic_text}\n"
        f"🕒 {bot_now().strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )


def _read_traffic_state() -> tuple[dict[str, dict[str, int | str]], list[dict[str, int | str]]]:  # noqa: C901
    return get_state_store().load_traffic_state()


def _persist_traffic_state(anchors: dict[str, dict[str, int | str]], history: list[dict[str, int | str]]) -> None:
    get_state_store().save_traffic_state(anchors, history)


def _period_keys(now: datetime) -> dict[str, str]:
    iso_year, iso_week, _ = now.isocalendar()
    return {
        "day": now.strftime("%Y-%m-%d"),
        "week": f"{iso_year}-W{iso_week:02d}",
        "month": now.strftime("%Y-%m"),
    }


def _normalize_traffic_counters(
    anchors: dict[str, dict[str, int | str]],
    downloaded: int,
    uploaded: int,
) -> tuple[int, int, bool]:
    counter = anchors.get("_counter")
    if not isinstance(counter, dict):
        logical_downloaded = downloaded
        logical_uploaded = uploaded
    else:
        last_downloaded = _non_negative_int(counter.get("last_downloaded"))
        last_uploaded = _non_negative_int(counter.get("last_uploaded"))
        last_downloaded = downloaded if last_downloaded is None else last_downloaded
        last_uploaded = uploaded if last_uploaded is None else last_uploaded
        previous_logical_downloaded = _non_negative_int(counter.get("logical_downloaded"))
        previous_logical_uploaded = _non_negative_int(counter.get("logical_uploaded"))
        previous_logical_downloaded = (
            last_downloaded if previous_logical_downloaded is None else previous_logical_downloaded
        )
        previous_logical_uploaded = last_uploaded if previous_logical_uploaded is None else previous_logical_uploaded
        download_delta = downloaded - last_downloaded if downloaded >= last_downloaded else downloaded
        upload_delta = uploaded - last_uploaded if uploaded >= last_uploaded else uploaded
        logical_downloaded = previous_logical_downloaded + max(0, download_delta)
        logical_uploaded = previous_logical_uploaded + max(0, upload_delta)

    updated_counter: dict[str, int | str] = {
        "key": "logical-v1",
        "last_downloaded": downloaded,
        "last_uploaded": uploaded,
        "logical_downloaded": logical_downloaded,
        "logical_uploaded": logical_uploaded,
    }
    changed = counter != updated_counter
    anchors["_counter"] = updated_counter
    return logical_downloaded, logical_uploaded, changed


def _effective_traffic_totals(
    anchors: dict[str, dict[str, int | str]],
    fallback_downloaded: int,
    fallback_uploaded: int,
) -> tuple[int, int]:
    counter = anchors.get("_counter")
    if not isinstance(counter, dict):
        return fallback_downloaded, fallback_uploaded
    logical_downloaded = counter.get("logical_downloaded")
    logical_uploaded = counter.get("logical_uploaded")
    if not isinstance(logical_downloaded, int) or not isinstance(logical_uploaded, int):
        return fallback_downloaded, fallback_uploaded
    return logical_downloaded, logical_uploaded


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
    if len(history) > 365:
        del history[:-365]
    return True


async def update_traffic_state(
    now: datetime,
    downloaded: int,
    uploaded: int,
) -> tuple[dict[str, dict[str, int | str]], list[dict[str, int | str]]]:
    async with _get_traffic_state_lock():
        anchors, history = await asyncio.to_thread(_read_traffic_state)
        effective_downloaded, effective_uploaded, counters_changed = _normalize_traffic_counters(
            anchors,
            downloaded,
            uploaded,
        )
        anchors_changed = _ensure_traffic_anchors(anchors, now, effective_downloaded, effective_uploaded)
        history_changed = _ensure_daily_traffic_history(history, now, effective_downloaded, effective_uploaded)
        if counters_changed or anchors_changed or history_changed:
            try:
                await asyncio.to_thread(_persist_traffic_state, anchors, history)
            except (OSError, sqlite3.Error):
                log.warning("Failed to persist traffic state", exc_info=True)
            else:
                period_key = _period_keys(now)["day"]
                log.debug("Traffic state persisted: period_key=%s history_size=%d", period_key, len(history))
        return anchors, history


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


def _build_basic_traffic_chart_png(down_values: list[float], up_values: list[float]) -> bytes:
    """Строит резервный график без зависимостей для установок без matplotlib."""
    width, height = 940, 520
    left, top, right, bottom = 72, 38, 28, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    pixels = bytearray(b"\xff\xff\xff" * width * height)

    def fill_rect(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        row = bytes(color) * max(0, x2 - x1)
        for y in range(y1, y2):
            offset = (y * width + x1) * 3
            pixels[offset : offset + len(row)] = row

    # Сетка и цветные столбцы оставляют график читаемым даже без шрифтов matplotlib.
    for step in range(6):
        y = top + round(plot_height * step / 5)
        fill_rect(left, y, width - right, y + 1, (218, 226, 236))
    fill_rect(left - 1, top, left + 1, height - bottom + 1, (93, 109, 126))
    fill_rect(left, height - bottom, width - right, height - bottom + 2, (93, 109, 126))

    values_count = max(len(down_values), len(up_values), 1)
    maximum = max([*down_values, *up_values, 1.0])
    group_width = plot_width / values_count
    bar_width = max(3, min(34, int(group_width * 0.32)))
    for index in range(values_count):
        center = left + round(group_width * (index + 0.5))
        for value, x, color in (
            (down_values[index] if index < len(down_values) else 0.0, center - bar_width, (36, 123, 220)),
            (up_values[index] if index < len(up_values) else 0.0, center, (241, 139, 45)),
        ):
            bar_height = round(plot_height * max(0.0, value) / maximum)
            fill_rect(x, height - bottom - bar_height, x + bar_width, height - bottom, color)

    # Легенда: синий — загрузка, оранжевый — отдача; подписи остаются в caption сообщения.
    fill_rect(left, 12, left + 28, 26, (36, 123, 220))
    fill_rect(left + 46, 12, left + 74, 26, (241, 139, 45))

    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))

    def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)

    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=7))
        + png_chunk(b"IEND", b"")
    )


def _build_traffic_chart_last_7_days(
    points: list[dict[str, int | str]],
) -> tuple[Optional[bytes], Optional[str]]:
    if len(points) < 2:
        return None, "Недостаточно данных для графика. История заполняется раз в день."

    labels = [str(item["date"]) for item in points]
    down_values = [float(item["downloaded"]) / (1024 * 1024 * 1024) for item in points]
    up_values = [float(item["uploaded"]) / (1024 * 1024 * 1024) for item in points]

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return _build_basic_traffic_chart_png(down_values, up_values), None

    fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=130)
    try:
        _draw_traffic_chart(
            ax=ax,
            labels=labels,
            down_values=down_values,
            up_values=up_values,
            title="Трафик за последние 7 дней",
            y_label="GiB / день",
            annotate_last_points=2,
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

    result: list[dict[str, int | str]] = []
    for day in range(1, now.day + 1):
        date_value = now.replace(day=day)
        day_key = date_value.strftime("%Y-%m-%d")
        totals = daily_totals.get(day_key)
        result.append(
            {
                "date": date_value.strftime("%d.%m"),
                "day": day,
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

    labels = [str(item["day"]) for item in points]
    down_values = [float(item["downloaded"]) / (1024 * 1024 * 1024) for item in points]
    up_values = [float(item["uploaded"]) / (1024 * 1024 * 1024) for item in points]

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return points, _build_basic_traffic_chart_png(down_values, up_values), None

    month_title = f"{_month_name_ru(now.month)} {now.year}"

    fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=130)
    try:
        _draw_traffic_chart(
            ax=ax,
            labels=labels,
            down_values=down_values,
            up_values=up_values,
            title=month_title,
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
    lines = [f"🗓️ <b>{_month_name_ru(now.month)} {now.year}</b>"]
    points = _daily_totals_current_month(now, downloaded, uploaded, history)

    for day in points:
        lines.append(
            f"{day['date']}: ⇣ <b>{fmt_bytes(int(day['downloaded']))}</b> | ⇡ <b>{fmt_bytes(int(day['uploaded']))}</b>"
        )

    lines.append(f"🕒 {now.strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def _smooth_chart_points(
    x_values: list[float],
    y_values: list[float],
    *,
    samples_per_segment: int = 14,
) -> tuple[list[float], list[float]]:
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return x_values, y_values

    smooth_x: list[float] = []
    smooth_y: list[float] = []

    for idx in range(len(x_values) - 1):
        p0 = y_values[max(0, idx - 1)]
        p1 = y_values[idx]
        p2 = y_values[idx + 1]
        p3 = y_values[min(len(y_values) - 1, idx + 2)]
        x1 = x_values[idx]
        x2 = x_values[idx + 1]
        local_min = min(p0, p1, p2, p3)
        local_max = max(p0, p1, p2, p3)

        segment_samples = max(2, samples_per_segment)
        for sample in range(segment_samples):
            t = sample / segment_samples
            t2 = t * t
            t3 = t2 * t
            y_value = 0.5 * (
                (2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            smooth_x.append(x1 + (x2 - x1) * t)
            smooth_y.append(max(0.0, min(local_max, max(local_min, y_value))))

    smooth_x.append(x_values[-1])
    smooth_y.append(max(0.0, y_values[-1]))
    return smooth_x, smooth_y


def _format_chart_value(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _format_chart_total(value: float) -> str:
    if value >= 1024:
        return f"{value / 1024:.1f} TiB"
    return f"{_format_chart_value(value)} GiB"


def _add_chart_summary_cards(
    ax: Any,
    *,
    down_total: float,
    up_total: float,
    down_color: str,
    up_color: str,
) -> None:
    card_style = {
        "boxstyle": "round,pad=0.42,rounding_size=0.12",
        "facecolor": "#FFFFFF",
        "edgecolor": "#E2E8F0",
        "linewidth": 1.0,
        "alpha": 0.96,
    }
    ax.text(
        0.02,
        0.96,
        f"⇣ {_format_chart_total(down_total)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=down_color,
        fontsize=10,
        weight="bold",
        bbox=card_style,
    )
    ax.text(
        0.25,
        0.96,
        f"⇡ {_format_chart_total(up_total)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=up_color,
        fontsize=10,
        weight="bold",
        bbox=card_style,
    )


def _annotate_latest_chart_value(
    ax: Any,
    *,
    x_value: float,
    y_value: float,
    color: str,
    y_max: float,
) -> None:
    y_offset = 14 if y_value < y_max * 0.12 else 0
    ax.annotate(
        _format_chart_value(y_value),
        xy=(x_value, y_value),
        xytext=(-12, y_offset),
        textcoords="offset points",
        ha="right",
        va="center",
        color=color,
        fontsize=9,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.0,
            "alpha": 0.95,
        },
    )


def _draw_traffic_chart(
    ax: Any,
    labels: list[str],
    down_values: list[float],
    up_values: list[float],
    title: str,
    y_label: str,
    annotate_last_points: int = 1,
) -> None:
    down_color = "#2563EB"
    up_color = "#EA580C"
    grid_color = "#D8E0EA"
    text_color = "#0F172A"
    muted_text_color = "#64748B"
    y_max = max([*down_values, *up_values, 0.0])
    minor_threshold = max(0.05, y_max * 0.025)
    down_is_minor = max(down_values or [0.0]) <= minor_threshold
    up_is_minor = max(up_values or [0.0]) <= minor_threshold

    ax.set_facecolor("#F8FAFC")
    ax.figure.set_facecolor("#F1F5F9")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    x_values = [float(idx) for idx in range(len(labels))]
    smooth_down_x, smooth_down_y = _smooth_chart_points(x_values, down_values)
    smooth_up_x, smooth_up_y = _smooth_chart_points(x_values, up_values)
    _add_chart_summary_cards(
        ax,
        down_total=sum(down_values),
        up_total=sum(up_values),
        down_color=down_color,
        up_color=up_color,
    )

    ax.fill_between(
        smooth_down_x,
        smooth_down_y,
        alpha=0.04 if down_is_minor else 0.10,
        color=down_color,
    )
    ax.fill_between(
        smooth_up_x,
        smooth_up_y,
        alpha=0.04 if up_is_minor else 0.12,
        color=up_color,
    )

    ax.plot(
        smooth_down_x,
        smooth_down_y,
        linewidth=2.0 if down_is_minor else 2.9,
        color=down_color,
        alpha=0.42 if down_is_minor else 1.0,
        solid_capstyle="round",
        label="Скачано",
        zorder=4,
    )
    ax.plot(
        smooth_up_x,
        smooth_up_y,
        linewidth=2.0 if up_is_minor else 2.9,
        color=up_color,
        alpha=0.42 if up_is_minor else 1.0,
        solid_capstyle="round",
        label="Отдано",
        zorder=4,
    )

    ax.scatter(
        x_values,
        down_values,
        s=34 if down_is_minor else 52,
        color=down_color,
        alpha=0.46 if down_is_minor else 1.0,
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
    )
    ax.scatter(
        x_values,
        up_values,
        s=34 if up_is_minor else 52,
        color=up_color,
        alpha=0.46 if up_is_minor else 1.0,
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
    )

    if labels and annotate_last_points > 0:
        last_idx = len(labels) - 1
        if not down_is_minor or down_values[last_idx] > 0:
            _annotate_latest_chart_value(
                ax,
                x_value=x_values[last_idx],
                y_value=down_values[last_idx],
                color=down_color,
                y_max=max(1.0, y_max),
            )
        if not up_is_minor or up_values[last_idx] > 0:
            _annotate_latest_chart_value(
                ax,
                x_value=x_values[last_idx],
                y_value=up_values[last_idx],
                color=up_color,
                y_max=max(1.0, y_max),
            )

    ax.set_title(title, fontsize=15, weight="bold", color=text_color, pad=18, loc="left")
    ax.set_ylabel(y_label, color=muted_text_color)

    tick_step = max(1, (len(labels) + 15) // 16)
    tick_indexes = list(range(0, len(labels), tick_step))
    if tick_indexes[-1] != len(labels) - 1:
        tick_indexes.append(len(labels) - 1)
    ax.set_xticks(tick_indexes)
    ax.set_xticklabels([labels[idx] for idx in tick_indexes])

    ax.set_ylim(bottom=0, top=max(1.0, y_max * 1.12))
    ax.margins(x=0.035)
    ax.tick_params(axis="x", rotation=0, labelsize=9, colors=muted_text_color, pad=6)
    ax.tick_params(axis="y", labelsize=9, colors=muted_text_color)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.9, alpha=0.72, color=grid_color)
    ax.grid(False, axis="x")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        framealpha=0.92,
        fontsize=9,
    )


def _month_name_ru(month: int) -> str:
    months = (
        "Январь",
        "Февраль",
        "Март",
        "Апрель",
        "Май",
        "Июнь",
        "Июль",
        "Август",
        "Сентябрь",
        "Октябрь",
        "Ноябрь",
        "Декабрь",
    )
    month_index = max(1, min(month, 12)) - 1
    return months[month_index]


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
    downloaded, uploaded = _effective_traffic_totals(anchors, downloaded, uploaded)
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


def _read_torrent_history_state() -> dict[str, dict[str, Any]]:
    return get_state_store().load_torrent_history()


def _persist_torrent_history_state(items: dict[str, dict[str, Any]]) -> None:
    get_state_store().save_torrent_history(items)


def _history_entry_int(entry: Optional[dict[str, Any]], field: str, default: int = 0) -> int:
    if not isinstance(entry, dict):
        return default
    value = entry.get(field)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return default


def _history_entry_optional_int(entry: Optional[dict[str, Any]], field: str) -> Optional[int]:
    if not isinstance(entry, dict):
        return None
    value = entry.get(field)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        as_int = int(value)
        return as_int if as_int > 0 else None
    return None


def _history_entry_float(entry: Optional[dict[str, Any]], field: str, default: float = 0.0) -> float:
    if not isinstance(entry, dict):
        return default
    value = entry.get(field)
    normalized = _non_negative_float(value)
    return default if normalized is None else normalized


def _non_negative_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            normalized = float(value)
        except OverflowError:
            return None
        if not isfinite(normalized):
            return None
        return max(0.0, normalized)
    return None


def _torrent_history_hash(torrent: Any) -> Optional[str]:
    value = _get_mapping_or_attr_value(
        torrent,
        (
            "hash_string",
            "hashString",
            "hash-string",
            "hash",
        ),
    )
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _torrent_history_key(torrent: Any) -> Optional[str]:
    torrent_hash = _torrent_history_hash(torrent)
    if torrent_hash:
        return f"hash:{torrent_hash}"

    torrent_id = _non_negative_int(getattr(torrent, "id", None))
    if torrent_id is not None and torrent_id > 0:
        return f"id:{torrent_id}"
    return None


def _torrent_history_stat_int(torrent: Any, names: Sequence[str]) -> int:
    value = _get_mapping_or_attr_value(torrent, names)
    normalized = _non_negative_int(value)
    return normalized if normalized is not None else 0


def _torrent_uploaded_ever(torrent: Any) -> int:
    return _torrent_history_stat_int(torrent, ("uploaded_ever", "uploadedEver", "uploaded-ever"))


def _torrent_downloaded_ever(torrent: Any) -> int:
    return _torrent_history_stat_int(torrent, ("downloaded_ever", "downloadedEver", "downloaded-ever"))


def _torrent_upload_ratio(torrent: Any, uploaded: int, downloaded: int, total_size: int) -> float:
    ratio = _non_negative_float(_get_mapping_or_attr_value(torrent, ("upload_ratio", "uploadRatio", "upload-ratio")))
    if ratio is not None:
        return ratio
    if downloaded > 0:
        return uploaded / downloaded
    if total_size > 0:
        return uploaded / total_size
    return 0.0


def _torrent_history_timepoint(torrent: Any, names: Sequence[str]) -> Optional[int]:
    return _torrent_timepoint_to_ts(_get_mapping_or_attr_value(torrent, names))


def _torrent_history_entry_from_torrent(
    torrent: Any,
    existing: Optional[dict[str, Any]],
    now_ts: int,
) -> Optional[dict[str, Any]]:
    key = _torrent_history_key(torrent)
    if key is None:
        return None

    torrent_id = _non_negative_int(getattr(torrent, "id", None))
    torrent_hash = _torrent_history_hash(torrent)
    total_size = max(torrent_total_size(torrent), _history_entry_int(existing, "total_size"))
    downloaded = max(_torrent_downloaded_ever(torrent), _history_entry_int(existing, "downloaded_ever"))
    uploaded = max(_torrent_uploaded_ever(torrent), _history_entry_int(existing, "uploaded_ever"))
    upload_ratio = max(
        _torrent_upload_ratio(torrent, uploaded, downloaded, total_size),
        _history_entry_float(existing, "upload_ratio"),
    )
    first_seen_at = _history_entry_optional_int(existing, "first_seen_at") or now_ts
    added_at = _torrent_history_timepoint(torrent, ("added_date", "date_added", "addedDate", "added-date"))
    done_at = _torrent_history_timepoint(torrent, ("done_date", "date_done", "doneDate", "done-date"))
    existing_name = str(existing.get("name", "")) if isinstance(existing, dict) else ""
    name = str(getattr(torrent, "name", "") or existing_name or "<без названия>")
    status = str(getattr(torrent, "status", "") or "").strip().lower()

    return {
        "key": key,
        "id": torrent_id,
        "hash": torrent_hash,
        "name": name,
        "status": status,
        "total_size": total_size,
        "downloaded_ever": downloaded,
        "uploaded_ever": uploaded,
        "upload_ratio": round(upload_ratio, 4),
        "progress": round(torrent_progress_percent(torrent), 2),
        "first_seen_at": first_seen_at,
        "last_seen_at": now_ts,
        "added_at": added_at or _history_entry_optional_int(existing, "added_at"),
        "done_at": done_at or _history_entry_optional_int(existing, "done_at"),
        "removed_at": None,
        "removed_with_data": None,
    }


def _mark_missing_torrent_history_entries(
    items: dict[str, dict[str, Any]],
    seen_keys: set[str],
    now_ts: int,
) -> bool:
    changed = False
    for key, entry in items.items():
        if key in seen_keys:
            continue
        if _history_entry_optional_int(entry, "removed_at") is not None:
            continue

        entry["removed_at"] = now_ts
        entry["status"] = "removed"
        changed = True
    return changed


def _merge_torrent_history_entries(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)

    for field in ("total_size", "downloaded_ever", "uploaded_ever"):
        merged[field] = max(_history_entry_int(primary, field), _history_entry_int(secondary, field))

    merged["upload_ratio"] = max(
        _history_entry_float(primary, "upload_ratio"),
        _history_entry_float(secondary, "upload_ratio"),
    )

    primary_first_seen = _history_entry_optional_int(primary, "first_seen_at")
    secondary_first_seen = _history_entry_optional_int(secondary, "first_seen_at")
    if primary_first_seen is not None and secondary_first_seen is not None:
        merged["first_seen_at"] = min(primary_first_seen, secondary_first_seen)
    elif secondary_first_seen is not None:
        merged["first_seen_at"] = secondary_first_seen

    primary_last_seen = _history_entry_optional_int(primary, "last_seen_at")
    secondary_last_seen = _history_entry_optional_int(secondary, "last_seen_at")
    if secondary_last_seen is not None and (primary_last_seen is None or secondary_last_seen > primary_last_seen):
        merged["last_seen_at"] = secondary_last_seen

    if not merged.get("name") and secondary.get("name"):
        merged["name"] = secondary["name"]

    return merged


def _pop_fallback_torrent_history_entry(
    items: dict[str, dict[str, Any]],
    torrent: Any,
    key: str,
) -> Optional[dict[str, Any]]:
    if not key.startswith("hash:"):
        return None

    torrent_id = _non_negative_int(getattr(torrent, "id", None))
    if torrent_id is None or torrent_id <= 0:
        return None

    fallback_key = f"id:{torrent_id}"
    if fallback_key == key:
        return None

    return items.pop(fallback_key, None)


async def sync_torrent_history(
    torrents: Sequence[Any],
    *,
    mark_missing: bool = True,
) -> list[dict[str, Any]]:
    now_ts = int(time_module.time())
    async with _get_torrent_history_lock():
        items = await asyncio.to_thread(_read_torrent_history_state)
        seen_keys: set[str] = set()
        changed = False

        for torrent in torrents:
            key = _torrent_history_key(torrent)
            if key is None:
                continue

            seen_keys.add(key)
            existing = items.get(key)
            fallback_entry = _pop_fallback_torrent_history_entry(items, torrent, key)
            if fallback_entry is not None:
                changed = True
                existing = (
                    fallback_entry if existing is None else _merge_torrent_history_entries(existing, fallback_entry)
                )

            updated = _torrent_history_entry_from_torrent(torrent, existing, now_ts)
            if updated is None:
                continue

            if items.get(key) != updated:
                items[key] = updated
                changed = True

        if mark_missing:
            changed = _mark_missing_torrent_history_entries(items, seen_keys, now_ts) or changed

        if changed:
            try:
                await asyncio.to_thread(_persist_torrent_history_state, items)
            except (OSError, sqlite3.Error):
                log.warning("Failed to persist torrent history", exc_info=True)

        return list(items.values())


async def mark_torrent_history_removed(torrent: Any, *, with_data: bool) -> None:
    now_ts = int(time_module.time())
    key = _torrent_history_key(torrent)
    if key is None:
        return

    async with _get_torrent_history_lock():
        items = await asyncio.to_thread(_read_torrent_history_state)
        entry = _torrent_history_entry_from_torrent(torrent, items.get(key), now_ts)
        if entry is None:
            return

        entry["removed_at"] = now_ts
        entry["removed_with_data"] = with_data
        entry["status"] = "removed"
        items[key] = entry

        try:
            await asyncio.to_thread(_persist_torrent_history_state, items)
        except (OSError, sqlite3.Error):
            log.warning("Failed to mark torrent history entry as removed", exc_info=True)


async def load_torrent_history_entries() -> list[dict[str, Any]]:
    async with _get_torrent_history_lock():
        items = await asyncio.to_thread(_read_torrent_history_state)
        return list(items.values())


def _shorten_text(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[: max(0, max_len - 1)]}…"


def _format_history_ts(value: Optional[int]) -> str:
    if value is None:
        return "неизвестно"
    return datetime.fromtimestamp(value, tz=get_config().timezone).strftime("%Y-%m-%d %H:%M")


def _torrent_status_label_ru(status: str) -> str:
    labels = {
        "downloading": "скачивается",
        "download pending": "ожидает скачивания",
        "seeding": "раздаётся",
        "seed pending": "ожидает раздачи",
        "checking": "проверяется",
        "check pending": "ожидает проверки",
        "stopped": "остановлен",
        "removed": "удалён из Transmission",
    }
    return labels.get(status, status or "в Transmission")


def _history_entry_removed(entry: dict[str, Any]) -> bool:
    return _history_entry_optional_int(entry, "removed_at") is not None


def _history_entry_sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    return (
        -_history_entry_int(entry, "uploaded_ever"),
        -_history_entry_int(entry, "last_seen_at"),
        str(entry.get("name", "")).lower(),
    )


def _torrent_history_keyboard(page: int, total_pages: int) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    nav: list[InlineKeyboardButton] = []

    if page > 0:
        nav.append(InlineKeyboardButton("◀️ Назад", callback_data=f"{TORRENT_HISTORY_CB_PREFIX}page:{page - 1}"))
    if page + 1 < total_pages:
        nav.append(InlineKeyboardButton("Вперёд ▶️", callback_data=f"{TORRENT_HISTORY_CB_PREFIX}page:{page + 1}"))
    if nav:
        rows.append(nav)

    rows.append(
        [InlineKeyboardButton("🔄 Обновить историю", callback_data=f"{TORRENT_HISTORY_CB_PREFIX}refresh:{page}")]
    )
    return InlineKeyboardMarkup(rows)


def _build_torrent_history_text(
    entries: Sequence[dict[str, Any]],
    *,
    page: int,
    warning: Optional[str] = None,
) -> tuple[str, InlineKeyboardMarkup, int]:
    sorted_entries = sorted(entries, key=_history_entry_sort_key)
    total = len(sorted_entries)
    page_size = min(max(1, get_config().list_limit), 8)
    total_pages = max(1, ceil(total / page_size))
    current_page = max(0, min(page, total_pages - 1))
    start = current_page * page_size
    page_entries = sorted_entries[start : start + page_size]

    lines = ["📚 <b>История раздач</b>"]
    if warning:
        lines.append(f"⚠️ {html.escape(warning)}")

    if total == 0:
        lines.append("История пока пустая. Она начнёт пополняться при следующем опросе Transmission.")
        return "\n".join(lines), _torrent_history_keyboard(0, 1), 0

    total_uploaded = sum(_history_entry_int(entry, "uploaded_ever") for entry in sorted_entries)
    total_downloaded = sum(_history_entry_int(entry, "downloaded_ever") for entry in sorted_entries)
    removed_count = sum(1 for entry in sorted_entries if _history_entry_removed(entry))
    active_count = total - removed_count
    lines.append(f"Всего: <b>{total}</b> | В Transmission: <b>{active_count}</b> | Удалены: <b>{removed_count}</b>")
    lines.append(f"Суммарно: ⇣ <b>{fmt_bytes(total_downloaded)}</b> | ⇡ <b>{fmt_bytes(total_uploaded)}</b>")
    lines.append(f"Страница <b>{current_page + 1}</b>/<b>{total_pages}</b>")

    for index, entry in enumerate(page_entries, start=start + 1):
        name = html.escape(_shorten_text(str(entry.get("name") or "<без названия>"), 72))
        uploaded = _history_entry_int(entry, "uploaded_ever")
        downloaded = _history_entry_int(entry, "downloaded_ever")
        total_size = _history_entry_int(entry, "total_size")
        ratio = _history_entry_float(entry, "upload_ratio")
        status = str(entry.get("status") or "").strip().lower()
        removed_at = _history_entry_optional_int(entry, "removed_at")
        last_seen_at = _history_entry_optional_int(entry, "last_seen_at")
        if removed_at:
            status_text = "🗑️ удалён из Transmission"
        else:
            status_text = f"{status_icon(status)} {_torrent_status_label_ru(status)}"
        date_label = "удалён" if removed_at else "обновлено"
        date_value = removed_at or last_seen_at
        torrent_id = _history_entry_optional_int(entry, "id")
        id_text = f"ID {torrent_id} | " if torrent_id is not None else ""

        lines.append(
            f"\n<b>{index}. {name}</b>\n"
            f"   ⇡ Раздано: <b>{fmt_bytes(uploaded)}</b> | Ratio <b>{ratio:.2f}</b>\n"
            f"   ⇣ Скачано: <b>{fmt_bytes(downloaded)}</b> | Размер: <b>{fmt_bytes(total_size)}</b>\n"
            f"   {id_text}{html.escape(status_text)} | {date_label}: <b>{_format_history_ts(date_value)}</b>"
        )

    return "\n".join(lines), _torrent_history_keyboard(current_page, total_pages), current_page


async def send_torrent_history(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    page: int = 0,
    edit_existing: bool = False,
) -> None:
    warning: Optional[str] = None
    try:
        torrents = await tr_call(lambda c: c.get_torrents(arguments=TORRENT_HISTORY_FIELDS))
    except (TransmissionError, TRCallError) as exc:
        entries = await load_torrent_history_entries()
        warning = f"Transmission сейчас недоступен, показываю сохранённую историю: {exc}"
    else:
        entries = await sync_torrent_history(torrents, mark_missing=True)

    text, keyboard, current_page = _build_torrent_history_text(entries, page=page, warning=warning)
    _require_user_data(ctx)[TORRENT_HISTORY_LAST_PAGE_KEY] = current_page

    if edit_existing and update.callback_query is not None:
        await update.callback_query.edit_message_text(text=text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        return

    await reply_chunks(update, text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


async def on_torrent_history_view(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    if not await callback_user_allowed(update):
        return

    data = query.data or ""
    if not data.startswith(TORRENT_HISTORY_CB_PREFIX):
        await query.answer()
        return

    try:
        action, page_raw = data[len(TORRENT_HISTORY_CB_PREFIX) :].split(":", 1)
        page = int(page_raw)
    except (ValueError, TypeError):
        await query.answer("Некорректная страница", show_alert=True)
        return

    if action not in {"page", "refresh"}:
        await query.answer("Неизвестный режим", show_alert=True)
        return

    await query.answer("Обновляю историю…")
    set_menu(ctx, MENU_HISTORY)
    set_wait(ctx, WAIT_NONE)
    await send_torrent_history(update, ctx, page=page, edit_existing=True)


async def send_traffic_stats(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        stats = await tr_call(lambda c: c.session_stats())
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Ошибка Transmission: {html.escape(str(exc))}", reply_markup=KB_MAIN)
        return

    now = bot_now()
    downloaded = int(max(0, getattr(stats.cumulative_stats, "downloaded_bytes", 0)))
    uploaded = int(max(0, getattr(stats.cumulative_stats, "uploaded_bytes", 0)))

    anchors, history = await update_traffic_state(now, downloaded, uploaded)

    text = _build_traffic_stats_text(now, downloaded, uploaded, anchors, history)
    await reply_chunks(update, text, parse_mode=ParseMode.HTML, reply_markup=TRAFFIC_OVERVIEW_KEYBOARD)


async def send_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        stats, free_space, torrents = await asyncio.gather(
            tr_call(lambda c: c.session_stats()),
            _get_download_dir_free_space(),
            tr_call(lambda c: c.get_torrents()),
        )
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Ошибка Transmission: {html.escape(str(exc))}", reply_markup=KB_MAIN)
        return

    await sync_torrent_history(torrents, mark_missing=True)
    text = _build_status_text(stats, free_space, torrents)
    await reply_chunks(update, text, parse_mode=ParseMode.HTML, reply_markup=STATUS_KEYBOARD)


async def on_status_refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    if not await callback_user_allowed(update):
        return

    await query.answer()

    try:
        stats, free_space, torrents = await asyncio.gather(
            tr_call(lambda c: c.session_stats()),
            _get_download_dir_free_space(),
            tr_call(lambda c: c.get_torrents()),
        )
    except (TransmissionError, TRCallError) as exc:
        await query.edit_message_text(
            text=f"❌ Ошибка Transmission: {html.escape(str(exc))}",
            reply_markup=STATUS_KEYBOARD,
        )
        return

    await sync_torrent_history(torrents, mark_missing=True)
    await query.edit_message_text(
        text=_build_status_text(stats, free_space, torrents),
        parse_mode=ParseMode.HTML,
        reply_markup=STATUS_KEYBOARD,
    )


def _search_revision_from_view_mode(view_mode: str) -> Optional[str]:
    if not view_mode.startswith(TORRENT_LIST_SEARCH_VIEW_PREFIX):
        return None
    revision = view_mode[len(TORRENT_LIST_SEARCH_VIEW_PREFIX) :]
    return revision if TORRENT_LIST_SEARCH_REVISION_RE.fullmatch(revision) else None


def _is_torrent_list_view_mode(view_mode: str) -> bool:
    return view_mode in TORRENT_LIST_VIEW_MODES or _search_revision_from_view_mode(view_mode) is not None


def _parse_list_view_payload(payload: str) -> tuple[str, int]:
    parts = payload.split(":")
    if len(parts) == 1:
        view_mode, page_raw = parts[0], "0"
    elif len(parts) == 2:
        view_mode, page_raw = parts
    else:
        raise ValueError("invalid list callback payload")

    if not _is_torrent_list_view_mode(view_mode) or not page_raw.isdigit():
        raise ValueError("invalid list callback payload")
    return view_mode, int(page_raw)


def _resolve_list_view(
    ctx: ContextTypes.DEFAULT_TYPE,
    view_mode: str,
) -> Optional[tuple[str, Optional[str], Optional[str]]]:
    if view_mode in TORRENT_LIST_MODES:
        return view_mode, None, None

    search_revision = _search_revision_from_view_mode(view_mode)
    if search_revision is None:
        return None

    user_data = _require_user_data(ctx)
    stored_mode = user_data.get(TORRENT_LIST_LAST_MODE_KEY)
    stored_query = user_data.get(TORRENT_LIST_LAST_QUERY_KEY)
    stored_revision = user_data.get(TORRENT_LIST_SEARCH_REVISION_KEY)
    if (
        stored_mode not in TORRENT_LIST_MODES
        or not isinstance(stored_query, str)
        or not stored_query.strip()
        or stored_revision != search_revision
    ):
        return None
    return str(stored_mode), stored_query, search_revision


async def on_list_refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    if not await callback_user_allowed(update):
        return

    data = query.data or ""
    if not data.startswith(LIST_REFRESH_CB_PREFIX):
        await query.answer()
        return

    try:
        view_mode, page = _parse_list_view_payload(data[len(LIST_REFRESH_CB_PREFIX) :])
    except ValueError:
        await query.answer("Неизвестный тип списка", show_alert=True)
        return

    resolved_view = _resolve_list_view(ctx, view_mode)
    if resolved_view is None:
        await query.answer("Поиск устарел. Запусти его заново.", show_alert=True)
        return

    mode, search_query, search_revision = resolved_view
    await query.answer("Обновляю список…")
    await send_torrent_list(
        update,
        ctx,
        mode=mode,
        query=search_query,
        search_revision=search_revision,
        page=page,
        edit_existing=True,
    )


def _parse_torrent_action_payload(payload: str) -> tuple[str, int, str, int]:
    parts = payload.split(":")
    if len(parts) == 3:
        action, torrent_id_raw, view_mode = parts
        page_raw = "0"
    elif len(parts) == 4:
        action, torrent_id_raw, view_mode, page_raw = parts
    else:
        raise ValueError("invalid torrent action payload")

    if not torrent_id_raw.isdigit() or not _is_torrent_list_view_mode(view_mode) or not page_raw.isdigit():
        raise ValueError("invalid torrent action payload")
    return action, int(torrent_id_raw), view_mode, int(page_raw)


async def _set_torrent_running_state(action: str, torrent_id: int) -> str:
    if action == "pause":
        await tr_call(
            lambda c: c.stop_torrent(torrent_id),
            retry_on_connection=False,
            operation="stop_torrent",
        )
        return f"⏸️ Торрент {torrent_id} остановлен"
    if action == "start":
        await tr_call(
            lambda c: c.start_torrent(torrent_id),
            retry_on_connection=False,
            operation="start_torrent",
        )
        return f"▶️ Торрент {torrent_id} запущен"
    raise ValueError("unknown torrent running-state action")


async def on_torrent_action(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    if not await callback_user_allowed(update):
        return

    payload = query.data or ""
    if not payload.startswith(TORRENT_ACTION_CB_PREFIX):
        return

    try:
        action, torrent_id, view_mode, page = _parse_torrent_action_payload(payload[len(TORRENT_ACTION_CB_PREFIX) :])
    except ValueError:
        await query.answer("Некорректные данные", show_alert=True)
        return

    if action in {"del_keep", "del_data"}:
        await query.answer("Готовлю подтверждение…")
        await _request_delete_confirmation(update, ctx, action=action, torrent_id=torrent_id)
        return

    if action not in {"pause", "start"}:
        await query.answer("Неизвестное действие", show_alert=True)
        return

    resolved_view = _resolve_list_view(ctx, view_mode)
    if resolved_view is None:
        await query.answer("Поиск устарел. Запусти его заново.", show_alert=True)
        return
    mode, search_query, search_revision = resolved_view

    await query.answer("Выполняю…")
    try:
        await _set_torrent_running_state(action, torrent_id)
    except KeyError:
        await reply_chunks(
            update,
            f"❌ Торрент {torrent_id} больше не найден.",
            reply_markup=KB_TORRENTS,
        )
        return
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(
            update,
            f"❌ Ошибка Transmission: {html.escape(str(exc))}",
            parse_mode=ParseMode.HTML,
            reply_markup=KB_TORRENTS,
        )
        return

    await send_torrent_list(
        update,
        ctx,
        mode=mode,
        query=search_query,
        search_revision=search_revision,
        page=page,
        edit_existing=True,
    )


async def _edit_traffic_message(query: Any, text: str) -> None:
    """Edit callback message text/caption depending on message type."""
    message = query.message
    if message is not None and message.text is None and message.caption is not None:
        await query.edit_message_caption(text, parse_mode=ParseMode.HTML, reply_markup=TRAFFIC_OVERVIEW_KEYBOARD)
        return

    await query.edit_message_text(text=text, parse_mode=ParseMode.HTML, reply_markup=TRAFFIC_OVERVIEW_KEYBOARD)


async def on_traffic_view(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: C901
    query = update.callback_query
    if query is None:
        return

    if not await callback_user_allowed(update):
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

    now = bot_now()
    downloaded = int(max(0, getattr(stats.cumulative_stats, "downloaded_bytes", 0)))
    uploaded = int(max(0, getattr(stats.cumulative_stats, "uploaded_bytes", 0)))
    anchors, history = await update_traffic_state(now, downloaded, uploaded)
    downloaded, uploaded = _effective_traffic_totals(anchors, downloaded, uploaded)

    if mode == "refresh":
        text = _build_traffic_stats_text(now, downloaded, uploaded, anchors, history)
        await _edit_traffic_message(query, text)
        return

    if mode == "4w":
        try:
            day_points, chart_payload, chart_error = await asyncio.to_thread(
                _build_traffic_chart_current_month,
                now,
                downloaded,
                uploaded,
                history,
            )
        except Exception:
            log.exception("Failed to build monthly traffic chart")
            await _edit_traffic_message(query, "❌ Не удалось построить график за месяц. Попробуйте позже.")
            return

        if chart_payload is None or day_points is None:
            text = _build_last_4_weeks_text(now, downloaded, uploaded, history)
            if chart_error:
                text = f"{text}\n\n⚠️ {chart_error}"
            await _edit_traffic_message(query, text)
            return

        caption = (
            f"🗓️ <b>{_month_name_ru(now.month)} {now.year}</b>\n"
            f"Сумма: ⇣ <b>{fmt_bytes(sum(int(item['downloaded']) for item in day_points))}</b> "
            f"| ⇡ <b>{fmt_bytes(sum(int(item['uploaded']) for item in day_points))}</b>"
        )

        message = query.message
        if not isinstance(message, Message):
            await query.answer("Не удалось отправить график", show_alert=True)
            return

        image_file = InputFile(io.BytesIO(chart_payload), filename="traffic_month.png")
        try:
            await message.reply_photo(
                photo=image_file,
                caption=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=TRAFFIC_OVERVIEW_KEYBOARD,
            )
        except TelegramError:
            log.exception("Failed to send monthly traffic chart")
            await _edit_traffic_message(query, "❌ Не удалось отправить график за месяц. Попробуйте позже.")
            return

        await query.answer("График отправлен")
        return

    chart_points = _traffic_points_last_7_days(now, downloaded, uploaded, history)
    try:
        chart_payload, chart_error = await asyncio.to_thread(_build_traffic_chart_last_7_days, chart_points)
    except Exception:
        log.exception("Failed to build 7-day traffic chart")
        await _edit_traffic_message(query, "❌ Не удалось построить график за 7 дней. Попробуйте позже.")
        return

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

    message = query.message
    if not isinstance(message, Message):
        await query.answer("Не удалось отправить график", show_alert=True)
        return

    image_file = InputFile(io.BytesIO(chart_payload), filename="traffic_7d.png")
    try:
        await message.reply_photo(
            photo=image_file,
            caption=caption,
            parse_mode=ParseMode.HTML,
            reply_markup=TRAFFIC_OVERVIEW_KEYBOARD,
        )
    except TelegramError:
        log.exception("Failed to send 7-day traffic chart")
        await _edit_traffic_message(query, "❌ Не удалось отправить график за 7 дней. Попробуйте позже.")
        return

    await query.answer("График отправлен")


def _is_active(status: str) -> bool:
    return status in ACTIVE_STATUSES


def _is_downloading(status: str) -> bool:
    return status in DOWNLOADING_STATUSES


def _is_torrent_completed(torrent: Any) -> bool:
    status = str(getattr(torrent, "status", "") or "").strip().lower()
    if status in COMPLETED_STATUSES:
        return True

    if torrent_progress_percent(torrent) >= 100.0:
        return True

    left_until_done = _torrent_left_until_done(torrent)
    return left_until_done == 0 and torrent_total_size(torrent) > 0


def _torrent_list_title(mode: str) -> tuple[str, str]:
    return {
        "all": ("📋", "Все торренты"),
        "downloading": ("⬇️", "Скачиваются"),
        "stopped": ("⏸️", "Остановленные"),
        "done": ("✅", "Завершённые"),
    }.get(mode, ("📋", "Торренты"))


def _format_search_query_for_display(query: str) -> str:
    compact_query = " ".join(query.split())
    return html.escape(_shorten_text(compact_query, 96))


def _build_torrent_list_header(
    mode: str,
    *,
    total: int,
    page: int,
    total_pages: int,
    query: Optional[str],
) -> str:
    icon, title = _torrent_list_title(mode)
    lines = [f"{icon} <b>{title}</b> · всего <b>{total}</b>"]
    if query:
        lines.append(f"🔎 Поиск: <code>{_format_search_query_for_display(query)}</code>")
    if total_pages > 1:
        lines.append(f"Страница <b>{page + 1}</b> из <b>{total_pages}</b>")
    return "\n".join(lines)


def _build_empty_torrent_list_text(mode: str, query: Optional[str]) -> str:
    if query:
        return (
            "🔎 <b>Ничего не найдено</b>\n\n"
            f"По запросу <code>{_format_search_query_for_display(query)}</code> совпадений нет. "
            "Попробуй другое название."
        )

    messages = {
        "downloading": "Сейчас нет активных скачиваний.",
        "stopped": "Сейчас нет остановленных торрентов.",
        "done": "Завершённых торрентов пока нет.",
        "all": "В Transmission пока нет торрентов.",
    }
    icon, title = _torrent_list_title(mode)
    return f"{icon} <b>{title}</b>\n\n{messages.get(mode, 'Список пуст.')}"


async def _edit_torrent_list_message(
    query: Any,
    *,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    try:
        await query.edit_message_text(
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
    except BadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return
        raise


async def send_torrent_list(  # noqa: C901
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    mode: str,
    query: Optional[str] = None,
    search_revision: Optional[str] = None,
    page: int = 0,
    edit_existing: bool = False,
) -> None:
    search_query = query.strip() if isinstance(query, str) and query.strip() else None
    active_search_revision: Optional[str] = None
    if search_query:
        active_search_revision = (
            search_revision
            if isinstance(search_revision, str) and TORRENT_LIST_SEARCH_REVISION_RE.fullmatch(search_revision)
            else secrets.token_hex(4)
        )
        view_mode = f"{TORRENT_LIST_SEARCH_VIEW_PREFIX}{active_search_revision}"
    else:
        view_mode = mode

    try:
        torrents = await tr_call(lambda c: c.get_torrents())
    except (TransmissionError, TRCallError) as exc:
        error_text = f"❌ Ошибка Transmission: {html.escape(str(exc))}"
        if edit_existing and update.callback_query is not None:
            await _edit_torrent_list_message(
                update.callback_query,
                text=error_text,
                reply_markup=TORRENT_LIST_KEYBOARD,
            )
        else:
            await reply_chunks(
                update,
                error_text,
                parse_mode=ParseMode.HTML,
                reply_markup=TORRENT_LIST_KEYBOARD,
            )
        return

    await sync_torrent_history(torrents, mark_missing=True)
    items = list(torrents)
    if mode == "downloading":
        items = [t for t in items if _is_downloading(str(getattr(t, "status", "")))]
    elif mode == "stopped":
        items = [t for t in items if str(getattr(t, "status", "")) == "stopped"]
    elif mode == "done":
        items = [t for t in items if _is_torrent_completed(t)]

    if search_query:
        normalized_query = search_query.casefold()
        items = [t for t in items if normalized_query in str(getattr(t, "name", "") or "").casefold()]

    items = _sort_torrents(items)
    total = len(items)
    page_size = min(max(1, get_config().list_limit), TORRENT_LIST_PAGE_SIZE_MAX)
    total_pages = max(1, ceil(total / page_size))
    current_page = max(0, min(page, total_pages - 1))
    page_start = current_page * page_size
    page_items = items[page_start : page_start + page_size]

    user_data = _require_user_data(ctx)
    user_data[TORRENT_LIST_LAST_MODE_KEY] = mode
    user_data[TORRENT_LIST_LAST_QUERY_KEY] = search_query
    user_data[TORRENT_LIST_SEARCH_REVISION_KEY] = active_search_revision

    list_keyboard = _torrent_actions_keyboard(
        page_items,
        mode,
        page=current_page,
        total_pages=total_pages,
        view_mode=view_mode,
    )

    if total == 0:
        empty_text = _build_empty_torrent_list_text(mode, search_query)
        if edit_existing and update.callback_query is not None:
            await _edit_torrent_list_message(
                update.callback_query,
                text=empty_text,
                reply_markup=list_keyboard,
            )
        else:
            await reply_chunks(
                update,
                empty_text,
                parse_mode=ParseMode.HTML,
                reply_markup=list_keyboard,
            )
        return

    lines: list[str] = []
    for torrent in page_items:
        status = str(getattr(torrent, "status", "") or "").strip().lower()
        name = _shorten_text(str(getattr(torrent, "name", "") or "<без названия>"), 82)
        safe_name = html.escape(name)
        torrent_id = _non_negative_int(_get_mapping_or_attr_value(torrent, ("id",)))
        id_text = str(torrent_id) if torrent_id is not None else "?"
        rate_download = _get_mapping_or_attr_value(torrent, ("rate_download", "rateDownload")) or 0
        rate_upload = _get_mapping_or_attr_value(torrent, ("rate_upload", "rateUpload")) or 0
        upload_ratio = _non_negative_float(_get_mapping_or_attr_value(torrent, ("upload_ratio", "uploadRatio")))
        size_text = fmt_bytes(torrent_total_size(torrent))
        lines.append(
            f"<b>{id_text} · {safe_name}</b>\n"
            f"{status_icon(status)} {html.escape(_torrent_status_label_ru(status))} · <b>{size_text}</b>\n"
            f"{_format_progress_summary(torrent, hide_completed_bar=True)}\n"
            f"⇣ {fmt_rate(rate_download)} · ⇡ {fmt_rate(rate_upload)} · "
            f"ratio {(upload_ratio or 0.0):.2f}"
        )

    header = _build_torrent_list_header(
        mode,
        total=total,
        page=current_page,
        total_pages=total_pages,
        query=search_query,
    )

    if edit_existing and update.callback_query is not None:
        await _edit_torrent_list_message(
            update.callback_query,
            text=_build_single_torrent_message(header, lines, ""),
            reply_markup=list_keyboard,
        )
        return

    messages = _build_torrent_messages(header, lines, "")
    for idx, text in enumerate(messages):
        await reply_chunks(
            update,
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=list_keyboard if idx == len(messages) - 1 else None,
        )


async def _hydrate_added_torrent(torrent: Any, *, operation: str) -> tuple[Any, bool]:
    selector: str | int | None = _torrent_history_hash(torrent)
    if selector is None:
        torrent_id = _non_negative_int(_get_mapping_or_attr_value(torrent, ("id",)))
        selector = torrent_id if torrent_id is not None and torrent_id > 0 else None

    if selector is None:
        log.warning("Torrent was added but its details cannot be refreshed because the response has no id or hash")
        return torrent, False

    try:
        hydrated = await tr_call(
            lambda c: c.get_torrent(selector),
            operation=operation,
        )
    except (KeyError, TransmissionError, TRCallError):
        log.warning("Torrent %s was added but its details refresh failed during %s", selector, operation, exc_info=True)
        return torrent, False

    return hydrated, True


def _added_torrent_display_values(torrent: Any) -> tuple[str, str]:
    name = str(_get_mapping_or_attr_value(torrent, ("name",)) or "<без названия>")
    torrent_id = _non_negative_int(_get_mapping_or_attr_value(torrent, ("id",)))
    return name, str(torrent_id) if torrent_id is not None and torrent_id > 0 else "неизвестен"


async def _register_added_torrent_watch_safe(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: Optional[int],
    torrent: Any,
) -> None:
    try:
        await _register_torrent_start_watch(ctx, chat_id, torrent)
    except Exception:
        log.exception("Torrent was added, but its start watch could not be persisted")


async def _sync_added_torrent_history_safe(torrent: Any) -> None:
    try:
        await sync_torrent_history([torrent], mark_missing=False)
    except Exception:
        log.exception("Torrent was added, but its history snapshot could not be persisted")


def _schedule_added_torrent_watch_safe(ctx: ContextTypes.DEFAULT_TYPE, torrent: Any) -> None:
    try:
        _schedule_torrent_start_watch(ctx, torrent)
    except Exception:
        log.exception("Torrent was added, but its quick start watch could not be scheduled")


async def add_magnet_or_url(update: Update, ctx: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    link = text.strip()
    if not is_magnet_or_torrent_link(link):
        await reply_chunks(update, "❌ Нужна magnet-ссылка или http(s) URL на .torrent.", reply_markup=KB_ADD)
        return

    free_space_before = await _get_download_dir_free_space()

    try:
        added_torrent = await tr_call(
            lambda c: c.add_torrent(link),
            retry_on_connection=False,
            operation="add_torrent_url",
        )
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Не удалось добавить: {html.escape(str(exc))}", reply_markup=KB_ADD)
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    await _register_added_torrent_watch_safe(ctx, chat_id, added_torrent)
    torrent, details_refreshed = await _hydrate_added_torrent(
        added_torrent,
        operation="hydrate_added_torrent_url",
    )
    if details_refreshed:
        await _sync_added_torrent_history_safe(torrent)

    torrent_name, torrent_id_text = _added_torrent_display_values(torrent)
    details_warning = ""
    if not details_refreshed:
        details_warning = "⚠️ Торрент добавлен, но свежие детали пока не удалось получить.\n"

    await reply_chunks(
        update,
        (
            f"✅ Торрент добавлен: <b>{html.escape(torrent_name)}</b>\n"
            f"ID: <b>{torrent_id_text}</b>\n"
            f"{details_warning}"
            f"{_build_projected_free_space_text(free_space_before, torrent)}"
        ),
        parse_mode=ParseMode.HTML,
        reply_markup=KB_ADD,
    )
    _schedule_added_torrent_watch_safe(ctx, torrent)


def is_magnet_or_torrent_link(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith("magnet:") or normalized.startswith("http://") or normalized.startswith("https://")


def _validate_downloaded_torrent_file(path: Path) -> None:
    downloaded_size = path.stat().st_size
    if downloaded_size <= 0:
        raise ValueError("получен пустой .torrent файл")
    if downloaded_size > TORRENT_FILE_MAX_BYTES:
        raise ValueError("размер .torrent файла превышает лимит 10 MiB")


def _delete_temporary_file_safe(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("Failed to remove temporary torrent file %s", path, exc_info=True)


async def add_torrent_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or message.document is None:
        await reply_chunks(update, "Пришли .torrent файлом.", reply_markup=KB_ADD)
        return

    doc = message.document
    if not (doc.file_name or "").lower().endswith(".torrent"):
        await reply_chunks(update, "Это не .torrent файл.", reply_markup=KB_ADD)
        return

    declared_size = doc.file_size
    if isinstance(declared_size, int) and declared_size > TORRENT_FILE_MAX_BYTES:
        await reply_chunks(
            update,
            "❌ .torrent файл слишком большой. Максимальный размер: 10 MiB.",
            reply_markup=KB_ADD,
        )
        return

    free_space_before = await _get_download_dir_free_space()
    tmp_path: Optional[Path] = None

    try:
        tg_file = await doc.get_file()
        with tempfile.NamedTemporaryFile(prefix="tg_", suffix=".torrent", delete=False) as temp_file:
            tmp_path = Path(temp_file.name)

        await tg_file.download_to_drive(custom_path=str(tmp_path))
        _validate_downloaded_torrent_file(tmp_path)

        def _add(client: Client):
            if tmp_path is None:
                raise RuntimeError("Temporary torrent file path is missing")
            with tmp_path.open("rb") as rf:
                return client.add_torrent(rf)

        added_torrent = await tr_call(_add, retry_on_connection=False, operation="add_torrent_file")

    except (TelegramError, OSError, ValueError, TransmissionError, TRCallError) as exc:
        await reply_chunks(update, f"❌ Не удалось добавить .torrent: {html.escape(str(exc))}", reply_markup=KB_ADD)
        return
    finally:
        _delete_temporary_file_safe(tmp_path)

    chat_id = update.effective_chat.id if update.effective_chat else None
    await _register_added_torrent_watch_safe(ctx, chat_id, added_torrent)
    torrent, details_refreshed = await _hydrate_added_torrent(
        added_torrent,
        operation="hydrate_added_torrent_file",
    )
    if details_refreshed:
        await _sync_added_torrent_history_safe(torrent)

    torrent_name, torrent_id_text = _added_torrent_display_values(torrent)
    details_warning = ""
    if not details_refreshed:
        details_warning = "⚠️ Торрент добавлен, но свежие детали пока не удалось получить.\n"

    await reply_chunks(
        update,
        (
            f"✅ Торрент добавлен из файла: <b>{html.escape(torrent_name)}</b>\n"
            f"ID: <b>{torrent_id_text}</b>\n"
            f"{details_warning}"
            f"{_build_projected_free_space_text(free_space_before, torrent)}"
        ),
        parse_mode=ParseMode.HTML,
        reply_markup=KB_ADD,
    )
    _schedule_added_torrent_watch_safe(ctx, torrent)


async def _delete_torrent_action(
    update: Update,
    *,
    action: str,
    torrent_id: int,
    expected_torrent_hash: Optional[str],
    reply_markup: ReplyKeyboardMarkup,
) -> Optional[str]:
    torrent = await tr_call(
        lambda c: c.get_torrent(torrent_id, arguments=TORRENT_HISTORY_FIELDS),
        operation="get_torrent_for_delete",
    )
    torrent_hash = _torrent_history_hash(torrent)
    expected_hash = expected_torrent_hash.strip().lower() if isinstance(expected_torrent_hash, str) else None
    if torrent_hash is None:
        await reply_chunks(
            update,
            "❌ Не удалось определить стабильный hash торрента. Удаление отменено.",
            reply_markup=reply_markup,
        )
        return None
    if expected_hash is not None and torrent_hash != expected_hash:
        await reply_chunks(
            update,
            "⚠️ Запрос подтверждения устарел: под этим ID уже другой торрент. Удаление отменено.",
            reply_markup=reply_markup,
        )
        return None

    with_data = action == "del_data"
    try:
        await sync_torrent_history([torrent], mark_missing=False)
    except Exception:
        log.exception("Torrent history snapshot failed before deletion")
    await tr_call(
        lambda c: c.remove_torrent(torrent_hash, delete_data=with_data),
        retry_on_connection=False,
        operation="remove_torrent_with_data" if with_data else "remove_torrent_keep_data",
    )
    try:
        await mark_torrent_history_removed(torrent, with_data=with_data)
    except Exception:
        log.exception("Torrent was removed, but its history entry could not be updated")

    if with_data:
        return f"💥 Удалено вместе с данными: ID {torrent_id} | {torrent.name}"
    return f"🗑️ Удалено (данные сохранены): ID {torrent_id} | {torrent.name}"


async def ctrl_action(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    action: str,
    torrent_id: int,
    *,
    expected_torrent_hash: Optional[str] = None,
) -> None:
    ctrl_keyboard = _ctrl_keyboard_for_chat(ctx, update.effective_chat.id if update.effective_chat else None)

    try:
        if action in {"pause", "start"}:
            msg = await _set_torrent_running_state(action, torrent_id)
        elif action in {"del_keep", "del_data"}:
            delete_message = await _delete_torrent_action(
                update,
                action=action,
                torrent_id=torrent_id,
                expected_torrent_hash=expected_torrent_hash,
                reply_markup=ctrl_keyboard,
            )
            if delete_message is None:
                return
            msg = delete_message
        else:
            msg = "❌ Неизвестное действие"
    except KeyError:
        await reply_chunks(
            update,
            f"❌ Торрент с ID {torrent_id} больше не найден.",
            reply_markup=ctrl_keyboard,
        )
        return
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(
            update,
            f"❌ Ошибка Transmission: {html.escape(str(exc))}",
            reply_markup=ctrl_keyboard,
        )
        return

    await reply_chunks(update, msg, reply_markup=ctrl_keyboard)


def _build_delete_confirm_keyboard(action: str, torrent_id: int) -> InlineKeyboardMarkup:
    if action == "del_data":
        confirm_cb = f"{CONFIRM_DEL_DATA_CB_PREFIX}{torrent_id}"
        cancel_cb = CANCEL_DEL_DATA_CB
        confirm_label = "✅ Да, удалить с данными"
    else:
        confirm_cb = f"{CONFIRM_DEL_KEEP_CB_PREFIX}{torrent_id}"
        cancel_cb = CANCEL_DEL_KEEP_CB
        confirm_label = "✅ Да, удалить"

    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(confirm_label, callback_data=confirm_cb)],
            [InlineKeyboardButton("↩️ Отмена", callback_data=cancel_cb)],
        ]
    )


def _build_delete_confirm_text(torrent: Any, *, with_data: bool) -> str:
    title = "⚠️ Подтверждение удаления (с данными)" if with_data else "⚠️ Подтверждение удаления"
    return (
        f"{title}\n\n"
        f"ID: <b>{torrent.id}</b>\n"
        f"Название: <b>{html.escape(torrent.name or '<без названия>')}</b>\n"
        f"Размер: <b>{fmt_bytes(torrent_total_size(torrent))}</b>\n"
        f"Прогресс: {_format_progress_summary(torrent)}"
    )


async def _request_delete_confirmation(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    action: str,
    torrent_id: int,
) -> None:
    ctrl_keyboard = _ctrl_keyboard_for_chat(ctx, update.effective_chat.id if update.effective_chat else None)
    _require_user_data(ctx).pop(PENDING_CTRL_ACTION_KEY, None)

    try:
        torrent = await tr_call(lambda c: c.get_torrent(torrent_id))
    except KeyError:
        await reply_chunks(
            update,
            f"❌ Торрент с ID {torrent_id} больше не найден.",
            reply_markup=ctrl_keyboard,
        )
        return
    except (TransmissionError, TRCallError) as exc:
        await reply_chunks(
            update,
            f"❌ Ошибка Transmission: {html.escape(str(exc))}",
            reply_markup=ctrl_keyboard,
        )
        return

    torrent_hash = _torrent_history_hash(torrent)
    if torrent_hash is None:
        await reply_chunks(
            update,
            "❌ Не удалось определить стабильный hash торрента. Удаление отменено.",
            reply_markup=ctrl_keyboard,
        )
        return

    set_wait(ctx, WAIT_NONE)
    set_menu(ctx, MENU_CTRL)
    _require_user_data(ctx)[PENDING_CTRL_ACTION_KEY] = {
        "action": action,
        "torrent_id": torrent_id,
        "torrent_hash": torrent_hash,
    }

    await reply_chunks(
        update,
        _build_delete_confirm_text(torrent, with_data=(action == "del_data")),
        parse_mode=ParseMode.HTML,
        reply_markup=_build_delete_confirm_keyboard(action, torrent_id),
    )


async def on_delete_confirmation(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    if not await callback_user_allowed(update):
        return

    data = query.data or ""
    pending = _require_user_data(ctx).get(PENDING_CTRL_ACTION_KEY)
    set_menu(ctx, MENU_CTRL)
    set_wait(ctx, WAIT_NONE)

    if data in (CANCEL_DEL_DATA_CB, CANCEL_DEL_KEEP_CB):
        _require_user_data(ctx).pop(PENDING_CTRL_ACTION_KEY, None)
        await query.answer("Отменено")
        await query.edit_message_reply_markup(reply_markup=None)
        ctrl_keyboard = _ctrl_keyboard_for_chat(ctx, update.effective_chat.id if update.effective_chat else None)
        await reply_chunks(update, "Ок, удаление отменено.", reply_markup=ctrl_keyboard)
        return

    action = "del_data" if data.startswith(CONFIRM_DEL_DATA_CB_PREFIX) else "del_keep"
    prefix = CONFIRM_DEL_DATA_CB_PREFIX if action == "del_data" else CONFIRM_DEL_KEEP_CB_PREFIX
    id_part = data[len(prefix) :]
    if not id_part.isdigit():
        await query.answer("Некорректный ID", show_alert=True)
        return

    torrent_id = int(id_part)
    if not isinstance(pending, dict):
        await query.answer("Запрос подтверждения устарел", show_alert=True)
        return

    pending_action = pending.get("action")
    pending_id = pending.get("torrent_id")
    if pending_action != action or pending_id != torrent_id:
        await query.answer("Запрос подтверждения устарел", show_alert=True)
        return

    pending_hash = pending.get("torrent_hash")
    if not isinstance(pending_hash, str) or not pending_hash.strip():
        _require_user_data(ctx).pop(PENDING_CTRL_ACTION_KEY, None)
        await query.answer("Запрос подтверждения устарел", show_alert=True)
        return

    _require_user_data(ctx).pop(PENDING_CTRL_ACTION_KEY, None)
    await query.answer("Выполняю…")
    await query.edit_message_reply_markup(reply_markup=None)
    await ctrl_action(
        update,
        ctx,
        action,
        torrent_id=torrent_id,
        expected_torrent_hash=pending_hash,
    )


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.type != "private":
        return

    if not user_allowed(update):
        await reply_chunks(update, "⛔️ Доступ запрещён.")
        return

    await _ensure_chat_notifications_initialized(ctx, update.effective_chat.id if update.effective_chat else None)

    set_menu(ctx, MENU_MAIN)
    set_wait(ctx, WAIT_NONE)
    await reply_chunks(
        update,
        "Привет! Я бот для управления Transmission.\n"
        "Выбирай пункт меню ниже 👇\n\n"
        "💡 Подсказка: если бот ждёт ввод, нажми «⬅️ Назад» — это сбросит текущий шаг.",
        reply_markup=KB_MAIN,
    )


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
        "• 📚 История раздач — сколько отдано и какой ratio у каждого торрента, включая удалённые\n"
        "• 📋 Торренты — постраничные списки, фильтры, поиск и быстрые действия\n"
        "• ➕ Добавить — magnet/URL или .torrent файл\n"
        "• ⚙️ Управление — пауза/старт/удаление по ID\n\n"
        "Подсказка: ID виден в списках торрентов.\n"
        "Отменить любой шаг ввода можно кнопкой «⬅️ Назад»."
    )
    await reply_chunks(update, text, parse_mode=ParseMode.HTML, reply_markup=KB_MAIN)


async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.type != "private":
        return

    if not user_allowed(update):
        return

    await _ensure_chat_notifications_initialized(ctx, update.effective_chat.id if update.effective_chat else None)

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
            chat_id = update.effective_chat.id if update.effective_chat else None
            await send_ephemeral(
                update,
                ctx,
                "Пришли числовой ID торрента (например: 12).",
                reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id, input_active=True),
            )
            return True

        if wait == WAIT_CTRL_DEL_DATA:
            await _request_delete_confirmation(update, ctx, action="del_data", torrent_id=torrent_id)
            return True

        if wait == WAIT_CTRL_DEL_KEEP and CONFIRM_DEL_KEEP_FLOW:
            await _request_delete_confirmation(update, ctx, action="del_keep", torrent_id=torrent_id)
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
        await send_ephemeral(
            update,
            ctx,
            "❌ Не удалось определить чат для настройки уведомлений.",
            reply_markup=KB_CTRL,
        )
        return

    enabled_chats = ctx.application.bot_data.setdefault(NOTIFY_ENABLED_CHATS_KEY, set())
    enabled = await asyncio.to_thread(get_state_store().toggle_chat, chat_id)
    if not enabled:
        enabled_chats.discard(chat_id)
        await asyncio.to_thread(get_state_store().cancel_pending_outbox, chat_id)
        status_text = "🔕 Уведомления о торрентах выключены."
    else:
        enabled_chats.add(chat_id)
        status_text = "🔔 Уведомления о торрентах включены."

    await send_ephemeral(update, ctx, status_text, reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id))


async def _handle_global_command(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    text: str,
    chat_id: Optional[int],
) -> bool:
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

    async def _open_history() -> None:
        set_menu(ctx, MENU_HISTORY)
        set_wait(ctx, WAIT_NONE)
        await send_torrent_history(update, ctx)

    handlers: dict[str, Callable[[], Awaitable[None]]] = {
        "📊 Статус": _open_main_status,
        "📈 Статистика": lambda: send_traffic_stats(update, ctx),
        "📚 История раздач": _open_history,
        "📋 Торренты": _open_torrents,
        "➕ Добавить": _open_add,
        "⚙️ Управление": _open_ctrl,
    }

    handler = handlers.get(text)
    if handler is None:
        return False

    await handler()
    return True


async def _handle_menu_command(  # noqa: C901
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
            reply_markup=KB_ADD_INPUT,
        )

    async def _ask_add_file() -> None:
        set_wait(ctx, WAIT_ADD_TORRENT_FILE)
        free_space = await _get_download_dir_free_space()
        await send_ephemeral(
            update,
            ctx,
            f"Ок, пришли .torrent файлом сюда в чат.\n{_build_free_space_text(free_space)}",
            reply_markup=KB_ADD_INPUT,
        )

    async def _ask_pause() -> None:
        set_wait(ctx, WAIT_CTRL_PAUSE)
        await send_ephemeral(
            update,
            ctx,
            "Пришли ID торрента для остановки:",
            reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id, input_active=True),
        )

    async def _ask_start() -> None:
        set_wait(ctx, WAIT_CTRL_START)
        await send_ephemeral(
            update,
            ctx,
            "Пришли ID торрента для запуска:",
            reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id, input_active=True),
        )

    async def _ask_del_keep() -> None:
        _require_user_data(ctx).pop(PENDING_CTRL_ACTION_KEY, None)
        set_wait(ctx, WAIT_CTRL_DEL_KEEP)
        await send_ephemeral(
            update,
            ctx,
            "Пришли ID торрента для удаления (данные останутся на диске):",
            reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id),
        )

    async def _ask_del_data() -> None:
        _require_user_data(ctx).pop(PENDING_CTRL_ACTION_KEY, None)
        set_wait(ctx, WAIT_CTRL_DEL_DATA)
        await send_ephemeral(
            update,
            ctx,
            "⚠️ Пришли ID торрента для удаления вместе с данными:",
            reply_markup=_ctrl_keyboard_for_chat(ctx, chat_id),
        )

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


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: C901
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
    await _ensure_chat_notifications_initialized(ctx, chat_id)

    try:
        if text == "⬅️ Назад":
            pending_delete = _require_user_data(ctx).pop(PENDING_CTRL_ACTION_KEY, None)
            set_menu(ctx, MENU_MAIN)
            set_wait(ctx, WAIT_NONE)
            status_text = (
                "Удаление отменено. Возвращаюсь в главное меню." if pending_delete else "Ок, назад в главное меню."
            )
            await send_ephemeral(update, ctx, status_text, reply_markup=KB_MAIN)
            return

        if text == CANCEL_INPUT_BUTTON:
            pending_delete = _require_user_data(ctx).pop(PENDING_CTRL_ACTION_KEY, None)
            if get_wait(ctx) in {WAIT_NONE, None} and pending_delete is None:
                set_menu(ctx, MENU_MAIN)
                await send_ephemeral(update, ctx, "Сейчас нет активного ввода 🙂", reply_markup=KB_MAIN)
                return
            set_wait(ctx, WAIT_NONE)
            set_menu(ctx, MENU_MAIN)
            status_text = (
                "Удаление отменено. Выбери следующее действие."
                if pending_delete
                else "Ввод отменён. Выбери следующее действие."
            )
            await send_ephemeral(update, ctx, status_text, reply_markup=KB_MAIN)
            return

        menu = get_menu(ctx)
        wait = get_wait(ctx)

        if await _handle_wait_state(update, ctx, wait, text):
            return

        if is_magnet_or_torrent_link(text):
            set_wait(ctx, WAIT_NONE)
            set_menu(ctx, MENU_ADD)
            await add_magnet_or_url(update, ctx, text)
            return

        if await _handle_global_command(update, ctx, text, chat_id):
            return

        if await _handle_menu_command(update, ctx, menu, text, chat_id):
            return

        await send_ephemeral(update, ctx, "Не понял. Выбери пункт меню 🙂", reply_markup=KB_MAIN)
    finally:
        await _delete_user_message(update, ctx)


async def _enqueue_for_enabled_chats(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    event_key: str,
    kind: str,
    text: str,
) -> None:
    enabled_chats = ctx.application.bot_data.get(NOTIFY_ENABLED_CHATS_KEY)
    if not isinstance(enabled_chats, set):
        return
    for chat_id in list(enabled_chats):
        await _enqueue_notification(
            event_key=event_key,
            chat_id=chat_id,
            kind=kind,
            text=text,
        )


def _build_no_peers_notification_text(torrent_id: int, name: str) -> str:
    safe_name = html.escape(name or "<без названия>")
    return (
        "⚠️ <b>Торрент не начал скачиваться за 10 минут</b>\n"
        "Возможно, сейчас нет раздающих.\n"
        f"Торрент: <b>{safe_name}</b>\n"
        f"ID: <b>{torrent_id}</b>"
    )


async def _process_torrent_notifications(  # noqa: C901
    ctx: ContextTypes.DEFAULT_TYPE,
    torrents: Sequence[Any],
) -> None:
    store = get_state_store()
    now_ts = time_module.time()
    snapshots, watches = await asyncio.gather(
        asyncio.to_thread(store.get_snapshots),
        asyncio.to_thread(store.list_start_watches),
    )
    watches_by_key: dict[str, list[StartWatch]] = {}
    for watch in watches:
        watches_by_key.setdefault(watch.torrent_key, []).append(watch)

    current_by_key: dict[str, Any] = {}
    next_snapshots: list[Snapshot] = []
    for torrent in torrents:
        torrent_key = _notification_torrent_key(torrent)
        if torrent_key is None:
            continue
        current_by_key[torrent_key] = torrent
        previous = snapshots.get(torrent_key)
        completed = _is_torrent_completed(torrent)
        started = _torrent_start_detected(torrent)
        generation = previous.generation if previous is not None else 0
        watched_completion_pending = any(not watch.completion_notified for watch in watches_by_key.get(torrent_key, ()))
        completion_transition = completed and (
            (previous is not None and previous.present and not previous.completed) or watched_completion_pending
        )
        if completion_transition:
            generation += 1
            torrent_id = int(getattr(torrent, "id", 0))
            name = str(getattr(torrent, "name", "") or "<без названия>")
            await _enqueue_for_enabled_chats(
                ctx,
                event_key=f"completion:{torrent_key}:{generation}",
                kind="completion",
                text=_build_completion_notification_text(torrent_id, name, torrent),
            )

        next_snapshots.append(
            Snapshot(
                torrent_key=torrent_key,
                completed=completed,
                started=started,
                present=True,
                generation=generation,
                updated_at=now_ts,
            )
        )

    for torrent_key, previous in snapshots.items():
        if torrent_key in current_by_key:
            continue
        next_snapshots.append(
            Snapshot(
                torrent_key=torrent_key,
                completed=previous.completed,
                started=previous.started,
                present=False,
                generation=previous.generation,
                updated_at=now_ts,
            )
        )

    enabled_chats = ctx.application.bot_data.get(NOTIFY_ENABLED_CHATS_KEY)
    if not isinstance(enabled_chats, set):
        enabled_chats = set()

    for watch in watches:
        torrent = current_by_key.get(watch.torrent_key)
        if torrent is None:
            await asyncio.to_thread(store.delete_start_watch, watch.torrent_key, watch.chat_id)
            continue

        completed = _is_torrent_completed(torrent)
        started = _torrent_start_detected(torrent)
        start_notified = watch.start_notified
        no_peers_notified = watch.no_peers_notified

        if completed:
            # Completion was enqueued above for every currently enabled chat.
            await asyncio.to_thread(store.delete_start_watch, watch.torrent_key, watch.chat_id)
            continue

        if started and not start_notified:
            if watch.chat_id in enabled_chats:
                await _enqueue_notification(
                    event_key=f"start:{watch.torrent_key}:{int(watch.added_at * 1000)}",
                    chat_id=watch.chat_id,
                    kind="start",
                    text=_build_torrent_start_notification_text(watch.torrent_id, watch.name),
                )
            start_notified = True
        elif (
            not started
            and _torrent_is_attempting_download(torrent)
            and not no_peers_notified
            and now_ts - watch.added_at >= NOTIFY_NO_PEERS_DELAY_SEC
        ):
            if watch.chat_id in enabled_chats:
                await _enqueue_notification(
                    event_key=f"no-peers:{watch.torrent_key}:{int(watch.added_at * 1000)}",
                    chat_id=watch.chat_id,
                    kind="no_peers",
                    text=_build_no_peers_notification_text(watch.torrent_id, watch.name),
                )
            no_peers_notified = True

        if start_notified != watch.start_notified or no_peers_notified != watch.no_peers_notified:
            await asyncio.to_thread(
                store.update_start_watch,
                watch.torrent_key,
                watch.chat_id,
                start_notified=start_notified,
                no_peers_notified=no_peers_notified,
            )

    # Events are inserted before the snapshot. If the process stops between
    # these operations, the next tick repeats the same idempotent event keys.
    await asyncio.to_thread(store.save_snapshots, next_snapshots)


async def _monitor_tick(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _drain_notification_outbox(ctx)

    try:
        torrents = await tr_call(
            lambda c: c.get_torrents(arguments=TORRENT_HISTORY_FIELDS),
            operation="monitor.get_torrents",
        )
    except TRCallError:
        log.warning("Skipping torrent monitor tick because Transmission is unavailable", exc_info=True)
    else:
        await sync_torrent_history(torrents, mark_missing=True)
        await _process_torrent_notifications(ctx, torrents)

    try:
        stats = await tr_call(lambda c: c.session_stats(), operation="monitor.session_stats")
    except TRCallError:
        log.warning("Skipping traffic snapshot because Transmission is unavailable", exc_info=True)
    else:
        downloaded = int(max(0, getattr(stats.cumulative_stats, "downloaded_bytes", 0)))
        uploaded = int(max(0, getattr(stats.cumulative_stats, "uploaded_bytes", 0)))
        await update_traffic_state(bot_now(), downloaded, uploaded)

    await _drain_notification_outbox(ctx)


def main() -> None:  # noqa: C901
    initialize_runtime()

    async def monitor_fallback(app: Application) -> None:
        while True:
            fake_ctx = cast(ContextTypes.DEFAULT_TYPE, SimpleNamespace(application=app, bot=app.bot))
            try:
                await _monitor_tick(fake_ctx)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Unexpected error in fallback monitor tick")
            await asyncio.sleep(NOTIFY_POLL_INTERVAL_SEC)

    async def on_post_init(app: Application) -> None:
        enabled_chats = await asyncio.to_thread(get_state_store().list_enabled_chats)
        app.bot_data[NOTIFY_ENABLED_CHATS_KEY] = enabled_chats
        app.bot_data[NOTIFY_KNOWN_CHATS_KEY] = set(enabled_chats)
        if app.job_queue is None:
            app.bot_data["notify_poll_task"] = app.create_task(
                monitor_fallback(app),
                name="transmission-monitor-fallback",
            )
            log.warning("python-telegram-bot job queue is unavailable; using fallback monitor task.")

    async def on_post_shutdown(app: Application) -> None:
        task = app.bot_data.pop("notify_poll_task", None)
        if isinstance(task, asyncio.Task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        start_tasks = app.bot_data.pop(NOTIFY_START_TASKS_KEY, None)
        if isinstance(start_tasks, set):
            active_tasks = [task for task in start_tasks if isinstance(task, asyncio.Task)]
            for start_task in active_tasks:
                start_task.cancel()
            for start_task in active_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await start_task

    app = build_telegram_application(post_init=on_post_init, post_shutdown=on_post_shutdown)

    if app.job_queue is None:
        log.info("Job queue is unavailable; fallback polling task will be used.")
    else:
        app.job_queue.run_repeating(
            _monitor_tick,
            interval=NOTIFY_POLL_INTERVAL_SEC,
            first=5.0,
            name="transmission-monitor",
        )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(on_status_refresh, pattern=f"^{STATUS_REFRESH_CB}$"))
    app.add_handler(
        CallbackQueryHandler(
            on_list_refresh,
            pattern=(f"^{LIST_REFRESH_CB_PREFIX}(all|downloading|stopped|done|search|search\\.[0-9a-f]{{8}})(:\\d+)?$"),
        )
    )
    app.add_handler(CallbackQueryHandler(on_torrent_action, pattern=f"^{TORRENT_ACTION_CB_PREFIX}"))
    app.add_handler(CallbackQueryHandler(on_traffic_view, pattern=f"^{TRAFFIC_VIEW_CB_PREFIX}(refresh|7d|4w)$"))
    app.add_handler(CallbackQueryHandler(on_torrent_history_view, pattern=f"^{TORRENT_HISTORY_CB_PREFIX}"))
    app.add_handler(
        CallbackQueryHandler(
            on_delete_confirmation,
            pattern=(
                f"^({CONFIRM_DEL_DATA_CB_PREFIX}\\d+|{CANCEL_DEL_DATA_CB}|"
                f"{CONFIRM_DEL_KEEP_CB_PREFIX}\\d+|{CANCEL_DEL_KEEP_CB})$"
            ),
        )
    )
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    cfg = get_config()
    log.info("Bot started (timezone=%s, state_dir=%s)", cfg.timezone_name, cfg.state_dir)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
