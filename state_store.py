"""Durable SQLite state for the Transmission Telegram bot.

The module has no import-time side effects.  Call :meth:`SQLiteStateStore.initialize`
explicitly before using a store.  Every public operation opens and closes its own
SQLite connection, so a store instance can safely be shared by worker threads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple, Union

TrafficValue = Union[int, str]
TrafficAnchors = Dict[str, Dict[str, TrafficValue]]
TrafficHistory = List[Dict[str, TrafficValue]]
TorrentHistory = Dict[str, Dict[str, Any]]


class StateStoreError(RuntimeError):
    """Raised when persisted state cannot be decoded safely."""


@dataclass(frozen=True)
class OutboxItem:
    """A notification waiting for delivery (or retained after delivery)."""

    id: int
    event_key: str
    chat_id: int
    kind: str
    text: str
    status: str
    attempts: int
    next_attempt_at: float
    created_at: float
    updated_at: float
    delivered_at: Optional[float]
    last_error: Optional[str]


@dataclass(frozen=True)
class Snapshot:
    """Last observed state of a torrent, addressed by a stable torrent key."""

    torrent_key: str
    completed: bool
    started: bool
    present: bool
    generation: int
    updated_at: float = dataclass_field(default_factory=time.time)


@dataclass(frozen=True)
class StartWatch:
    """A per-chat torrent notification lifecycle surviving process restarts."""

    torrent_key: str
    chat_id: int
    torrent_id: int
    name: str
    added_at: float
    start_notified: bool = False
    no_peers_notified: bool = False
    completion_notified: bool = False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traffic_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    anchors_json TEXT NOT NULL,
    history_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS torrent_history (
    torrent_key TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS notification_chats (
    chat_id INTEGER PRIMARY KEY,
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS notification_outbox (
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

CREATE INDEX IF NOT EXISTS notification_outbox_due_idx
    ON notification_outbox (status, next_attempt_at, id);
CREATE INDEX IF NOT EXISTS notification_outbox_chat_idx
    ON notification_outbox (chat_id, status);

CREATE TABLE IF NOT EXISTS monitor_snapshots (
    torrent_key TEXT PRIMARY KEY,
    completed INTEGER NOT NULL CHECK (completed IN (0, 1)),
    started INTEGER NOT NULL CHECK (started IN (0, 1)),
    present INTEGER NOT NULL CHECK (present IN (0, 1)),
    generation INTEGER NOT NULL CHECK (generation >= 0),
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS monitor_snapshots_generation_idx
    ON monitor_snapshots (generation);

CREATE TABLE IF NOT EXISTS start_watches (
    torrent_key TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    torrent_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    added_at REAL NOT NULL,
    start_notified INTEGER NOT NULL DEFAULT 0 CHECK (start_notified IN (0, 1)),
    no_peers_notified INTEGER NOT NULL DEFAULT 0 CHECK (no_peers_notified IN (0, 1)),
    completion_notified INTEGER NOT NULL DEFAULT 0 CHECK (completion_notified IN (0, 1)),
    PRIMARY KEY (torrent_key, chat_id)
);

CREATE INDEX IF NOT EXISTS start_watches_chat_idx
    ON start_watches (chat_id);
"""


class SQLiteStateStore:
    """Thread-safe, connection-per-operation persistent application state."""

    _TRAFFIC_MIGRATION_KEY = "legacy.traffic_anchors.v1"
    _HISTORY_MIGRATION_KEY = "legacy.torrent_history.v1"

    def __init__(
        self,
        db_path: Union[str, Path],
        *,
        legacy_traffic_path: Optional[Union[str, Path]] = None,
        legacy_torrent_history_path: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.legacy_traffic_path = (
            Path(legacy_traffic_path).expanduser()
            if legacy_traffic_path is not None
            else self.db_path.with_name("traffic_anchors.json")
        )
        self.legacy_torrent_history_path = (
            Path(legacy_torrent_history_path).expanduser()
            if legacy_torrent_history_path is not None
            else self.db_path.with_name("torrent_history.json")
        )
        if busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms must be positive")
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.log = logger or logging.getLogger(__name__)
        self._initialize_lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> None:
        """Create the schema and migrate adjacent legacy JSON files once."""

        with self._initialize_lock:
            if self._initialized:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                self._migrate_outbox_claim_status(connection)
                connection.executescript("BEGIN IMMEDIATE;\n{}\nPRAGMA user_version=2;\nCOMMIT;".format(_SCHEMA))
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()

            # State may contain chat identifiers, torrent names and delivery
            # errors. Keep it private even outside the hardened systemd unit.
            self.db_path.chmod(0o600)

            self._migrate_legacy_traffic()
            self._migrate_legacy_torrent_history()
            self._initialized = True

    def load_traffic_state(self) -> Tuple[TrafficAnchors, TrafficHistory]:
        """Return ``(anchors, history)`` in the legacy bot-compatible shape."""

        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT anchors_json, history_json FROM traffic_state WHERE singleton = 1"
            ).fetchone()
        if row is None:
            return {}, []
        try:
            anchors_raw = json.loads(str(row["anchors_json"]))
            history_raw = json.loads(str(row["history_json"]))
            return self._normalize_traffic_state(anchors_raw, history_raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise StateStoreError("SQLite traffic state is invalid") from exc

    def save_traffic_state(
        self,
        anchors: Mapping[str, Mapping[str, TrafficValue]],
        history: Iterable[Mapping[str, TrafficValue]],
    ) -> None:
        """Atomically replace traffic anchors and daily history."""

        self._require_initialized()
        normalized_anchors, normalized_history = self._normalize_traffic_state(dict(anchors), list(history))
        with self._connection(write=True) as connection:
            self._save_traffic_state(connection, normalized_anchors, normalized_history)

    def load_torrent_history(self) -> TorrentHistory:
        """Load torrent history as a dictionary keyed by stable torrent key."""

        self._require_initialized()
        items: TorrentHistory = {}
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT torrent_key, payload_json FROM torrent_history ORDER BY torrent_key"
            ).fetchall()
        for row in rows:
            key = str(row["torrent_key"])
            try:
                payload = json.loads(str(row["payload_json"]))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise StateStoreError("SQLite torrent history entry {!r} is invalid".format(key)) from exc
            if not isinstance(payload, dict):
                raise StateStoreError("SQLite torrent history entry {!r} is not an object".format(key))
            items[key] = payload
        return items

    def save_torrent_history(self, items: Mapping[str, Mapping[str, Any]]) -> None:
        """Atomically replace all torrent-history entries."""

        self._require_initialized()
        normalized = self._normalize_torrent_history(items)
        updated_at = time.time()
        encoded = [(key, self._dump_json(payload), updated_at) for key, payload in sorted(normalized.items())]
        with self._connection(write=True) as connection:
            connection.execute("DELETE FROM torrent_history")
            connection.executemany(
                "INSERT INTO torrent_history (torrent_key, payload_json, updated_at) VALUES (?, ?, ?)",
                encoded,
            )

    def ensure_chat(self, chat_id: int, default_enabled: bool = True) -> bool:
        """Persist a chat preference if unknown and return its effective value."""

        self._require_initialized()
        now = time.time()
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO notification_chats
                    (chat_id, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (int(chat_id), int(bool(default_enabled)), now, now),
            )
            row = connection.execute(
                "SELECT enabled FROM notification_chats WHERE chat_id = ?", (int(chat_id),)
            ).fetchone()
        if row is None:  # Defensive: the transaction above must have created it.
            raise StateStoreError("Failed to initialize notification chat")
        return bool(row["enabled"])

    def notifications_enabled(self, chat_id: int) -> bool:
        """Return the persisted preference, creating an enabled-by-default row."""

        return self.ensure_chat(chat_id, default_enabled=True)

    def set_notification_enabled(self, chat_id: int, enabled: bool) -> bool:
        """Set and return a chat's notification preference."""

        self._require_initialized()
        now = time.time()
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO notification_chats
                    (chat_id, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (int(chat_id), int(bool(enabled)), now, now),
            )
            if not enabled:
                self._cancel_pending_for_chat(connection, int(chat_id), now)
                self._cancel_start_watches_for_chat(connection, int(chat_id))
        return bool(enabled)

    def toggle_chat(self, chat_id: int) -> bool:
        """Toggle and return a chat's preference in one transaction."""

        self._require_initialized()
        chat_id = int(chat_id)
        now = time.time()
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO notification_chats
                    (chat_id, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, 1, now, now),
            )
            row = connection.execute("SELECT enabled FROM notification_chats WHERE chat_id = ?", (chat_id,)).fetchone()
            if row is None:
                raise StateStoreError("Failed to load notification chat")
            enabled = not bool(row["enabled"])
            connection.execute(
                "UPDATE notification_chats SET enabled = ?, updated_at = ? WHERE chat_id = ?",
                (int(enabled), now, chat_id),
            )
            if not enabled:
                self._cancel_pending_for_chat(connection, chat_id, now)
                self._cancel_start_watches_for_chat(connection, chat_id)
        return enabled

    def list_enabled_chats(self) -> Set[int]:
        """Return all chats whose notifications are enabled."""

        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT chat_id FROM notification_chats WHERE enabled = 1 ORDER BY chat_id"
            ).fetchall()
        return {int(row["chat_id"]) for row in rows}

    def enqueue_outbox(
        self,
        event_key: str,
        chat_id: int,
        kind: str,
        text: str,
    ) -> bool:
        """Enqueue once per ``(event_key, chat_id)``; return whether inserted."""

        self._require_initialized()
        event_key = self._non_empty(event_key, "event_key")
        kind = self._non_empty(kind, "kind")
        text = str(text)
        if not text:
            raise ValueError("text must not be empty")
        now = time.time()
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO notification_outbox
                    (event_key, chat_id, kind, text, status, attempts,
                     next_attempt_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (event_key, int(chat_id), kind, text, now, now, now),
            )
            return cursor.rowcount == 1

    def list_due_outbox(self, now_ts: Optional[float] = None, limit: int = 100) -> List[OutboxItem]:
        """List pending items whose retry/delivery time has arrived."""

        self._require_initialized()
        if limit < 1:
            raise ValueError("limit must be positive")
        due_at = time.time() if now_ts is None else float(now_ts)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, event_key, chat_id, kind, text, status,
                       attempts, next_attempt_at, created_at, updated_at,
                       delivered_at, last_error
                  FROM notification_outbox
                 WHERE status = 'pending' AND next_attempt_at <= ?
                 ORDER BY next_attempt_at, id
                 LIMIT ?
                """,
                (due_at, int(limit)),
            ).fetchall()
        return [self._outbox_item_from_row(row) for row in rows]

    def claim_due_outbox(
        self,
        now_ts: Optional[float] = None,
        limit: int = 100,
        *,
        lease_seconds: float = 120.0,
    ) -> List[OutboxItem]:
        """Atomically lease due items so parallel drainers cannot send duplicates."""

        self._require_initialized()
        if limit < 1:
            raise ValueError("limit must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        claimed_at = time.time() if now_ts is None else float(now_ts)
        lease_until = claimed_at + float(lease_seconds)
        with self._connection(write=True) as connection:
            connection.execute(
                """
                UPDATE notification_outbox
                   SET status = 'pending', updated_at = ?
                 WHERE status = 'sending' AND next_attempt_at <= ?
                """,
                (claimed_at, claimed_at),
            )
            rows = connection.execute(
                """
                SELECT id
                  FROM notification_outbox
                 WHERE status = 'pending' AND next_attempt_at <= ?
                 ORDER BY next_attempt_at, id
                 LIMIT ?
                """,
                (claimed_at, int(limit)),
            ).fetchall()
            item_ids = [int(row["id"]) for row in rows]
            if not item_ids:
                return []
            placeholders = ",".join("?" for _ in item_ids)
            connection.execute(
                """
                UPDATE notification_outbox
                   SET status = 'sending', next_attempt_at = ?, updated_at = ?
                 WHERE id IN ({}) AND status = 'pending'
                """.format(placeholders),
                [lease_until, claimed_at, *item_ids],
            )
            claimed_rows = connection.execute(
                """
                SELECT id, event_key, chat_id, kind, text, status,
                       attempts, next_attempt_at, created_at, updated_at,
                       delivered_at, last_error
                  FROM notification_outbox
                 WHERE id IN ({}) AND status = 'sending'
                 ORDER BY id
                """.format(placeholders),
                item_ids,
            ).fetchall()
        return [self._outbox_item_from_row(row) for row in claimed_rows]

    def mark_outbox_delivered(self, item_id: int, *, delivered_at: Optional[float] = None) -> bool:
        """Mark a pending item delivered; return whether it changed state."""

        self._require_initialized()
        delivered = time.time() if delivered_at is None else float(delivered_at)
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE notification_outbox
                   SET status = 'delivered', delivered_at = ?, updated_at = ?, last_error = NULL
                 WHERE id = ? AND status IN ('pending', 'sending')
                """,
                (delivered, delivered, int(item_id)),
            )
            return cursor.rowcount == 1

    def mark_outbox_failed(
        self,
        item_id: int,
        attempts: int,
        next_attempt_at: float,
        *,
        error: Optional[str] = None,
    ) -> bool:
        """Record a failed attempt and schedule the next delivery attempt."""

        self._require_initialized()
        attempts = int(attempts)
        if attempts < 1:
            raise ValueError("attempts must be positive")
        now = time.time()
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE notification_outbox
                   SET status = 'pending',
                       attempts = MAX(attempts, ?),
                       next_attempt_at = ?,
                       updated_at = ?,
                       last_error = ?
                 WHERE id = ? AND status IN ('pending', 'sending')
                """,
                (
                    attempts,
                    float(next_attempt_at),
                    now,
                    None if error is None else str(error)[:4096],
                    int(item_id),
                ),
            )
            return cursor.rowcount == 1

    def cancel_pending_outbox(self, chat_id: int) -> int:
        """Cancel all queued deliveries for a chat and return the affected count."""

        self._require_initialized()
        with self._connection(write=True) as connection:
            return self._cancel_pending_for_chat(connection, int(chat_id), time.time())

    def load_monitor_snapshots(self) -> Dict[str, Snapshot]:
        """Return all last-known torrent monitor states keyed by stable key."""

        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT torrent_key, completed, started, present, generation, updated_at
                  FROM monitor_snapshots
                 ORDER BY torrent_key
                """
            ).fetchall()
        snapshots = [self._snapshot_from_row(row) for row in rows]
        return {snapshot.torrent_key: snapshot for snapshot in snapshots}

    def get_monitor_snapshot(self, torrent_key: str) -> Optional[Snapshot]:
        """Return one last-known torrent monitor state, if present."""

        self._require_initialized()
        torrent_key = self._non_empty(torrent_key, "torrent_key")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT torrent_key, completed, started, present, generation, updated_at
                  FROM monitor_snapshots
                 WHERE torrent_key = ?
                """,
                (torrent_key,),
            ).fetchone()
        return None if row is None else self._snapshot_from_row(row)

    def save_monitor_snapshot(
        self,
        torrent_key: str,
        *,
        completed: bool,
        started: bool,
        present: bool,
        generation: int,
        updated_at: Optional[float] = None,
    ) -> Snapshot:
        """Insert or replace one last-known torrent state."""

        self._require_initialized()
        snapshot = Snapshot(
            torrent_key=self._non_empty(torrent_key, "torrent_key"),
            completed=bool(completed),
            started=bool(started),
            present=bool(present),
            generation=self._generation(generation),
            updated_at=time.time() if updated_at is None else float(updated_at),
        )
        with self._connection(write=True) as connection:
            self._upsert_snapshot(connection, snapshot)
        return snapshot

    def replace_monitor_snapshots(self, snapshots: Iterable[Snapshot]) -> None:
        """Atomically replace the complete monitor snapshot set."""

        self._require_initialized()
        normalized = [self._normalize_snapshot(snapshot) for snapshot in snapshots]
        with self._connection(write=True) as connection:
            connection.execute("DELETE FROM monitor_snapshots")
            for snapshot in normalized:
                self._upsert_snapshot(connection, snapshot)

    def get_snapshots(self) -> Dict[str, Snapshot]:
        """Compatibility name for loading the complete snapshot map."""

        return self.load_monitor_snapshots()

    def save_snapshots(
        self,
        snapshots: Union[Mapping[str, Snapshot], Iterable[Snapshot]],
    ) -> None:
        """Atomically save the complete snapshot set."""

        values = snapshots.values() if isinstance(snapshots, Mapping) else snapshots
        self.replace_monitor_snapshots(values)

    def delete_monitor_snapshot(self, torrent_key: str) -> bool:
        """Delete one monitor snapshot."""

        self._require_initialized()
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM monitor_snapshots WHERE torrent_key = ?",
                (self._non_empty(torrent_key, "torrent_key"),),
            )
            return cursor.rowcount == 1

    def add_start_watch(
        self,
        torrent_key: str,
        chat_id: int,
        torrent_id: int,
        name: str,
        *,
        added_at: Optional[float] = None,
    ) -> StartWatch:
        """Idempotently add a per-torrent, per-chat start watch."""

        self._require_initialized()
        torrent_key = self._non_empty(torrent_key, "torrent_key")
        torrent_id = int(torrent_id)
        if torrent_id <= 0:
            raise ValueError("torrent_id must be positive")
        watch = StartWatch(
            torrent_key=torrent_key,
            chat_id=int(chat_id),
            torrent_id=torrent_id,
            name=str(name) or "<без названия>",
            added_at=time.time() if added_at is None else float(added_at),
            start_notified=False,
            no_peers_notified=False,
            completion_notified=False,
        )
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO start_watches
                    (torrent_key, chat_id, torrent_id, name, added_at,
                     start_notified, no_peers_notified, completion_notified)
                VALUES (?, ?, ?, ?, ?, 0, 0, 0)
                ON CONFLICT(torrent_key, chat_id) DO UPDATE SET
                    torrent_id = excluded.torrent_id,
                    name = excluded.name,
                    added_at = MIN(start_watches.added_at, excluded.added_at)
                """,
                (watch.torrent_key, watch.chat_id, watch.torrent_id, watch.name, watch.added_at),
            )
            row = connection.execute(
                """
                SELECT torrent_key, chat_id, torrent_id, name, added_at,
                       start_notified, no_peers_notified, completion_notified
                  FROM start_watches
                 WHERE torrent_key = ? AND chat_id = ?
                """,
                (watch.torrent_key, watch.chat_id),
            ).fetchone()
        if row is None:
            raise StateStoreError("Failed to persist start watch")
        return self._start_watch_from_row(row)

    def list_start_watches(
        self,
        *,
        torrent_key: Optional[str] = None,
        chat_id: Optional[int] = None,
    ) -> List[StartWatch]:
        """List watches, optionally filtered by torrent and/or chat."""

        self._require_initialized()
        normalized_key = None if torrent_key is None else self._non_empty(torrent_key, "torrent_key")
        normalized_chat_id = None if chat_id is None else int(chat_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT torrent_key, chat_id, torrent_id, name, added_at,
                       start_notified, no_peers_notified, completion_notified
                  FROM start_watches
                 WHERE (? IS NULL OR torrent_key = ?)
                   AND (? IS NULL OR chat_id = ?)
                 ORDER BY added_at, torrent_key, chat_id
                """,
                (normalized_key, normalized_key, normalized_chat_id, normalized_chat_id),
            ).fetchall()
        return [self._start_watch_from_row(row) for row in rows]

    def update_start_watch(
        self,
        torrent_key: str,
        chat_id: int,
        *,
        start_notified: Optional[bool] = None,
        no_peers_notified: Optional[bool] = None,
        completion_notified: Optional[bool] = None,
    ) -> Optional[StartWatch]:
        """Update lifecycle flags and return the watch, or ``None`` if missing."""

        self._require_initialized()
        torrent_key = self._non_empty(torrent_key, "torrent_key")
        start_value = None if start_notified is None else int(bool(start_notified))
        no_peers_value = None if no_peers_notified is None else int(bool(no_peers_notified))
        completion_value = None if completion_notified is None else int(bool(completion_notified))
        with self._connection(write=True) as connection:
            connection.execute(
                """
                UPDATE start_watches
                   SET start_notified = COALESCE(?, start_notified),
                       no_peers_notified = COALESCE(?, no_peers_notified),
                       completion_notified = COALESCE(?, completion_notified)
                 WHERE torrent_key = ? AND chat_id = ?
                """,
                (start_value, no_peers_value, completion_value, torrent_key, int(chat_id)),
            )
            row = connection.execute(
                """
                SELECT torrent_key, chat_id, torrent_id, name, added_at,
                       start_notified, no_peers_notified, completion_notified
                  FROM start_watches
                 WHERE torrent_key = ? AND chat_id = ?
                """,
                (torrent_key, int(chat_id)),
            ).fetchone()
        return None if row is None else self._start_watch_from_row(row)

    def delete_start_watch(self, torrent_key: str, chat_id: int) -> bool:
        """Delete one watch after its completion notification was enqueued."""

        self._require_initialized()
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM start_watches WHERE torrent_key = ? AND chat_id = ?",
                (self._non_empty(torrent_key, "torrent_key"), int(chat_id)),
            )
            return cursor.rowcount == 1

    remove_start_watch = delete_start_watch

    def remove_start_watches(self, torrent_key: str) -> int:
        """Remove all watches for a torrent and return the affected count."""

        self._require_initialized()
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM start_watches WHERE torrent_key = ?",
                (self._non_empty(torrent_key, "torrent_key"),),
            )
            return max(0, cursor.rowcount)

    def cancel_start_watches_for_chat(self, chat_id: int) -> int:
        """Remove all start watches belonging to a chat."""

        self._require_initialized()
        with self._connection(write=True) as connection:
            return self._cancel_start_watches_for_chat(connection, int(chat_id))

    # Compatibility aliases keep call sites concise while the JSON-backed bot is migrated.
    read_traffic_state = load_traffic_state
    persist_traffic_state = save_traffic_state
    load_torrent_history_state = load_torrent_history
    persist_torrent_history_state = save_torrent_history
    ensure_notification_chat = ensure_chat
    get_notification_enabled = notifications_enabled
    toggle_notification = toggle_chat
    toggle_notifications = toggle_chat
    list_enabled_chat_ids = list_enabled_chats
    cancel_pending_for_chat = cancel_pending_outbox

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout={}".format(self.busy_timeout_ms))
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @staticmethod
    def _migrate_outbox_claim_status(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notification_outbox'"
        ).fetchone()
        if row is None or "'sending'" in str(row["sql"]):
            return

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

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("SQLiteStateStore.initialize() must be called first")

    def _migrate_legacy_traffic(self) -> None:
        status = self._metadata_value(self._TRAFFIC_MIGRATION_KEY)
        if status is not None:
            self._finish_imported_file(status, self._TRAFFIC_MIGRATION_KEY, self.legacy_traffic_path)
            return
        path = self.legacy_traffic_path
        if not path.exists():
            self._set_metadata(self._TRAFFIC_MIGRATION_KEY, "absent")
            return

        try:
            raw = self._load_json_file(path)
            if not isinstance(raw, dict):
                raise ValueError("traffic state root must be an object")
            anchors_raw = {key: value for key, value in raw.items() if key != "history"}
            history_container = raw.get("history", {})
            if not isinstance(history_container, dict):
                raise ValueError("traffic history must be an object")
            history_raw = history_container.get("days", [])
            anchors, history = self._normalize_traffic_state(anchors_raw, history_raw)
        except FileNotFoundError:
            self._set_metadata(self._TRAFFIC_MIGRATION_KEY, "absent")
            return
        except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
            self._quarantine_corrupt(path, self._TRAFFIC_MIGRATION_KEY, exc)
            return

        with self._connection(write=True) as connection:
            self._save_traffic_state(connection, anchors, history)
            self._set_metadata_in_connection(connection, self._TRAFFIC_MIGRATION_KEY, "imported")
        try:
            migrated_path = self._move_legacy_file(path, "migrated")
            final_status = "done:{}".format(migrated_path.name)
        except FileNotFoundError:
            final_status = "done:file-already-moved"
        self._set_metadata(self._TRAFFIC_MIGRATION_KEY, final_status)
        self.log.info("Migrated legacy traffic state from %s to SQLite", path)

    def _migrate_legacy_torrent_history(self) -> None:
        status = self._metadata_value(self._HISTORY_MIGRATION_KEY)
        if status is not None:
            self._finish_imported_file(status, self._HISTORY_MIGRATION_KEY, self.legacy_torrent_history_path)
            return
        path = self.legacy_torrent_history_path
        if not path.exists():
            self._set_metadata(self._HISTORY_MIGRATION_KEY, "absent")
            return

        try:
            raw = self._load_json_file(path)
            if not isinstance(raw, dict):
                raise ValueError("torrent history root must be an object")
            raw_items = raw.get("items")
            if not isinstance(raw_items, dict):
                raise ValueError("torrent history items must be an object")
            items = self._normalize_torrent_history(raw_items)
        except FileNotFoundError:
            self._set_metadata(self._HISTORY_MIGRATION_KEY, "absent")
            return
        except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
            self._quarantine_corrupt(path, self._HISTORY_MIGRATION_KEY, exc)
            return

        updated_at = time.time()
        encoded = [(key, self._dump_json(payload), updated_at) for key, payload in sorted(items.items())]
        with self._connection(write=True) as connection:
            connection.execute("DELETE FROM torrent_history")
            connection.executemany(
                "INSERT INTO torrent_history (torrent_key, payload_json, updated_at) VALUES (?, ?, ?)",
                encoded,
            )
            self._set_metadata_in_connection(connection, self._HISTORY_MIGRATION_KEY, "imported")
        try:
            migrated_path = self._move_legacy_file(path, "migrated")
            final_status = "done:{}".format(migrated_path.name)
        except FileNotFoundError:
            final_status = "done:file-already-moved"
        self._set_metadata(self._HISTORY_MIGRATION_KEY, final_status)
        self.log.info("Migrated legacy torrent history from %s to SQLite", path)

    def _finish_imported_file(self, status: str, key: str, path: Path) -> None:
        if status != "imported":
            return
        if path.exists():
            try:
                migrated_path = self._move_legacy_file(path, "migrated")
                final_status = "done:{}".format(migrated_path.name)
            except FileNotFoundError:
                final_status = "done:file-already-moved"
        else:
            final_status = "done:file-already-moved"
        self._set_metadata(key, final_status)

    def _quarantine_corrupt(self, path: Path, key: str, exc: BaseException) -> None:
        try:
            corrupt_path = self._move_legacy_file(path, "corrupt-{}".format(self._utc_timestamp()))
        except FileNotFoundError:
            self.log.warning("Legacy state file %s is invalid and was already moved: %s", path, exc)
            self._set_metadata(key, "corrupt:file-already-moved")
            return
        self.log.warning(
            "Legacy state file %s is invalid and was preserved as %s: %s",
            path,
            corrupt_path,
            exc,
        )
        self._set_metadata(key, "corrupt:{}".format(corrupt_path.name))

    def _move_legacy_file(self, path: Path, suffix: str) -> Path:
        target = path.with_name("{}.{}".format(path.name, suffix))
        counter = 1
        while target.exists():
            target = path.with_name("{}.{}.{}".format(path.name, counter, suffix))
            counter += 1
        path.rename(target)
        return target

    @staticmethod
    def _load_json_file(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _metadata_value(self, key: str) -> Optional[str]:
        with self._connection() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_metadata(self, key: str, value: str) -> None:
        with self._connection(write=True) as connection:
            self._set_metadata_in_connection(connection, key, value)

    @staticmethod
    def _set_metadata_in_connection(connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            """
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _save_traffic_state(
        self,
        connection: sqlite3.Connection,
        anchors: TrafficAnchors,
        history: TrafficHistory,
    ) -> None:
        connection.execute(
            """
            INSERT INTO traffic_state
                (singleton, anchors_json, history_json, updated_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                anchors_json = excluded.anchors_json,
                history_json = excluded.history_json,
                updated_at = excluded.updated_at
            """,
            (self._dump_json(anchors), self._dump_json(history), time.time()),
        )

    @classmethod
    def _normalize_traffic_state(cls, anchors_raw: Any, history_raw: Any) -> Tuple[TrafficAnchors, TrafficHistory]:
        if not isinstance(anchors_raw, dict):
            raise ValueError("traffic anchors must be an object")
        if not isinstance(history_raw, list):
            raise ValueError("traffic history must be a list")

        anchors = cls._normalize_traffic_anchors(anchors_raw)
        history = cls._normalize_traffic_history(history_raw)
        return anchors, history

    @classmethod
    def _normalize_traffic_anchors(cls, anchors_raw: Mapping[Any, Any]) -> TrafficAnchors:
        anchors: TrafficAnchors = {}
        for anchor_name, row in anchors_raw.items():
            if not isinstance(anchor_name, str) or not anchor_name:
                raise ValueError("traffic anchor names must be non-empty strings")
            anchors[anchor_name] = cls._normalize_traffic_anchor_row(anchor_name, row)
        return anchors

    @classmethod
    def _normalize_traffic_anchor_row(cls, anchor_name: str, row: Any) -> Dict[str, TrafficValue]:
        if not isinstance(row, Mapping):
            raise ValueError("traffic anchor {!r} must be an object".format(anchor_name))

        normalized: Dict[str, TrafficValue] = {}
        for field, value in row.items():
            if not isinstance(field, str) or not field:
                raise ValueError("traffic anchor fields must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError("traffic anchor {}.{} must be a string or integer".format(anchor_name, field))
            normalized[field] = max(0, value) if isinstance(value, int) else value

        if anchor_name in {"day", "week", "month"}:
            key = normalized.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("traffic anchor {!r} has an invalid key".format(anchor_name))
            cls._counter(normalized.get("downloaded"), "{}.downloaded".format(anchor_name))
            cls._counter(normalized.get("uploaded"), "{}.uploaded".format(anchor_name))
        elif anchor_name == "_counter":
            cls._validate_logical_counter(normalized)
        return normalized

    @classmethod
    def _validate_logical_counter(cls, row: Mapping[str, TrafficValue]) -> None:
        for field in (
            "last_downloaded",
            "last_uploaded",
            "logical_downloaded",
            "logical_uploaded",
        ):
            cls._counter(row.get(field), "_counter.{}".format(field))

    @classmethod
    def _normalize_traffic_history(cls, history_raw: Iterable[Any]) -> TrafficHistory:
        history: TrafficHistory = []
        for index, raw_item in enumerate(history_raw):
            if not isinstance(raw_item, Mapping):
                raise ValueError("traffic history item {} must be an object".format(index))
            date = raw_item.get("date")
            if not isinstance(date, str) or not date:
                raise ValueError("traffic history item {} has an invalid date".format(index))
            downloaded = cls._counter(raw_item.get("downloaded"), "history.downloaded")
            uploaded = cls._counter(raw_item.get("uploaded"), "history.uploaded")
            history.append({"date": date, "downloaded": downloaded, "uploaded": uploaded})
        return history

    @staticmethod
    def _normalize_torrent_history(items: Mapping[str, Mapping[str, Any]]) -> TorrentHistory:
        if not isinstance(items, Mapping):
            raise ValueError("torrent history must be an object")
        normalized: TorrentHistory = {}
        for key, payload in items.items():
            if not isinstance(key, str) or not key:
                raise ValueError("torrent history keys must be non-empty strings")
            if not isinstance(payload, Mapping):
                raise ValueError("torrent history entry {!r} must be an object".format(key))
            normalized[key] = dict(payload)
        return normalized

    @staticmethod
    def _counter(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("{} must be an integer".format(field))
        return max(0, value)

    @staticmethod
    def _non_empty(value: str, field: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("{} must not be empty".format(field))
        return normalized

    @staticmethod
    def _generation(value: int) -> int:
        generation = int(value)
        if generation < 0:
            raise ValueError("generation must not be negative")
        return generation

    @staticmethod
    def _dump_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    @classmethod
    def _outbox_item_from_row(cls, row: sqlite3.Row) -> OutboxItem:
        return OutboxItem(
            id=int(row["id"]),
            event_key=str(row["event_key"]),
            chat_id=int(row["chat_id"]),
            kind=str(row["kind"]),
            text=str(row["text"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            next_attempt_at=float(row["next_attempt_at"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            delivered_at=None if row["delivered_at"] is None else float(row["delivered_at"]),
            last_error=None if row["last_error"] is None else str(row["last_error"]),
        )

    @staticmethod
    def _snapshot_from_row(row: sqlite3.Row) -> Snapshot:
        return Snapshot(
            torrent_key=str(row["torrent_key"]),
            completed=bool(row["completed"]),
            started=bool(row["started"]),
            present=bool(row["present"]),
            generation=int(row["generation"]),
            updated_at=float(row["updated_at"]),
        )

    @classmethod
    def _normalize_snapshot(cls, snapshot: Snapshot) -> Snapshot:
        if not isinstance(snapshot, Snapshot):
            raise TypeError("snapshots must contain Snapshot instances")
        return Snapshot(
            torrent_key=cls._non_empty(snapshot.torrent_key, "torrent_key"),
            completed=bool(snapshot.completed),
            started=bool(snapshot.started),
            present=bool(snapshot.present),
            generation=cls._generation(snapshot.generation),
            updated_at=float(snapshot.updated_at),
        )

    @staticmethod
    def _upsert_snapshot(connection: sqlite3.Connection, snapshot: Snapshot) -> None:
        connection.execute(
            """
            INSERT INTO monitor_snapshots
                (torrent_key, completed, started, present, generation, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(torrent_key) DO UPDATE SET
                completed = excluded.completed,
                started = excluded.started,
                present = excluded.present,
                generation = excluded.generation,
                updated_at = excluded.updated_at
            """,
            (
                snapshot.torrent_key,
                int(snapshot.completed),
                int(snapshot.started),
                int(snapshot.present),
                snapshot.generation,
                snapshot.updated_at,
            ),
        )

    @staticmethod
    def _start_watch_from_row(row: sqlite3.Row) -> StartWatch:
        return StartWatch(
            torrent_key=str(row["torrent_key"]),
            chat_id=int(row["chat_id"]),
            torrent_id=int(row["torrent_id"]),
            name=str(row["name"]),
            added_at=float(row["added_at"]),
            start_notified=bool(row["start_notified"]),
            no_peers_notified=bool(row["no_peers_notified"]),
            completion_notified=bool(row["completion_notified"]),
        )

    @staticmethod
    def _cancel_pending_for_chat(connection: sqlite3.Connection, chat_id: int, now: float) -> int:
        cursor = connection.execute(
            """
            UPDATE notification_outbox
               SET status = 'cancelled', updated_at = ?
             WHERE chat_id = ? AND status IN ('pending', 'sending')
            """,
            (now, chat_id),
        )
        return max(0, cursor.rowcount)

    @staticmethod
    def _cancel_start_watches_for_chat(connection: sqlite3.Connection, chat_id: int) -> int:
        cursor = connection.execute("DELETE FROM start_watches WHERE chat_id = ?", (chat_id,))
        return max(0, cursor.rowcount)


__all__ = [
    "OutboxItem",
    "SQLiteStateStore",
    "Snapshot",
    "StartWatch",
    "StateStoreError",
    "TorrentHistory",
    "TrafficAnchors",
    "TrafficHistory",
    "TrafficValue",
]
