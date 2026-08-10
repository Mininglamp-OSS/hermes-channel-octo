"""Durable polling and trusted dispatch for Octo ``card_action`` events."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


from gateway.platforms.base import MessageEvent, MessageType as HermesMessageType
from gateway.session import build_session_key

from . import api, cards
from .cards import CardRenderResult
from .types import ChannelType

MAX_SAFE_EVENT_ID = (1 << 53) - 1
DEFAULT_EVENT_INTERVAL_SECONDS = 2.0
DEFAULT_EVENT_WAIT_SECONDS = 25
MAX_EVENT_WAIT_SECONDS = 30
MIN_EVENT_WAIT_SECONDS = 5
MAX_EVENT_BACKOFF_SECONDS = 30.0
_HELD_FRACTION = 0.5
_CARD_SESSION_TTL_SECONDS = 24 * 60 * 60
_MAX_CARD_SESSIONS = 1000
_CARD_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_MAX_EVENT_INPUT_FIELDS = 128
_MAX_EVENT_DATA_FIELDS = 128
_MAX_EVENT_INPUT_VALUE_BYTES = 16 << 10
_MAX_EVENT_ENVELOPE_BYTES = 64 << 10
_ACK_ATTEMPTS = 3
_ACK_RETRY_SECONDS = 0.1

logger = logging.getLogger(__name__)
_OWNER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class EventCursorStore(Protocol):
    async def load(self) -> int: ...

    async def save(self, event_id: int) -> None: ...


def _safe_event_id(value: object) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_SAFE_EVENT_ID
    ):
        return None
    return value


class FileEventCursorStore:
    """One monotonic event cursor persisted by fsync plus atomic replacement."""

    def __init__(self, *, owner_id: str, base_dir: Path | None = None) -> None:
        owner = owner_id
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("invalid Octo owner id")
        owner_path = (
            owner
            if _OWNER_ID_RE.fullmatch(owner)
            else f"robot-{hashlib.sha256(owner.encode('utf-8')).hexdigest()}"
        )
        if base_dir is None:
            from hermes_constants import get_hermes_home

            base_dir = Path(get_hermes_home()) / "workspace" / "octo"
        self.path = Path(base_dir) / owner_path / "events.cursor.json"
        self._lock = threading.Lock()

    async def load(self) -> int:
        return await asyncio.to_thread(self._load_sync)

    async def save(self, event_id: int) -> None:
        if _safe_event_id(event_id) is None:
            raise ValueError("invalid event cursor")
        await asyncio.to_thread(self._save_sync, event_id)

    def _load_sync(self) -> int:
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> int:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return 0
        if not isinstance(raw, dict):
            return 0
        event_id = _safe_event_id(raw.get("event_id"))
        return event_id if event_id is not None else 0

    def _save_sync(self, event_id: int) -> None:
        with self._lock:
            current = self._read_unlocked()
            if event_id < current:
                raise ValueError("event cursor cannot move backwards")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.parent / (
                f".events.cursor.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary.open("x", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"event_id": event_id},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                try:
                    directory_fd = os.open(self.path.parent, os.O_RDONLY)
                except OSError:
                    directory_fd = None
                if directory_fd is not None:
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


@dataclass(frozen=True)
class CardAction:
    event_id: int
    message_id: str
    channel_id: str
    channel_type: ChannelType
    action_id: str
    inputs: dict[str, str]
    operator_uid: str
    data: dict[str, object]


def _bounded_string(value: object, *, max_chars: int = 256) -> str:
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if (
        not candidate
        or len(candidate) > max_chars
        or any(character in candidate for character in "\x00\r\n")
    ):
        return ""
    return candidate


def _channel_type(value: object) -> ChannelType | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str) and value in {"1", "2", "5"}:
        candidate = int(value)
    else:
        return None
    if candidate not in {
        int(ChannelType.DM),
        int(ChannelType.Group),
        int(ChannelType.CommunityTopic),
    }:
        return None
    return ChannelType(candidate)


def parse_card_action(event: object) -> CardAction | None:
    """Parse only the server-authoritative card-action envelope shape."""
    if not isinstance(event, Mapping):
        return None
    event_id = _safe_event_id(event.get("event_id"))
    if event_id is None or event.get("event_type") != "card_action":
        return None
    raw = event.get("event_data")
    if not isinstance(raw, Mapping):
        return None
    message_id = _bounded_string(raw.get("message_id"))
    channel_id = _bounded_string(raw.get("channel_id"))
    channel_type = _channel_type(raw.get("channel_type"))
    action_id = _bounded_string(raw.get("action_id"), max_chars=64)
    operator_uid = _bounded_string(raw.get("operator_uid"), max_chars=128)
    if not all((message_id, channel_id, action_id, operator_uid)) or channel_type is None:
        return None

    inputs: dict[str, str] = {}
    raw_inputs = raw.get("inputs")
    if raw_inputs is not None:
        if not isinstance(raw_inputs, Mapping) or len(raw_inputs) > _MAX_EVENT_INPUT_FIELDS:
            return None
        for key, value in raw_inputs.items():
            if not isinstance(key, str) or not _CARD_ID_RE.fullmatch(key):
                return None
            if isinstance(value, str):
                normalized = value
            elif isinstance(value, bool):
                normalized = "true" if value else "false"
            elif isinstance(value, int):
                if abs(value) > MAX_SAFE_EVENT_ID:
                    return None
                normalized = str(value)
            elif isinstance(value, float):
                if not math.isfinite(value):
                    return None
                normalized = str(value)
            else:
                continue
            try:
                value_bytes = normalized.encode("utf-8")
            except UnicodeError:
                return None
            if len(value_bytes) > _MAX_EVENT_INPUT_VALUE_BYTES:
                return None
            inputs[key] = normalized

    raw_data = raw.get("data")
    if raw_data is None:
        data: dict[str, object] = {}
    elif isinstance(raw_data, Mapping) and len(raw_data) <= _MAX_EVENT_DATA_FIELDS:
        if not all(isinstance(key, str) and _CARD_ID_RE.fullmatch(key) for key in raw_data):
            return None
        data = dict(raw_data)
    else:
        return None
    try:
        envelope_size = len(
            json.dumps(
                {"inputs": inputs, "data": data},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, UnicodeError):
        return None
    if envelope_size > _MAX_EVENT_ENVELOPE_BYTES:
        return None
    return CardAction(
        event_id=event_id,
        message_id=message_id,
        channel_id=channel_id,
        channel_type=channel_type,
        action_id=action_id,
        inputs=inputs,
        operator_uid=operator_uid,
        data=data,
    )


@dataclass(frozen=True)
class CardSession:
    message_id: str
    binding_id: str
    session_key: str
    chat_id: str
    channel_id: str
    channel_type: ChannelType
    requester_uid: str
    card: dict[str, Any]
    plain: str
    action_labels: dict[str, str]
    input_ids: tuple[str, ...]
    max_input_text_bytes: int | None = None
    max_inputs_bytes: int | None = None


@dataclass
class _CardSessionEntry:
    session: CardSession
    expires_at: float
    state: str = "pending"
    claimed_event_id: int | None = None
    attempt_event_id: int | None = None
    dispatch_attempts: int = 0
    card_seq: int = 0



@dataclass(frozen=True)
class CardClaim:
    status: str
    session: CardSession | None = None
    attempts: int = 0


class CardSessionRegistry:
    """Bounded, thread-safe session claims for cards sent from tool workers."""

    def __init__(
        self,
        *,
        max_sessions: int = _MAX_CARD_SESSIONS,
        ttl_seconds: float = _CARD_SESSION_TTL_SECONDS,
        max_dispatch_attempts: int = 3,
    ) -> None:
        self._max_sessions = max(1, max_sessions)
        self._ttl_seconds = max(1.0, ttl_seconds)
        self.max_dispatch_attempts = max(1, max_dispatch_attempts)
        self._entries: OrderedDict[str, _CardSessionEntry] = OrderedDict()
        self._lock = threading.Lock()

    def register(self, session: CardSession) -> None:
        if not _bounded_string(session.message_id):
            return
        with self._lock:
            self._prune_locked()
            self._entries.pop(session.message_id, None)
            while len(self._entries) >= self._max_sessions:
                self._entries.popitem(last=False)
            self._entries[session.message_id] = _CardSessionEntry(
                session=session,
                expires_at=time.monotonic() + self._ttl_seconds,
            )

    def claim_edit(
        self,
        *,
        message_id: str,
        card_seq: int,
        session_key: str,
        channel_id: str,
        channel_type: ChannelType,
        requester_uid: str,
    ) -> bool:
        """Claim one terminal edit for the exact trusted interactive session."""
        if _safe_event_id(card_seq) is None or card_seq == 0:
            return False
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is None or entry.state != "pending":
                return False
            session = entry.session
            if (
                session.session_key != session_key
                or session.channel_id != channel_id
                or session.channel_type != channel_type
                or session.requester_uid != requester_uid
            ):
                return False
            entry.state = "processing"
            entry.claimed_event_id = -card_seq
            return True

    def claim(self, message_id: str, event_id: int) -> CardClaim:
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is None:
                return CardClaim("missing")
            if entry.state != "pending":
                return CardClaim("duplicate", entry.session)
            entry.state = "processing"
            entry.claimed_event_id = event_id
            if entry.attempt_event_id != event_id:
                entry.attempt_event_id = event_id
                entry.dispatch_attempts = 0
            entry.dispatch_attempts += 1
            return CardClaim("claimed", entry.session, entry.dispatch_attempts)

    def next_card_seq(self, message_id: str) -> int | None:
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is None:
                return None
            entry.card_seq += 1
            return entry.card_seq


    def release(self, message_id: str, event_id: int) -> None:
        with self._lock:
            entry = self._entry_locked(message_id)
            if (
                entry is not None
                and entry.state == "processing"
                and entry.claimed_event_id == event_id
            ):
                entry.state = "pending"
                entry.claimed_event_id = None

    def complete(self, message_id: str, event_id: int) -> None:
        with self._lock:
            entry = self._entry_locked(message_id)
            if (
                entry is not None
                and entry.state == "processing"
                and entry.claimed_event_id == event_id
            ):
                entry.state = "completed"

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _entry_locked(self, message_id: str) -> _CardSessionEntry | None:
        entry = self._entries.get(message_id)
        if entry is not None and entry.expires_at <= time.monotonic():
            self._entries.pop(message_id, None)
            return None
        return entry

    def _prune_locked(self) -> None:
        now = time.monotonic()
        for message_id in [
            message_id
            for message_id, entry in self._entries.items()
            if entry.expires_at <= now
        ]:
            self._entries.pop(message_id, None)


def _action_matches_session(action: CardAction, session: CardSession) -> bool:
    channel_matches = (
        action.channel_type == ChannelType.DM
        or action.channel_id == session.channel_id
    )
    if (
        action.message_id != session.message_id
        or not channel_matches
        or action.channel_type != session.channel_type
        or action.operator_uid != session.requester_uid
        or action.action_id not in session.action_labels
        or action.data.get("_octo_binding") != session.binding_id
    ):
        return False
    allowed_inputs = set(session.input_ids)
    max_text = session.max_input_text_bytes or 4096
    for key, value in action.inputs.items():
        if (
            not _CARD_ID_RE.fullmatch(key)
            or key not in allowed_inputs
            or cards.is_sensitive(key, generic=True)
            or cards.is_sensitive(value, generic=True)
            or len(value.encode("utf-8")) > max_text
        ):
            return False
    max_inputs = session.max_inputs_bytes or 16384
    serialized = json.dumps(
        action.inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(serialized) <= max_inputs

def _neutralize_action_echo(value: str) -> str:
    reduced = cards.reduce_urls_in_text(value)
    return re.sub(r"([\\`*_~\[\]<>])", r"\\\1", reduced)


def _freeze_action_node(
    node: Mapping[str, Any],
    inputs: Mapping[str, str],
) -> dict[str, Any] | None:
    node_type = node.get("type")
    if isinstance(node_type, str) and node_type.startswith("Input."):
        input_id = node.get("id")
        if not isinstance(input_id, str) or input_id not in inputs:
            return None
        label = node.get("label")
        safe_label = label.strip() if isinstance(label, str) and label.strip() else input_id
        return {
            "type": "TextBlock",
            "text": f"{safe_label}: {_neutralize_action_echo(inputs[input_id])}",
            "wrap": True,
            "spacing": "Small",
        }
    frozen: dict[str, Any] = {}
    for key, value in node.items():
        if key == "actions":
            continue
        if isinstance(value, list):
            children: list[Any] = []
            for item in value:
                if isinstance(item, Mapping):
                    child = _freeze_action_node(item, inputs)
                    if child is not None:
                        children.append(child)
                else:
                    children.append(item)
            frozen[key] = children
        else:
            frozen[key] = value
    return frozen


def render_card_action_status(
    session: CardSession,
    action: CardAction,
    status: str,
) -> CardRenderResult:
    """Freeze submitted controls and append a disclosure-safe terminal status."""
    source_body = session.card.get("body")
    body: list[dict[str, Any]] = []
    if isinstance(source_body, list):
        for item in source_body:
            if not isinstance(item, Mapping):
                continue
            frozen = _freeze_action_node(item, action.inputs)
            if frozen is not None:
                body.append(frozen)
    label = session.action_labels.get(action.action_id, action.action_id)
    operator = _neutralize_action_echo(action.operator_uid)
    status_line = (
        f"Processing {label} for {operator}"
        if status == "processing"
        else (
            f"Completed {label} for {operator}"
            if status == "completed"
            else f"Failed {label} for {operator}"
        )
    )
    body.append(
        {
            "type": "TextBlock",
            "text": status_line,
            "wrap": True,
            "spacing": "Medium",
            "separator": True,
        }
    )
    card = {key: value for key, value in session.card.items() if key != "actions"}
    card["body"] = body
    plain_base = session.plain.strip()
    plain = "\n".join(part for part in (plain_base, status_line) if part)
    return CardRenderResult(card=card, plain=plain)


async def _update_action_status(
    callback: Callable[..., Awaitable[None]] | None,
    session: CardSession,
    action: CardAction,
    status: str,
    *,
    transient: bool,
) -> None:
    if callback is None:
        return
    try:
        await callback(session, action, status, transient=transient)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Octo card action status update failed", exc_info=True)




async def handle_card_action(
    registry: CardSessionRegistry,
    action: CardAction,
    dispatch: Callable[[CardSession, CardAction], Awaitable[bool]],
    *,
    update_status: Callable[..., Awaitable[None]] | None = None,
) -> str:
    """Claim, validate, dispatch, and bound retries for one card action."""
    claim = registry.claim(action.message_id, action.event_id)
    if claim.session is None:
        return claim.status
    if claim.status != "claimed":
        try:
            return "duplicate" if _action_matches_session(action, claim.session) else "ignored"
        except (TypeError, ValueError, UnicodeError):
            return "ignored"
    session = claim.session
    try:
        matches = _action_matches_session(action, session)
    except (TypeError, ValueError, UnicodeError):
        registry.release(action.message_id, action.event_id)
        return "ignored"
    if not matches:
        registry.release(action.message_id, action.event_id)
        return "ignored"
    try:
        await _update_action_status(
            update_status,
            session,
            action,
            "processing",
            transient=True,
        )
        accepted = await dispatch(session, action)
    except asyncio.CancelledError:
        registry.release(action.message_id, action.event_id)
        raise
    except Exception:
        if claim.attempts >= registry.max_dispatch_attempts:
            registry.complete(action.message_id, action.event_id)
            await _update_action_status(
                update_status,
                session,
                action,
                "failed",
                transient=False,
            )
            return "dead_letter"
        registry.release(action.message_id, action.event_id)
        raise
    if not accepted:
        registry.release(action.message_id, action.event_id)
        return "ignored"
    registry.complete(action.message_id, action.event_id)
    await _update_action_status(
        update_status,
        session,
        action,
        "completed",
        transient=False,
    )
    return "completed"


def format_card_action_text(action: CardAction) -> str:
    action_id = json.dumps(action.action_id, ensure_ascii=False)
    inputs = json.dumps(
        action.inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    public_data = {
        key: value
        for key, value in action.data.items()
        if not key.startswith("_octo_")
    }
    data = json.dumps(
        public_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"[Octo card action]\naction_id={action_id}\n"
        f"inputs={inputs}\ndata={data}"
    )


async def dispatch_card_action_event(
    adapter: Any,
    session: CardSession,
    action: CardAction,
) -> bool:
    """Inject through the public platform message bridge after exact key verification."""
    if getattr(adapter, "_message_handler", None) is None:
        return False
    chat_type = "dm" if session.channel_type == ChannelType.DM else "group"
    source = adapter.build_source(
        chat_id=session.chat_id,
        chat_type=chat_type,
        user_id=session.requester_uid,
        user_name=session.requester_uid,
    )
    extra = getattr(adapter.config, "extra", None) or {}
    derived_session_key = build_session_key(
        source,
        group_sessions_per_user=extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
    )
    if derived_session_key != session.session_key:
        return False
    event = MessageEvent(
        text=format_card_action_text(action),
        message_type=HermesMessageType.TEXT,
        source=source,
        message_id=f"card_action:{action.event_id}",
    )
    await adapter.handle_message(event)
    return True


class EventPoller:
    """Single sequential event poller with durable progress and bounded pacing."""

    def __init__(
        self,
        *,
        session: Any,
        api_url: str,
        bot_token: str,
        cursor_store: EventCursorStore,
        on_card_action: Callable[[CardAction], Awaitable[str]],
        interval_seconds: float = DEFAULT_EVENT_INTERVAL_SECONDS,
        wait_seconds: int = DEFAULT_EVENT_WAIT_SECONDS,
        limit: int = 50,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._session = session
        self._api_url = api_url
        self._bot_token = bot_token
        self._cursor_store = cursor_store
        self._on_card_action = on_card_action
        self._interval_seconds = max(0.5, float(interval_seconds))
        self._wait_seconds = (
            min(MAX_EVENT_WAIT_SECONDS, max(MIN_EVENT_WAIT_SECONDS, int(wait_seconds)))
            if wait_seconds > 0
            else 0
        )
        self._limit = max(1, min(100, int(limit)))
        self._clock = clock
        self._cursor = 0
        self._consecutive_errors = 0
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def cursor(self) -> int:
        return self._cursor

    async def initialize(self) -> None:
        try:
            loaded = await self._cursor_store.load()
        except Exception:
            loaded = 0
        self._cursor = loaded if _safe_event_id(loaded) is not None else 0

    async def _ack(self, event_id: int) -> None:
        for attempt in range(_ACK_ATTEMPTS):
            try:
                await api.ack_bot_event(
                    self._session,
                    self._api_url,
                    self._bot_token,
                    event_id=event_id,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt + 1 >= _ACK_ATTEMPTS:
                    logger.warning(
                        "Octo event ack failed after %d attempts (%s)",
                        _ACK_ATTEMPTS,
                        type(exc).__name__,
                    )
                    return
                await asyncio.sleep(_ACK_RETRY_SECONDS * (attempt + 1))

    async def poll_once(self) -> float:
        started_at = self._clock()
        try:
            events = await api.fetch_bot_events(
                self._session,
                self._api_url,
                self._bot_token,
                since_event_id=self._cursor,
                limit=self._limit,
                wait_seconds=self._wait_seconds or None,
            )
            ordered = sorted(
                (
                    event
                    for event in events
                    if isinstance(event, dict)
                    and (event_id := _safe_event_id(event.get("event_id"))) is not None
                    and event_id > self._cursor
                ),
                key=lambda event: int(event["event_id"]),
            )
            for event in ordered:
                event_id = int(event["event_id"])
                action = parse_card_action(event)
                status = await self._on_card_action(action) if action is not None else None
                await self._cursor_store.save(event_id)
                self._cursor = event_id
                if status in {"completed", "dead_letter", "duplicate"}:
                    await self._ack(event_id)
            self._consecutive_errors = 0
            if self._wait_seconds == 0:
                return self._interval_seconds
            if ordered:
                return 0.0
            elapsed = self._clock() - started_at
            return (
                self._interval_seconds
                if elapsed < self._wait_seconds * _HELD_FRACTION
                else 0.0
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._consecutive_errors += 1
            exponent = min(self._consecutive_errors - 1, 30)
            return min(
                MAX_EVENT_BACKOFF_SECONDS,
                self._interval_seconds * (2**exponent),
            )

    async def run(self) -> None:
        await self.initialize()
        delay = 0.0 if self._wait_seconds > 0 else self._interval_seconds
        while not self._stop_event.is_set():
            if delay > 0:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                except TimeoutError:
                    pass
                if self._stop_event.is_set():
                    return
            delay = await self.poll_once()

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())
        return self._task

    def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
