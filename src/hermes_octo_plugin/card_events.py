"""Durable polling and trusted dispatch for Octo ``card_action`` events."""

from __future__ import annotations

import asyncio
import hashlib
import json
import inspect
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
from typing import Any, Protocol, cast

from .card_sessions import (
    CardClaim,
    CardSession,
    CardSessionRegistry,
    ClarifySession,
)

from gateway.platforms.base import MessageEvent, MessageType as HermesMessageType
from gateway.session import build_session_key

from . import api, cards, clarify as clarify_integration
from .cards import CardRenderResult
from .types import ChannelType

MAX_SAFE_EVENT_ID = (1 << 53) - 1
DEFAULT_EVENT_INTERVAL_SECONDS = 2.0
DEFAULT_EVENT_WAIT_SECONDS = 25
MAX_EVENT_WAIT_SECONDS = 30
MIN_EVENT_WAIT_SECONDS = 5
MAX_EVENT_BACKOFF_SECONDS = 30.0
_HELD_FRACTION = 0.5
_CARD_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_MAX_EVENT_INPUT_FIELDS = 128
_MAX_EVENT_DATA_FIELDS = 128
_MAX_EVENT_INPUT_VALUE_BYTES = 16 << 10
_MAX_EVENT_ENVELOPE_BYTES = 64 << 10
_ACK_ATTEMPTS = 3
_ACK_RETRY_SECONDS = 0.1
_MAX_PENDING_ACK_FLUSH_FAILURES = 3
_MAX_REJECTION_LOG_KEYS = 256
_MAX_REJECTION_LOGS = 10
_REJECTION_STATUSES = frozenset(
    {
        "dead_letter",
        "expired",
        "failed",
        "ignored",
        "invalid",
        "missing",
        "unsupported",
    }
)

logger = logging.getLogger(__name__)
_OWNER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class EventCursorStore(Protocol):
    async def load(self) -> int: ...

    async def load_pending_ack(self) -> int | None: ...

    async def save(
        self,
        event_id: int,
        *,
        pending_ack_event_id: int | None = None,
    ) -> None: ...


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

    async def load_pending_ack(self) -> int | None:
        return await asyncio.to_thread(self._load_pending_ack_sync)

    async def save(
        self,
        event_id: int,
        *,
        pending_ack_event_id: int | None = None,
    ) -> None:
        if _safe_event_id(event_id) is None:
            raise ValueError("invalid event cursor")
        if pending_ack_event_id is not None and (
            _safe_event_id(pending_ack_event_id) is None
            or pending_ack_event_id > event_id
        ):
            raise ValueError("invalid pending ack event id")
        await asyncio.to_thread(
            self._save_sync,
            event_id,
            pending_ack_event_id,
        )

    def _load_sync(self) -> int:
        with self._lock:
            event_id, _ = self._read_state_unlocked()
            return event_id

    def _load_pending_ack_sync(self) -> int | None:
        with self._lock:
            _, pending_ack_event_id = self._read_state_unlocked()
            return pending_ack_event_id

    def _read_state_unlocked(self) -> tuple[int, int | None]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return 0, None
        if not isinstance(raw, dict):
            return 0, None
        event_id = _safe_event_id(raw.get("event_id"))
        if event_id is None:
            return 0, None
        pending_ack_event_id = _safe_event_id(raw.get("pending_ack_event_id"))
        if (
            pending_ack_event_id is None
            or pending_ack_event_id > event_id
        ):
            pending_ack_event_id = None
        return event_id, pending_ack_event_id

    def _save_sync(
        self,
        event_id: int,
        pending_ack_event_id: int | None,
    ) -> None:
        with self._lock:
            current, _ = self._read_state_unlocked()
            if event_id < current:
                raise ValueError("event cursor cannot move backwards")
            payload: dict[str, int] = {"event_id": event_id}
            if pending_ack_event_id is not None:
                payload["pending_ack_event_id"] = pending_ack_event_id
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.parent / (
                f".events.cursor.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary.open("x", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            payload,
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




def _action_matches_session(action: CardAction, session: CardSession) -> bool:
    channel_matches = (
        action.channel_id in (session.action_channel_ids or (session.channel_id,))
        if action.channel_type == ChannelType.DM
        else action.channel_id == session.channel_id
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


def _reasoning_action_matches_session(
    action: CardAction,
    session: CardSession,
) -> bool:
    channel_matches = (
        action.channel_id in (session.action_channel_ids or (session.channel_id,))
        if action.channel_type == ChannelType.DM
        else action.channel_id == session.channel_id
    )
    return (
        session.kind == "reasoning"
        and action.message_id == session.message_id
        and channel_matches
        and action.channel_type == session.channel_type
        and action.operator_uid == session.requester_uid
        and action.action_id in session.action_labels
        and not action.inputs
        and action.data.get("reasoningId") == session.binding_id
    )


def handle_reasoning_card_action(
    registry: CardSessionRegistry,
    action: CardAction,
) -> str | None:
    """Consume an owned Registry control without injecting a user turn."""
    session = registry.peek(action.message_id)
    if session is None or session.kind != "reasoning":
        return None
    claim = registry.claim(action.message_id, action.event_id)
    if claim.session is None:
        return claim.status
    try:
        matches = _reasoning_action_matches_session(action, claim.session)
    except (TypeError, ValueError, UnicodeError):
        matches = False
    if claim.status != "claimed":
        return "duplicate" if matches else "ignored"
    if not matches:
        registry.release(action.message_id, action.event_id)
        return "ignored"
    registry.complete(action.message_id, action.event_id)
    return "unsupported"

def _neutralize_action_echo(value: str) -> str:
    return re.sub(r"([\\`*_~\[\]<>])", r"\\\1", value)


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
            "text": f"{cards.literal_card_text(safe_label)}: "
            f"{_neutralize_action_echo(inputs[input_id])}",

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



def _render_clarify_action_status(
    session: CardSession,
    action: CardAction,
    status: str,
) -> CardRenderResult:
    clarify = session.clarify
    if clarify is None:
        raise ValueError("clarify session is required")
    if status == "invalid":
        status_line = "请选择至少一个选项后再提交"
        card = dict(session.card)
        source_body = session.card.get("body")
        body = list(source_body) if isinstance(source_body, list) else []
        body.append({
            "type": "TextBlock",
            "text": cards.literal_card_text(status_line),
            "wrap": True,
            "spacing": "Medium",
            "color": "Attention",
        })
        card["body"] = body
        return CardRenderResult(
            card=card,
            plain="\n".join(
                part for part in (session.plain.strip(), status_line) if part
            ),
        )
    status_line = {
        "processing": "正在提交…",
        "completed": "已提交",
        "awaiting_text": "请直接发送文字回复",
        "expired": "该确认已失效或已处理",
        "failed": "提交失败，请重试",
    }.get(status, "提交失败，请重试")
    lines = ["需要确认", clarify.question]
    selected = (
        clarify_integration.selected_choices(clarify, action)
        if status in {"processing", "completed"}
        else []
    )
    if selected:
        lines.extend(("已选择", "、".join(selected)))
    lines.append(status_line)
    body = [
        {
            "type": "TextBlock",
            "text": cards.literal_card_text(text),
            "wrap": True,
            **(
                {"weight": "Bolder", "size": "Medium"}
                if index == 0
                else {"spacing": "Medium" if text in {"已选择", status_line} else "Small"}
            ),
        }
        for index, text in enumerate(lines)
    ]
    card = {key: value for key, value in session.card.items() if key != "actions"}
    card["body"] = body
    return CardRenderResult(card=card, plain="\n".join(lines))



def render_card_action_status(
    session: CardSession,
    action: CardAction,
    status: str,
) -> CardRenderResult:
    """Freeze submitted controls and append a disclosure-safe terminal status."""
    if session.clarify is not None:
        return _render_clarify_action_status(session, action, status)
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
    if status == "processing":
        status_line = f"Processing {label} for {operator}"
    elif status == "completed":
        status_line = f"Completed {label} for {operator}"
    elif status == "awaiting_text":
        status_line = f"Waiting for a typed response from {operator}"
    elif status == "expired":
        status_line = "This clarification expired or was already resolved"
    else:
        status_line = f"Failed {label} for {operator}"
    body.append(
        {
            "type": "TextBlock",
            "text": cards.literal_card_text(status_line),
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
    dispatch: Callable[[CardSession, CardAction], Awaitable[bool | str]],
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
    clarify = session.clarify
    if (
        clarify is not None
        and clarify.multi_select
        and action.action_id == clarify.confirm_action_id
        and not clarify_integration.selected_choices(clarify, action)
    ):
        registry.release(action.message_id, action.event_id)
        await _update_action_status(
            update_status,
            session,
            action,
            "invalid",
            transient=False,
        )
        return "invalid"
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
    terminal_status = accepted if isinstance(accepted, str) else (
        "completed" if accepted else "ignored"
    )
    if terminal_status not in {
        "completed",
        "awaiting_text",
        "expired",
        "failed",
    }:
        registry.release(action.message_id, action.event_id)
        await _update_action_status(
            update_status,
            session,
            action,
            "failed",
            transient=False,
        )
        return "ignored"
    registry.complete(action.message_id, action.event_id)
    await _update_action_status(
        update_status,
        session,
        action,
        terminal_status,
        transient=False,
    )
    return terminal_status


async def dispatch_clarify_action(
    session: CardSession,
    action: CardAction,
) -> bool | str | None:
    """Resolve an owned clarify card without creating a new user turn."""
    return await clarify_integration.dispatch_action(session, action)


async def dispatch_card_session_action(
    adapter: Any,
    session: CardSession,
    action: CardAction,
) -> bool | str:
    """Route clarify controls to Hermes primitives; other cards stay model-bound."""
    clarify_status = await dispatch_clarify_action(session, action)
    if clarify_status is not None:
        return clarify_status
    return await dispatch_card_action_event(adapter, session, action)


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
    group_sessions_per_user = bool(
        extra.get("group_sessions_per_user", True)
    )
    thread_sessions_per_user = bool(
        extra.get("thread_sessions_per_user", False)
    )
    source_profile = getattr(source, "profile", None)
    if (
        isinstance(source_profile, str)
        and source_profile
        and "profile" in inspect.signature(build_session_key).parameters
    ):
        profiled_builder = cast(Callable[..., str], build_session_key)
        derived_session_key = profiled_builder(
            source,
            group_sessions_per_user=group_sessions_per_user,
            thread_sessions_per_user=thread_sessions_per_user,
            profile=source_profile,
        )
    else:
        derived_session_key = build_session_key(
            source,
            group_sessions_per_user=group_sessions_per_user,
            thread_sessions_per_user=thread_sessions_per_user,
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
        self._pending_ack_event_id: int | None = None
        self._consecutive_errors = 0
        self._pending_ack_flush_failures = 0
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._logged_rejections: OrderedDict[tuple[int, str, str, str], None] = (
            OrderedDict()
        )
        self._rejection_logs_remaining = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    async def initialize(self) -> None:
        try:
            loaded = await self._cursor_store.load()
        except Exception:
            loaded = 0
        self._cursor = loaded if _safe_event_id(loaded) is not None else 0
        try:
            pending_ack_event_id = await self._cursor_store.load_pending_ack()
        except Exception:
            pending_ack_event_id = None
        safe_pending_ack_event_id = _safe_event_id(pending_ack_event_id)
        self._pending_ack_event_id = (
            safe_pending_ack_event_id
            if (
                safe_pending_ack_event_id is not None
                and safe_pending_ack_event_id <= self._cursor
            )
            else None
        )

    def _warn_rejection(
        self,
        *,
        event: Mapping[str, object],
        event_id: int,
        action: CardAction | None,
        reason: str,
    ) -> None:
        raw = event.get("event_data")
        if action is not None:
            message_id = action.message_id
            action_id = action.action_id
        elif isinstance(raw, Mapping):
            message_id = _bounded_string(raw.get("message_id")) or "<invalid>"
            action_id = (
                _bounded_string(raw.get("action_id"), max_chars=64) or "<invalid>"
            )
        else:
            message_id = "<invalid>"
            action_id = "<invalid>"
        key = (event_id, message_id, action_id, reason)
        if key in self._logged_rejections:
            return
        if self._rejection_logs_remaining <= 0:
            return
        self._rejection_logs_remaining -= 1
        self._logged_rejections[key] = None
        if len(self._logged_rejections) > _MAX_REJECTION_LOG_KEYS:
            self._logged_rejections.popitem(last=False)
        logger.warning(
            "Octo card action rejected event_id=%d message_id=%s action_id=%s reason=%s",
            event_id,
            message_id,
            action_id,
            reason,
        )

    async def _ack(self, event_id: int) -> bool:
        for attempt in range(_ACK_ATTEMPTS):
            try:
                await api.ack_bot_event(
                    self._session,
                    self._api_url,
                    self._bot_token,
                    event_id=event_id,
                )
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt + 1 >= _ACK_ATTEMPTS:
                    logger.warning(
                        "Octo event ack failed after %d attempts (%s)",
                        _ACK_ATTEMPTS,
                        type(exc).__name__,
                    )
                    return False
                await asyncio.sleep(_ACK_RETRY_SECONDS * (attempt + 1))
        return False

    async def _flush_pending_ack(self) -> bool:
        event_id = self._pending_ack_event_id
        if event_id is None:
            self._pending_ack_flush_failures = 0
            return True
        if not await self._ack(event_id):
            self._pending_ack_flush_failures += 1
            if (
                self._pending_ack_flush_failures
                < _MAX_PENDING_ACK_FLUSH_FAILURES
            ):
                return False
            await self._cursor_store.save(self._cursor)
            self._pending_ack_event_id = None
            self._pending_ack_flush_failures = 0
            logger.error(
                "Octo event poller abandoning pending ack for event %d "
                "after repeated failures",
                event_id,
            )
            return True
        await self._cursor_store.save(self._cursor)
        self._pending_ack_event_id = None
        self._pending_ack_flush_failures = 0
        return True

    async def poll_once(self) -> float:
        self._rejection_logs_remaining = _MAX_REJECTION_LOGS
        started_at = self._clock()
        try:
            if not await self._flush_pending_ack():
                self._consecutive_errors = 0
                return self._interval_seconds
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
                    and (event_id := _safe_event_id(event.get("event_id")))
                    is not None
                    and event_id > self._cursor
                ),
                key=lambda event: int(event["event_id"]),
            )
            ack_failed = False
            for event in ordered:
                event_id = int(event["event_id"])
                action = parse_card_action(event)
                status = (
                    await self._on_card_action(action)
                    if action is not None
                    else None
                )
                if action is None:
                    self._warn_rejection(
                        event=event,
                        event_id=event_id,
                        action=None,
                        reason="parse_invalid",
                    )
                elif status in _REJECTION_STATUSES:
                    self._warn_rejection(
                        event=event,
                        event_id=event_id,
                        action=action,
                        reason=status,
                    )
                elif status is None:
                    self._warn_rejection(
                        event=event,
                        event_id=event_id,
                        action=action,
                        reason="unhandled",
                    )
                should_ack = status in {
                    "completed",
                    "awaiting_text",
                    "expired",
                    "failed",
                    "dead_letter",
                    "invalid",
                    "duplicate",
                    "unsupported",
                }
                pending_ack_event_id = event_id if should_ack else None
                await self._cursor_store.save(
                    event_id,
                    pending_ack_event_id=pending_ack_event_id,
                )
                self._cursor = event_id
                self._pending_ack_event_id = pending_ack_event_id
                if should_ack and not await self._flush_pending_ack():
                    ack_failed = True
                    break
            self._consecutive_errors = 0
            if ack_failed or self._wait_seconds == 0:
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
        except Exception as exc:
            self._consecutive_errors += 1
            exponent = min(self._consecutive_errors - 1, 30)
            delay = min(
                MAX_EVENT_BACKOFF_SECONDS,
                self._interval_seconds * (2**exponent),
            )
            previous_delay = (
                min(
                    MAX_EVENT_BACKOFF_SECONDS,
                    self._interval_seconds * (2 ** min(exponent - 1, 30)),
                )
                if self._consecutive_errors > 1
                else None
            )
            if previous_delay is None or delay > previous_delay:
                logger.warning(
                    "Octo event polling failed (%s); retrying in %.1f seconds",
                    type(exc).__name__,
                    delay,
                )
            return delay

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
