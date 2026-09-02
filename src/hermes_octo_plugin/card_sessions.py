"""Shared card-session data models and bounded claim registry.

Bindings may be persisted per Octo identity so a deferred card action still
passes its original safety checks after a gateway restart or a token
rotation.  Only a whitelisted set of fields is written: action verification
data, controlled card/plain state, and the claim bookkeeping needed to avoid
re-running or permanently skipping an action.  Nothing credential-bearing is
ever persisted, and the store is never allowed to grow into a chat archive.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

from .types import ChannelType

logger = logging.getLogger(__name__)

_MAX_CARD_SESSIONS = 1024
_CARD_SESSION_TTL_SECONDS = 24 * 60 * 60


class CardBindingPersistence(Protocol):
    """Durable, identity-scoped storage for card-session bindings."""

    def load(self) -> list[dict[str, Any]]: ...

    def save(self, records: list[dict[str, Any]]) -> None: ...


def _bounded_message_id(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128


def _valid_sequence(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 2**63 - 1


@dataclass(frozen=True)
class ClarifySession:
    clarify_id: str
    entry: object
    multi_select: bool
    question: str
    choices: tuple[str, ...]
    action_choices: tuple[tuple[str, str], ...]
    input_id: str | None
    confirm_action_id: str | None
    other_action_id: str


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
    action_channel_ids: tuple[str, ...] = ()
    max_input_text_bytes: int | None = None
    max_inputs_bytes: int | None = None
    clarify: ClarifySession | None = None
    kind: str = "interactive"


@dataclass
class _CardSessionEntry:
    session: CardSession
    expires_at: float
    #: Absolute wall-clock expiry.  ``expires_at`` is monotonic and therefore
    #: meaningless across process restarts; this is what the durable store
    #: round-trips so a restored TTL keeps its original deadline.
    wall_expires_at: float = 0.0
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
    def __init__(
        self,
        *,
        max_sessions: int = _MAX_CARD_SESSIONS,
        ttl_seconds: float = _CARD_SESSION_TTL_SECONDS,
        max_dispatch_attempts: int = 3,
        persistence: CardBindingPersistence | None = None,
    ) -> None:
        self._max_sessions = max(1, max_sessions)
        self._ttl_seconds = max(1.0, ttl_seconds)
        self.max_dispatch_attempts = max(1, max_dispatch_attempts)
        self._entries: OrderedDict[str, _CardSessionEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._persistence = persistence

    def register(self, session: CardSession) -> None:
        if not _bounded_message_id(session.message_id):
            raise ValueError("invalid card session message_id")
        with self._lock:
            self._prune_locked()
            existing = self._entries.get(session.message_id)
            if existing is not None and existing.state != "completed":
                raise ValueError("card session message_id already active")

            self._entries.pop(session.message_id, None)
            while len(self._entries) >= self._max_sessions:
                completed_message_id = next(
                    (
                        message_id
                        for message_id, entry in self._entries.items()
                        if entry.state == "completed"
                    ),
                    None,
                )
                if completed_message_id is None:
                    raise ValueError("card session registry capacity exhausted")
                self._entries.pop(completed_message_id)
            now = time.monotonic()
            self._entries[session.message_id] = _CardSessionEntry(
                session=session,
                expires_at=now + self._ttl_seconds,
                wall_expires_at=time.time() + self._ttl_seconds,
            )
        self._persist()

    def refresh_reasoning(self, session: CardSession) -> None:
        """Refresh actions for the same pending reasoning-card identity only."""
        if not _bounded_message_id(session.message_id) or session.kind != "reasoning":
            raise ValueError("invalid reasoning card session")
        with self._lock:
            self._prune_locked()
            entry = self._entries.get(session.message_id)
            if entry is None or entry.state != "pending":
                raise ValueError("reasoning card session is not pending")
            existing = entry.session
            identity = (
                "binding_id",
                "session_key",
                "chat_id",
                "channel_id",
                "channel_type",
                "requester_uid",
                "action_channel_ids",
                "input_ids",
                "clarify",
                "kind",
            )
            if any(getattr(existing, field) != getattr(session, field) for field in identity):
                raise ValueError("reasoning card session identity mismatch")
            entry.session = session
            entry.expires_at = time.monotonic() + self._ttl_seconds
            entry.wall_expires_at = time.time() + self._ttl_seconds
            self._entries.move_to_end(session.message_id)



    def peek(self, message_id: str) -> CardSession | None:
        with self._lock:
            entry = self._entry_locked(message_id)
            return entry.session if entry is not None else None

    def discard(self, message_id: str) -> None:
        with self._lock:
            _ = self._entries.pop(message_id, None)
        self._persist()

    def claim_edit(
        self,
        *,
        message_id: str,
        session_key: str,
        channel_id: str,
        channel_type: ChannelType,
        requester_uid: str,
    ) -> int | None:
        """Claim a live card and allocate its next server-owned edit sequence."""
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is None or entry.state != "pending":
                return None
            session = entry.session
            if session.kind != "interactive" or session.clarify is not None:
                return None
            if (
                session.session_key != session_key
                or session.channel_id != channel_id
                or session.channel_type != channel_type
                or session.requester_uid != requester_uid
            ):
                return None
            previous = (entry.card_seq, entry.state, entry.claimed_event_id)
            entry.card_seq += 1
            card_seq = entry.card_seq
            entry.state = "processing"
            entry.claimed_event_id = -card_seq
            try:
                self._persist()
            except Exception:
                entry.card_seq, entry.state, entry.claimed_event_id = previous
                raise
            return card_seq

    def claim(self, message_id: str, event_id: int) -> CardClaim:
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is None:
                return CardClaim("missing")
            if entry.state != "pending":
                # A completed claim is only safe to acknowledge as duplicate
                # after its terminal state is durable. Retrying this write also
                # lets a transient disk failure recover without redispatch.
                self._persist()
                return CardClaim("duplicate", entry.session)
            previous = (
                entry.state,
                entry.claimed_event_id,
                entry.attempt_event_id,
                entry.dispatch_attempts,
            )
            entry.state = "processing"
            entry.claimed_event_id = event_id
            if entry.attempt_event_id != event_id:
                entry.attempt_event_id = event_id
                entry.dispatch_attempts = 0
            entry.dispatch_attempts += 1
            claim = CardClaim("claimed", entry.session, entry.dispatch_attempts)
            # Claim state must reach disk before the caller advances its event
            # cursor, otherwise a restart either replays or permanently skips
            # the action. Roll back a non-durable claim so a retry can dispatch.
            try:
                self._persist()
            except Exception:
                (
                    entry.state,
                    entry.claimed_event_id,
                    entry.attempt_event_id,
                    entry.dispatch_attempts,
                ) = previous
                raise
            return claim

    def next_card_seq(self, message_id: str) -> int | None:
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is None:
                return None
            entry.card_seq += 1
            card_seq = entry.card_seq
        self._persist()
        return card_seq

    def release(self, message_id: str, event_id: int) -> None:
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is not None and entry.state == "processing" and entry.claimed_event_id == event_id:
                entry.state = "pending"
                entry.claimed_event_id = None
        self._persist()

    def release_edit(self, message_id: str, card_seq: int) -> None:
        self.release(message_id, -card_seq)

    def complete(self, message_id: str, event_id: int) -> None:
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is not None and entry.state == "processing" and entry.claimed_event_id == event_id:
                entry.state = "completed"
        self._persist()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self._persist()

    # ── durable, identity-scoped bindings ──

    def set_persistence(self, persistence: CardBindingPersistence | None) -> None:
        """Attach identity-scoped durable storage once the identity is known."""
        with self._lock:
            self._persistence = persistence

    def attach_persistence(
        self,
        persistence: CardBindingPersistence,
    ) -> int:
        """Attach and restore one identity shard under a single lock."""
        with self._lock:
            self._persistence = persistence
            return self._restore_locked()

    def restore(self) -> int:
        """Rehydrate this identity's bindings before its event poller starts."""
        with self._lock:
            return self._restore_locked()

    def _restore_locked(self) -> int:
        if self._persistence is None:
            return 0
        try:
            records = self._persistence.load()
        except Exception:
            logger.error("octo: card-session restore failed", exc_info=True)
            return 0
        restored = 0
        wall_now = time.time()
        mono_now = time.monotonic()
        for record in records:
            entry = _entry_from_record(
                record,
                wall_now=wall_now,
                mono_now=mono_now,
            )
            if entry is None:
                continue
            if len(self._entries) >= self._max_sessions:
                break
            self._entries[entry.session.message_id] = entry
            restored += 1
        return restored

    def _persist(self) -> None:
        with self._lock:
            if self._persistence is None:
                return
            self._prune_locked()
            records = [
                record
                for record in (
                    _entry_to_record(entry) for entry in self._entries.values()
                )
                if record is not None
            ]
            try:
                self._persistence.save(records)
            except Exception:
                logger.error(
                    "octo: card-session persistence failed",
                    exc_info=True,
                )
                raise


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


#: Whitelisted card-session fields that may be persisted.  Anything outside
#: this list — tokens, credentials, extra chat copies — is never written.
_PERSISTED_SESSION_FIELDS: tuple[str, ...] = (
    "message_id",
    "binding_id",
    "session_key",
    "chat_id",
    "channel_id",
    "requester_uid",
    "plain",
    "kind",
)


def _entry_to_record(entry: _CardSessionEntry) -> dict[str, Any] | None:
    """Serialize one binding, or ``None`` when it must not be persisted.

    Reasoning/progress cards belong to a live turn and clarify sessions hold a
    non-serializable Hermes entry, so neither is durable.
    """
    session = entry.session
    if session.kind != "interactive" or session.clarify is not None:
        return None
    if not entry.wall_expires_at:
        return None
    record: dict[str, Any] = {
        field: getattr(session, field) for field in _PERSISTED_SESSION_FIELDS
    }
    record["channel_type"] = int(session.channel_type)
    record["card"] = session.card
    record["action_labels"] = dict(session.action_labels)
    record["input_ids"] = list(session.input_ids)
    record["action_channel_ids"] = list(session.action_channel_ids)
    record["max_input_text_bytes"] = session.max_input_text_bytes
    record["max_inputs_bytes"] = session.max_inputs_bytes
    record["expires_at"] = entry.wall_expires_at
    record["state"] = entry.state
    record["claimed_event_id"] = entry.claimed_event_id
    record["attempt_event_id"] = entry.attempt_event_id
    record["dispatch_attempts"] = entry.dispatch_attempts
    record["card_seq"] = entry.card_seq
    return record


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("card-session integer field is malformed")
    return value


def _entry_from_record(
    record: object,
    *,
    wall_now: float,
    mono_now: float,
) -> _CardSessionEntry | None:
    """Rebuild one binding, rejecting anything that fails its field contract."""
    if not isinstance(record, dict):
        return None
    try:
        expires_at = record["expires_at"]
        if not isinstance(expires_at, (int, float)) or expires_at <= wall_now:
            return None
        state = record["state"]
        if state not in {"pending", "processing", "completed"}:
            return None
        if not _bounded_message_id(record.get("message_id")):
            return None
        session = CardSession(
            message_id=record["message_id"],
            binding_id=str(record["binding_id"]),
            session_key=str(record["session_key"]),
            chat_id=str(record["chat_id"]),
            channel_id=str(record["channel_id"]),
            channel_type=ChannelType(int(record["channel_type"])),
            requester_uid=str(record["requester_uid"]),
            card=dict(record["card"]),
            plain=str(record["plain"]),
            action_labels={
                str(key): str(value)
                for key, value in dict(record["action_labels"]).items()
            },
            input_ids=tuple(str(item) for item in record["input_ids"]),
            action_channel_ids=tuple(
                str(item) for item in record["action_channel_ids"]
            ),
            max_input_text_bytes=_optional_int(record.get("max_input_text_bytes")),
            max_inputs_bytes=_optional_int(record.get("max_inputs_bytes")),
            clarify=None,
            kind=str(record.get("kind", "interactive")),
        )
        dispatch_attempts = record.get("dispatch_attempts", 0)
        card_seq = record.get("card_seq", 0)
        if not isinstance(dispatch_attempts, int) or isinstance(dispatch_attempts, bool):
            return None
        if not isinstance(card_seq, int) or isinstance(card_seq, bool):
            return None
        entry = _CardSessionEntry(
            session=session,
            expires_at=mono_now + (float(expires_at) - wall_now),
            wall_expires_at=float(expires_at),
            # A binding that was mid-dispatch when the gateway died is returned
            # to ``pending`` so the owning identity may safely retry it; the
            # attempt counter still bounds total retries.
            state="pending" if state == "processing" else state,
            claimed_event_id=None,
            attempt_event_id=_optional_int(record.get("attempt_event_id")),
            dispatch_attempts=dispatch_attempts,
            card_seq=card_seq,
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("octo: rejected a malformed persisted card binding")
        return None
    if session.kind != "interactive":
        return None
    return entry
