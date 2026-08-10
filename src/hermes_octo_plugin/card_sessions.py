"""Shared card-session data models and bounded claim registry."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .types import ChannelType

_MAX_CARD_SESSIONS = 1024
_CARD_SESSION_TTL_SECONDS = 30 * 60


def _bounded_message_id(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128


def _valid_sequence(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 2**63 - 1


@dataclass(frozen=True)
class ClarifySession:
    clarify_id: str
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
    max_input_text_bytes: int | None = None
    max_inputs_bytes: int | None = None
    clarify: ClarifySession | None = None


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
        if not _bounded_message_id(session.message_id):
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
        if not _valid_sequence(card_seq):
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
            if entry is not None and entry.state == "processing" and entry.claimed_event_id == event_id:
                entry.state = "pending"
                entry.claimed_event_id = None

    def release_edit(self, message_id: str, card_seq: int) -> None:
        self.release(message_id, -card_seq)

    def complete(self, message_id: str, event_id: int) -> None:
        with self._lock:
            entry = self._entry_locked(message_id)
            if entry is not None and entry.state == "processing" and entry.claimed_event_id == event_id:
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
