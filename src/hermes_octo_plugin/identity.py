"""Multi-token Octo identity routing.

One ``octo`` platform and one ``OctoAdapter`` may carry several Octo bot
tokens.  Each token becomes an independent identity runtime, and this module
owns the three-layer identity model that keeps conversations pinned to the
right one:

``Hermes SessionKey``
    Which conversation this is.  Never contains a token or ``robot_id``.
``robot_id``
    Which stable Octo bot identity must send.  Survives token rotation.
``bot token``
    The current credential used to connect and call the Bot API.

Every durable artefact here is profile-owned, schema-versioned, checksummed
and replaced atomically.  None of it may ever contain a bot token, an IM
token, a crypto key or a token hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .types import ChannelType

logger = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 1

# Bounded persistence: hostile or runaway traffic must not grow profile state
# without limit.  Routes are cheap (two short strings) so the cap is generous.
MAX_PERSISTED_ROUTES = 4096

_ROBOT_ID_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
_ROBOT_PATH_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_WIRE_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_@:-]{1,128}$")

#: ``phase=single`` — this profile has only ever run one identity.  The
#: adapter keeps the historical single-runtime fast path and writes no route
#: snapshot.
PHASE_SINGLE = "single"
#: ``phase=pending`` — a frozen legacy-migration plan exists but has not been
#: applied to the verified legacy primary yet.
PHASE_PENDING = "pending"
#: ``phase=migrated`` — legacy migration finished.  It never runs again.
PHASE_MIGRATED = "migrated"
_PHASES = frozenset({PHASE_SINGLE, PHASE_PENDING, PHASE_MIGRATED})

CHAT_TYPE_DM = "dm"
CHAT_TYPE_GROUP = "group"
_CHAT_TYPES = frozenset({CHAT_TYPE_DM, CHAT_TYPE_GROUP})

#: Trusted, adapter-internal identity of the turn currently being processed.
#: Set by the inbound listener that decrypted the message, never by the model
#: and never exposed in a tool schema.
current_robot_id: ContextVar[str] = ContextVar("octo_current_robot_id", default="")


class TokenConfigError(ValueError):
    """Malformed ``OCTO_BOT_TOKEN`` list.

    The message reports the offending entry's position and error class only —
    never the token text or any value derived from it.
    """


class IdentityStateError(RuntimeError):
    """Durable identity state is unusable and recovery would be ambiguous."""


class IdentityConflictError(RuntimeError):
    """Two configured tokens resolve to the same stable Octo identity.

    Only a failure state is kept for the losing runtime: starting a second
    connection for one bot makes the two sessions kick each other forever.
    """


def parse_bot_tokens(raw: object) -> tuple[str, ...]:
    """Parse ``OCTO_BOT_TOKEN`` into an ordered tuple of distinct tokens.

    ``token-a`` and ``token-a;token-b;token-c`` are both valid.  An empty or
    absent value yields ``()`` so an unconfigured platform still constructs
    and reports the existing "must be set" failure at connect time.
    """
    if raw is None:
        return ()
    if not isinstance(raw, str):
        raise TokenConfigError("OCTO_BOT_TOKEN must be a string")
    if not raw.strip():
        return ()
    seen: dict[str, int] = {}
    tokens: list[str] = []
    for index, item in enumerate(raw.split(";"), start=1):
        token = item.strip()
        if not token:
            raise TokenConfigError(
                f"OCTO_BOT_TOKEN entry #{index} is empty: separate tokens with "
                "one ';' and do not use a leading or trailing ';'"
            )
        first = seen.get(token)
        if first is not None:
            raise TokenConfigError(
                f"OCTO_BOT_TOKEN entry #{index} repeats entry #{first}: the same "
                "identity must not open two connections"
            )
        seen[token] = index
        tokens.append(token)
    return tuple(tokens)


def valid_robot_id(value: object) -> bool:
    """Is *value* a usable stable Octo identity?"""
    return isinstance(value, str) and _ROBOT_ID_RE.fullmatch(value) is not None


def robot_path_segment(robot_id: str) -> str:
    """Map a ``robot_id`` onto one safe filesystem segment."""
    if not valid_robot_id(robot_id):
        raise ValueError("invalid Octo robot_id")
    if _ROBOT_PATH_RE.fullmatch(robot_id):
        return robot_id
    digest = hashlib.sha256(robot_id.encode("utf-8")).hexdigest()
    return f"robot-{digest}"

def scoped_dm_chat_id(robot_id: str, wire_chat_id: str) -> str:
    """Opaque, rotation-stable Hermes DM id for a non-primary identity.

    Octo's live transport may expose the same bare user uid in several Spaces.
    The trusted receiving ``robot_id`` supplies the missing scope, but neither
    that id nor a credential is copied into the SessionKey.
    """
    if not valid_robot_id(robot_id) or not _WIRE_CHAT_ID_RE.fullmatch(wire_chat_id):
        raise ValueError("invalid Octo DM identity")
    robot_scope = hashlib.sha256(robot_id.encode("utf-8")).hexdigest()[:16]
    peer_scope = hashlib.sha256(wire_chat_id.encode("utf-8")).hexdigest()[:24]
    return f"octodm_{robot_scope}_{peer_scope}"


def chat_type_for_channel(channel_type: ChannelType | int) -> str:
    """Collapse an Octo channel type onto the Hermes ``chat_type`` vocabulary."""
    return CHAT_TYPE_DM if int(channel_type) == int(ChannelType.DM) else CHAT_TYPE_GROUP


def valid_channel_type(value: object) -> bool:
    """Is *value* one of the Octo channel types this adapter speaks?"""
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    try:
        ChannelType(value)
    except ValueError:
        return False
    return True


def channel_type_for_planned_route(chat_type: str, chat_id: str) -> int:
    """Derive the exact Octo wire type from a structured Hermes origin.

    Only structural facts are used: the Hermes ``chat_type`` decides DM versus
    group, and Octo's ``____`` thread separator decides group versus community
    topic.  No id-shape heuristic is involved.
    """
    if chat_type == CHAT_TYPE_DM:
        return int(ChannelType.DM)
    if "____" in chat_id:
        return int(ChannelType.CommunityTopic)
    return int(ChannelType.Group)


def derive_session_key(
    source: Any,
    *,
    group_sessions_per_user: bool,
    thread_sessions_per_user: bool,
) -> str:
    """Build the Hermes session key exactly as ``handle_message`` would.

    Route binding must agree with Hermes' own key derivation, including the
    optional ``profile`` parameter newer Hermes versions accept, otherwise a
    persisted route would never be found again.
    """
    import inspect

    from gateway.session import build_session_key

    profile = getattr(source, "profile", None)
    if (
        isinstance(profile, str)
        and profile
        and "profile" in inspect.signature(build_session_key).parameters
    ):
        profiled_builder = cast("Callable[..., str]", build_session_key)
        return profiled_builder(
            source,
            group_sessions_per_user=group_sessions_per_user,
            thread_sessions_per_user=thread_sessions_per_user,
            profile=profile,
        )
    return build_session_key(
        source,
        group_sessions_per_user=group_sessions_per_user,
        thread_sessions_per_user=thread_sessions_per_user,
    )


def identity_state_dir(base_dir: Path | None = None) -> Path:
    """Profile-owned directory for identity-routing state."""
    if base_dir is not None:
        return Path(base_dir)
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "workspace" / "octo" / "identity"


# ── Durable document primitives ───────────────────────────────────────────


def _checksum(body: Any) -> str:
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_state_document(path: Path, body: Any) -> None:
    """Persist *body* with a schema version, checksum, fsync and atomic replace."""
    document = {
        "schema": STATE_SCHEMA_VERSION,
        "checksum": _checksum(body),
        "body": body,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


#: ``read_state_document`` outcomes.  ``corrupt`` and ``unsupported`` are
#: deliberately distinct from ``missing``: only a genuinely absent file may be
#: treated as "this profile has no state yet".
READ_OK = "ok"
READ_MISSING = "missing"
READ_CORRUPT = "corrupt"
READ_UNSUPPORTED = "unsupported"


def read_state_document(path: Path) -> tuple[str, Any]:
    """Read one durable document, verifying schema version and checksum."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return READ_MISSING, None
    except OSError:
        return READ_CORRUPT, None
    try:
        document = json.loads(raw)
    except ValueError:
        return READ_CORRUPT, None
    if not isinstance(document, dict):
        return READ_CORRUPT, None
    schema = document.get("schema")
    if schema != STATE_SCHEMA_VERSION:
        return READ_UNSUPPORTED, None
    body = document.get("body")
    if document.get("checksum") != _checksum(body):
        return READ_CORRUPT, None
    return READ_OK, body


# ── Identity-routing metadata ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class IdentityMetadata:
    """Persistent sentinel for "has this profile entered identity routing?"."""

    phase: str
    legacy_robot_id: str = ""
    plan_size: int = 0

    def to_body(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "legacy_robot_id": self.legacy_robot_id,
            "plan_size": self.plan_size,
        }

    @classmethod
    def from_body(cls, body: Any) -> IdentityMetadata:
        if not isinstance(body, dict):
            raise IdentityStateError("identity metadata is not an object")
        phase = body.get("phase")
        if phase not in _PHASES:
            raise IdentityStateError("identity metadata phase is unknown")
        legacy = body.get("legacy_robot_id") or ""
        if legacy and not valid_robot_id(legacy):
            raise IdentityStateError("identity metadata legacy id is malformed")
        plan_size = body.get("plan_size", 0)
        if not isinstance(plan_size, int) or isinstance(plan_size, bool) or plan_size < 0:
            raise IdentityStateError("identity metadata plan size is malformed")
        return cls(
            phase=phase,
            legacy_robot_id=legacy,
            plan_size=plan_size,
        )


@dataclass(frozen=True, slots=True)
class PlannedRoute:
    """One pre-existing ``octo`` conversation frozen for legacy migration."""

    session_key: str
    chat_type: str
    chat_id: str
    channel_type: int

    def to_body(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "chat_type": self.chat_type,
            "chat_id": self.chat_id,
            "channel_type": self.channel_type,
        }

    @classmethod
    def from_body(cls, body: Any) -> PlannedRoute | None:
        if not isinstance(body, dict):
            return None
        session_key = body.get("session_key")
        chat_type = body.get("chat_type")
        chat_id = body.get("chat_id")
        channel_type = body.get("channel_type")
        if not isinstance(session_key, str) or not session_key:
            return None
        if chat_type not in _CHAT_TYPES:
            return None
        if not isinstance(chat_id, str) or not chat_id:
            return None
        if not valid_channel_type(channel_type):
            return None
        return cls(
            session_key=session_key,
            chat_type=chat_type,
            chat_id=chat_id,
            channel_type=int(channel_type),
        )


# ── Route registry ────────────────────────────────────────────────────────

BIND_BOUND = "bound"
BIND_CAPACITY = "capacity"
BIND_IDEMPOTENT = "idempotent"
BIND_CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class TargetLookup:
    """Unique reverse-index answer for a chat-id-only send.

    ``channel_type`` is the exact Octo wire type recorded when the route was
    bound, so a restarted gateway never has to guess DM versus group from the
    shape of a chat id.
    """

    chat_type: str
    channel_type: int
    robot_id: str
    wire_chat_id: str


class IdentityRouteRegistry:
    """Stable ``SessionRoute`` / ``TargetRoute`` bindings with fail-closed conflicts.

    Bindings are established only from a trusted inbound listener, a verified
    legacy-migration plan, or a restored snapshot.  A key claimed by a second
    ``robot_id`` becomes a permanent conflict: neither identity may serve it
    until configuration or durable state is repaired.  "Last writer wins" and
    token-order guessing are both refused.
    """

    def __init__(self, *, path: Path, backup_path: Path | None = None) -> None:
        self._path = Path(path)
        self._backup_path = (
            Path(backup_path)
            if backup_path is not None
            else self._path.with_suffix(self._path.suffix + ".bak")
        )
        self._lock = threading.RLock()
        self._flush_lock = threading.Lock()
        self._session_routes: dict[str, str] = {}
        self._target_routes: dict[tuple[str, str], str] = {}
        self._session_conflicts: set[str] = set()
        self._target_conflicts: set[tuple[str, str]] = set()
        self._chat_index: dict[str, set[tuple[str, str]]] = {}
        self._target_sessions: dict[tuple[str, str], set[str]] = {}
        self._target_channel_types: dict[tuple[str, str], int] = {}
        self._target_wire_ids: dict[tuple[str, str], str] = {}
        self._dirty = False
        self.restored_from_backup = False
        self.snapshot_corrupt = False

    # ── state predicates ──

    @property
    def path(self) -> Path:
        return self._path

    @property
    def backup_path(self) -> Path:
        return self._backup_path

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def has_state(self) -> bool:
        """Does any durable route evidence exist for this profile?"""
        return self._path.exists() or self._backup_path.exists()

    def route_count(self) -> int:
        with self._lock:
            return len(self._target_routes)

    def session_count(self) -> int:
        with self._lock:
            return len(self._session_routes)

    def capacity(self) -> int:
        return MAX_PERSISTED_ROUTES

    def remaining_capacity(self) -> int:
        with self._lock:
            return max(0, MAX_PERSISTED_ROUTES - len(self._target_routes))

    def capacity_exhausted(self) -> bool:
        with self._lock:
            return len(self._target_routes) >= MAX_PERSISTED_ROUTES

    # ── binding ──

    def bind(
        self,
        *,
        robot_id: str,
        session_key: str | None,
        chat_type: str,
        chat_id: str,
        channel_type: int,
        wire_chat_id: str | None = None,
        wire_chat_id_known: bool = True,
    ) -> str:
        """Atomically claim a session and/or transport target for *robot_id*."""
        if not valid_robot_id(robot_id):
            raise ValueError("invalid Octo robot_id")
        if chat_type not in _CHAT_TYPES:
            raise ValueError("invalid Octo chat_type")
        if not chat_id:
            raise ValueError("invalid Octo chat_id")
        if not valid_channel_type(channel_type):
            raise ValueError("invalid Octo channel_type")
        wire_target = (
            (chat_id if wire_chat_id is None else wire_chat_id)
            if wire_chat_id_known
            else None
        )
        if wire_target is not None and (
            not isinstance(wire_target, str)
            or not _WIRE_CHAT_ID_RE.fullmatch(wire_target)
        ):
            raise ValueError("invalid Octo wire chat_id")
        target_key = (chat_type, chat_id)
        with self._lock:
            if session_key and session_key in self._session_conflicts:
                return BIND_CONFLICT
            if target_key in self._target_conflicts:
                return BIND_CONFLICT

            existing_session = (
                self._session_routes.get(session_key) if session_key else None
            )
            existing_target = self._target_routes.get(target_key)
            if existing_session is not None and existing_session != robot_id:
                self._session_conflicts.add(session_key or "")
                if existing_target is not None and existing_target != robot_id:
                    self._target_conflicts.add(target_key)
                    self._session_conflicts.update(
                        self._target_sessions.get(target_key, ())
                    )
                logger.error(
                    "octo: session route conflict; %s claimed a route owned by %s",
                    robot_id,
                    existing_session,
                )
                self._dirty = True
                return BIND_CONFLICT
            if existing_target is not None and existing_target != robot_id:
                self._target_conflicts.add(target_key)
                self._session_conflicts.update(
                    self._target_sessions.get(target_key, ())
                )
                logger.error(
                    "octo: target route conflict; %s claimed a target owned by %s",
                    robot_id,
                    existing_target,
                )
                self._dirty = True
                return BIND_CONFLICT
            existing_wire_target = self._target_wire_ids.get(target_key)
            wire_target_fillable = (
                wire_target is not None and existing_wire_target is None
            )
            if (
                existing_target is not None
                and existing_wire_target is not None
                and not wire_target_fillable
                and wire_target is not None
                and existing_wire_target != wire_target
            ):
                self._target_conflicts.add(target_key)
                self._session_conflicts.update(
                    self._target_sessions.get(target_key, ())
                )
                logger.error(
                    "octo: target route wire identity conflict for robot %s",
                    robot_id,
                )
                self._dirty = True
                return BIND_CONFLICT

            if existing_target is not None and (
                not session_key or existing_session is not None
            ):
                if wire_target_fillable:
                    assert wire_target is not None
                    self._target_channel_types[target_key] = int(channel_type)
                    self._target_wire_ids[target_key] = wire_target
                    self._dirty = True
                    return BIND_BOUND
                return BIND_IDEMPOTENT

            if (
                existing_target is None
                and len(self._target_routes) >= MAX_PERSISTED_ROUTES
            ):
                logger.error(
                    "octo: route store full; refusing a new target without "
                    "evicting established ownership"
                )
                return BIND_CAPACITY
            if existing_target is None:
                self._target_routes[target_key] = robot_id
                self._chat_index.setdefault(chat_id, set()).add(
                    (chat_type, robot_id)
                )
            # A thread can turn into a plain group target (or the reverse) only
            # by changing its channel id, so refreshing the exact wire type of
            # an existing target is safe and keeps the reverse index accurate.
            self._target_channel_types[target_key] = int(channel_type)
            if wire_target is not None:
                self._target_wire_ids[target_key] = wire_target
            if session_key:
                while len(self._session_routes) >= MAX_PERSISTED_ROUTES:
                    self._evict_oldest_session_locked()
                self._session_routes[session_key] = robot_id
                self._target_sessions.setdefault(target_key, set()).add(session_key)
            self._dirty = True
            return BIND_BOUND

    def bind_plan(
        self,
        *,
        robot_id: str,
        plan: Iterable[PlannedRoute],
    ) -> bool:
        """Atomically bind a frozen legacy plan without leaving tombstones."""
        with self._lock:
            previous = self._body_locked()
            previous_dirty = self._dirty
            for route in plan:
                result = self.bind(
                    robot_id=robot_id,
                    session_key=route.session_key,
                    chat_type=route.chat_type,
                    chat_id=route.chat_id,
                    channel_type=route.channel_type,
                    wire_chat_id_known=route.chat_type != CHAT_TYPE_DM,
                )
                if result in {BIND_CONFLICT, BIND_CAPACITY}:
                    self._apply_body(previous)
                    self._dirty = previous_dirty
                    return False
            return True

    def forget_target(
        self,
        *,
        robot_id: str,
        chat_type: str,
        chat_id: str,
    ) -> bool:
        """Durably forget one exact route after an explicit operator decision."""
        if not valid_robot_id(robot_id):
            raise ValueError("invalid Octo robot_id")
        if chat_type not in _CHAT_TYPES:
            raise ValueError("invalid Octo chat_type")
        if not chat_id:
            raise ValueError("invalid Octo chat_id")
        target_key = (chat_type, chat_id)
        # Keep the same lock order as flush(): filesystem transaction first,
        # then registry state. Holding both through the explicit admin write
        # prevents another inbound from claiming the released target before
        # its deletion is durable.
        with self._flush_lock:
            with self._lock:
                if target_key in self._target_conflicts:
                    return False
                if self._target_routes.get(target_key) != robot_id:
                    return False
                linked_sessions = set(self._target_sessions.get(target_key, ()))
                if any(key in self._session_conflicts for key in linked_sessions):
                    return False
                previous = self._body_locked()
                previous_dirty = self._dirty
                self._target_routes.pop(target_key, None)
                self._target_sessions.pop(target_key, None)
                self._target_channel_types.pop(target_key, None)
                self._target_wire_ids.pop(target_key, None)
                holders = self._chat_index.get(chat_id)
                if holders is not None:
                    holders.discard((chat_type, robot_id))
                    if not holders:
                        self._chat_index.pop(chat_id, None)
                for session_key in linked_sessions:
                    if not any(
                        session_key in sessions
                        for sessions in self._target_sessions.values()
                    ):
                        self._session_routes.pop(session_key, None)
                body = self._body_locked()
                if not self._write_snapshot(body):
                    self._apply_body(previous)
                    self._dirty = previous_dirty
                    raise IdentityStateError(
                        "Octo route deletion could not be persisted"
                    )
                self._dirty = False
                return True


    def _evict_oldest_session_locked(self) -> None:
        try:
            victim = next(iter(self._session_routes))
        except StopIteration:
            return
        self._session_routes.pop(victim, None)
        for sessions in self._target_sessions.values():
            sessions.discard(victim)
        logger.warning("octo: route store full; evicted the oldest session route")

    # ── resolution ──

    def robot_id_for_session(self, session_key: str) -> str | None:
        if not session_key:
            return None
        with self._lock:
            if session_key in self._session_conflicts:
                return None
            return self._session_routes.get(session_key)

    def robot_id_for_target(self, chat_type: str, chat_id: str) -> str | None:
        key = (chat_type, chat_id)
        with self._lock:
            if key in self._target_conflicts:
                return None
            return self._target_routes.get(key)

    def lookup_chat_id(self, chat_id: str) -> TargetLookup | None:
        """Resolve a chat-id-only send, or ``None`` for zero/ambiguous matches."""
        with self._lock:
            holders = self._chat_index.get(chat_id)
            if not holders or len(holders) != 1:
                return None
            chat_type, robot_id = next(iter(holders))
            target_key = (chat_type, chat_id)
            if target_key in self._target_conflicts:
                return None
            channel_type = self._target_channel_types.get(target_key)
            wire_chat_id = self._target_wire_ids.get(target_key)
            if channel_type is None or wire_chat_id is None:
                return None
            return TargetLookup(
                chat_type=chat_type,
                channel_type=channel_type,
                robot_id=robot_id,
                wire_chat_id=wire_chat_id,
            )

    def session_conflict(self, session_key: str) -> bool:
        with self._lock:
            return session_key in self._session_conflicts

    def target_conflict(self, chat_type: str, chat_id: str) -> bool:
        with self._lock:
            return (chat_type, chat_id) in self._target_conflicts

    def chat_id_has_state(self, chat_id: str) -> bool:
        """Whether a chat id is bound, ambiguous, or explicitly conflicted."""
        with self._lock:
            if self._chat_index.get(chat_id):
                return True
            return any(
                target_chat_id == chat_id
                for _chat_type, target_chat_id in self._target_conflicts
            )

    def robot_ids(self) -> set[str]:
        with self._lock:
            return set(self._target_routes.values()) | set(
                self._session_routes.values()
            )

    # ── persistence ──

    def _body_locked(self) -> dict[str, Any]:
        session_targets: dict[str, tuple[str, str]] = {
            session_key: target_key
            for target_key, session_keys in self._target_sessions.items()
            for session_key in session_keys
        }
        sessions: list[dict[str, Any]] = []
        for key, robot_id in self._session_routes.items():
            record: dict[str, Any] = {"session_key": key, "robot_id": robot_id}
            target_key = session_targets.get(key)
            if target_key is not None:
                record["chat_type"], record["chat_id"] = target_key
            sessions.append(record)
        return {
            "sessions": sessions,
            "targets": [
                {
                    "chat_type": chat_type,
                    "chat_id": chat_id,
                    "robot_id": robot_id,
                    "channel_type": self._target_channel_types.get(
                        (chat_type, chat_id)
                    ),
                    "wire_chat_id": self._target_wire_ids.get(
                        (chat_type, chat_id)
                    ),
                    "wire_chat_id_known": (chat_type, chat_id)
                    in self._target_wire_ids,
                }
                for (chat_type, chat_id), robot_id in self._target_routes.items()
            ],
            "session_conflicts": sorted(self._session_conflicts),
            "target_conflicts": [
                [chat_type, chat_id]
                for chat_type, chat_id in sorted(self._target_conflicts)
            ],
        }

    def _write_snapshot(self, body: dict[str, Any]) -> bool:
        try:
            if (
                self._path.exists()
                and not self.restored_from_backup
                and not self.snapshot_corrupt
            ):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(self._path, self._backup_path)
            write_state_document(self._path, body)
        except OSError:
            logger.error("octo: route snapshot write failed", exc_info=True)
            return False
        self.restored_from_backup = False
        self.snapshot_corrupt = False
        return True

    def flush(self, *, force: bool = False) -> bool:
        """Persist the snapshot, retaining the previous generation as a backup."""
        # Snapshot generation and the current→backup→new-current file sequence
        # are one transaction. The registry lock protects the in-memory maps;
        # this lock separately serializes the slower filesystem transaction.
        with self._flush_lock:
            with self._lock:
                if not self._dirty and not force:
                    return False
                body = self._body_locked()
                self._dirty = False
            if not self._write_snapshot(body):
                with self._lock:
                    self._dirty = True
                return False
            return True

    def load(self) -> None:
        """Restore the current snapshot, falling back to the previous generation."""
        status, body = read_state_document(self._path)
        if status != READ_OK:
            backup_status, backup_body = read_state_document(self._backup_path)
            if backup_status == READ_OK:
                self.restored_from_backup = True
                logger.warning(
                    "octo: route snapshot unusable (%s); restored previous generation",
                    status,
                )
                status, body = backup_status, backup_body
            elif status == READ_MISSING and backup_status == READ_MISSING:
                return
            else:
                self.snapshot_corrupt = True
                logger.error(
                    "octo: route snapshot unusable (%s) and backup unusable (%s); "
                    "unknown routes fail closed until trusted inbound rebinds them",
                    status,
                    backup_status,
                )
                return
        self._apply_body(body)

    def _apply_body(self, body: Any) -> None:
        if not isinstance(body, dict):
            self.snapshot_corrupt = True
            return
        with self._lock:
            self._session_routes.clear()
            self._target_routes.clear()
            self._session_conflicts.clear()
            self._target_conflicts.clear()
            self._chat_index.clear()
            self._target_sessions.clear()
            self._target_channel_types.clear()
            self._target_wire_ids.clear()
            for record in body.get("sessions") or ():
                if not isinstance(record, dict):
                    continue
                key = record.get("session_key")
                robot_id = record.get("robot_id")
                if not (isinstance(key, str) and key and valid_robot_id(robot_id)):
                    continue
                self._session_routes[key] = robot_id
                chat_type = record.get("chat_type")
                chat_id = record.get("chat_id")
                if (
                    chat_type in _CHAT_TYPES
                    and isinstance(chat_id, str)
                    and chat_id
                ):
                    self._target_sessions.setdefault(
                        (chat_type, chat_id), set()
                    ).add(key)
            for record in body.get("targets") or ():
                if not isinstance(record, dict):
                    continue
                chat_type = record.get("chat_type")
                chat_id = record.get("chat_id")
                robot_id = record.get("robot_id")
                channel_type = record.get("channel_type")
                wire_chat_id = record.get("wire_chat_id", chat_id)
                wire_chat_id_known = record.get("wire_chat_id_known")
                if wire_chat_id_known is None:
                    wire_chat_id_known = not (
                        chat_type == CHAT_TYPE_DM and wire_chat_id == chat_id
                    )
                if not (
                    chat_type in _CHAT_TYPES
                    and isinstance(chat_id, str)
                    and chat_id
                    and valid_robot_id(robot_id)
                    and valid_channel_type(channel_type)
                    and isinstance(wire_chat_id_known, bool)
                    and (
                        wire_chat_id is None
                        or (
                            isinstance(wire_chat_id, str)
                            and _WIRE_CHAT_ID_RE.fullmatch(wire_chat_id)
                        )
                    )
                ):
                    continue
                target_key = (str(chat_type), chat_id)
                target_robot_id = str(robot_id)
                self._target_routes[target_key] = target_robot_id
                self._target_channel_types[target_key] = int(channel_type)
                if wire_chat_id_known and wire_chat_id is not None:
                    self._target_wire_ids[target_key] = wire_chat_id
                self._chat_index.setdefault(chat_id, set()).add(
                    (str(chat_type), target_robot_id)
                )
            for key in body.get("session_conflicts") or ():
                if isinstance(key, str) and key:
                    self._session_conflicts.add(key)
            for record in body.get("target_conflicts") or ():
                if (
                    isinstance(record, (list, tuple))
                    and len(record) == 2
                    and record[0] in _CHAT_TYPES
                    and isinstance(record[1], str)
                    and record[1]
                ):
                    self._target_conflicts.add((record[0], record[1]))
            self._dirty = False



@dataclass(frozen=True, slots=True)
class IdentityStartup:
    """Decision taken before any listener is allowed to receive inbound traffic."""

    phase: str
    legacy_robot_id: str
    plan: tuple[PlannedRoute, ...]
    sole_identity_mode: bool


class IdentityStateStore:
    """Own the identity-routing metadata, frozen plan and route snapshot.

    The metadata file is the persistent sentinel for "this profile has entered
    the new identity-routing state machine".  It lives in its own document so
    deleting or corrupting the route snapshot cannot erase that fact and let a
    legacy migration run a second time.
    """

    METADATA_NAME = "identity-routing.json"
    PLAN_NAME = "identity-migration-plan.json"
    ROUTES_NAME = "identity-routes.json"

    def __init__(self, *, base_dir: Path | None = None) -> None:
        self._dir = identity_state_dir(base_dir)
        self._metadata_path = self._dir / self.METADATA_NAME
        self._plan_path = self._dir / self.PLAN_NAME
        self.routes = IdentityRouteRegistry(path=self._dir / self.ROUTES_NAME)
        self._metadata: IdentityMetadata | None = None

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def metadata_path(self) -> Path:
        return self._metadata_path

    @property
    def plan_path(self) -> Path:
        return self._plan_path

    @property
    def metadata(self) -> IdentityMetadata | None:
        return self._metadata

    # ── metadata ──

    def load_metadata(self) -> IdentityMetadata | None:
        status, body = read_state_document(self._metadata_path)
        if status == READ_MISSING:
            self._metadata = None
            return None
        if status != READ_OK:
            raise IdentityStateError(
                f"Octo identity-routing metadata is unusable ({status})"
            )
        self._metadata = IdentityMetadata.from_body(body)
        return self._metadata

    def save_metadata(self, metadata: IdentityMetadata) -> None:
        write_state_document(self._metadata_path, metadata.to_body())
        self._metadata = metadata

    # ── frozen migration plan ──

    def load_plan(self) -> tuple[PlannedRoute, ...]:
        status, body = read_state_document(self._plan_path)
        if status == READ_MISSING:
            return ()
        if status != READ_OK:
            raise IdentityStateError(
                f"Octo legacy migration plan is unusable ({status})"
            )
        if not isinstance(body, dict):
            raise IdentityStateError("Octo legacy migration plan is malformed")
        planned: list[PlannedRoute] = []
        for record in body.get("entries") or ():
            route = PlannedRoute.from_body(record)
            if route is not None:
                planned.append(route)
        return tuple(planned)

    def load_pending_plan(
        self, metadata: IdentityMetadata
    ) -> tuple[PlannedRoute, ...]:
        plan = self.load_plan()
        if len(plan) != metadata.plan_size:
            raise IdentityStateError(
                "Octo legacy migration plan size does not match durable metadata"
            )
        return plan

    def save_plan(self, plan: tuple[PlannedRoute, ...]) -> None:
        write_state_document(
            self._plan_path, {"entries": [route.to_body() for route in plan]}
        )

    def clear_plan(self) -> None:
        try:
            self._plan_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            logger.debug("octo: migration plan removal failed", exc_info=True)

    # ── startup state machine ──

    def has_routing_evidence(self) -> bool:
        """Any durable evidence that identity routing already ran here."""
        if self.routes.has_state() or self._plan_path.exists():
            return True
        # Card bindings are sharded beside ``identity/`` in the real profile
        # layout (or below an explicitly supplied test root). Their survival
        # proves the multi-identity-capable plugin has run, even if somebody
        # deleted the sentinel and route snapshots.
        card_root = self._dir.parent if self._dir.name == "identity" else self._dir
        try:
            return next(card_root.glob("*/card-sessions.json"), None) is not None
        except OSError:
            # Inability to establish that no evidence exists is itself
            # ambiguous; recovery must fail closed.
            return True

    def begin(
        self,
        *,
        token_count: int,
        enumerate_legacy_sessions,
    ) -> IdentityStartup:
        """Advance durable state to the point where listeners may start.

        *enumerate_legacy_sessions* is a zero-argument callable returning the
        ``(session_key, chat_type, chat_id)`` triples that already exist for the
        ``octo`` platform.  It is only consulted when a first-time legacy
        migration is genuinely allowed.
        """
        metadata = self.load_metadata()
        self.routes.load()

        if metadata is None and self.has_routing_evidence():
            raise IdentityStateError(
                "Octo identity-routing metadata is missing while route state "
                "exists; refusing to guess the previous identity"
            )

        if token_count <= 1:
            return self._begin_single(metadata)
        return self._begin_multi(metadata, enumerate_legacy_sessions)

    def _begin_single(
        self,
        metadata: IdentityMetadata | None,
    ) -> IdentityStartup:
        if metadata is None:
            # First run of an identity-routing-capable plugin.  Record the
            # sentinel now, before any inbound traffic, so a later multi-token
            # start can tell "was single" from "unknown".
            metadata = IdentityMetadata(phase=PHASE_SINGLE)
            self.save_metadata(metadata)
            return IdentityStartup(
                phase=PHASE_SINGLE,
                legacy_robot_id="",
                plan=(),
                sole_identity_mode=True,
            )
        if metadata.phase == PHASE_SINGLE:
            return IdentityStartup(
                phase=PHASE_SINGLE,
                legacy_robot_id=metadata.legacy_robot_id,
                plan=(),
                sole_identity_mode=True,
            )
        # A profile that already entered multi-identity routing keeps honouring
        # its persisted routes even when the configuration drops back to one
        # token.
        plan = (
            self.load_pending_plan(metadata)
            if metadata.phase == PHASE_PENDING
            else ()
        )
        return IdentityStartup(
            phase=metadata.phase,
            legacy_robot_id=metadata.legacy_robot_id,
            plan=plan,
            sole_identity_mode=False,
        )

    def _begin_multi(
        self,
        metadata: IdentityMetadata | None,
        enumerate_legacy_sessions,
    ) -> IdentityStartup:
        if metadata is None or not metadata.legacy_robot_id:
            raise IdentityStateError(
                "first multi-token startup requires one successful start with "
                "the existing OCTO_BOT_TOKEN alone; refusing to guess the "
                "legacy Octo identity"
            )
        if metadata.phase == PHASE_MIGRATED:
            return IdentityStartup(
                phase=PHASE_MIGRATED,
                legacy_robot_id=metadata.legacy_robot_id,
                plan=(),
                sole_identity_mode=False,
            )
        if metadata.phase == PHASE_PENDING:
            return IdentityStartup(
                phase=PHASE_PENDING,
                legacy_robot_id=metadata.legacy_robot_id,
                plan=self.load_pending_plan(metadata),
                sole_identity_mode=False,
            )

        # ``phase=single`` now always carries the stable identity recorded by a
        # successful one-token connection. Without it, token order is only a
        # guess and no legacy session may be assigned.
        legacy_robot_id = metadata.legacy_robot_id
        plan = self._freeze_plan(enumerate_legacy_sessions)
        self.save_plan(plan)
        self.save_metadata(
            IdentityMetadata(
                phase=PHASE_PENDING,
                legacy_robot_id=legacy_robot_id,
                plan_size=len(plan),
            )
        )
        logger.info(
            "octo: froze %d legacy Octo session route(s) for first multi-token start",
            len(plan),
        )
        return IdentityStartup(
            phase=PHASE_PENDING,
            legacy_robot_id=legacy_robot_id,
            plan=plan,
            sole_identity_mode=False,
        )


    @staticmethod
    def _freeze_plan(
        enumerate_legacy_sessions: Callable[[], Iterable[tuple[str, str, str]]],
    ) -> tuple[PlannedRoute, ...]:
        planned: dict[str, PlannedRoute] = {}
        for session_key, chat_type, chat_id in enumerate_legacy_sessions():
            if not isinstance(session_key, str) or not session_key:
                continue
            if chat_type not in _CHAT_TYPES:
                continue
            if not isinstance(chat_id, str) or not chat_id:
                continue
            planned[session_key] = PlannedRoute(
                session_key=session_key,
                chat_type=chat_type,
                chat_id=chat_id,
                channel_type=channel_type_for_planned_route(chat_type, chat_id),
            )
            if len(planned) > MAX_PERSISTED_ROUTES:
                raise IdentityStateError(
                    "Octo legacy migration plan exceeds route capacity"
                )
        return tuple(planned.values())
    def record_single_identity(self, robot_id: str) -> None:
        """Remember the write-once stable identity of a single-token profile."""
        metadata = self._metadata or IdentityMetadata(phase=PHASE_SINGLE)
        if metadata.phase != PHASE_SINGLE:
            return
        if metadata.legacy_robot_id:
            if metadata.legacy_robot_id != robot_id:
                raise IdentityStateError(
                    "configured Octo token no longer registers as this "
                    "profile's stable single identity"
                )
            return
        self.save_metadata(
            IdentityMetadata(phase=PHASE_SINGLE, legacy_robot_id=robot_id)
        )

    def protect_replaced_single_identity(
        self,
        *,
        previous_robot_id: str,
        enumerate_legacy_sessions,
    ) -> int:
        """Pin legacy sessions before a single configured token changes bot."""
        metadata = self._metadata or self.load_metadata()
        if (
            metadata is None
            or metadata.phase != PHASE_SINGLE
            or metadata.legacy_robot_id != previous_robot_id
        ):
            raise IdentityStateError(
                "single-identity replacement state changed during protection"
            )
        plan = self._freeze_plan(enumerate_legacy_sessions)
        if not self.routes.bind_plan(
            robot_id=previous_robot_id,
            plan=plan,
        ):
            raise IdentityStateError(
                "legacy routes conflict while protecting replaced identity"
            )
        if not self.routes.flush(force=True):
            raise OSError("could not persist replaced identity routes")
        self.save_metadata(
            IdentityMetadata(
                phase=PHASE_MIGRATED,
                legacy_robot_id=previous_robot_id,
                plan_size=len(plan),
            )
        )
        return len(plan)

    def apply_legacy_plan(
        self,
        *,
        robot_id: str,
        plan: tuple[PlannedRoute, ...],
    ) -> None:
        """Bind the frozen plan to the verified legacy primary and finish."""
        if not self.routes.bind_plan(robot_id=robot_id, plan=plan):
            raise IdentityStateError(
                "legacy migration conflicts with an existing identity route"
            )
        if not self.routes.flush(force=True):
            raise OSError("could not persist legacy identity routes")
        self.save_metadata(
            IdentityMetadata(
                phase=PHASE_MIGRATED,
                legacy_robot_id=robot_id,
                plan_size=len(plan),
            )
        )
        self.clear_plan()
        logger.info(
            "octo: legacy identity migration complete for %d route(s)", len(plan)
        )


# ── Durable card-session bindings, sharded per identity ───────────────────


class CardBindingStore:
    """Per-``robot_id`` durable card-action bindings.

    A deferred card action must still pass its original safety checks after a
    gateway restart or a token rotation, and only the identity that owns the
    binding may claim it.  Sharding by ``robot_id`` makes that structural: a
    runtime restores nothing but its own bindings.
    """

    FILE_NAME = "card-sessions.json"

    def __init__(self, *, robot_id: str, base_dir: Path | None = None) -> None:
        if not valid_robot_id(robot_id):
            raise ValueError("invalid Octo robot_id")
        self._robot_id = robot_id
        if base_dir is None:
            from hermes_constants import get_hermes_home

            base_dir = Path(get_hermes_home()) / "workspace" / "octo"
        self._path = (
            Path(base_dir) / robot_path_segment(robot_id) / self.FILE_NAME
        )
        self._lock = threading.Lock()
        self.corrupt = False

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[dict[str, Any]]:
        """Return records that are still inside their TTL for this identity."""
        with self._lock:
            status, body = read_state_document(self._path)
        if status == READ_MISSING:
            return []
        if status != READ_OK or not isinstance(body, dict):
            self.corrupt = True
            logger.error(
                "octo: card-session store for this identity is unusable (%s); "
                "historical card actions fail closed",
                status,
            )
            return []
        if body.get("robot_id") != self._robot_id:
            self.corrupt = True
            logger.error(
                "octo: card-session store identity mismatch; refusing to restore"
            )
            return []
        now = time.time()
        records: list[dict[str, Any]] = []
        for record in body.get("records") or ():
            if not isinstance(record, dict):
                continue
            expires_at = record.get("expires_at")
            if not isinstance(expires_at, (int, float)) or expires_at <= now:
                continue
            records.append(record)
        return records

    def save(self, records: list[dict[str, Any]]) -> None:
        """Persist claim state before the caller advances the event cursor."""
        with self._lock:
            write_state_document(
                self._path,
                {"robot_id": self._robot_id, "records": records},
            )
