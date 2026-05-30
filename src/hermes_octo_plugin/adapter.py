"""Octo (WuKongIM) platform adapter for Hermes Agent."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import os
import random
import re
import time
from collections import OrderedDict
from typing import Any
from urllib.parse import urlparse

import aiohttp
import websockets
import websockets.exceptions
from agent.redact import redact_sensitive_text as _redact_raw
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

from . import api
from .mention import (
    convert_content_for_llm,
    convert_structured_mentions,
    extract_mention_uids,
    parse_structured_mentions,
)
from .protocol import (
    PROTO_VERSION,
    PacketType,
    aes_decrypt,
    compute_shared_secret,
    decode_packet,
    derive_aes_key,
    encode_connect_packet,
    encode_ping_packet,
    encode_recvack_packet,
    generate_device_id,
    generate_keypair,
    try_unpack_one,
)
from .types import (
    BotMessage,
    BotRegisterResp,
    ChannelType,
    MessagePayload,
)
from .types import (
    MessageType as OctoMessageType,
)

logger = logging.getLogger(__name__)


def redact_log(s: str) -> str:
    # Always pass force=True: octo is a safety boundary that must never
    # leak secrets into error returns / logs regardless of the user's
    # global HERMES_REDACT_SECRETS preference. See agent/redact.py:317.
    return _redact_raw(s, force=True)


MAX_MESSAGE_LENGTH = 5000
RECONNECT_BASE_DELAY = 3.0
RECONNECT_MAX_DELAY = 60.0
HEARTBEAT_INTERVAL = 60
PING_MAX_RETRY = 3
# Env var names that hermes considers mandatory for the octo channel —
# surfaced both in plugin registration (required_env) and in the tool
# registration (requires_env). Single source of truth so the two stay in
# sync as new vars are added.
_REQUIRED_ENV: tuple[str, ...] = ("OCTO_API_URL", "OCTO_BOT_TOKEN")
GROUP_CACHE_EXPIRY_MS = 60 * 60 * 1000
NAME_CACHE_MAX_SIZE = 1000
DEFAULT_HISTORY_LIMIT = 20
# Minimum gap between two forced /bot/register?force_refresh=true calls.
# Without this, a server-side token rotation can stampede many reconnect
# attempts into refreshing the IM token simultaneously, which the server
# punishes with 429s / further kicks.
TOKEN_REFRESH_COOLDOWN_S = 60.0
# Extra random delay added on top of the exponential backoff when reconnecting
# after a kick / connect failure — prevents thundering-herd reconnect when
# multiple bots refresh tokens at the same wall-clock moment.
RECONNECT_STAGGER_MAX_S = 5.0
# Periodic cache cleanup: drop per-channel caches (group history, GROUP.md,
# chat_kind, group_cache_timestamps) for channels we haven't seen activity on
# for CACHE_MAX_AGE_S. Catches abandoned groups / DMs so long-running gateways
# don't accumulate state forever.
CACHE_MAX_AGE_S = 4 * 60 * 60
CACHE_CLEANUP_INTERVAL_S = 30 * 60
# Inbound file handling.
# Text files smaller than this get their content inlined into the LLM
# message body verbatim. Above this we still download large files to a
# temp path and tell the agent where they live — never silently drop.
FILE_INLINE_MAX_BYTES = 20 * 1024  # 20 KB — small enough not to blow up
FILE_DOWNLOAD_MAX_BYTES = 500 * 1024 * 1024  # 500 MB hard cap
FILE_TEMP_DIR = "/tmp/octo-files"
FILE_TEMP_RETENTION_S = 60 * 60  # 1 hour
# Extensions worth attempting to inline as text. Anything outside this set
# (.png, .zip, .pdf, ...) falls through to the regular "[文件: name]\nurl"
# placeholder — the LLM gets a URL but doesn't see content.
_TEXT_FILE_EXTS = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".csv", ".tsv", ".tab",
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt", ".swift",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cxx",
    ".rb", ".php", ".pl", ".lua", ".scala", ".clj", ".ex", ".exs",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat",
    ".html", ".htm", ".xml", ".svg", ".css", ".scss", ".less",
    ".sql", ".graphql", ".proto",
    ".env", ".gitignore", ".dockerignore",
    ".diff", ".patch",
})
# Inbound media (Image/GIF/Voice/Video) gets streamed to a local temp file
# before being handed to hermes-core so vision/audio pipelines don't hang on
# slow remote URLs. Capped at 20 MB — anything larger falls back to the
# remote URL with a log warning.
MEDIA_TEMP_DIR = "/tmp/octo-media"
MEDIA_DOWNLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
MEDIA_DOWNLOAD_TIMEOUT_S = 120.0
MEDIA_TEMP_RETENTION_S = 60 * 60  # 1 hour
# GROUP.md / THREAD.md disk cache. Survives gateway restarts so the first @
# after a restart doesn't pay the round-trip; the in-memory cache primes from
# disk during startup prefetch, and writes happen on every ensure / refresh.
# Path: ${HERMES_HOME}/workspace/octo/{owner_short}/groups/{group_no}/GROUP.md
#       + .../GROUP.meta.json   (or .../threads/{short_id}/THREAD.md)
GROUP_MD_DISK_ROOT_NAME = "workspace/octo"
# Cross-segment coalescing window. Hermes' streaming pipeline treats each
# LLM "segment" (text block between tool calls / api calls) as a separate
# finalize=True boundary, calling send() again for the next segment. On
# platforms that support in-place edits, each segment becomes a separate
# bubble — fine. On octo (no real edit API) each segment becomes a fresh
# message bubble, which means a single logical response gets split mid-
# markdown (e.g. ```bash fence opened in segment 1, body in segment 2 →
# broken code block in client). We buffer across segments and only flush
# after this many seconds of true idle, so a multi-segment response lands
# as one clean bubble. Long enough to absorb sub-second to 2-second tool
# gaps; short enough that the user doesn't notice the delay.
STREAM_FLUSH_DELAY_S = 3.0
# Hermes' streaming cursor is appended to every mid-stream frame to signal
# "still typing" on platforms that render it as a visual hint (Telegram,
# etc). Octo clients render it as a literal ▉ block, which looks like a
# rendering glitch. Strip it before any outbound API call so the bubble
# never shows the marker — final frames will have the clean text either
# way, and intermediate edits look cleaner without the trailing block.
# Mirrors gateway.config.DEFAULT_STREAMING_CURSOR = " ▉".
HERMES_STREAM_CURSOR = " ▉"
DEFAULT_HISTORY_PROMPT_TEMPLATE = (
    "[Group Chat History ({count} messages)]\n{messages}\n---\n"
)

# Group is any of these channel types
_GROUP_CHANNEL_TYPES = frozenset([ChannelType.Group, ChannelType.CommunityTopic])


# SSRF / injection defenses for OCTO_API_URL, OCTO_CDN_URL, and chat_id.
# OCTO_* URLs come from env/yaml and would otherwise let a tampered
# config point the bot's HTTP/WebSocket traffic (including OCTO_BOT_TOKEN
# in Authorization headers) at internal services (metadata endpoints,
# Redis, etc.). chat_id is interpolated into URL paths so anything outside
# the documented charset can break out of the path segment.
_OCTO_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_\-@:]{1,128}$")
# Path-segment safe id: stricter than chat_id (no @/:) because these are
# spliced into filesystem paths. Anything outside this set could escape the
# expected cache root (e.g. ``../../etc/passwd``).
_OCTO_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _validate_octo_path_segment(seg: str, kind: str) -> str:
    if not _OCTO_SAFE_ID_RE.fullmatch(seg):
        raise ValueError(f"Octo server returned malformed {kind}: {seg!r}")
    return seg


# Hostnames that resolve to cloud / container metadata services (token theft
# vectors when an attacker controls OCTO_API_URL / OCTO_CDN_URL).
_METADATA_HOSTS = frozenset({
    "169.254.169.254",                # AWS / GCP / Azure IMDS
    "fd00:ec2::254",                  # AWS IMDSv6
    "metadata.google.internal",       # GCP
    "metadata.goog",                  # GCP
    "metadata",                       # GCP short form
    "100.100.100.200",                # Aliyun
})


def _is_private_or_metadata_host(hostname: str) -> bool:
    """Return True if the host should be blocked as an SSRF target.

    Catches loopback, RFC1918, link-local, multicast, and well-known cloud
    metadata endpoints. Only literal IPs are inspected — hostnames are
    not DNS-resolved (TOCTOU + slows __init__), but suffixes commonly used
    for private services (``.local``, ``.internal``) and the explicit
    metadata host names are blocked.
    """
    if not hostname:
        return True
    lowered = hostname.lower()
    if lowered in _METADATA_HOSTS:
        return True
    if lowered.endswith((".local", ".internal", ".localhost")) or lowered == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(lowered.strip("[]"))
    except ValueError:
        return False  # Not a literal IP — let DNS layer / network stack decide
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_octo_url(url: str, name: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"{name} must be http(s): got {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError(f"{name} missing host: {url!r}")
    allow_private = os.getenv("OCTO_ALLOW_PRIVATE_HOSTS", "").lower() in ("1", "true", "yes")
    if not allow_private and _is_private_or_metadata_host(parsed.hostname or ""):
        raise ValueError(
            f"{name} points at private/loopback/metadata host {parsed.hostname!r} "
            f"(SSRF guard). Set OCTO_ALLOW_PRIVATE_HOSTS=true for dev/self-hosted setups."
        )
    return url


def check_octo_requirements() -> bool:
    """Are runtime deps importable? Hermes plugin discovery uses this as the
    gate for whether to register the platform at all — it must reflect
    deps only, not user configuration (that's ``_is_connected``'s job)."""
    try:
        import aiohttp  # noqa: F401
        import websockets  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import x25519  # noqa: F401
    except ImportError:
        return False
    return True


def _format_size(n: int | None) -> str:
    """Format a byte count as a short human label (B / KB / MB / GB)."""
    if n is None:
        return "?"
    units = ("B", "KB", "MB", "GB", "TB")
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}".replace(".0 ", " ")
        f /= 1024
    return f"{n} B"


_SPACE_PREFIX_RE = re.compile(r"^s(\d+)_(.+)$")


def _extract_base_uid(uid: str) -> str:
    """Strip the Octo Space prefix from a uid.

    Octo carries Space membership inline in the uid: ``s14_abc123`` means
    "user abc123 in Space 14". The same human shows up with different Space
    prefixes in different conversations, breaking direct uid→name lookups
    that were populated under one prefix.  Resolving against the base uid
    lets cross-space lookups still find a cached name.

    Bare uids (no ``s<digits>_`` prefix) are returned unchanged.
    """
    if not uid:
        return uid
    m = _SPACE_PREFIX_RE.match(uid)
    return m.group(2) if m else uid


def _extract_space_id(channel_id: str) -> str | None:
    """Extract spaceId from a channel_id starting with ``s{digits}_``.

    Handles both single-peer (``s14_uid``) and compound DM
    (``s14_uid1@s14_uid2``) shapes. Returns ``None`` when no Space prefix
    is detectable. Used to namespace DM session ids per-Space so the same
    user in two Spaces gets two distinct hermes sessions.
    """
    if not channel_id or not channel_id.startswith("s"):
        return None
    first_part = channel_id.split("@", 1)[0]
    if not first_part.startswith("s"):
        return None
    last_us = first_part.rfind("_")
    if last_us <= 1:
        return None
    candidate = first_part[1:last_us]
    return candidate if candidate.isdigit() else None


def _strip_emoji(s: str) -> str:
    """Remove common emoji ranges from *s*. Used for emoji-tolerant member
    matching: '陈皮皮🎀' and '陈皮皮' should resolve to the same uid even
    though only one is the canonical display name.

    Returns the stripped string trimmed of whitespace. If the result is
    empty (the whole string was emoji), returns "" so callers can skip
    adding a useless alias.
    """
    if not s:
        return ""
    # Coverage: most emoji families + dingbats + variation selectors +
    # playing card / mahjong / domino blocks. Trade-off is breadth, not
    # surgical accuracy — false positives only mean we add an alias that's
    # never queried.
    pattern = (
        "[\U0001F300-\U0001F9FF"   # most emoji (faces, symbols)
        "☀-➿"             # misc symbols + dingbats
        "︀-️"             # variation selectors
        "\U0001F000-\U0001F02F"     # mahjong / dominos
        "\U0001F0A0-\U0001F0FF]"    # playing cards
    )
    return re.sub(pattern, "", s).strip()


def _is_connected(cfg) -> bool:
    """Is the platform fully configured to connect? Both bot_token and
    api_url must be present — token alone isn't useful without an endpoint."""
    extra = getattr(cfg, "extra", None) or {}
    has_token = bool(extra.get("bot_token") or os.getenv("OCTO_BOT_TOKEN"))
    has_url = bool(extra.get("api_url") or os.getenv("OCTO_API_URL"))
    return has_token and has_url


def _env_enablement() -> dict | None:
    bot_token = os.getenv("OCTO_BOT_TOKEN")
    if not bot_token:
        return None
    result: dict = {"bot_token": bot_token}
    if api_url := os.getenv("OCTO_API_URL"):
        result["api_url"] = api_url
    if cdn_url := os.getenv("OCTO_CDN_URL"):
        result["cdn_url"] = cdn_url
    if home := os.getenv("OCTO_HOME_CHANNEL"):
        # gateway/config.py unpacks this into a HomeChannel dataclass when
        # synthesising the platform config from env.
        result["home_channel"] = {"chat_id": home}
    return result


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: str | None = None,
    media_files=None,
    force_document: bool = False,
) -> dict:
    """Out-of-process sender for cron jobs that run separately from the gateway.

    Opens an ephemeral aiohttp session, posts a single text message to the
    Octo bot API, then closes. Media/threads are not yet supported on this
    path (cron rarely needs them; live adapter handles richer payloads).
    """
    import aiohttp as _aiohttp

    from .api import send_message as _octo_send
    from .types import ChannelType as _ChannelType

    extra = getattr(pconfig, "extra", None) or {}
    api_url = extra.get("api_url") or os.getenv("OCTO_API_URL", "")
    bot_token = (
        extra.get("bot_token")
        or getattr(pconfig, "token", "")
        or os.getenv("OCTO_BOT_TOKEN", "")
    )
    if not api_url or not bot_token:
        return {"error": "OCTO_API_URL and OCTO_BOT_TOKEN must be configured"}

    try:
        _validate_octo_url(api_url, "OCTO_API_URL")
    except ValueError as e:
        return {"error": str(e)}

    if not _OCTO_CHAT_ID_RE.fullmatch(str(chat_id)):
        return {"error": f"invalid chat_id format: {chat_id!r}"}

    try:
        async with _aiohttp.ClientSession() as session:
            # Convert LLM-emitted @[uid:name] markers into wire format so the
            # client renders an @ pill. Same logic as OctoAdapter._send_normal;
            # cron-delivered messages benefit equally.
            from .mention import (
                convert_structured_mentions as _convert_sm,
            )
            from .mention import (
                parse_structured_mentions as _parse_sm,
            )
            send_content = message
            send_uids = None
            send_entities = None
            structured = _parse_sm(message)
            if structured:
                send_content, send_entities, send_uids = _convert_sm(message, structured)
            await _octo_send(
                session, api_url, bot_token,
                channel_id=chat_id,
                channel_type=(
                    _ChannelType.CommunityTopic
                    if "____" in str(chat_id)
                    else _ChannelType.Group
                ),
                content=send_content,
                mention_uids=send_uids,
                mention_entities=send_entities,
            )
        return {"success": True}
    except Exception as e:  # pragma: no cover — best-effort cron path
        # Use redact_log on the message text (force=True via our alias) so
        # secrets in aiohttp error messages can't leak. Do NOT pass exc_info=
        # because traceback frames may contain raw URLs / headers / token
        # bytes that bypass the redact pipeline.
        logger.error("[octo] standalone send failed: %s", redact_log(str(e)))
        return {"error": "Octo API send failed (see logs)"}


class LRUCache:
    def __init__(self, max_size: int = 1000) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


class OctoAdapter(BasePlatformAdapter):
    """
    Octo (WuKongIM) platform adapter for Hermes Agent.

    Connects via WuKongIM WebSocket binary protocol for message reception
    and HTTP API for message sending.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig) -> None:
        # Platform("octo") works via _missing_() dynamic enum creation
        super().__init__(config, Platform("octo"))

        extra = config.extra or {}
        self._api_url: str = extra.get("api_url") or os.getenv("OCTO_API_URL", "")
        self._bot_token: str = extra.get("bot_token") or os.getenv("OCTO_BOT_TOKEN", "")
        # Optional CDN base URL. When set, _build_media_url returns
        # <cdn_url>/<relative_path> instead of <api_url>/file/<relative_path>
        # — lets the agent fetch media from a public-read CDN without
        # routing every download through the bot API server.
        self._cdn_url: str = extra.get("cdn_url") or os.getenv("OCTO_CDN_URL", "")

        if self._api_url:
            _validate_octo_url(self._api_url, "OCTO_API_URL")
        if self._cdn_url:
            _validate_octo_url(self._cdn_url, "OCTO_CDN_URL")

        # Connection state
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._registration: BotRegisterResp | None = None

        # Crypto state (set after CONNACK)
        self._aes_key: str = ""
        self._aes_iv: str = ""
        self._dh_private_key: bytes | None = None
        self._server_version: int = 0

        # Tasks
        self._recv_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._cache_cleanup_task: asyncio.Task | None = None

        # Cache activity tracker: channel_id → last-touched monotonic seconds.
        # _cleanup_caches() uses this to age out stale per-channel state so
        # long-running gateways don't accumulate forever-pinned dicts for
        # groups / DMs the bot hasn't seen in a while.
        self._cache_activity: dict[str, float] = {}

        # Streaming buffers keyed by chat_id. Each value:
        #   {
        #     "message_id": str,             # synthetic id returned to consumer
        #     "channel_type": ChannelType,
        #     "segments": list[str],         # finalized text segments
        #     "current_segment": str,        # in-progress (cursor-frame) text
        #     "flush_task": asyncio.Task | None,
        #   }
        # Why this lives here (and not in stream_consumer): octo's wire
        # protocol has no real edit_in_place API, so every send() to its
        # /v1/bot/sendMessage spawns a fresh client bubble. Hermes' streaming
        # pipeline assumes the platform can edit, and splits a single
        # logical response into N send()s — one per LLM segment (text block
        # between api_calls / tool calls). Without coalescing, a response
        # whose markdown spans a segment boundary (e.g. ```bash opened in
        # segment 1, body in segment 2) renders as two bubbles with broken
        # formatting at the boundary. So we buffer EVERYTHING for this chat
        # and only flush after STREAM_FLUSH_DELAY_S of true idle. Hermes'
        # typing indicator (sent independently) gives the user feedback
        # while we wait.
        self._active_streams: dict[str, dict[str, Any]] = {}

        # Reconnection
        self._need_reconnect: bool = True
        self._reconnect_attempts: int = 0
        self._connected: bool = False
        # Dedup guard: prevents two concurrent _schedule_reconnect() coroutines
        # racing each other into self-kick storms (concurrent reconnects open
        # multiple WS sessions and the "one connection per uid+device_flag"
        # rule causes them to kick each other in a loop). Cleared
        # only when the reconnect attempt either succeeds (connect → set False
        # at the bottom of _schedule_reconnect) or fails (caught in the
        # try/except and respawned, but the old task is done at that point).
        self._reconnect_in_progress: bool = False
        # Token refresh cooldown: last time we hit register(force_refresh=true).
        # Bare reconnects don't rotate the IM token; only Kicked/Connect-failed
        # events trigger a forced refresh, and we throttle them.
        self._last_token_refresh: float = 0.0

        # Heartbeat
        self._ping_retry_count: int = 0

        # Sticky packet buffer
        self._temp_buffer: bytearray = bytearray()

        # Bot identity (populated after registration)
        self._robot_id: str = ""
        self._owner_uid: str = ""

        # All group_no this bot is currently a member of (used by
        # permission.parse_target to disambiguate bare IDs as group vs DM).
        # Populated by startup prefetch (P1-3); also self-heals from inbound
        # group messages.
        self._known_group_ids: set[str] = set()

        # Sender name resolution
        self._name_cache = LRUCache(max_size=NAME_CACHE_MAX_SIZE)
        self._uid_to_name: dict[str, str] = {}
        # Reverse index: base_uid (uid with Space prefix stripped) → name.
        # Used by _resolve_sender_name to answer cross-Space fallbacks in
        # O(1) instead of scanning _uid_to_name every miss. Written in lock-
        # step with _uid_to_name via _record_uid_name().
        self._base_uid_to_name: dict[str, str] = {}
        self._member_map: dict[str, str] = {}   # displayName → uid
        self._group_cache_timestamps: dict[str, int] = {}
        # Reverse index for shared-groups queries: uid → set of group_no the
        # bot has observed this uid as a member of. Updated whenever we
        # refresh a group's member roster (startup prefetch + on-demand
        # refresh). Used by the octo_management search-shared-groups action.
        self._user_group_index: dict[str, set[str]] = {}
        # Group display names, populated alongside the reverse index so
        # search-shared-groups results carry the group name not just group_no.
        self._group_names: dict[str, str] = {}

        # Group history: channel_id → list of {sender, body, mention, timestamp}
        self._group_histories: dict[str, list[dict[str, Any]]] = {}

        # Outbound channel-type memory: chat_id → ChannelType.
        # When hermes calls send(chat_id, ...) it doesn't know whether the
        # chat is a DM or a group, so we remember the type from inbound
        # messages and use it on outbound sends.
        self._chat_kind: dict[str, ChannelType] = {}

        # Streaming config
        self._stream_threshold: int = extra.get("stream_threshold", 500)

        # GROUP.md cache: channel_id → {content, version}
        self._group_md_cache: dict[str, dict[str, Any]] = {}
        self._group_md_checked: set[str] = set()

        # Config
        self._history_limit: int = extra.get("history_limit", DEFAULT_HISTORY_LIMIT)
        self._require_mention: bool = extra.get("require_mention", True)
        # When True, group @all (mention.all) does NOT trigger this bot.
        # Useful when many bots share a group — operators @all to notify
        # humans, not to fan-out to every assistant.
        self._ignore_mention_all: bool = bool(extra.get("ignore_mention_all", False))
        # Per-account override for the prompt template used to inject group
        # chat history. {messages} (JSON-encoded list) and {count} (int)
        # placeholders are substituted. Falls back to the module-level
        # default when not set.
        self._history_prompt_template: str = (
            extra.get("history_prompt_template")
            or DEFAULT_HISTORY_PROMPT_TEMPLATE
        )
        # Per-account heartbeat / reconnect tuning. Bots on flaky links can
        # raise ping_max_retry; bots in latency-sensitive deployments can
        # lower heartbeat_interval_s. Falls back to module defaults.
        self._heartbeat_interval_s: float = float(
            extra.get("heartbeat_interval_s", HEARTBEAT_INTERVAL)
        )
        self._ping_max_retry: int = int(
            extra.get("ping_max_retry", PING_MAX_RETRY)
        )

    @property
    def name(self) -> str:
        return "Octo"

    # ── Connection Lifecycle ──────────────────────────────────────────────

    async def connect(self) -> bool:
        if not self._api_url or not self._bot_token:
            logger.error("[%s] OCTO_API_URL and OCTO_BOT_TOKEN must be set", self.name)
            return False

        self._http_session = aiohttp.ClientSession()
        self._need_reconnect = True

        try:
            return await self._do_connect()
        except Exception as e:
            logger.error("[%s] Connection failed: %s", self.name, e)
            return False

    async def _do_connect(self) -> bool:
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()

        try:
            # force_refresh decision: only worth burning a token rotation if
            # this is a retry AND we haven't refreshed too recently. Cooldown
            # protects the server from refresh storms when many bots disconnect
            # at the same moment (e.g. after a server restart).
            now = time.monotonic()
            should_force_refresh = (
                self._reconnect_attempts > 0
                and (now - self._last_token_refresh) > TOKEN_REFRESH_COOLDOWN_S
            )
            if should_force_refresh:
                self._last_token_refresh = now
                logger.info("[%s] Forcing IM token refresh (attempt %d)",
                            self.name, self._reconnect_attempts)
            elif self._reconnect_attempts > 0:
                logger.info(
                    "[%s] Skipping token refresh (in cooldown, %.0fs since last)",
                    self.name, now - self._last_token_refresh,
                )
            self._registration = await api.register_bot(
                self._http_session,
                self._api_url,
                self._bot_token,
                force_refresh=should_force_refresh,
            )
            self._robot_id = self._registration.robot_id
            self._owner_uid = self._registration.owner_uid or ""
            logger.info("[%s] Bot registered: robot_id=%s owner=%s",
                        self.name, self._robot_id, self._owner_uid or "<none>")
        except Exception as e:
            logger.error("[%s] Bot registration failed: %s", self.name, redact_log(str(e)))
            raise

        ws_url = self._registration.ws_url
        try:
            self._ws = await websockets.connect(
                ws_url,
                max_size=8 * 1024 * 1024,   # 8 MiB cap — defends against malformed/malicious frames
                ping_interval=20,
                ping_timeout=20,
            )
        except Exception as e:
            logger.error("[%s] WebSocket connection failed: %s", self.name, redact_log(str(e)))
            raise

        self._temp_buffer = bytearray()
        priv_key, pub_key = generate_keypair()
        self._dh_private_key = priv_key
        pub_key_b64 = base64.b64encode(pub_key).decode("ascii")

        device_id = generate_device_id() + "W"
        connect_packet = encode_connect_packet(
            version=PROTO_VERSION,
            device_flag=0,
            device_id=device_id,
            uid=self._registration.robot_id,
            token=self._registration.im_token,
            client_timestamp=int(time.time() * 1000),
            client_key=pub_key_b64,
        )
        await self._ws.send(connect_packet)

        connack_success = False
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        except TimeoutError:
            logger.error("[%s] Timeout waiting for CONNACK", self.name)
            await self._ws.close()
            raise RuntimeError("CONNACK timeout") from None

        data = raw if isinstance(raw, bytes) else raw.encode("latin-1")
        self._temp_buffer.extend(data)

        while self._temp_buffer:
            frame, self._temp_buffer = try_unpack_one(self._temp_buffer)
            if frame is None:
                break

            pkt_type, result = decode_packet(frame)
            if pkt_type == PacketType.CONNACK:
                if result.reason_code == 1:
                    server_pub_key = base64.b64decode(result.server_key)
                    shared_secret = compute_shared_secret(self._dh_private_key, server_pub_key)
                    self._aes_key = derive_aes_key(shared_secret)
                    salt = result.salt or ""
                    # AES-CBC requires a 16-byte IV. WuKongIM v4 CONNACK salt
                    # is in practice always ≥ 16 chars, but if a future server
                    # ever shortens it the downstream aes_decrypt would raise
                    # ValueError mid-frame. Right-pad with NUL so the IV is
                    # always a stable 16 bytes.
                    if len(salt) >= 16:
                        self._aes_iv = salt[:16]
                    else:
                        logger.warning(
                            "[%s] CONNACK salt shorter than 16 bytes (len=%d); padding IV",
                            self.name, len(salt),
                        )
                        self._aes_iv = salt.ljust(16, "\x00")
                    self._server_version = result.server_version
                    self._connected = True
                    self._ping_retry_count = 0
                    self._reconnect_attempts = 0
                    connack_success = True
                    logger.info(
                        "[%s] Connected (server_version=%d)",
                        self.name, self._server_version,
                    )
                elif result.reason_code == 0:
                    logger.error("[%s] Kicked by server", self.name)
                    self._need_reconnect = False
                    raise RuntimeError("Kicked by server")
                else:
                    logger.error(
                        "[%s] Connect failed: reasonCode=%d",
                        self.name, result.reason_code,
                    )
                    raise RuntimeError(f"Connect failed: reasonCode={result.reason_code}")

        if not connack_success:
            raise RuntimeError("CONNACK not received")

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._recv_task = asyncio.create_task(self._receive_loop())
        # Only one cleanup loop per adapter lifetime; survives reconnects.
        if self._cache_cleanup_task is None or self._cache_cleanup_task.done():
            self._cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        # Warm caches in the background — fetch_bot_groups + GROUP.md +
        # member list — so the first inbound @ doesn't pay the round-trip.
        # Fire-and-forget; failures are logged but never block connect().
        asyncio.create_task(self._prefetch_groups_and_members())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._need_reconnect = False
        self._connected = False

        # Close any in-flight outbound streams before we tear down the http
        # session — otherwise the octo server keeps the bubble in a
        # "streaming" state indefinitely.
        for chat_id in list(self._active_streams.keys()):
            try:
                await self._close_active_stream(chat_id)
            except Exception:
                pass

        for task in (self._heartbeat_task, self._recv_task, self._cache_cleanup_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._heartbeat_task = None
        self._recv_task = None
        self._cache_cleanup_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        self._mark_disconnected()
        logger.info("[%s] Disconnected", self.name)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_media_url(self, rel_url: str | None) -> str | None:
        """Build a full media URL from a relative storage path.

        Routing priority: full http(s) URL → returned verbatim; otherwise
        prepend ``cdn_url`` when configured (public-read CDN, no auth),
        else fall back to ``<api_url>/file/<path>`` which routes through
        the bot API server.
        """
        if not rel_url:
            return None
        if rel_url.startswith("http"):
            return rel_url
        path = rel_url
        for prefix in ("file/preview/", "file/"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        if getattr(self, "_cdn_url", ""):
            return f"{self._cdn_url.rstrip('/')}/{path}"
        base = (self._api_url or "").rstrip("/")
        return f"{base}/file/{path}"

    # ── Sender Name Resolution ────────────────────────────────────────────

    def _record_uid_name(self, uid: str, name: str, *, overwrite: bool = True) -> None:
        """Store uid → name and keep the base-uid reverse index in sync.

        ``overwrite=False`` mirrors ``dict.setdefault`` semantics: only writes
        when ``uid`` isn't already mapped. The reverse index always learns the
        latest name for a given base_uid (last writer wins), since cross-Space
        fallback only needs *some* display name for the user.
        """
        if not uid or not name:
            return
        if overwrite or uid not in self._uid_to_name:
            self._uid_to_name[uid] = name
        base = _extract_base_uid(uid)
        if not base:
            return
        # Bare uid (no Space prefix): already recorded in _uid_to_name above;
        # cross-Space fallback only consults the reverse index when base != uid.
        if base == uid:
            return
        self._base_uid_to_name[base] = name

    async def _resolve_sender_name(self, uid: str) -> str:
        name = self._uid_to_name.get(uid)
        if name:
            return name
        # I+J: cross-space fallback. The same user appears with different
        # Space prefixes (s14_xxx, s27_xxx) in different conversations; if a
        # direct hit misses, look up the base form in the reverse index.
        base = _extract_base_uid(uid)
        if base != uid:
            base_name = self._base_uid_to_name.get(base) or self._uid_to_name.get(base)
            if base_name:
                return base_name
        cached = self._name_cache.get(uid)
        if cached is not None:
            return cached if cached else uid
        if self._http_session:
            info = await api.fetch_user_info(
                self._http_session, self._api_url, self._bot_token, uid
            )
            if info and info.get("name"):
                self._name_cache.set(uid, info["name"])
                self._record_uid_name(uid, info["name"])
                return info["name"]
            else:
                self._name_cache.set(uid, "")
        return uid

    def _update_user_group_index(self, group_no: str, members) -> None:
        """Sync the uid → set[group_no] reverse index with a fresh roster.

        Drops the bot's own uid before recording — the bot is technically a
        member of every group it watches, but it's never a useful answer to
        "groups I share with the bot".

        Idempotent: removes stale (uid, group_no) pairs that aren't in the
        fresh roster, so a member kicked from a group no longer shows up
        in shared-groups results after the next refresh.
        """
        if not group_no:
            return
        # Compute fresh uid set
        fresh_uids: set[str] = set()
        for m in members:
            if not getattr(m, "uid", None):
                continue
            if m.uid == self._robot_id:
                continue
            fresh_uids.add(m.uid)

        # Add fresh entries
        for uid in fresh_uids:
            self._user_group_index.setdefault(uid, set()).add(group_no)

        # Remove stale entries: scan all uids known to be in this group and
        # drop those not in the fresh set.
        for uid, groups in list(self._user_group_index.items()):
            if group_no in groups and uid not in fresh_uids:
                groups.discard(group_no)
                if not groups:
                    self._user_group_index.pop(uid, None)

    def find_shared_groups(self, uid: str) -> list[dict[str, Any]]:
        """Return the list of groups *uid* and this bot are both in.

        Looked up from the reverse index populated by prefetch and
        per-message refresh. Returns ``[]`` when the uid is unknown or
        shares no group — callers should treat that as "nothing to show"
        rather than "lookup failed".
        """
        if not uid:
            return []
        groups = self._user_group_index.get(uid)
        if not groups:
            return []
        return [
            {"group_no": gid, "name": self._group_names.get(gid, gid)}
            for gid in sorted(groups)
        ]

    async def _refresh_group_member_cache(self, group_no: str, force: bool = False) -> bool:
        if not self._http_session:
            return False
        now = int(time.time() * 1000)
        last_fetched = self._group_cache_timestamps.get(group_no, 0)
        if not force and (now - last_fetched) <= GROUP_CACHE_EXPIRY_MS and last_fetched > 0:
            return False
        try:
            members = await api.get_group_members(
                self._http_session, self._api_url, self._bot_token, group_no
            )
            if members:
                for m in members:
                    if m.name and m.uid:
                        self._member_map[m.name] = m.uid
                        # H: emoji-tolerant alias — store the de-emoji'd form
                        # too so @陈皮皮 matches a member named 陈皮皮🎀.
                        stripped = _strip_emoji(m.name)
                        if stripped and stripped != m.name:
                            self._member_map.setdefault(stripped, m.uid)
                        self._record_uid_name(m.uid, m.name)
                        self._name_cache.set(m.uid, m.name)
                self._update_user_group_index(group_no, members)
                self._group_cache_timestamps[group_no] = now
                logger.info("[%s] Group member cache refreshed: %s (%d members)",
                            self.name, group_no, len(members))
                return True
            else:
                self._group_cache_timestamps[group_no] = now - GROUP_CACHE_EXPIRY_MS + 30000
                return False
        except Exception as e:
            logger.error("[%s] Group member cache refresh failed: %s", self.name, e)
            self._group_cache_timestamps[group_no] = now - GROUP_CACHE_EXPIRY_MS + 30000
            return False

    # ── Group History ─────────────────────────────────────────────────────

    def _record_history_entry(
        self, channel_id: str, from_uid: str, body: str, mention: Any = None
    ) -> None:
        """Record a non-@ message in the group history buffer."""
        if channel_id not in self._group_histories:
            self._group_histories[channel_id] = []
        entries = self._group_histories[channel_id]
        entries.append({
            "sender": from_uid,
            "body": body,
            "mention": mention,
            "timestamp": int(time.time() * 1000),
        })
        while len(entries) > self._history_limit:
            entries.pop(0)

    def _resolve_api_message_placeholder(
        self, msg_type: int | None, name: str | None = None
    ) -> str:
        """Return a text placeholder for non-text API history messages."""
        t = OctoMessageType
        mapping = {
            t.Image: "[图片]",
            t.GIF: "[GIF]",
            t.Voice: "[语音消息]",
            t.Video: "[视频]",
            t.Location: "[位置信息]",
            t.Card: "[名片]",
            t.MultipleForward: "[合并转发]",
        }
        if msg_type == t.File:
            return f"[文件: {name or '未知文件'}]"
        return mapping.get(msg_type, "[消息]") if msg_type is not None else "[消息]"

    def _build_member_list_prefix(self) -> str:
        """Build the [Group Members] prefix that tells the LLM who's in the
        room and how to @ them via the structured ``@[uid:name]`` form.

        ≤10 members → list every name(uid) explicitly, with a one-line teach
        on how to mention. >10 members → just announce the count and point
        at the octo_management tool — listing everyone would blow up the
        prompt and the LLM rarely needs the full set up-front.
        """
        size = len(self._uid_to_name)
        if size == 0:
            return ""
        if size <= 10:
            members = list(self._uid_to_name.items())  # (uid, name)
            lines = "\n".join(f"  {name} ({uid})" for uid, name in members)
            example_uid, example_name = members[0]
            return (
                f"[Group Members]\n{lines}\n\n"
                f"When mentioning a group member, use the format "
                f"@[uid:displayName] (e.g. @[{example_uid}:{example_name}]). "
                "I will convert it to the correct format before sending."
            )
        return (
            f"[Group Info] This group has {size} members. Use the "
            "octo_management tool (action=group-members) to look up member "
            "info when needed. When mentioning a group member, use the "
            "format @[uid:displayName]."
        )

    async def _build_history_context(self, channel_id: str, bot_uid: str) -> str:
        """
        Build history context string for injection on @mention.

        Uses in-memory cache first. Falls back to API if cache is insufficient
        (less than half of history_limit).
        """
        entries = list(self._group_histories.get(channel_id, []))
        half_limit = max(1, self._history_limit // 2)

        if len(entries) < half_limit and self._http_session:
            logger.info("[%s] [HISTORY] Cache insufficient (%d), fetching from API...",
                        self.name, len(entries))
            try:
                fetch_limit = min(self._history_limit, 100)
                api_messages = await api.get_channel_messages(
                    self._http_session,
                    self._api_url,
                    self._bot_token,
                    channel_id=channel_id,
                    channel_type=ChannelType.Group,
                    limit=fetch_limit,
                )
                api_entries = []
                for m in api_messages:
                    if m.get("from_uid") == bot_uid:
                        continue
                    msg_type = m.get("type")
                    body = m.get("content") or self._resolve_api_message_placeholder(
                        msg_type, m.get("name")
                    )
                    # For media types, resolve and append full URL
                    if msg_type in (
                        OctoMessageType.Image, OctoMessageType.File,
                        OctoMessageType.Voice, OctoMessageType.Video,
                    ) and m.get("url"):
                        full_url = self._build_media_url(m["url"])
                        if full_url:
                            body = f"{body}\n{full_url}".strip()
                    api_entries.append({
                        "sender": m.get("from_uid", "unknown"),
                        "body": body,
                        "mention": m.get("payload", {}).get("mention"),
                        "timestamp": m.get("timestamp", 0),
                    })
                if api_entries:
                    entries = api_entries[-self._history_limit:]
                    logger.info("[%s] [HISTORY] Fetched %d from API for %s",
                                self.name, len(entries), channel_id)
            except Exception as e:
                logger.error("[%s] [HISTORY] API fetch failed: %s", self.name, e)

        if not entries:
            return ""

        # Apply sliding window
        entries = entries[-self._history_limit:]
        formatted = []
        for e in entries:
            sender_uid = e.get("sender", "unknown")
            sender_name = self._uid_to_name.get(sender_uid, sender_uid)
            sender_label = (
                f"{sender_name}({sender_uid})" if sender_name != sender_uid else sender_uid
            )
            body = e.get("body", "")
            # Convert mentions to @[uid:name] format for LLM
            mention = e.get("mention")
            if mention:
                body = convert_content_for_llm(body, mention, dict(self._member_map))
            formatted.append({"sender": sender_label, "body": body})

        messages_json = json.dumps(formatted, ensure_ascii=False, indent=2)
        return self._history_prompt_template.format(
            messages=messages_json, count=len(formatted)
        )

    # ── GROUP.md / THREAD.md disk persistence ─────────────────────────────

    def _md_dir(self, key: str) -> os.PathLike[str] | None:
        """Compute the on-disk directory for a GROUP.md / THREAD.md cache
        entry. *key* is either ``<group_no>`` or ``<group_no>____<short_id>``.

        Returns ``None`` when HERMES_HOME isn't resolvable (e.g. the adapter
        is being constructed outside a hermes context — tests), or when the
        adapter hasn't been fully initialised yet (object.__new__ in unit
        tests skips __init__).
        """
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        except Exception:
            return None
        from pathlib import Path
        # Owner namespace so multi-bot-on-different-tenants doesn't collide.
        try:
            owner_short = _validate_octo_path_segment(
                (getattr(self, "_owner_uid", None) or "unknown")[:16],
                "owner_uid",
            )
            if "____" in key:
                group_no, short_id = key.split("____", 1)
                group_no = _validate_octo_path_segment(group_no, "group_no")
                short_id = _validate_octo_path_segment(short_id, "short_id")
                return (
                    Path(home) / GROUP_MD_DISK_ROOT_NAME / owner_short
                    / "groups" / group_no / "threads" / short_id
                )
            key = _validate_octo_path_segment(key, "group_key")
            return Path(home) / GROUP_MD_DISK_ROOT_NAME / owner_short / "groups" / key
        except ValueError as e:
            logger.warning("[%s] refusing MD disk path: %s", self.name, e)
            return None

    def _write_md_to_disk(self, key: str, content: str, version: int) -> None:
        """Persist a single GROUP.md / THREAD.md entry. Errors are logged
        and swallowed — the in-memory cache is the source of truth, disk
        is an optimisation for cold start."""
        d = self._md_dir(key)
        if d is None:
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
            md_path = d / ("THREAD.md" if "____" in key else "GROUP.md")
            meta_path = d / ("THREAD.meta.json" if "____" in key else "GROUP.meta.json")
            md_path.write_text(content, encoding="utf-8")
            meta = {
                "version": version,
                "fetched_at": time.time(),
                "owner_uid": self._owner_uid,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except Exception as e:
            logger.debug("[%s] MD disk write failed for %s: %s", self.name, key, e)

    def _delete_md_from_disk(self, key: str) -> None:
        d = self._md_dir(key)
        if d is None:
            return
        try:
            md_path = d / ("THREAD.md" if "____" in key else "GROUP.md")
            meta_path = d / ("THREAD.meta.json" if "____" in key else "GROUP.meta.json")
            if md_path.exists():
                md_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
        except Exception as e:
            logger.debug("[%s] MD disk delete failed for %s: %s", self.name, key, e)

    def _read_md_from_disk(self, key: str) -> dict[str, Any] | None:
        """Read a cached MD entry from disk. Returns ``{content, version}``
        or ``None`` if absent / unreadable."""
        d = self._md_dir(key)
        if d is None or not d.exists():
            return None
        try:
            md_path = d / ("THREAD.md" if "____" in key else "GROUP.md")
            meta_path = d / ("THREAD.meta.json" if "____" in key else "GROUP.meta.json")
            if not md_path.exists():
                return None
            content = md_path.read_text(encoding="utf-8")
            version = 0
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    version = int(meta.get("version", 0))
                except Exception:
                    pass
            return {"content": content, "version": version}
        except Exception as e:
            logger.debug("[%s] MD disk read failed for %s: %s", self.name, key, e)
            return None

    def _hydrate_md_cache_from_disk(self, key: str) -> bool:
        """Prime the in-memory cache from disk if available. Returns True if
        a cache entry was loaded. Used by prefetch to skip API on cold start
        when the local cache is already fresh; the version is then validated
        on the next 'updated' event."""
        cached = self._read_md_from_disk(key)
        if not cached or not cached.get("content"):
            return False
        self._group_md_cache[key] = cached
        self._group_md_checked.add(key)
        return True

    # ── GROUP.md / THREAD.md ──────────────────────────────────────────────

    @staticmethod
    def _split_thread_channel_id(channel_id: str) -> tuple[str, str | None]:
        """Parse a channel_id into (parent_group_no, short_id_or_None).

        For group channels the parent_group_no is the channel_id itself and
        short_id is None. For thread channels (CommunityTopic) the format is
        ``<group_no>____<short_id>``.
        """
        if "____" in channel_id:
            parent, _, short = channel_id.partition("____")
            return parent, (short or None)
        return channel_id, None

    async def _ensure_group_md(self, group_no: str) -> None:
        """Fetch+cache GROUP.md (keyed on ``group_no``) once per session.

        Callers must pass the *parent* group_no, never a thread channel_id —
        the API endpoint /v1/bot/groups/{group_no}/md only knows group_no.
        """
        if group_no in self._group_md_checked:
            return
        self._group_md_checked.add(group_no)
        if not self._http_session:
            return
        try:
            md_data = await api.get_group_md(
                self._http_session, self._api_url, self._bot_token, group_no
            )
            if md_data and md_data.get("content"):
                self._group_md_cache[group_no] = {
                    "content": md_data["content"],
                    "version": md_data.get("version", 0),
                }
                self._write_md_to_disk(group_no, md_data["content"], md_data.get("version", 0))
                logger.info("[%s] GROUP.md cached for %s (v%d)",
                            self.name, group_no, md_data.get("version", 0))
        except Exception as e:
            logger.debug("[%s] GROUP.md fetch failed for %s: %s", self.name, group_no, e)

    async def _ensure_thread_md(self, group_no: str, short_id: str) -> None:
        """Fetch+cache THREAD.md (keyed on the composite ``group_no____short_id``).

        THREAD.md is sub-channel scope, so it co-exists with the parent
        GROUP.md in the same _group_md_cache dict, distinguished by the
        composite key.
        """
        key = f"{group_no}____{short_id}"
        if key in self._group_md_checked:
            return
        self._group_md_checked.add(key)
        if not self._http_session:
            return
        try:
            md_data = await api.get_thread_md(
                self._http_session, self._api_url, self._bot_token,
                group_no=group_no, short_id=short_id,
            )
            if md_data and md_data.get("content"):
                self._group_md_cache[key] = {
                    "content": md_data["content"],
                    "version": md_data.get("version", 0),
                }
                self._write_md_to_disk(key, md_data["content"], md_data.get("version", 0))
                logger.info("[%s] THREAD.md cached for %s (v%d)",
                            self.name, key, md_data.get("version", 0))
        except Exception as e:
            logger.debug("[%s] THREAD.md fetch failed for %s: %s", self.name, key, e)

    def _handle_group_md_event(self, channel_id: str, event_type: str) -> None:
        """Mark GROUP.md or THREAD.md stale on update/delete events.

        For thread-scoped MD the server sends the composite channel_id
        (``<group_no>____<short_id>``); for group-scoped MD it sends the
        bare group_no. We just use the channel_id as the cache key — the
        ensure_* / refresh_* methods write under the same key so an event
        for a thread invalidates only that thread's MD, not the parent's.
        """
        if event_type == "group_md_deleted":
            self._group_md_cache.pop(channel_id, None)
            self._group_md_checked.discard(channel_id)
            self._delete_md_from_disk(channel_id)
        elif event_type == "group_md_updated":
            # Force re-fetch on next message. The actual refresh task is
            # scheduled by the caller (which has access to the running loop /
            # http session) so this method stays sync-safe for unit tests.
            self._group_md_checked.discard(channel_id)

    async def _refresh_group_md(self, channel_id: str) -> None:
        """Re-fetch MD after an update event. Routes to the right API based
        on whether the channel_id encodes a thread."""
        if not self._http_session:
            return
        parent, short_id = self._split_thread_channel_id(channel_id)
        try:
            if short_id is None:
                md_data = await api.get_group_md(
                    self._http_session, self._api_url, self._bot_token, parent
                )
            else:
                md_data = await api.get_thread_md(
                    self._http_session, self._api_url, self._bot_token,
                    group_no=parent, short_id=short_id,
                )
            if md_data and md_data.get("content"):
                self._group_md_cache[channel_id] = {
                    "content": md_data["content"],
                    "version": md_data.get("version", 0),
                }
                self._write_md_to_disk(channel_id, md_data["content"], md_data.get("version", 0))
            else:
                self._group_md_cache.pop(channel_id, None)
                self._delete_md_from_disk(channel_id)
        except Exception as e:
            logger.debug("[%s] MD refresh failed for %s: %s", self.name, channel_id, e)

    # ── Message Reception ─────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        try:
            while self._connected and self._ws:
                try:
                    raw = await self._ws.recv()
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning("[%s] WebSocket closed: %s", self.name, redact_log(str(e)))
                    break
                except Exception as e:
                    logger.error("[%s] WebSocket recv error: %s", self.name, redact_log(str(e)))
                    break

                data = raw if isinstance(raw, bytes) else raw.encode("latin-1")
                self._temp_buffer.extend(data)

                # Process all complete frames; tolerate per-frame errors so a
                # single bad message doesn't tear down the WebSocket.
                while self._temp_buffer:
                    try:
                        frame, self._temp_buffer = try_unpack_one(self._temp_buffer)
                    except Exception as e:
                        logger.error("[%s] Frame unpack error, resetting buffer: %s",
                                     self.name, redact_log(str(e)))
                        self._temp_buffer = bytearray()
                        break
                    if frame is None:
                        break
                    try:
                        await self._handle_frame(frame)
                    except Exception as e:
                        logger.error("[%s] Frame handle error (skipping): %s",
                                     self.name, redact_log(str(e)))
        except asyncio.CancelledError:
            return
        finally:
            self._connected = False
            if self._need_reconnect:
                logger.warning("[%s] Receive loop exited — scheduling reconnect", self.name)
                asyncio.create_task(self._schedule_reconnect())

    async def _handle_frame(self, frame: bytes) -> None:
        pkt_type, result = decode_packet(frame)
        if pkt_type == PacketType.PONG:
            self._ping_retry_count = 0
        elif pkt_type == PacketType.RECV:
            await self._handle_recv(result)
        elif pkt_type == PacketType.DISCONNECT:
            # Server kicked us — usually because another client with the same
            # uid+device_flag connected (WuKongIM "kicked by another login").
            # Keep _need_reconnect=True so the bot recovers automatically once
            # the competing connection goes away.
            logger.warning("[%s] Server sent DISCONNECT — will attempt reconnect", self.name)
            self._connected = False
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass

    async def _handle_recv(self, recv: Any) -> None:
        # Send RECVACK immediately
        if self._ws:
            try:
                ack = encode_recvack_packet(recv.message_id, recv.message_seq)
                await self._ws.send(ack)
            except Exception as e:
                logger.debug("[%s] RECVACK failed: %s", self.name, redact_log(str(e)))

        # Decrypt payload
        try:
            decrypted = aes_decrypt(recv.encrypted_payload, self._aes_key, self._aes_iv)
            payload_dict = json.loads(decrypted.decode("utf-8"))
        except Exception as e:
            logger.debug("[%s] Payload decrypt/parse error: %s", self.name, redact_log(str(e)))
            return

        payload = MessagePayload.from_dict(payload_dict)

        msg = BotMessage(
            message_id=recv.message_id,
            message_seq=recv.message_seq,
            from_uid=recv.from_uid,
            channel_id=recv.channel_id,
            channel_type=recv.channel_type,
            timestamp=recv.timestamp,
            payload=payload,
        )

        # [DIAG] dump raw recv to find why some DMs early-return at empty content
        logger.debug(
            "[%s] [DIAG recv] from_uid=%r channel_id=%r channel_type=%r "
            "payload.type=%r content=%r url=%r name=%r event=%r extra_keys=%r",
            self.name,
            recv.from_uid,
            recv.channel_id,
            recv.channel_type,
            payload.type,
            payload.content,
            payload.url,
            payload.name,
            payload.event,
            list(payload_dict.keys()),
        )

        # Skip self-messages
        if msg.from_uid == self._robot_id:
            return

        is_group = msg.channel_type in _GROUP_CHANNEL_TYPES

        # Handle GROUP.md events — don't pass to LLM
        event_type = payload.event.get("type") if isinstance(payload.event, dict) else None
        if event_type in ("group_md_updated", "group_md_deleted") and msg.channel_id:
            self._handle_group_md_event(msg.channel_id, event_type)
            if event_type == "group_md_updated" and self._http_session:
                asyncio.create_task(self._refresh_group_md(msg.channel_id))
            return

        channel_id = msg.channel_id if is_group else msg.from_uid
        channel_type_enum = (
            (msg.channel_type or ChannelType.Group) if is_group else ChannelType.DM
        )

        # Remember the chat kind so outbound send() can pick the right
        # channel_type without depending on hermes-provided metadata.
        self._chat_kind[channel_id] = channel_type_enum

        # Self-heal known_group_ids from inbound traffic so cross-channel
        # `read` can disambiguate bare ids even without the startup prefetch
        # having completed (or having missed a newly-joined group).
        if is_group and msg.channel_id:
            parent = msg.channel_id.split("____", 1)[0]
            if parent:
                self._known_group_ids.add(parent)

        # Touch cache activity so the cleanup task doesn't evict an active
        # channel mid-conversation.
        self._touch_cache(channel_id)
        if is_group and msg.channel_id and msg.channel_id != channel_id:
            self._touch_cache(msg.channel_id)

        # Send read receipt (fire-and-forget)
        if self._http_session:
            asyncio.create_task(self._send_read_receipt_safe(
                channel_id, channel_type_enum,
                [msg.message_id] if msg.message_id else [],
            ))

        # Resolve content
        content = self._resolve_content(payload)
        if not content:
            return

        # File messages: try to inline text content or download to local temp.
        # Replaces the bare "[文件: name]\n<url>" with either the file's
        # body (small text) or a local path the agent can read (anything else).
        # Failures fall back to the original placeholder gracefully.
        if payload.type == OctoMessageType.File:
            file_url = self._build_media_url(payload.url)
            file_name = payload.name or "未知文件"
            known_size = payload.extra.get("size") if payload.extra else None
            if file_url:
                resolved = await self._resolve_inbound_file(
                    file_url, file_name,
                    known_size if isinstance(known_size, int) else None,
                )
                if resolved:
                    content = resolved

        # Refresh group member cache
        if is_group and msg.channel_id:
            await self._refresh_group_member_cache(msg.channel_id)

        # ── Mention detection ──
        is_mentioned = False
        if is_group and payload.mention:
            mention_uids = extract_mention_uids(payload.mention)
            mention_all = getattr(payload.mention, "all", False)
            # ignore_mention_all: treat @all as if it wasn't there. Lets
            # operators @all humans without fan-out to every bot.
            mention_all_effective = bool(mention_all) and not self._ignore_mention_all
            is_mentioned = (self._robot_id in mention_uids) or mention_all_effective

        # Defensive fallback (G): some senders (older clients, bot-to-bot)
        # don't populate payload.mention. If the message body contains
        # "@<BotName>" with a CJK-aware word boundary, treat it as an
        # explicit @ of this bot. Conservative regex — we'd rather miss
        # a corner case than activate the bot on false positives like
        # "你好@BotName" (no space before @).
        if (
            is_group
            and not is_mentioned
            and payload.type == OctoMessageType.Text
            and content
        ):
            bot_name = self._uid_to_name.get(self._robot_id, "")
            if bot_name.strip():
                bot_name_re = re.escape(bot_name)
                pat = (
                    r"(?:^|(?<=[^\w一-鿿぀-ヿ가-힯À-ɏ]))"
                    rf"@{bot_name_re}"
                    r"(?![\w一-鿿぀-ヿ가-힯À-ɏ.\-])"
                )
                if re.search(pat, content):
                    is_mentioned = True

        # ── Mention gating ──
        # Non-@ group messages are cached in history but NOT dispatched to agent
        if is_group and self._require_mention and not is_mentioned:
            self._record_history_entry(
                msg.channel_id, msg.from_uid, content, payload.mention
            )
            logger.info("[%s] [HISTORY] Non-@ message cached | from=%s | channel=%s",
                        self.name, msg.from_uid, msg.channel_id)
            return

        # ── Resolve sender name ──
        sender_name = await self._resolve_sender_name(msg.from_uid)

        # Cache reply sender name
        if (
            payload.reply
            and getattr(payload.reply, "from_uid", None)
            and getattr(payload.reply, "from_name", None)
        ):
            self._record_uid_name(payload.reply.from_uid, payload.reply.from_name)
            self._name_cache.set(payload.reply.from_uid, payload.reply.from_name)

        # ── Build reply context ──
        reply_text: str | None = None
        if payload.reply:
            reply_from = (
                getattr(payload.reply, "from_name", None)
                or getattr(payload.reply, "from_uid", None)
                or "unknown"
            )
            reply_payload = getattr(payload.reply, "payload", None)
            reply_content = ""
            if isinstance(reply_payload, dict):
                reply_content = reply_payload.get("content", "")
            if reply_content:
                reply_text = f"[Quoted message from {reply_from}]: {reply_content}"

        # ── Convert mentions for LLM ──
        llm_content = content
        if payload.mention:
            llm_content = convert_content_for_llm(content, payload.mention, dict(self._member_map))

        # Build body with quote prefix
        body = (reply_text + "\n---\n" + llm_content) if reply_text else llm_content

        # ── Group history context (injected on @mention) ──
        history_context: str | None = None
        if is_group and msg.channel_id:
            history_prefix = await self._build_history_context(msg.channel_id, self._robot_id)
            if history_prefix:
                history_context = history_prefix

        # ── Ensure GROUP.md (and THREAD.md if in a thread) cached ──
        # GROUP.md is keyed on the parent group_no; THREAD.md on the composite
        # channel_id. Both refresh fire-and-forget so they don't block the
        # turn — first message in a fresh group pays the round-trip *after*
        # the LLM call starts. Startup prefetch warms most of this already.
        parent_group_no = None
        thread_short_id = None
        if is_group and msg.channel_id:
            parent_group_no, thread_short_id = self._split_thread_channel_id(msg.channel_id)
            asyncio.create_task(self._ensure_group_md(parent_group_no))
            if thread_short_id:
                asyncio.create_task(
                    self._ensure_thread_md(parent_group_no, thread_short_id)
                )

        # ── Compose channel_prompt = GROUP.md + (optional) THREAD.md ──
        # Thread-scoped MD complements (does not replace) the parent group's
        # MD. We always send group context first, then append thread context.
        group_system_prompt: str | None = None
        if is_group and parent_group_no:
            parts: list[str] = []
            gmd = self._group_md_cache.get(parent_group_no)
            if gmd and gmd.get("content"):
                parts.append(gmd["content"])
            if thread_short_id:
                tkey = f"{parent_group_no}____{thread_short_id}"
                tmd = self._group_md_cache.get(tkey)
                if tmd and tmd.get("content"):
                    parts.append(f"[THREAD CONTEXT]\n{tmd['content']}")
            if parts:
                group_system_prompt = "\n\n".join(parts)

        chat_type = "dm" if not is_group else "group"
        source = self.build_source(
            chat_id=channel_id,
            chat_type=chat_type,
            user_id=msg.from_uid,
            user_name=sender_name,
        )

        # Map Octo media types to Hermes MessageType
        hermes_msg_type = MessageType.TEXT
        media_urls: list[str] = []
        media_types: list[str] = []

        # Map Octo media types to Hermes MessageType.
        # Image/GIF/Voice/Video are streamed to /tmp/octo-media first and
        # the local path is handed to hermes-core (avoids the vision/audio
        # pipeline hanging on slow remote URLs). The
        # download is bounded (20 MB / 120s); on cap-exceeded or error we
        # fall back to the remote URL so the LLM at least has *something*.
        async def _local_or_remote(rel_url: str | None, mime: str) -> str | None:
            url = self._build_media_url(rel_url)
            if not url:
                return None
            local = await self._download_inbound_media_to_local(url, mime)
            return local or url

        if payload.type == OctoMessageType.Image:
            hermes_msg_type = MessageType.PHOTO
            u = await _local_or_remote(payload.url, "image/jpeg")
            if u:
                media_urls.append(u)
                media_types.append("image/jpeg")
        elif payload.type == OctoMessageType.Voice:
            hermes_msg_type = MessageType.VOICE
            u = await _local_or_remote(payload.url, "audio/ogg")
            if u:
                media_urls.append(u)
                media_types.append("audio/ogg")
        elif payload.type == OctoMessageType.Video:
            hermes_msg_type = MessageType.VIDEO
            u = await _local_or_remote(payload.url, "video/mp4")
            if u:
                media_urls.append(u)
                media_types.append("video/mp4")
        elif payload.type == OctoMessageType.File:
            hermes_msg_type = MessageType.DOCUMENT
            url = self._build_media_url(payload.url)
            if url:
                media_urls.append(url)
                media_types.append("application/octet-stream")

        # Send typing indicator (fire-and-forget)
        asyncio.create_task(self._send_typing_safe(channel_id, channel_type_enum))

        # Inject history + member list into channel_context when available
        # (hermes v0.13+), otherwise prepend to text body so older hermes
        # versions still work. Member list is built fresh per turn so it
        # reflects the latest cache contents (prefetch / refresh updates).
        member_list_prefix = (
            self._build_member_list_prefix() if is_group else ""
        )
        context_parts: list[str] = []
        if member_list_prefix:
            context_parts.append(member_list_prefix)
        if history_context:
            context_parts.append(history_context)
        combined_context = "\n\n".join(context_parts) if context_parts else ""

        final_text = body
        event_kwargs: dict = {}
        if combined_context:
            event_kwargs["channel_context"] = combined_context

        event = MessageEvent(
            text=final_text,
            message_type=hermes_msg_type,
            source=source,
            message_id=msg.message_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=None,
            reply_to_text=reply_text,
            channel_prompt=group_system_prompt,
            **event_kwargs,
        )

        # Pre-populate hermes' session_context ContextVars BEFORE handing
        # the event over. hermes itself sets these later in its agent flow
        # (gateway/run.py `_set_session_env`), but slash-command dispatch
        # runs **earlier** than that — so without this pre-fill our owner
        # gate (`/octo_refresh`, `/octo_audit`) sees an empty user_id
        # and refuses. Setting the same values twice is harmless because
        # hermes will overwrite with the same identity it derives from the
        # SessionSource. Wrap in try/except so absence of hermes (test
        # imports) doesn't crash.
        _session_tokens = None
        try:
            from gateway.session_context import set_session_vars
            _session_tokens = set_session_vars(
                platform="octo",
                chat_id=channel_id,
                chat_name="",
                thread_id=thread_short_id or "",
                user_id=msg.from_uid or "",
                user_name=sender_name or "",
            )
        except Exception:
            pass

        try:
            await self.handle_message(event)
        finally:
            if _session_tokens is not None:
                try:
                    from gateway.session_context import clear_session_vars
                    clear_session_vars(_session_tokens)
                except Exception:
                    pass

    # ── Cross-Channel Permission + Read ───────────────────────────────────

    async def _fetch_group_members_for_permission(self, group_no: str):
        """Fetcher passed to permission.check_permission.

        Hits the API directly (no caching) so the check reflects current
        membership. Returns [] on failure — permission.check_permission turns
        that into a denial.
        """
        if not self._http_session:
            return []
        return await api.get_group_members(
            self._http_session, self._api_url, self._bot_token, group_no
        )

    async def check_read_permission(
        self,
        requester_uid: str | None,
        target: str,
    ):
        """Authorize a cross-channel read against a parsed target.

        Returns (PermissionResult, channel_id, channel_type) so callers can
        reuse the parse result for the subsequent API call.
        """
        from .permission import check_permission, parse_target

        channel_id, channel_type = parse_target(
            target, known_group_ids=self._known_group_ids
        )
        result = await check_permission(
            requester_uid=requester_uid,
            channel_id=channel_id,
            channel_type=channel_type,
            owner_uid=self._owner_uid,
            fetch_group_members=self._fetch_group_members_for_permission,
        )
        return result, channel_id, channel_type

    async def read_channel_messages(
        self,
        *,
        requester_uid: str | None,
        target: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Read recent messages from any DM/Group/Thread, gated by permission.

        Returns a dict ready to be serialised back to the LLM:
          - {"ok": True, "messages": [...], "channel_id": str, "channel_type": int}
          - {"ok": False, "error": "..."}  on permission denial or API failure
        """
        if not self._http_session:
            return {"ok": False, "error": "Not connected"}

        result, channel_id, channel_type = await self.check_read_permission(
            requester_uid, target
        )
        if not result.allowed:
            return {"ok": False, "error": result.reason or "permission denied"}

        try:
            messages = await api.get_channel_messages(
                self._http_session,
                self._api_url,
                self._bot_token,
                channel_id=channel_id,
                channel_type=ChannelType(channel_type),
                limit=max(1, min(int(limit), 100)),
            )
        except Exception as e:
            logger.error("[%s] read_channel_messages failed: %s", self.name, e)
            return {"ok": False, "error": f"API call failed: {e}"}

        return {
            "ok": True,
            "channel_id": channel_id,
            "channel_type": int(channel_type),
            "messages": messages,
        }

    # ── Inbound file handling ─────────────────────────────────────────────

    @staticmethod
    def _cleanup_temp_dir(dir_path: str, retention_s: float) -> None:
        """Best-effort sweep of temp files older than retention_s. Failures
        are swallowed — cleanup is opportunistic, never blocks the call."""
        try:
            import os as _os
            cutoff = time.time() - retention_s
            for entry in _os.listdir(dir_path):
                p = _os.path.join(dir_path, entry)
                try:
                    st = _os.stat(p)
                    if st.st_mtime < cutoff:
                        _os.unlink(p)
                except Exception:
                    pass
        except FileNotFoundError:
            pass
        except Exception:
            pass

    async def _resolve_inbound_file(
        self, url: str, filename: str, known_size: int | None
    ) -> str | None:
        """Try to inline a small text file or download a large one to temp.

        Returns the replacement message body (str) on success, or ``None`` if
        the caller should fall through to the regular "[文件: name]\\n<url>"
        placeholder (non-text extensions, or download errors after retries).

        - Text-like extensions ≤ FILE_INLINE_MAX_BYTES → inline content
        - Larger / non-text → stream-download to /tmp/octo-files/ and
          return ``[文件: name (size) - 已下载到本地: /tmp/...]``
        - 4xx errors → don't retry; return None
        - 5xx / timeout → retry up to 3x with backoff
        """
        if not self._http_session or not url:
            return None
        import os as _os
        from pathlib import PurePosixPath
        ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

        # Hard size cap — never download files above the limit even if we
        # cannot inline (saves disk + network on huge attachments).
        if known_size is not None and known_size > FILE_DOWNLOAD_MAX_BYTES:
            return (
                f"[文件: {filename} ({_format_size(known_size)}) — "
                f"超过最大下载限制 ({_format_size(FILE_DOWNLOAD_MAX_BYTES)})]"
            )

        _os.makedirs(FILE_TEMP_DIR, exist_ok=True)
        # Sweep old temp files opportunistically; never block.
        try:
            self._cleanup_temp_dir(FILE_TEMP_DIR, FILE_TEMP_RETENTION_S)
        except Exception:
            pass

        max_retries = 3
        last_err: str | None = None

        for attempt in range(1, max_retries + 1):
            try:
                # Cap connection + read timeout based on known size, with a
                # generous floor for small files and a 30 min ceiling.
                timeout_s = max(60.0, min(1800.0, (known_size or 0) / (256 * 1024) + 30))
                timeout = aiohttp.ClientTimeout(total=timeout_s)
                async with self._http_session.get(
                    url,
                    headers={"Authorization": f"Bearer {self._bot_token}"},
                    timeout=timeout,
                ) as resp:
                    if resp.status >= 400 and resp.status < 500:
                        # Permanent failure (auth, 404, ...) — don't retry
                        return f"[文件: {filename} - 下载失败 HTTP {resp.status}]"
                    if not resp.ok:
                        raise RuntimeError(f"HTTP {resp.status}")

                    # Decide inline vs download path
                    inline_eligible = (
                        ext in _TEXT_FILE_EXTS
                        and (known_size is None or known_size <= FILE_INLINE_MAX_BYTES)
                    )
                    if inline_eligible:
                        # Stream up to FILE_INLINE_MAX_BYTES, if exceeded fall
                        # through to disk download (one HTTP request — costs
                        # a re-fetch but keeps the simple path simple).
                        buf = bytearray()
                        async for chunk in resp.content.iter_chunked(8192):
                            buf.extend(chunk)
                            if len(buf) > FILE_INLINE_MAX_BYTES:
                                break
                        if len(buf) <= FILE_INLINE_MAX_BYTES:
                            try:
                                text = buf.decode("utf-8")
                            except UnicodeDecodeError:
                                text = buf.decode("utf-8", errors="replace")
                            return (
                                f"[文件: {filename}]\n\n--- 文件内容 ---\n"
                                f"{text}\n--- 文件结束 ---"
                            )
                        # Exceeded inline budget mid-stream; treat as large file
                        # — fall through to download path.

                    # Download to temp file
                    import uuid as _uuid
                    safe_name = "".join(
                        c if c.isalnum() or c in "._-" else "_"
                        for c in PurePosixPath(filename).name
                    ) or "file"
                    tmp_name = f"{_uuid.uuid4().hex}-{safe_name}"
                    tmp_path = _os.path.join(FILE_TEMP_DIR, tmp_name)
                    total = 0
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(64 * 1024):
                            total += len(chunk)
                            if total > FILE_DOWNLOAD_MAX_BYTES:
                                try:
                                    _os.unlink(tmp_path)
                                except Exception:
                                    pass
                                return (
                                    f"[文件: {filename} ({_format_size(total)}+) — "
                                    f"超过最大下载限制 ({_format_size(FILE_DOWNLOAD_MAX_BYTES)})]"
                                )
                            f.write(chunk)
                    size_label = f" ({_format_size(total)})"
                    return f"[文件: {filename}{size_label} - 已下载到本地: {tmp_path}]"

            except TimeoutError as e:
                last_err = f"下载超时 (attempt {attempt}/{max_retries})"
                logger.warning("[%s] file download timeout for %s: %s",
                               self.name, filename, e)
            except Exception as e:
                last_err = str(e)
                logger.warning("[%s] file download error for %s (attempt %d/%d): %s",
                               self.name, filename, attempt, max_retries, e)
            if attempt < max_retries:
                await asyncio.sleep(1.0 * attempt)

        size_info = f" ({_format_size(known_size)})" if known_size else ""
        return f"[文件: {filename}{size_info} - {last_err or '下载失败'}]"

    async def _download_inbound_media_to_local(
        self, url: str, mime: str | None
    ) -> str | None:
        """Stream a remote media URL to a local temp file. Returns the path
        on success, ``None`` on any failure (caller keeps the remote URL).

        Hermes' vision/audio pipelines can stall on slow CDNs or hosts that
        don't honour Range requests; downloading once locally is the same
        trick the upstream reference uses. Capped at MEDIA_DOWNLOAD_MAX_BYTES; oversize
        files quietly fall back to the remote URL so big videos still work
        (LLM just won't get a local file path).
        """
        if not self._http_session or not url:
            return None
        import os as _os
        import uuid as _uuid
        _os.makedirs(MEDIA_TEMP_DIR, exist_ok=True)
        # Opportunistic sweep of old temp files; failures swallowed.
        try:
            self._cleanup_temp_dir(MEDIA_TEMP_DIR, MEDIA_TEMP_RETENTION_S)
        except Exception:
            pass

        # Derive an extension from MIME or URL so the temp file looks like
        # what it is (helps hermes-core sniffing + makes debugging easier).
        ext = ""
        if mime:
            sub = mime.split("/", 1)[-1].split(";", 1)[0]
            if sub:
                ext = "." + "".join(c for c in sub if c.isalnum())[:10]
        if not ext:
            url_path = url.split("?", 1)[0]
            dot = url_path.rfind(".")
            if dot >= 0:
                tail = url_path[dot:]
                ext = "".join(c for c in tail if c.isalnum() or c == ".")[:10]

        tmp_path = _os.path.join(MEDIA_TEMP_DIR, f"{_uuid.uuid4().hex}{ext}")

        try:
            timeout = aiohttp.ClientTimeout(total=MEDIA_DOWNLOAD_TIMEOUT_S)
            async with self._http_session.get(url, timeout=timeout) as resp:
                if not resp.ok:
                    logger.warning("[%s] inbound media download HTTP %d for %s",
                                   self.name, resp.status, url)
                    return None
                total = 0
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        total += len(chunk)
                        if total > MEDIA_DOWNLOAD_MAX_BYTES:
                            f.close()
                            try:
                                _os.unlink(tmp_path)
                            except Exception:
                                pass
                            logger.info(
                                "[%s] inbound media too large (>%s) — using remote URL",
                                self.name, _format_size(MEDIA_DOWNLOAD_MAX_BYTES),
                            )
                            return None
                        f.write(chunk)
            return tmp_path
        except Exception as e:
            logger.warning("[%s] inbound media download failed for %s: %s",
                           self.name, url, e)
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass
            return None

    def _resolve_inner_message_text(self, inner: dict[str, Any]) -> str:
        """Render a single inner message inside a MultipleForward payload.

        Each inner ``msg`` has a ``payload`` block with the same
        ``type``/``content``/``url``/``name`` shape as a top-level message;
        we reuse the same placeholders so the LLM sees consistent labels.
        """
        payload = inner.get("payload") or {}
        msg_type = payload.get("type")
        content = payload.get("content")
        url = self._build_media_url(payload.get("url"))
        t = OctoMessageType

        if msg_type == t.Text:
            return content or ""
        if msg_type == t.Image:
            return f"[图片]\n{url}".strip() if url else "[图片]"
        if msg_type == t.GIF:
            return f"[GIF]\n{url}".strip() if url else "[GIF]"
        if msg_type == t.Voice:
            return f"[语音]\n{url}".strip() if url else "[语音]"
        if msg_type == t.Video:
            return f"[视频]\n{url}".strip() if url else "[视频]"
        if msg_type == t.Location:
            return "[位置信息]"
        if msg_type == t.Card:
            return "[名片]"
        if msg_type == t.File:
            name = payload.get("name")
            label = f"[文件: {name}]" if name else "[文件]"
            return f"{label}\n{url}".strip() if url else label
        if msg_type == t.MultipleForward:
            # Nested forward — recurse
            return self._resolve_multiple_forward_text(payload)
        return content or "[消息]"

    def _resolve_multiple_forward_text(self, payload: Any) -> str:
        """Expand a MultipleForward payload into readable text.

        Each inner message is rendered as ``<senderName>: <body>`` (sender
        resolved from the forward's own ``users[]`` array first, falling
        back to the uid). The ``users`` list is also used to opportunistically
        fill the long-lived uid→name map, so downstream display benefits even
        after the forward message itself is processed.

        Accepts either a MessagePayload (top-level call) or a plain dict
        (recursive call from _resolve_inner_message_text).
        """
        if isinstance(payload, MessagePayload):
            users = (payload.extra or {}).get("users") or []
            msgs = (payload.extra or {}).get("msgs") or []
        elif isinstance(payload, dict):
            users = payload.get("users") or []
            msgs = payload.get("msgs") or []
        else:
            return "[合并转发]"

        user_map: dict[str, str] = {}
        for u in users:
            if not isinstance(u, dict):
                continue
            uid = u.get("uid")
            name = u.get("name")
            if uid and name:
                user_map[uid] = name
                # Pollinate the long-lived map so subsequent messages from
                # the same uid render with a name without a separate API call.
                self._record_uid_name(uid, name, overwrite=False)

        if not msgs:
            # Forward with no inner messages: nothing to render — keep the
            # bare placeholder (matches pre-expansion behaviour, and the
            # existing test contract).
            return "[合并转发]"

        lines: list[str] = ["[合并转发: 聊天记录]"]
        for m in msgs:
            if not isinstance(m, dict):
                continue
            from_uid = m.get("from_uid", "unknown")
            sender_name = (
                user_map.get(from_uid)
                or self._uid_to_name.get(from_uid)
                or from_uid
            )
            inner_payload = m.get("payload") or {}
            if inner_payload.get("type") == OctoMessageType.MultipleForward:
                nested = self._resolve_multiple_forward_text(inner_payload)
                lines.append(f"{sender_name}: [合并转发]")
                lines.append(nested)
            else:
                body = self._resolve_inner_message_text(m)
                lines.append(f"{sender_name}: {body}")
        return "\n".join(lines)

    def _resolve_content(self, payload: MessagePayload) -> str:
        """Resolve message payload to display text.

        Note: text content is left-stripped because some Octo clients
        prepend whitespace (sometimes a single space) before messages
        that start with ``/`` — almost certainly an anti-slash-misclick
        autocorrect. The leading space defeats hermes' strict
        ``text.startswith("/")`` slash detection, so we trim it back.
        """
        if payload.type == OctoMessageType.Text:
            raw = payload.content or ""
            return raw.lstrip() if raw else ""
        elif payload.type == OctoMessageType.Image:
            url = self._build_media_url(payload.url) or ""
            return f"[图片]\n{url}".strip()
        elif payload.type == OctoMessageType.GIF:
            url = self._build_media_url(payload.url) or ""
            return f"[GIF]\n{url}".strip()
        elif payload.type == OctoMessageType.Voice:
            url = self._build_media_url(payload.url) or ""
            return f"[语音消息]\n{url}".strip()
        elif payload.type == OctoMessageType.Video:
            url = self._build_media_url(payload.url) or ""
            return f"[视频]\n{url}".strip()
        elif payload.type == OctoMessageType.File:
            name = payload.name or "未知文件"
            url = self._build_media_url(payload.url) or ""
            return f"[文件: {name}]\n{url}".strip()
        elif payload.type == OctoMessageType.Location:
            return "[位置信息]"
        elif payload.type == OctoMessageType.Card:
            return f"[名片: {payload.name or '未知'}]"
        elif payload.type == OctoMessageType.MultipleForward:
            return self._resolve_multiple_forward_text(payload)
        else:
            return payload.content or ""

    async def _send_read_receipt_safe(
        self, channel_id: str, channel_type: ChannelType, message_ids: list[str]
    ) -> None:
        if not self._http_session:
            return
        try:
            await api.send_read_receipt(
                self._http_session, self._api_url, self._bot_token,
                channel_id, channel_type, message_ids,
            )
        except Exception as e:
            logger.debug("[%s] Read receipt failed: %s", self.name, e)

    async def _send_typing_safe(self, channel_id: str, channel_type: ChannelType) -> None:
        if not self._http_session:
            return
        try:
            await api.send_typing(
                self._http_session, self._api_url, self._bot_token,
                channel_id=channel_id, channel_type=channel_type,
            )
        except Exception as e:
            logger.debug("[%s] Typing indicator failed: %s", self.name, e)

    # ── Heartbeat ─────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        try:
            while self._connected:
                await asyncio.sleep(self._heartbeat_interval_s)
                self._ping_retry_count += 1
                if self._ping_retry_count > self._ping_max_retry:
                    logger.warning("[%s] Ping timeout, reconnecting...", self.name)
                    self._connected = False
                    if self._ws:
                        try:
                            await self._ws.close()
                        except Exception:
                            pass
                    break
                if self._ws:
                    try:
                        await self._ws.send(encode_ping_packet())
                    except Exception as e:
                        logger.debug("[%s] Ping failed: %s", self.name, e)
                        break
        except asyncio.CancelledError:
            return

    # ── Reconnection ──────────────────────────────────────────────────────

    async def _cache_cleanup_loop(self) -> None:
        """Periodically prune per-channel caches for inactive channels.

        Runs every CACHE_CLEANUP_INTERVAL_S. Survives reconnects (the task is
        only re-created if cancelled or finished). Cancelled cleanly during
        disconnect().
        """
        try:
            while True:
                try:
                    await asyncio.sleep(CACHE_CLEANUP_INTERVAL_S)
                    removed = self._cleanup_caches()
                    if removed:
                        logger.info(
                            "[%s] cache cleanup: evicted %d inactive channel(s)",
                            self.name, removed,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # Don't let an exception kill the cleanup loop — bounded
                    # caches are a stability feature, not a critical path.
                    logger.warning("[%s] cache cleanup error: %s", self.name, e)
        except asyncio.CancelledError:
            return

    def _touch_cache(self, channel_id: str) -> None:
        """Mark a channel as recently active so it survives the next sweep."""
        if not channel_id:
            return
        self._cache_activity[channel_id] = time.monotonic()

    async def _prefetch_groups_and_members(self) -> None:
        """Warm caches at startup.

        Concretely:
          1. List all groups the bot belongs to → seed known_group_ids so
             parse_target on a bare id can disambiguate group vs DM.
          2. For each group, fetch GROUP.md once and stash in the cache so
             the first @ message doesn't pay the round-trip.
          3. For each group, fetch member list once and stash uid↔name maps
             so mention resolution works on the very first message.

        Fire-and-forget; every step is wrapped in try/except so a single
        slow / missing endpoint doesn't poison the warmup. Activity is
        recorded for every prefetched channel so the cleanup loop doesn't
        evict cold-but-just-fetched state on its first pass.
        """
        if not self._http_session:
            return
        try:
            groups = await api.fetch_bot_groups(
                self._http_session, self._api_url, self._bot_token
            )
        except Exception as e:
            logger.warning("[%s] prefetch: fetch_bot_groups failed: %s",
                           self.name, e)
            return

        md_count = 0
        member_count = 0
        for g in groups:
            if hasattr(g, "group_no"):
                gid = g.group_no
            elif isinstance(g, dict):
                gid = g.get("group_no")
            else:
                gid = None
            if not gid:
                continue
            self._known_group_ids.add(gid)
            self._touch_cache(gid)
            self._chat_kind[gid] = ChannelType.Group

            # GROUP.md prefetch — best effort.
            # First hydrate from disk so a restart serves the cached MD
            # immediately even if the API is slow / down. The next inbound
            # group_md_updated event invalidates and refetches.
            self._hydrate_md_cache_from_disk(gid)
            try:
                md = await api.get_group_md(
                    self._http_session, self._api_url, self._bot_token, gid,
                )
                if md and md.get("content"):
                    self._group_md_cache[gid] = {
                        "content": md["content"],
                        "version": md.get("version", 0),
                    }
                    self._group_md_checked.add(gid)
                    self._write_md_to_disk(gid, md["content"], md.get("version", 0))
                    md_count += 1
            except Exception as e:
                logger.debug("[%s] prefetch GROUP.md for %s skipped: %s",
                             self.name, gid, e)

            # Member list prefetch — best effort; fills uid↔name maps so
            # the first @-mention in this group doesn't have to do a
            # blocking refresh.
            try:
                members = await api.get_group_members(
                    self._http_session, self._api_url, self._bot_token, gid,
                )
                for m in members:
                    if m.name and m.uid:
                        self._member_map[m.name] = m.uid
                        # H: emoji-tolerant alias
                        stripped = _strip_emoji(m.name)
                        if stripped and stripped != m.name:
                            self._member_map.setdefault(stripped, m.uid)
                        self._record_uid_name(m.uid, m.name)
                        self._name_cache.set(m.uid, m.name)
                        member_count += 1
                self._update_user_group_index(gid, members)
                # Also record the group's display name for shared-groups
                # answers — fall back to group_no when the API omits it.
                gname = getattr(g, "name", None) or gid
                self._group_names[gid] = gname
                self._group_cache_timestamps[gid] = int(time.time() * 1000)
            except Exception as e:
                logger.debug("[%s] prefetch members for %s skipped: %s",
                             self.name, gid, e)

        if groups:
            logger.info(
                "[%s] prefetch complete: %d groups, %d GROUP.md cached, %d member names",
                self.name, len(groups), md_count, member_count,
            )

    def _cleanup_caches(self) -> int:
        """Evict per-channel state for channels untouched for > CACHE_MAX_AGE_S.

        Returns the number of channels evicted (for logging).
        """
        cutoff = time.monotonic() - CACHE_MAX_AGE_S
        # Channels with no recorded activity OR activity older than cutoff.
        # We only sweep the activity-tracked dicts; global maps
        # (_member_map / _uid_to_name / _name_cache) are bounded by the LRU
        # itself and aren't keyed per-channel.
        stale: list[str] = []
        for cid, ts in list(self._cache_activity.items()):
            if ts < cutoff:
                stale.append(cid)

        for cid in stale:
            self._group_histories.pop(cid, None)
            self._group_md_cache.pop(cid, None)
            self._group_md_checked.discard(cid)
            self._chat_kind.pop(cid, None)
            self._group_cache_timestamps.pop(cid, None)
            # Thread channel_ids carry their parent group's group_no as the
            # prefix; the parent's group-level entries are keyed on the bare
            # group_no, which gets its own activity record on inbound traffic.
            self._cache_activity.pop(cid, None)

        return len(stale)

    async def _schedule_reconnect(self) -> None:
        # Dedup: if a reconnect is already in flight, refuse to start a
        # second one. Without this, every per-frame error in the receive
        # loop AND every heartbeat failure can each spawn an independent
        # reconnect task — each opens a new WS, and WuKongIM's
        # "one connection per uid+device_flag" rule means later opens
        # kick earlier opens, leaving the bot in a permanent self-kick
        # storm.
        if self._reconnect_in_progress:
            return
        if not self._need_reconnect:
            return
        self._reconnect_in_progress = True
        try:
            exponential = min(
                RECONNECT_BASE_DELAY * (2 ** self._reconnect_attempts),
                RECONNECT_MAX_DELAY,
            )
            # Jitter the exponential delay (±25%) AND add a flat stagger drawn
            # from [0, RECONNECT_STAGGER_MAX_S]. The flat stagger is what
            # de-syncs reconnects across multiple bots that all dropped at the
            # same instant (server restart, network blip) — without it, every
            # bot lands its next attempt within ms of every other bot.
            delay = exponential * (0.75 + random.random() * 0.5)
            delay += random.random() * RECONNECT_STAGGER_MAX_S
            self._reconnect_attempts += 1
            logger.info("[%s] Reconnecting in %.1fs (attempt %d)...",
                        self.name, delay, self._reconnect_attempts)
            await asyncio.sleep(delay)
            if not self._need_reconnect:
                return
            try:
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
                self._temp_buffer = bytearray()
                await self._do_connect()
            except Exception as e:
                logger.error("[%s] Reconnection failed: %s", self.name, e)
                if self._need_reconnect:
                    # Reset the in-progress flag BEFORE re-scheduling so the
                    # next attempt's dedup check sees a clear lane.
                    self._reconnect_in_progress = False
                    asyncio.create_task(self._schedule_reconnect())
                    return
        finally:
            self._reconnect_in_progress = False

    # ── Sending ───────────────────────────────────────────────────────────

    def _resolve_channel_type(
        self, chat_id: str, metadata: dict[str, Any] | None = None
    ) -> ChannelType:
        """Pick the channel_type for an outbound message.

        Priority: explicit metadata > remembered inbound mapping > guess based
        on whether chat_id looks like a group_no (rough heuristic).
        """
        if metadata and metadata.get("channel_type"):
            return ChannelType(metadata["channel_type"])
        cached = self._chat_kind.get(chat_id)
        if cached is not None:
            return cached
        # Thread channel IDs always contain the "____" separator
        if "____" in chat_id:
            return ChannelType.CommunityTopic
        # Heuristic fallback: Octo user uids are 32-char hex strings;
        # anything shorter or with non-hex chars is likely a group_no.
        if len(chat_id) == 32 and all(c in "0123456789abcdef" for c in chat_id.lower()):
            return ChannelType.DM
        return ChannelType.Group

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        if not self._http_session:
            return SendResult(success=False, error="Not connected")

        if not _OCTO_CHAT_ID_RE.fullmatch(str(chat_id)):
            return SendResult(success=False, error=f"invalid chat_id format: {chat_id!r}")

        channel_type = self._resolve_channel_type(chat_id, metadata)

        # Only metadata.no_stream forces a direct (un-coalesced) send. The
        # buffer ignores reply_to — octo doesn't render reply-threads in
        # a way that's worth giving up coalescing for, and consumer's
        # _send_or_edit always passes _initial_reply_to_id which would
        # otherwise bypass every streaming response's first send().
        if metadata and metadata.get("no_stream"):
            return await self._send_normal(chat_id, content, channel_type, reply_to)

        # Everything else goes into the per-chat coalescing buffer:
        # streaming cursor frames, segment-first frames (no cursor), AND
        # one-shot commentary. The buffer flushes after a quiet period,
        # so multiple consecutive sends within a "burst" merge into one
        # octo message.
        return await self._buffer_streamed(chat_id, content, channel_type)

    async def _buffer_streamed(
        self,
        chat_id: str,
        content: str,
        channel_type: ChannelType,
    ) -> SendResult:
        """Append a new send() to the chat's coalescing buffer.

        - If no buffer exists: open one with this content as ``current_segment``.
        - If a buffer exists: close its existing ``current_segment`` into the
          ``segments`` list, then start a fresh ``current_segment`` with
          this content. Re-arms the flush timer either way.
        """
        cleaned = self._strip_hermes_cursor(content)

        state = self._active_streams.get(chat_id)
        if state is not None:
            # Existing buffer — close out the in-progress segment, then
            # start a new one with the incoming content. Cancel any flush
            # task; it'll be re-armed below.
            self._cancel_stream_timeout(state)
            if state["current_segment"]:
                state["segments"].append(state["current_segment"])
            state["current_segment"] = cleaned
            state["channel_type"] = channel_type  # in case it changed
            message_id = state["message_id"]
        else:
            import uuid
            message_id = (
                f"octo-buf-{chat_id}-{time.monotonic_ns()}-{uuid.uuid4().hex[:8]}"
            )
            state = {
                "message_id": message_id,
                "channel_type": channel_type,
                "segments": [],
                "current_segment": cleaned,
                "flush_task": None,
            }
            self._active_streams[chat_id] = state

        logger.debug(
            "[%s] buffer SEND chat=%s id=%s segs=%d cur=%d",
            self.name, chat_id[:8], message_id[-12:],
            len(state["segments"]), len(state["current_segment"]),
        )
        self._arm_stream_timeout(chat_id, message_id)
        return SendResult(success=True, message_id=message_id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Update the in-progress segment of the chat's buffer.

        Non-finalize edits replace ``current_segment``. finalize=True closes
        ``current_segment`` into ``segments`` (the segment is "done"), but
        does NOT flush — the flush waits until STREAM_FLUSH_DELAY_S of true
        idle, so a follow-on segment (next LLM round) can join the buffer.
        """
        if not self._http_session:
            return SendResult(success=False, error="Not connected")

        state = self._active_streams.get(chat_id)
        if not state or state.get("message_id") != message_id:
            return SendResult(
                success=False,
                error="No active Octo buffer for this message",
            )

        self._cancel_stream_timeout(state)

        # Always strip hermes' cursor — buffered text feeds straight into a
        # future _send_normal, no cursor should ever reach octo.
        cleaned = self._strip_hermes_cursor(content)

        if finalize:
            # Close this segment off into the segments list, clear in-progress.
            if cleaned:
                state["segments"].append(cleaned)
            state["current_segment"] = ""
            logger.debug(
                "[%s] buffer FINALIZE chat=%s id=%s segs=%d (waiting for idle)",
                self.name, chat_id[:8], message_id[-12:], len(state["segments"]),
            )
        else:
            state["current_segment"] = cleaned
            logger.debug(
                "[%s] buffer EDIT chat=%s id=%s cur=%d",
                self.name, chat_id[:8], message_id[-12:], len(cleaned),
            )

        self._arm_stream_timeout(chat_id, message_id)
        return SendResult(success=True, message_id=message_id)

    def _arm_stream_timeout(self, chat_id: str, message_id: str) -> None:
        """Re-arm the deferred-flush task for this chat's buffer.

        Called after every send / edit / finalize. If STREAM_FLUSH_DELAY_S
        seconds elapse with no further activity, the buffer is flushed as
        a single normal message — coalescing whatever segments accumulated.
        """
        state = self._active_streams.get(chat_id)
        if not state or state.get("message_id") != message_id:
            return
        self._cancel_stream_timeout(state)
        try:
            task = asyncio.create_task(
                self._close_stream_after_idle(chat_id, message_id)
            )
        except RuntimeError:
            return
        state["flush_task"] = task

    @staticmethod
    def _cancel_stream_timeout(state: dict) -> None:
        task = state.get("flush_task")
        if task and not task.done():
            task.cancel()
        state["flush_task"] = None

    @staticmethod
    def _strip_hermes_cursor(content: str) -> str:
        """Remove hermes' streaming cursor marker from outbound content."""
        if not content:
            return content
        if HERMES_STREAM_CURSOR in content:
            content = content.replace(HERMES_STREAM_CURSOR, "")
        bare = HERMES_STREAM_CURSOR.lstrip()
        if bare and bare in content:
            content = content.replace(bare, "")
        return content

    @staticmethod
    def _joined_buffer(state: dict) -> str:
        """Concatenate finalized segments + current segment into one body.

        Segments are joined as-is (no separator) — each segment is a complete
        LLM accumulation that already starts/ends with whatever whitespace
        the model emitted. Adding artificial separators would distort the
        markdown structure (e.g. mid-list `\\n\\n` would close a list).
        """
        parts = list(state.get("segments", []))
        cur = state.get("current_segment", "")
        if cur:
            parts.append(cur)
        return "".join(parts)

    async def _close_stream_after_idle(self, chat_id: str, message_id: str) -> None:
        """Deferred-flush watchdog. Fires STREAM_FLUSH_DELAY_S after the
        most recent buffer activity and ships the coalesced content."""
        try:
            await asyncio.sleep(STREAM_FLUSH_DELAY_S)
        except asyncio.CancelledError:
            return
        state = self._active_streams.get(chat_id)
        if not state or state.get("message_id") != message_id:
            return
        channel_type = state.get("channel_type")
        body = self._joined_buffer(state)
        # Pop FIRST so a follow-up send() arriving mid-flush opens a fresh
        # buffer instead of racing with our flushing entry.
        self._active_streams.pop(chat_id, None)
        if not body or channel_type is None:
            return
        logger.debug(
            "[%s] buffer FLUSH chat=%s id=%s segs=%d total=%d",
            self.name, chat_id[:8], message_id[-12:],
            len(state.get("segments", [])), len(body),
        )
        try:
            await self._send_normal(chat_id, body, channel_type, None)
        except Exception as e:
            logger.debug("[%s] idle flush failed for %s: %s",
                         self.name, chat_id, e)

    async def _close_active_stream(self, chat_id: str) -> None:
        """Flush whatever buffer is open for *chat_id* immediately.

        Used by disconnect() to drain in-flight responses before tearing
        down the http session. Idempotent — no-op when there's nothing
        buffered.
        """
        state = self._active_streams.pop(chat_id, None)
        if not state:
            return
        self._cancel_stream_timeout(state)
        body = self._joined_buffer(state)
        channel_type = state.get("channel_type")
        if not body or channel_type is None:
            return
        logger.debug(
            "[%s] buffer FORCE-flush chat=%s total=%d",
            self.name, chat_id[:8], len(body),
        )
        try:
            await self._send_normal(chat_id, body, channel_type, None)
        except Exception as e:
            logger.debug("[%s] proactive flush failed for %s: %s",
                         self.name, chat_id, e)

    async def _send_normal(
        self,
        chat_id: str,
        content: str,
        channel_type: ChannelType,
        reply_to: str | None = None,
    ) -> SendResult:
        if not self._http_session:
            return SendResult(success=False, error="Not connected")
        content = self._strip_hermes_cursor(content)
        chunks = self.truncate_message(content, MAX_MESSAGE_LENGTH)
        logger.debug(
            "[%s] _send_normal chat=%s chunks=%d total=%d",
            self.name, chat_id[:8], len(chunks), len(content),
        )
        # Threads: the server accepts messages from non-members (HTTP 2xx)
        # but does not push them to subscribers. Best-effort join before
        # send so newly-created threads receive M1. Already-joined / race
        # errors are expected and swallowed. Centralised here so every
        # outbound path (no_stream direct, buffer flush, edit flush, and
        # callers like the generic send_message tool) is covered.
        if channel_type == ChannelType.CommunityTopic and "____" in chat_id:
            group_no, short_id = chat_id.split("____", 1)
            try:
                await api.join_thread(
                    self._http_session, self._api_url, self._bot_token,
                    group_no=group_no, short_id=short_id,
                )
            except Exception as exc:
                logger.debug(
                    "[%s] _send_normal thread auto-join failed "
                    "(likely already joined): %s",
                    self.name, exc,
                )
        try:
            for chunk in chunks:
                # Convert LLM-emitted @[uid:name] markers into wire format:
                # plain @name in content + mention.entities/uids sidecar.
                # Without this, the Octo client renders the literal template
                # text instead of an @ pill (see review #2026-05-19).
                send_content = chunk
                send_uids: list[str] | None = None
                send_entities: list | None = None
                structured = parse_structured_mentions(chunk)
                if structured:
                    send_content, send_entities, send_uids = convert_structured_mentions(
                        chunk, structured,
                    )
                await api.send_message(
                    self._http_session, self._api_url, self._bot_token,
                    channel_id=chat_id, channel_type=channel_type,
                    content=send_content, reply_msg_id=reply_to,
                    mention_uids=send_uids, mention_entities=send_entities,
                )
            return SendResult(success=True)
        except Exception as e:
            logger.error("[%s] Send failed: %s", self.name, e)
            return SendResult(
                success=False,
                error=str(e),
                retryable="connect" in str(e).lower() or "timeout" in str(e).lower(),
            )

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        if not self._http_session:
            return
        channel_type = self._resolve_channel_type(
            chat_id, metadata if isinstance(metadata, dict) else None
        )
        await self._send_typing_safe(chat_id, channel_type)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: str | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        if not self._http_session:
            return SendResult(success=False, error="Not connected")
        channel_type = self._resolve_channel_type(chat_id, metadata)
        try:
            final_url = image_url
            width, height = None, None
            if image_url.startswith(("http://", "https://")):
                try:
                    file_data, content_type, filename = await api.download_file(
                        self._http_session, image_url, max_size=20 * 1024 * 1024
                    )
                    dims = api.parse_image_dimensions(file_data, content_type)
                    if dims:
                        width, height = dims
                    final_url = await api.upload_and_get_url(
                        self._http_session, self._api_url, self._bot_token,
                        filename, file_data, content_type,
                    )
                except Exception as upload_err:
                    logger.warning("[%s] Image upload failed, using URL directly: %s",
                                   self.name, upload_err)

            await api.send_media_message(
                self._http_session, self._api_url, self._bot_token,
                channel_id=chat_id, channel_type=channel_type,
                msg_type=OctoMessageType.Image, url=final_url,
                width=width, height=height,
            )
            if caption:
                await api.send_message(
                    self._http_session, self._api_url, self._bot_token,
                    channel_id=chat_id, channel_type=channel_type,
                    content=caption,
                )
            return SendResult(success=True)
        except Exception as e:
            logger.error("[%s] send_image failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: str | None = None,
        file_name: str | None = None,
        reply_to: str | None = None,
        **kwargs: Any,
    ) -> SendResult:
        if not self._http_session:
            return SendResult(success=False, error="Not connected")
        metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else None
        channel_type = self._resolve_channel_type(chat_id, metadata)
        try:
            if file_path.startswith(("http://", "https://")):
                file_data, content_type, filename = await api.download_file(
                    self._http_session, file_path
                )
            else:
                import aiofiles
                filename = file_name or os.path.basename(file_path)
                content_type = api.infer_content_type(filename)
                async with aiofiles.open(file_path, "rb") as f:
                    file_data = await f.read()

            if file_name:
                filename = file_name

            uploaded_url = await api.upload_and_get_url(
                self._http_session, self._api_url, self._bot_token,
                filename, file_data, content_type,
            )
            await api.send_media_message(
                self._http_session, self._api_url, self._bot_token,
                channel_id=chat_id, channel_type=channel_type,
                msg_type=OctoMessageType.File, url=uploaded_url,
                name=filename, size=len(file_data),
            )
            if caption:
                await api.send_message(
                    self._http_session, self._api_url, self._bot_token,
                    channel_id=chat_id, channel_type=channel_type, content=caption,
                )
            return SendResult(success=True)
        except Exception as e:
            logger.error("[%s] send_document failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: str | None = None,
        reply_to: str | None = None,
        **kwargs: Any,
    ) -> SendResult:
        if not self._http_session:
            return SendResult(success=False, error="Not connected")
        metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else None
        channel_type = self._resolve_channel_type(chat_id, metadata)
        try:
            if audio_path.startswith(("http://", "https://")):
                file_data, content_type, filename = await api.download_file(
                    self._http_session, audio_path
                )
            else:
                import aiofiles
                filename = os.path.basename(audio_path)
                content_type = api.infer_content_type(filename)
                async with aiofiles.open(audio_path, "rb") as f:
                    file_data = await f.read()
            uploaded_url = await api.upload_and_get_url(
                self._http_session, self._api_url, self._bot_token,
                filename, file_data, content_type,
            )
            await api.send_media_message(
                self._http_session, self._api_url, self._bot_token,
                channel_id=chat_id, channel_type=channel_type,
                msg_type=OctoMessageType.Voice, url=uploaded_url, name=filename,
            )
            return SendResult(success=True)
        except Exception as e:
            logger.error("[%s] send_voice failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: str | None = None,
        reply_to: str | None = None,
        **kwargs: Any,
    ) -> SendResult:
        if not self._http_session:
            return SendResult(success=False, error="Not connected")
        metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else None
        channel_type = self._resolve_channel_type(chat_id, metadata)
        try:
            if video_path.startswith(("http://", "https://")):
                file_data, content_type, filename = await api.download_file(
                    self._http_session, video_path
                )
            else:
                import aiofiles
                filename = os.path.basename(video_path)
                content_type = api.infer_content_type(filename)
                async with aiofiles.open(video_path, "rb") as f:
                    file_data = await f.read()
            uploaded_url = await api.upload_and_get_url(
                self._http_session, self._api_url, self._bot_token,
                filename, file_data, content_type,
            )
            await api.send_media_message(
                self._http_session, self._api_url, self._bot_token,
                channel_id=chat_id, channel_type=channel_type,
                msg_type=OctoMessageType.Video, url=uploaded_url, name=filename,
            )
            if caption:
                await api.send_message(
                    self._http_session, self._api_url, self._bot_token,
                    channel_id=chat_id, channel_type=channel_type, content=caption,
                )
            return SendResult(success=True)
        except Exception as e:
            logger.error("[%s] send_video failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        if not self._http_session:
            return {"name": chat_id, "type": "group", "chat_id": chat_id}
        try:
            info = await api.get_group_info(
                self._http_session, self._api_url, self._bot_token, group_no=chat_id,
            )
            return {
                "name": info.name,
                "type": "group",
                "chat_id": info.group_no,
                **info.extra,
            }
        except Exception as e:
            logger.debug("[%s] get_chat_info failed: %s", self.name, e)
            return {"name": chat_id, "type": "unknown", "chat_id": chat_id}


# ── Plugin Registration ───────────────────────────────────────────────────────


def register(ctx) -> None:
    """Hermes plugin entry point — registers the Octo platform adapter.

    Filters keyword arguments against the installed ``PlatformEntry`` to stay
    compatible across hermes versions (older versions may not support newer
    fields like ``env_enablement_fn``).

    LLM-callable surface (tool / skill / slash commands) is gated on BOTH
    ``OCTO_API_URL`` and ``OCTO_BOT_TOKEN`` being set. Without those the
    adapter cannot connect, so polluting the global registry with octo
    schemas / commands just wastes LLM context tokens and clutters
    cross-platform slash menus. ``register_platform`` always runs so the
    platform stays visible in ``hermes setup gateway`` / ``hermes config``
    UIs (matching IRC/Teams/Line behaviour). Once both env vars are set,
    a gateway restart will rerun this register() and add the full set.
    """
    _bot_configured = bool(os.getenv("OCTO_BOT_TOKEN") and os.getenv("OCTO_API_URL"))
    if not _bot_configured:
        logger.info(
            "[Octo] OCTO_API_URL/OCTO_BOT_TOKEN not both set — skipping "
            "tool/skill/command registration; platform entry still "
            "registered so it appears in hermes setup gateway / hermes config."
        )

    # Register the LLM-callable management tool first so it lives in the
    # registry by the time the gateway builds its toolset list.
    #
    # Why a Tool (not a Skill + generic HTTP tool):
    #   1. Target parsing must happen plugin-side. Octo's 32-char hex
    #      channel_ids and {group_no}____{short_id} thread format are not
    #      recognized by the core send_message tool — using it would silently
    #      drop messages onto the bot's home channel with a fake message_id.
    #   2. OWNER_ONLY mutating actions (create-group, *-md-update,
    #      voice-context-*, …) enforce requester_uid == _owner_uid in the
    #      handler. Prompt-only enforcement is unacceptable against injection.
    #   3. The handler depends on live adapter runtime state (_owner_uid,
    #      _api_url, _bot_token, _known_group_ids, check_read_permission()),
    #      not configuration — a generic HTTP tool cannot see it.
    #   4. Event-loop / aiohttp-session compatibility with the gateway is a
    #      plugin black box (see agent_tools.py module docstring).
    if _bot_configured:
        try:
            from .agent_tools import (
                TOOL_SCHEMA,
                octo_management_handler,
            )
            ctx.register_tool(
                name="octo_management",
                toolset="hermes-octo",
                schema=TOOL_SCHEMA,
                handler=octo_management_handler,
                check_fn=check_octo_requirements,
                requires_env=list(_REQUIRED_ENV),
                is_async=True,
                description=TOOL_SCHEMA.get("description", ""),
                emoji="💬",
            )
        except Exception as e:  # pragma: no cover — be defensive at import time
            logger.warning("[Octo] register_tool(octo_management) failed: %s", e)

    # Register the bundled Octo Bot API skill so the LLM can load it on
    # demand via skill_view("octo:octo-bot-api"). Not auto-injected into
    # every system prompt — opt-in keeps token usage down.
    if _bot_configured:
        try:
            from pathlib import Path
            skill_path = Path(__file__).parent / "skills" / "octo-bot-api" / "SKILL.md"
            if skill_path.exists():
                ctx.register_skill(
                    name="octo-bot-api",
                    path=skill_path,
                    description=(
                        "Octo Bot API reference: register, send/receive messages, "
                        "group/thread management, GROUP.md/THREAD.md, voice context, "
                        "file upload via COS. Load this skill before composing "
                        "non-trivial Octo operations."
                    ),
                )
            else:
                logger.warning("[Octo] bundled SKILL.md missing at %s", skill_path)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("[Octo] register_skill(octo-bot-api) failed: %s", e)

    # Register ops slash commands (/octo_doctor, /octo_info, etc.).
    # Failures here are non-fatal — commands are a nicety, not load-bearing.
    if _bot_configured:
        try:
            from .commands import register_all as _register_commands
            _register_commands(ctx)
        except Exception as e:
            logger.warning("[Octo] slash command registration failed: %s", e)

    candidate_kwargs = dict(
        name="octo",
        label="Octo",
        adapter_factory=lambda cfg: OctoAdapter(cfg),
        check_fn=check_octo_requirements,
        validate_config=_is_connected,
        is_connected=_is_connected,
        required_env=list(_REQUIRED_ENV),
        install_hint="hermes plugins install Mininglamp-OSS/hermes-channel-octo",
        env_enablement_fn=_env_enablement,
        allowed_users_env="OCTO_ALLOWED_USERS",
        allow_all_env="OCTO_ALLOW_ALL_USERS",
        # Cron home-channel delivery support (deliver=octo).
        cron_deliver_env_var="OCTO_HOME_CHANNEL",
        # Out-of-process cron sender — used when cron runs in a separate
        # process from the gateway and the in-process adapter weakref is None.
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💬",
        platform_hint=(
            "You are chatting via Octo (WuKongIM). "
            "In group chats you are only activated when @mentioned."
        ),
    )

    ctx.register_platform(**candidate_kwargs)
