"""
Octo Bot HTTP API client.

All API calls use aiohttp with Bearer token authentication.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
import logging
import os
import re
import socket
import stat
import struct
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote, unquote_to_bytes, urlencode, urljoin, urlparse

import aiohttp
from .transport import (
    TransportPolicy,
    canonical_url_host,
    is_private_or_metadata_host,
    new_guarded_http_session,
)

from .types import (
    CARD_VERSION,
    CardProfileManifest,
    CardTemplateCapability,
    CardTemplateViewCapability,
    CardTemplatingCapability,
    BotRegisterResp,
    ChannelType,
    GroupInfo,
    GroupMember,
    MentionEntity,
    MessageType,
    RichTextBlock,
    SendMessageResult,
    resolve_card_profile,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)
MAX_EVENT_WAIT_SECONDS = 30
EVENT_POLL_WAIT_MARGIN_SECONDS = 10
_API_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _api_path_segment(value: str, name: str) -> str:
    """Validate and encode one server resource identifier for a URL path."""
    if not isinstance(value, str) or not _API_PATH_SEGMENT_RE.fullmatch(value):
        raise ValueError(f"invalid Octo {name}")
    return quote(value, safe="")


class OctoApiError(RuntimeError):
    """A safe, structured failure from an authenticated Octo API request.

    Response bodies can contain implementation details and should never be
    copied into agent-tool output.  Callers that need compatibility handling
    can inspect ``status`` without parsing a response body.
    """

    def __init__(
        self,
        path: str,
        *,
        status: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.path = path
        self.status = status
        suffix = f" (HTTP {status})" if status is not None else ""
        detail = f": {reason}" if reason else ""
        super().__init__(f"Octo API request failed{suffix}{detail}")


def _response_error(path: str, response: aiohttp.ClientResponse) -> OctoApiError:
    """Create a secret-free error for a non-success HTTP response."""
    return OctoApiError(path, status=response.status)


# ─── MIME Type Helpers ───────────────────────────────────────────────────────
MAX_OUTBOUND_MEDIA_BYTES = 100 * 1024 * 1024
MAX_MEDIA_FILENAME_BYTES = 255


def safe_media_filename(value: object) -> str | None:
    """Return a safe basename for Octo media metadata, or ``None``."""
    if not isinstance(value, str) or value.endswith((".", " ")):
        return None
    candidate = value.strip()
    if (
        not candidate
        or candidate in {".", ".."}
        or "/" in candidate
        or "\\" in candidate
        or candidate != Path(candidate).name
        or any(unicodedata.category(char) in {"Cc", "Cf"} for char in candidate)
        or candidate.endswith((".", " "))
        or len(candidate.encode("utf-8")) > MAX_MEDIA_FILENAME_BYTES
    ):
        return None
    return candidate

_MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".opus": "audio/opus",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".html": "text/html",
    ".json": "application/json",
}
_DATA_URI_EXTENSION_MAP = {
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/json": ".json",
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "video/mp4": ".mp4",
}


def infer_content_type(filename: str) -> str:
    """Infer MIME type from filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    return _MIME_MAP.get(ext, "application/octet-stream")


def decode_media_data_uri(
    source: str,
    *,
    max_size: int,
) -> tuple[bytes, str, str]:
    """Decode one OpenClaw-compatible outbound media data URI."""
    if not source.startswith("data:") or "," not in source or max_size <= 0:
        raise ValueError("invalid media data URI")
    header, encoded = source[5:].split(",", 1)
    parts = header.split(";")
    content_type = parts[0].strip().lower() or "application/octet-stream"
    is_base64 = any(part.strip().lower() == "base64" for part in parts[1:])
    try:
        if is_base64:
            compact = re.sub(r"\s+", "", encoded)
            padding = 2 if compact.endswith("==") else 1 if compact.endswith("=") else 0
            estimated_size = (len(compact) * 3 // 4) - padding
            if estimated_size > max_size:
                raise ValueError(f"media exceeds the {max_size}-byte limit")
            data = base64.b64decode(compact, validate=True)
        else:
            if len(encoded) > max_size * 3:
                raise ValueError(f"media exceeds the {max_size}-byte limit")
            data = unquote_to_bytes(encoded)
    except (binascii.Error, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("media exceeds"):
            raise
        raise ValueError("invalid media data URI") from exc
    if len(data) > max_size:
        raise ValueError(f"media exceeds the {max_size}-byte limit")
    extension = _DATA_URI_EXTENSION_MAP.get(content_type, ".bin")
    return data, content_type, f"file{extension}"


def authorize_local_media_path(source: str) -> str | None:
    """Return the local path authorized by the installed Hermes runtime."""
    if not isinstance(source, str) or not source:
        return None
    if source.startswith("file://"):
        parsed = urlparse(source)
        if parsed.netloc not in {"", "localhost"}:
            return None
        source = unquote(parsed.path)
    try:
        from gateway.platforms.base import BasePlatformAdapter
    except ImportError:
        return None
    validator = getattr(BasePlatformAdapter, "validate_media_delivery_path", None)
    if not callable(validator):
        return None
    authorized = cast(Callable[[str], object], validator)(source)
    return authorized if isinstance(authorized, str) and authorized else None


def read_authorized_local_media(
    source: str,
    *,
    max_size: int,
) -> tuple[bytes, str]:
    """Authorize one local source with Hermes before opening it."""
    authorized = authorize_local_media_path(source)
    if authorized is None:
        raise PermissionError("local media source is not authorized")
    return read_local_media(authorized, max_size=max_size)


def read_local_media(
    source: str,
    *,
    max_size: int,
) -> tuple[bytes, str]:
    """Read a regular local file up to Octo's outbound upload limit."""
    if source.startswith("file://"):
        parsed = urlparse(source)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("local media source is unavailable")
        source = unquote(parsed.path)
    if not isinstance(source, str) or not source or max_size <= 0:
        raise ValueError("local media source is unavailable")
    try:
        candidate = Path(source).expanduser()
        before = candidate.lstat()
    except OSError as exc:
        raise ValueError("local media source is unavailable") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("local media source is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ValueError("local media source is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError("local media source is unavailable")
        if opened.st_size > max_size:
            raise ValueError(f"media exceeds the {max_size}-byte limit")
        data = bytearray()
        while len(data) <= max_size:
            chunk = os.read(descriptor, min(1 << 20, max_size + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > max_size:
            raise ValueError(f"media exceeds the {max_size}-byte limit")
        return bytes(data), candidate.name
    finally:
        os.close(descriptor)


def parse_image_dimensions(data: bytes, mime: str) -> tuple[int, int] | None:
    """
    Parse image dimensions from buffer header bytes (PNG/JPEG/GIF/WebP).

    Returns:
        (width, height) or None if parsing fails.
    """
    try:
        if mime == "image/png" and len(data) > 24:
            width = struct.unpack(">I", data[16:20])[0]
            height = struct.unpack(">I", data[20:24])[0]
            return width, height
        if mime in ("image/jpeg", "image/jpg") and len(data) > 2:
            offset = 2
            while offset < len(data) - 8:
                if data[offset] != 0xFF:
                    break
                marker = data[offset + 1]
                if marker in (0xC0, 0xC2):
                    height = struct.unpack(">H", data[offset + 5 : offset + 7])[0]
                    width = struct.unpack(">H", data[offset + 7 : offset + 9])[0]
                    return width, height
                seg_len = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
                offset += 2 + seg_len
        if mime == "image/gif" and len(data) > 10:
            width = struct.unpack("<H", data[6:8])[0]
            height = struct.unpack("<H", data[8:10])[0]
            return width, height
        if mime == "image/webp" and len(data) > 30:
            if data[12:16] == b"VP8 " and len(data) > 29:
                width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
                height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
                return width, height
    except Exception:
        pass
    return None


# ─── Core HTTP Helpers ───────────────────────────────────────────────────────


async def post_json(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: aiohttp.ClientTimeout | None = None,
) -> Any | None:
    """
    POST JSON to a Octo API endpoint with Bearer auth.

    Args:
        session: aiohttp client session.
        api_url: Base API URL (e.g. https://api.botgate.cn).
        bot_token: Bot authentication token.
        path: API path (e.g. /v1/bot/sendMessage).
        payload: JSON body dict.
        timeout: Optional request-specific timeout.

    Returns:
        Parsed JSON response dict, or None if empty response.

    Raises:
        OctoApiError: On non-2xx responses, without exposing response text.
    """
    url = f"{api_url.rstrip('/')}{path}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bot_token}",
    }
    async with session.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout or DEFAULT_TIMEOUT,
        allow_redirects=False,
    ) as resp:
        if not 200 <= resp.status < 300:
            raise _response_error(path, resp)
        text = await resp.text()
        if not text:
            return None
        return await resp.json(content_type=None)


async def get_json(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    path: str,
) -> Any | None:
    """
    GET JSON from a Octo API endpoint with Bearer auth.

    Returns:
        Parsed JSON response value, or None for an empty successful response.
    """
    url = f"{api_url.rstrip('/')}{path}"
    headers = {"Authorization": f"Bearer {bot_token}"}
    async with session.get(
        url,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
        allow_redirects=False,
    ) as resp:
        if not 200 <= resp.status < 300:
            raise _response_error(path, resp)
        text = await resp.text()
        if not text:
            return None
        return await resp.json(content_type=None)


def _string_items(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _parse_card_templating(value: Any) -> CardTemplatingCapability | None:
    if not isinstance(value, dict):
        return None
    templates: list[CardTemplateCapability] = []
    raw_templates = value.get("templates")
    if isinstance(raw_templates, list):
        for raw_template in raw_templates:
            if not isinstance(raw_template, dict):
                continue
            template_id = raw_template.get("id")
            version = raw_template.get("version")
            if not isinstance(template_id, str) or not isinstance(version, str):
                continue
            views: list[CardTemplateViewCapability] = []
            raw_views = raw_template.get("views")
            if isinstance(raw_views, list):
                for raw_view in raw_views:
                    if not isinstance(raw_view, dict):
                        continue
                    name = raw_view.get("name")
                    wire_profile = raw_view.get("wire_profile")
                    if not isinstance(name, str) or not isinstance(wire_profile, str):
                        continue
                    views.append(
                        CardTemplateViewCapability(
                            name=name,
                            wire_profile=wire_profile,
                            states=_string_items(raw_view.get("states")),
                            submit_actions=_string_items(
                                raw_view.get("submit_actions")
                            ),
                        )
                    )
            templates.append(
                CardTemplateCapability(
                    id=template_id,
                    version=version,
                    views=tuple(views),
                )
            )
    wire = value.get("wire")
    return CardTemplatingCapability(
        supported=value.get("supported") is True,
        wire=wire if isinstance(wire, str) else "",
        templates=tuple(templates),
    )


async def get_card_profile(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
) -> CardProfileManifest:
    """Read and normalize the optional Type-17 capability manifest."""
    path = "/v1/bot/card/profile"
    try:
        raw = await get_json(session, api_url, bot_token, path)
    except OctoApiError as exc:
        if exc.status == 404:
            return CardProfileManifest(available=False, enabled=False)
        raise
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise OctoApiError(path, reason="transport failure") from exc
    except (TypeError, ValueError):
        raw = None

    if not isinstance(raw, dict):
        return CardProfileManifest(available=True, enabled=False)

    def string_tuple(name: str) -> tuple[str, ...] | None:
        value = raw.get(name)
        if not isinstance(value, list):
            return None
        return tuple(item for item in value if isinstance(item, str))

    card_version = raw.get("card_version")
    limits = raw.get("limits")
    return CardProfileManifest(
        available=True,
        enabled=raw.get("enabled") is True or raw.get("enabled") == 1,
        profiles=string_tuple("profiles"),
        card_version=card_version if isinstance(card_version, str) else None,
        elements=string_tuple("elements"),
        inputs=string_tuple("inputs"),
        actions=string_tuple("actions"),
        limits=limits if isinstance(limits, dict) else {},
        templating=_parse_card_templating(raw.get("templating")),
    )


async def fetch_bot_events(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    since_event_id: int = 0,
    limit: int = 20,
    wait_seconds: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch bot events strictly after the supplied server cursor."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("event limit must be an integer")
    if wait_seconds is not None and (
        isinstance(wait_seconds, bool) or not isinstance(wait_seconds, int)
    ):
        raise ValueError("event wait_seconds must be an integer")

    bounded_limit = max(1, min(100, limit))
    bounded_wait = (
        min(MAX_EVENT_WAIT_SECONDS, wait_seconds)
        if wait_seconds is not None and wait_seconds > 0
        else 0
    )
    body: dict[str, Any] = {
        "event_id": since_event_id,
        "limit": bounded_limit,
    }
    if bounded_wait > 0:
        body["wait"] = bounded_wait
    request_timeout = aiohttp.ClientTimeout(
        total=max(
            DEFAULT_TIMEOUT.total or 0,
            bounded_wait + EVENT_POLL_WAIT_MARGIN_SECONDS,
        )
    )
    result = await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/events",
        body,
        timeout=request_timeout,
    )
    if not isinstance(result, dict):
        return []
    events = result.get("results")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


async def ack_bot_event(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    event_id: int,
) -> None:
    """Acknowledge a bot event after the caller has accepted it locally."""
    if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
        raise ValueError("event_id must be a non-negative integer")
    await post_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/events/{event_id}/ack",
        {},
    )


async def set_commands(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    commands: list[dict[str, str]],
) -> None:
    """Replace the Bot's Octo command menu with a complete snapshot."""
    await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/setCommands",
        {"commands": commands},
    )


# ─── Bot Registration ────────────────────────────────────────────────────────


async def register_bot(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    force_refresh: bool = False,
) -> BotRegisterResp:
    """
    Register bot and obtain WS connection credentials.

    Args:
        session: aiohttp client session.
        api_url: Base API URL.
        bot_token: Bot authentication token.
        force_refresh: If True, force token refresh.

    Returns:
        BotRegisterResp with ws_url, im_token, robot_id, etc.
    """
    path = "/v1/bot/register"
    if force_refresh:
        path += "?force_refresh=true"

    result = await post_json(session, api_url, bot_token, path, {})
    if not result:
        raise RuntimeError("Octo bot registration returned empty response")

    return BotRegisterResp(
        robot_id=result["robot_id"],
        im_token=result["im_token"],
        ws_url=result["ws_url"],
        api_url=result.get("api_url", api_url),
        owner_uid=result["owner_uid"],
        owner_channel_id=result["owner_channel_id"],
    )


# ─── Message Sending ─────────────────────────────────────────────────────────


def _parse_send_message_result(
    result: object,
    *,
    requested_client_msg_no: str,
) -> SendMessageResult:
    data = result if isinstance(result, dict) else {}
    raw_message_id = data.get("message_id")
    message_id = (
        str(raw_message_id)
        if isinstance(raw_message_id, (str, int))
        and not isinstance(raw_message_id, bool)
        else None
    )
    if not message_id:
        raise OctoApiError(
            "/v1/bot/sendMessage",
            reason="missing message_id in success response",
        )
    raw_message_seq = data.get("message_seq")
    try:
        message_seq = (
            int(raw_message_seq)
            if isinstance(raw_message_seq, (str, int))
            and not isinstance(raw_message_seq, bool)
            else None
        )
    except (TypeError, ValueError):
        message_seq = None
    raw_client_msg_no = data.get("client_msg_no")
    response_client_msg_no = (
        str(raw_client_msg_no)
        if isinstance(raw_client_msg_no, (str, int))
        and not isinstance(raw_client_msg_no, bool)
        else requested_client_msg_no
    )
    return SendMessageResult(
        message_id=message_id,
        message_seq=message_seq,
        client_msg_no=response_client_msg_no,
    )


async def send_message(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    channel_id: str,
    channel_type: ChannelType,
    content: str,
    mention_uids: list[str] | None = None,
    mention_entities: list[MentionEntity] | None = None,
    mention_all: bool = False,
    stream_no: str | None = None,
    reply_msg_id: str | None = None,
    client_msg_no: str | None = None,
    on_behalf_of: str | None = None,
) -> SendMessageResult:
    """
    Send a text message to a channel.

    Args:
        session: aiohttp client session.
        api_url: Base API URL.
        bot_token: Bot authentication token.
        channel_id: Target channel ID.
        channel_type: DM (1) or Group (2).
        content: Message text content.
        mention_uids: UIDs to @mention.
        mention_entities: Precise mention entities.
        mention_all: If True, @all.
        stream_no: Optional stream number for streaming messages.
        reply_msg_id: Optional message ID to reply to.
    """
    payload: dict[str, Any] = {
        "type": MessageType.Text,
        "content": content,
    }

    # Build mention field
    if mention_uids or mention_entities or mention_all:
        mention: dict[str, Any] = {}
        if mention_uids:
            mention["uids"] = mention_uids
        if mention_entities:
            mention["entities"] = [
                {"uid": e.uid, "offset": e.offset, "length": e.length}
                for e in mention_entities
            ]
        if mention_all:
            mention["all"] = 1
        payload["mention"] = mention

    # Add reply field
    if reply_msg_id:
        payload["reply"] = {"message_id": reply_msg_id}

    body: dict[str, Any] = {
        "channel_id": channel_id,
        "channel_type": channel_type,
        "payload": payload,
        "client_msg_no": client_msg_no or str(uuid.uuid4()),
    }
    if on_behalf_of:
        body["on_behalf_of"] = on_behalf_of
    if stream_no:
        body["stream_no"] = stream_no

    result = await post_json(session, api_url, bot_token, "/v1/bot/sendMessage", body)
    return _parse_send_message_result(
        result,
        requested_client_msg_no=body["client_msg_no"],
    )


def _validate_template_frame(
    template_ref: Mapping[str, str],
    state: str,
    data: dict[str, Any],
) -> None:
    if not isinstance(template_ref, Mapping) or set(template_ref) != {"id", "version"}:
        raise ValueError("template_ref must contain exactly id and version")
    template_id = template_ref.get("id")
    version = template_ref.get("version")
    if (
        not isinstance(template_id, str)
        or not template_id.strip()
        or not isinstance(version, str)
        or not version.strip()
    ):
        raise ValueError("template_ref id/version are required")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("template state is required")
    if not isinstance(data, dict) or "state" not in data:
        raise ValueError("data must be a plain object with own state")
    if data["state"] != state:
        raise ValueError("data.state must match state")


async def send_template_card_message(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    channel_id: str,
    channel_type: ChannelType,
    template_ref: Mapping[str, str],
    state: str,
    data: dict[str, Any],
    client_msg_no: str | None = None,
) -> SendMessageResult:
    """Send one server-Registry Type-17 card frame."""
    if not isinstance(channel_id, str) or not channel_id.strip():
        raise ValueError("channel_id is required")
    _validate_template_frame(template_ref, state, data)
    requested_client_msg_no = client_msg_no or str(uuid.uuid4())
    result = await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/sendMessage",
        {
            "channel_id": channel_id,
            "channel_type": channel_type,
            "client_msg_no": requested_client_msg_no,
            "payload": {
                "type": MessageType.InteractiveCard,
                "template_ref": dict(template_ref),
                "state": state,
                "data": data,
            },
        },
    )
    return _parse_send_message_result(
        result,
        requested_client_msg_no=requested_client_msg_no,
    )


async def edit_template_card_message(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    channel_id: str,
    channel_type: ChannelType,
    message_id: str,
    template_ref: Mapping[str, str],
    state: str,
    data: dict[str, Any],
    card_seq: int,
    transient: bool | None = None,
) -> Any | None:
    """Replace a server-Registry Type-17 card frame."""
    if not isinstance(message_id, str) or not message_id:
        raise ValueError("message_id is required")
    if not isinstance(channel_id, str) or not channel_id.strip():
        raise ValueError("channel_id is required")
    if isinstance(card_seq, bool) or not isinstance(card_seq, int) or card_seq <= 0:
        raise ValueError("card_seq must be a positive integer")
    _validate_template_frame(template_ref, state, data)
    body: dict[str, Any] = {
        "message_id": message_id,
        "channel_id": channel_id,
        "channel_type": channel_type,
        "template_ref": dict(template_ref),
        "state": state,
        "data": data,
        "card_seq": card_seq,
    }
    if transient is not None:
        body["transient"] = transient
    return await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/message/edit",
        body,
    )


async def send_card_message(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    channel_id: str,
    channel_type: ChannelType,
    card: dict[str, Any],
    plain: str | None = None,
    client_msg_no: str | None = None,
    profile: str | None = None,
    on_behalf_of: str | None = None,
) -> SendMessageResult:
    """Send an Adaptive Card using Octo's Type-17 envelope."""
    from . import cards as card_renderer

    resolved_profile = resolve_card_profile(card, profile)
    card_renderer.validate_card_limits(
        card,
        plain,
        None,
        profile=resolved_profile,
    )
    payload: dict[str, Any] = {
        "type": MessageType.InteractiveCard,
        "card": card,
        "profile": resolved_profile,
        "card_version": CARD_VERSION,
    }
    if plain is not None:
        payload["plain"] = plain

    requested_client_msg_no = client_msg_no or str(uuid.uuid4())
    body: dict[str, Any] = {
        "channel_id": channel_id,
        "channel_type": channel_type,
        "client_msg_no": requested_client_msg_no,
        "payload": payload,
    }
    if on_behalf_of:
        body["on_behalf_of"] = on_behalf_of

    result = await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/sendMessage",
        body,
    )
    return _parse_send_message_result(
        result,
        requested_client_msg_no=requested_client_msg_no,
    )


async def edit_card_message(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    channel_id: str,
    channel_type: ChannelType,
    message_id: str,
    card: dict[str, Any],
    card_seq: int,
    plain: str | None = None,
    transient: bool | None = None,
    profile: str | None = None,
    on_behalf_of: str | None = None,
) -> Any | None:
    """Replace a Type-17 card frame using a positive card sequence."""
    if isinstance(card_seq, bool) or not isinstance(card_seq, int) or card_seq <= 0:
        raise ValueError("card_seq must be a positive integer")
    from . import cards as card_renderer

    resolved_profile = resolve_card_profile(card, profile)
    card_renderer.validate_card_limits(
        card,
        plain,
        None,
        profile=resolved_profile,
        card_seq=card_seq,
        transient=transient,
    )
    frame: dict[str, Any] = {
        "type": MessageType.InteractiveCard,
        "card": card,
        "profile": resolved_profile,
        "card_version": CARD_VERSION,
    }
    if plain is not None:
        frame["plain"] = plain
    frame["card_seq"] = card_seq
    if transient is not None:
        frame["transient"] = transient

    content_edit = json.dumps(
        frame,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if (
        len(content_edit.encode("utf-8"))
        > card_renderer.DEFAULT_MAX_CARD_PAYLOAD_BYTES
    ):
        raise card_renderer.CardLimitError(
            "card exceeds max_payload_bytes"
        )
    body: dict[str, Any] = {
        "message_id": str(message_id),
        "channel_id": channel_id,
        "channel_type": channel_type,
        "content_edit": content_edit,
    }
    if on_behalf_of:
        body["on_behalf_of"] = on_behalf_of
    return await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/message/edit",
        body,
    )


async def edit_message(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    channel_id: str,
    channel_type: ChannelType,
    message_id: str,
    content: str,
    finalize: bool = False,
    on_behalf_of: str | None = None,
) -> Any | None:
    """Edit a text message using Octo's native edit envelope.

    ``finalize`` is a Hermes lifecycle argument.  The current Octo contract
    has no corresponding wire field, so it is deliberately accepted but not
    serialized.
    """
    del finalize
    frame: dict[str, Any] = {"type": MessageType.Text, "content": content}
    body: dict[str, Any] = {
        "message_id": str(message_id),
        "channel_id": channel_id,
        "channel_type": channel_type,
        "content_edit": json.dumps(frame),
    }
    if on_behalf_of:
        body["on_behalf_of"] = on_behalf_of
    return await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/message/edit",
        body,
    )


async def send_typing(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    channel_id: str,
    channel_type: ChannelType,
    on_behalf_of: str | None = None,
) -> None:
    """Send typing indicator to a channel."""
    body: dict[str, Any] = {
        "channel_id": channel_id,
        "channel_type": channel_type,
    }
    if on_behalf_of:
        body["on_behalf_of"] = on_behalf_of
    await post_json(session, api_url, bot_token, "/v1/bot/typing", body)


async def send_heartbeat(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
) -> Any | None:
    """Record an authenticated Bot API heartbeat (empty envelope by contract)."""
    return await post_json(session, api_url, bot_token, "/v1/bot/heartbeat", {})


async def send_media_message(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    channel_id: str,
    channel_type: ChannelType,
    msg_type: MessageType,
    url: str,
    name: str | None = None,
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    duration: int | None = None,
    reply_msg_id: str | None = None,
    client_msg_no: str | None = None,
    on_behalf_of: str | None = None,
) -> SendMessageResult:
    """
    Send a media message (image, file, etc.) to a channel.

    Args:
        msg_type: MessageType.Image, MessageType.File, etc.
        url: Media file URL.
        name: Filename (for File type).
        size: File size in bytes (for File type).
        width: Image/GIF/Video width.
        height: Image/GIF/Video height.
        duration: Voice/Video duration, when the caller has it.
        reply_msg_id: Optional message ID to reply to.
    """
    dimension_types = {MessageType.Image, MessageType.GIF, MessageType.Video}
    duration_types = {MessageType.Voice, MessageType.Video}
    if (width is not None or height is not None) and msg_type not in dimension_types:
        raise ValueError(f"width/height are not supported for media type {msg_type}")
    if duration is not None and msg_type not in duration_types:
        raise ValueError(f"duration is not supported for media type {msg_type}")

    payload: dict[str, Any] = {
        "type": msg_type,
        "url": url,
    }
    if name:
        payload["name"] = name
    if size is not None:
        payload["size"] = size
    if width is not None:
        payload["width"] = width
    if height is not None:
        payload["height"] = height
    if duration is not None:
        payload["duration"] = duration
    if reply_msg_id:
        payload["reply"] = {"message_id": reply_msg_id}

    requested_client_msg_no = client_msg_no or str(uuid.uuid4())
    body: dict[str, Any] = {
        "channel_id": channel_id,
        "channel_type": channel_type,
        "client_msg_no": requested_client_msg_no,
        "payload": payload,
    }
    if on_behalf_of:
        body["on_behalf_of"] = on_behalf_of
    result = await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/sendMessage",
        body,
    )
    return _parse_send_message_result(
        result,
        requested_client_msg_no=requested_client_msg_no,
    )


async def send_rich_text_message(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    channel_id: str,
    channel_type: ChannelType,
    blocks: list[RichTextBlock],
    plain: str | None = None,
    mention_uids: list[str] | None = None,
    mention_entities: list[MentionEntity] | None = None,
    mention_all: bool = False,
    reply_msg_id: str | None = None,
    client_msg_no: str | None = None,
    on_behalf_of: str | None = None,
) -> SendMessageResult:
    """
    Send a RichText(=14) 图文混排 message.

    Replaces the "text sendMessage + loop uploadMedia" pattern with a
    single POST carrying an ordered ``content`` block array — the caller
    uploads images first (to obtain url + width + height), assembles a
    mix of text and image blocks in display order, and passes them here.

    Contract (see octo-lib common/richtext.go):
      - `blocks` MUST be non-empty; text blocks require non-empty `text`,
        image blocks require http/https `url` and positive `width`/`height`.
        Validation is enforced by octo-server; this function only assembles.
      - `plain` is an optional redundant plain-text rendering for legacy
        clients; the server regenerates it authoritatively from `content`.

    Args:
        blocks: Ordered RichText block list (text/image interleaved).
        plain: Optional redundant plain-text (server-overridden).
        mention_uids/mention_entities/mention_all: Same shape as send_message.
        reply_msg_id: Optional message ID to reply to.
    """
    payload: dict[str, Any] = {
        "type": MessageType.RichText,
        "content": [b.to_dict() for b in blocks],
    }
    if plain is not None:
        payload["plain"] = plain

    if mention_uids or mention_entities or mention_all:
        mention: dict[str, Any] = {}
        if mention_uids:
            mention["uids"] = mention_uids
        if mention_entities:
            mention["entities"] = [
                {"uid": e.uid, "offset": e.offset, "length": e.length}
                for e in mention_entities
            ]
        if mention_all:
            mention["all"] = 1
        payload["mention"] = mention

    if reply_msg_id:
        payload["reply"] = {"message_id": reply_msg_id}

    body: dict[str, Any] = {
        "channel_id": channel_id,
        "channel_type": channel_type,
        "payload": payload,
        "client_msg_no": client_msg_no or str(uuid.uuid4()),
    }
    if on_behalf_of:
        body["on_behalf_of"] = on_behalf_of
    result = await post_json(session, api_url, bot_token, "/v1/bot/sendMessage", body)
    return _parse_send_message_result(
        result,
        requested_client_msg_no=body["client_msg_no"],
    )


async def send_read_receipt(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    channel_id: str,
    channel_type: ChannelType,
    message_ids: list[str] | None = None,
) -> None:
    """
    Send a read receipt to a channel.

    Args:
        channel_id: Channel to mark as read.
        channel_type: DM or Group.
        message_ids: Optional specific message IDs to acknowledge.
    """
    body: dict[str, Any] = {
        "channel_id": channel_id,
        "channel_type": channel_type,
    }
    if message_ids:
        body["message_ids"] = message_ids
    await post_json(session, api_url, bot_token, "/v1/bot/readReceipt", body)


# ─── Stream API ──────────────────────────────────────────────────────────────


async def stream_start(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    channel_id: str,
    channel_type: ChannelType,
    initial_content: str,
) -> str | None:
    """
    Start a streaming message.

    The initial content is sent as a base64-encoded JSON payload.

    Args:
        channel_id: Target channel.
        channel_type: DM or Group.
        initial_content: Initial text content.

    Returns:
        stream_no (stream identifier) or None on failure.
    """
    import json

    payload_b64 = base64.b64encode(
        json.dumps({"type": 1, "content": initial_content}).encode("utf-8")
    ).decode("ascii")

    result = await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/stream/start",
        {
            "channel_id": channel_id,
            "channel_type": channel_type,
            "payload": payload_b64,
        },
    )
    return result.get("stream_no") if result else None


async def stream_end(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    stream_no: str,
    channel_id: str,
    channel_type: ChannelType,
) -> None:
    """
    End a streaming message.

    Args:
        stream_no: Stream identifier from stream_start.
        channel_id: Target channel.
        channel_type: DM or Group.
    """
    await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/stream/end",
        {
            "stream_no": stream_no,
            "channel_id": channel_id,
            "channel_type": channel_type,
        },
    )


# ─── Backend-agnostic presigned upload ───────────────────────────────────────


async def get_upload_presign(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    filename: str,
    file_size: int,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Fetch a server-issued presigned PUT target for the exact file body."""
    if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size <= 0:
        raise ValueError("file_size must be a positive integer")
    query = urlencode({
        "filename": filename,
        "fileSize": str(file_size),
        **({"contentType": content_type} if content_type else {}),
    })
    path = "/v1/bot/upload/presigned"
    url = f"{api_url.rstrip('/')}{path}?{query}"
    headers = {"Authorization": f"Bearer {bot_token}"}

    async with session.get(
        url,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
        allow_redirects=False,
    ) as resp:
        if not 200 <= resp.status < 300:
            raise _response_error(path, resp)
        data = await resp.json()

    if not isinstance(data, dict):
        raise RuntimeError(f"Octo API {path} returned an invalid response")
    raw_headers = data.get("headers", {})
    if raw_headers is None:
        raw_headers = {}
    if not isinstance(raw_headers, dict) or any(
        not isinstance(key, str)
        or not key
        or "\r" in key
        or "\n" in key
        or not isinstance(value, str)
        or "\r" in value
        or "\n" in value
        for key, value in raw_headers.items()
    ):
        raise RuntimeError(f"Octo API {path} returned invalid signed headers")
    signed_headers = dict(raw_headers)
    for field in ("uploadUrl", "downloadUrl"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise RuntimeError(
                f"Octo API {path} returned incomplete response: missing {field}"
            )

    header_lookup = {key.lower(): value for key, value in signed_headers.items()}
    content_type = data.get("contentType")
    if not isinstance(content_type, str) or not content_type:
        content_type = header_lookup.get("content-type") or "application/octet-stream"
    result: dict[str, Any] = {
        "uploadUrl": data["uploadUrl"],
        "downloadUrl": data["downloadUrl"],
        "contentType": content_type,
    }
    content_disposition = data.get("contentDisposition")
    if not isinstance(content_disposition, str):
        content_disposition = header_lookup.get("content-disposition")
    if content_disposition:
        result["contentDisposition"] = content_disposition
    if signed_headers:
        result["headers"] = signed_headers
    return result

def _validate_presigned_upload_origin(
    policy: TransportPolicy | None,
    upload_url: str,
) -> None:
    """Allow public presigns and exact policy-configured private origins."""
    try:
        parsed = urlparse(upload_url)
        host = canonical_url_host(upload_url)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("unsafe presigned upload URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("unsafe presigned upload URL")

    literal = _canonical_download_ip(host)
    if host in _DOWNLOAD_METADATA_HOSTS or (
        literal is not None
        and (
            literal.is_link_local
            or literal.is_multicast
            or literal.is_reserved
            or literal.is_unspecified
        )
    ):
        raise RuntimeError("unsafe presigned upload URL")
    if (
        policy is not None
        and policy.is_download_endpoint_trusted(upload_url)
        and not policy.is_download_url_trusted(upload_url)
    ):
        raise RuntimeError("unsafe presigned upload URL")

    if is_private_or_metadata_host(host) and (
        policy is None or not policy.is_download_url_trusted(upload_url)
    ):
        raise RuntimeError("unsafe presigned upload URL")

async def upload_file_to_presigned_url(
    *,
    upload_url: str,
    download_url: str,
    file_data: bytes,
    content_type: str,
    content_disposition: str | None = None,
    headers: Mapping[str, str] | None = None,
    policy: TransportPolicy | None = None,
) -> str:
    """PUT one exact body while replaying the server-signed request headers."""
    _validate_presigned_upload_origin(policy, upload_url)
    put_headers = dict(headers or ())
    if any(
        not isinstance(key, str)
        or not key
        or "\r" in key
        or "\n" in key
        or not isinstance(value, str)
        or "\r" in value
        or "\n" in value
        for key, value in put_headers.items()
    ):
        raise ValueError("presigned headers must be safe strings")
    header_names = {key.lower(): key for key in put_headers}
    expected_length = str(len(file_data))
    signed_length_key = header_names.get("content-length")
    if (
        signed_length_key is not None
        and put_headers[signed_length_key] != expected_length
    ):
        raise ValueError("presigned Content-Length does not match file body")
    if signed_length_key is None:
        put_headers["Content-Length"] = expected_length
    if "content-type" not in header_names:
        put_headers["Content-Type"] = content_type
    if content_disposition is not None and "content-disposition" not in header_names:
        put_headers["Content-Disposition"] = content_disposition

    upload_timeout = aiohttp.ClientTimeout(total=300)
    upload_policy = policy or TransportPolicy()
    owned_upload_session = new_guarded_http_session(policy=upload_policy)
    upload_session = owned_upload_session
    try:
        async with upload_session.put(
            upload_url,
            data=file_data,
            headers=put_headers,
            timeout=upload_timeout,
            allow_redirects=False,
        ) as resp:
            if not 200 <= resp.status < 300:
                raise RuntimeError(
                    f"Presigned PUT upload failed (HTTP {resp.status})"
                )
    finally:
        if owned_upload_session is not None:
            await owned_upload_session.close()
    return download_url


async def upload_and_get_url(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    filename: str,
    file_data: bytes,
    content_type: str,
    policy: TransportPolicy | None = None,
) -> str:
    """Presign and upload a file through the server-selected storage backend."""
    presign = await get_upload_presign(
        session,
        api_url,
        bot_token,
        filename=filename,
        file_size=len(file_data),
        content_type=content_type,
    )
    _validate_presigned_upload_origin(policy, presign["uploadUrl"])
    return await upload_file_to_presigned_url(
        upload_url=presign["uploadUrl"],
        download_url=presign["downloadUrl"],
        file_data=file_data,
        content_type=presign["contentType"],
        content_disposition=presign.get("contentDisposition"),
        headers=presign.get("headers"),
        policy=policy,
    )


_DOWNLOAD_METADATA_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.goog",
    "100.100.100.200",
    "fd00:ec2::254",
}
_DOWNLOAD_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_DOWNLOAD_REDIRECTS = 5


def _canonical_download_ip(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    normalized = host.lower().strip("[]").rstrip(".")
    try:
        return ipaddress.ip_address(normalized)
    except ValueError:
        try:
            return ipaddress.ip_address(socket.inet_aton(normalized))
        except OSError:
            return None


def _validate_download_url(
    url: str,
    *,
    policy: TransportPolicy | None = None,
) -> str:
    """Validate one download hop before any network I/O."""
    try:
        parsed = urlparse(url)
        host = canonical_url_host(url)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("unsafe download URL") from exc
    if parsed.scheme not in {"http", "https"} or not host:
        raise RuntimeError("unsafe download URL")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("unsafe download URL")
    if host in _DOWNLOAD_METADATA_HOSTS:
        raise RuntimeError("unsafe download URL")

    literal = _canonical_download_ip(host)
    if literal is not None and (
        literal.is_link_local
        or literal.is_multicast
        or literal.is_reserved
        or literal.is_unspecified
    ):
        raise RuntimeError("unsafe download URL")
    trusted = policy is not None and policy.is_download_url_trusted(url)
    if (
        policy is not None
        and policy.is_download_endpoint_trusted(url)
        and not trusted
    ):
        raise RuntimeError("unsafe download URL")
    if (
        is_private_or_metadata_host(host)
        or (
            literal is not None
            and (literal.is_loopback or literal.is_private)
        )
    ) and not trusted:
        raise RuntimeError("unsafe download URL")
    return url


def _content_disposition_filename(value: str) -> str | None:
    """Extract RFC 5987 ``filename*`` or legacy ``filename``."""
    fallback: str | None = None
    for raw_part in value.split(";")[1:]:
        key, separator, raw_value = raw_part.strip().partition("=")
        if not separator:
            continue
        candidate = raw_value.strip().strip('"').strip("'")
        if not candidate or "\r" in candidate or "\n" in candidate:
            continue
        if key.lower() == "filename*":
            charset, marker, encoded = candidate.partition("''")
            if marker and charset.lower() == "utf-8":
                decoded = unquote(encoded)
                if decoded and "\r" not in decoded and "\n" not in decoded:
                    return decoded
        elif key.lower() == "filename":
            fallback = candidate
    return fallback



async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    max_size: int = 500 * 1024 * 1024,
    timeout_seconds: int = 300,
    *,
    policy: TransportPolicy | None = None,
) -> tuple[bytes, str, str]:
    """
    Download a file from a URL.

    Args:
        url: URL to download.
        timeout_seconds: Total download timeout across every redirect hop.

    Returns:
        (file_data, content_type, filename)

    Raises:
        RuntimeError: If file is too large or download fails.
    """
    deadline = time.monotonic() + timeout_seconds
    current_url = url
    trusted_policy = policy
    async with asyncio.timeout(timeout_seconds):
        for redirect_count in range(_MAX_DOWNLOAD_REDIRECTS + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Download failed (timeout)")
            _validate_download_url(
                current_url,
                policy=trusted_policy,
            )
            hop_timeout = aiohttp.ClientTimeout(total=remaining)
            async with session.get(
                current_url,
                timeout=hop_timeout,
                allow_redirects=False,
            ) as resp:
                if resp.status in _DOWNLOAD_REDIRECT_STATUSES:
                    location = resp.headers.get("Location")
                    if not location or redirect_count >= _MAX_DOWNLOAD_REDIRECTS:
                        raise RuntimeError("Download failed (invalid redirect)")
                    current_url = urljoin(current_url, location)
                    _validate_download_url(
                        current_url,
                        policy=trusted_policy,
                    )
                    continue
                if not 200 <= resp.status < 300:
                    # Source URLs are frequently pre-signed and must not be copied into
                    # exceptions that can reach logs or SendResult.error.
                    raise RuntimeError(f"Download failed (HTTP {resp.status})")

                content_type = resp.headers.get("Content-Type", "application/octet-stream")

                # Prefer a safe RFC 5987 filename, then a safe URL basename.
                cd_filename = safe_media_filename(
                    _content_disposition_filename(
                        resp.headers.get("Content-Disposition", "")
                    )
                )
                url_filename = safe_media_filename(
                    unquote(urlparse(current_url).path.split("/")[-1])
                )
                filename = cd_filename or url_filename or "file"

                cl = resp.headers.get("Content-Length")
                if cl:
                    try:
                        content_length = int(cl)
                    except (TypeError, ValueError):
                        raise RuntimeError(
                            "Download failed (invalid Content-Length)"
                        ) from None
                    if content_length < 0:
                        raise RuntimeError("Download failed (invalid Content-Length)")
                    if content_length > max_size:
                        raise RuntimeError(
                            f"File too large ({content_length} bytes, max {max_size})"
                        )

                data = bytearray()
                async for chunk in resp.content.iter_any():
                    data.extend(chunk)
                    if len(data) > max_size:
                        raise RuntimeError(f"File too large (>{max_size} bytes)")

                return bytes(data), content_type, filename

    raise RuntimeError("Download failed (too many redirects)")


# ─── Channel History ─────────────────────────────────────────────────────────


async def get_channel_messages(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    channel_id: str,
    channel_type: ChannelType,
    limit: int = 20,
    start_message_seq: int = 0,
    end_message_seq: int = 0,
) -> list[dict[str, Any]]:
    """
    Fetch channel history messages.

    Uses /v1/bot/messages/sync API. Payloads are base64-encoded JSON.

    Args:
        channel_id: Channel to fetch history from.
        channel_type: DM or Group.
        limit: Max messages to fetch (default 20).
        start_message_seq: Start sequence (0 = from beginning).
        end_message_seq: End sequence (0 = to latest).

    Returns:
        URL-free message summaries safe to expose to the model.
    """
    result = await post_json(
        session,
        api_url,
        bot_token,
        "/v1/bot/messages/sync",
        {
            "channel_id": channel_id,
            "channel_type": channel_type,
            "limit": limit,
            "start_message_seq": start_message_seq,
            "end_message_seq": end_message_seq,
            "pull_mode": 1,  # 1 = pull up (newer messages)
        },
    )

    if not result:
        return []

    messages = result.get("messages", [])
    parsed = []
    for m in messages:
        payload: dict[str, Any] = {}
        raw_payload = m.get("payload")
        if raw_payload:
            try:
                decoded = base64.b64decode(raw_payload).decode("utf-8")
                import json

                payload = json.loads(decoded)
            except Exception:
                if isinstance(raw_payload, dict):
                    payload = raw_payload

        message_type = payload.get("type")
        raw_name = payload.get("name")
        name = safe_media_filename(raw_name)
        if isinstance(raw_name, str) and name is None:
            normalized_name = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
            name = safe_media_filename(normalized_name)
        mention = payload.get("mention")
        if isinstance(mention, dict):
            from .mention import MAX_MENTIONS_PER_MESSAGE

            raw_entities = mention.get("entities")
            if isinstance(raw_entities, list):
                mention = {
                    **mention,
                    "entities": raw_entities[:MAX_MENTIONS_PER_MESSAGE],
                }
        content = payload.get("content", "")
        if message_type == MessageType.File:
            name = name or "未知文件"
            content = f"[文件: {name}]"
        elif message_type in {
            MessageType.Image,
            MessageType.GIF,
            MessageType.Voice,
            MessageType.Video,
        }:
            content = f"[{MessageType(message_type).name}]"
        elif message_type == MessageType.RichText:
            content = "[图文消息]"
        elif message_type == MessageType.MultipleForward:
            content = "[合并转发消息]"
        if isinstance(content, str):
            from .mention import neutralize_structured_mention_envelopes

            content = neutralize_structured_mention_envelopes(content)

        parsed.append({
            "from_uid": m.get("from_uid", "unknown"),
            "type": message_type,
            "name": name,
            "content": content if isinstance(content, str) else "",
            "mention": mention if isinstance(mention, dict) else None,
            # API returns seconds, convert to ms
            "timestamp": (m.get("timestamp", int(time.time()))) * 1000,
        })
    return parsed


# ─── Group API ────────────────────────────────────────────────────────────────


async def fetch_bot_groups(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
) -> list[GroupInfo]:
    """Fetch and normalize the groups visible to the bot."""
    data = await get_json(session, api_url, bot_token, "/v1/bot/groups")
    raw_groups = data.get("groups") if isinstance(data, dict) else data
    if not isinstance(raw_groups, list):
        raise RuntimeError("malformed group snapshot")
    groups: list[GroupInfo] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise RuntimeError("malformed group snapshot")
        group_no = raw_group.get("group_no")
        if not isinstance(group_no, str) or not group_no:
            raise RuntimeError("malformed group snapshot")
        raw_name = raw_group.get("name")
        groups.append(
            GroupInfo(
                group_no=group_no,
                name=raw_name if isinstance(raw_name, str) else "",
                extra={
                    key: value
                    for key, value in raw_group.items()
                    if key not in {"group_no", "name"}
                },
            )
        )
    return groups


async def get_group_members(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    group_no: str,
) -> list[GroupMember]:
    """
    Get members of a group.

    Args:
        group_no: Group ID (channel_id).

    Returns:
        List of GroupMember objects.
    """
    data = await get_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/members",
    )
    # Normalize: API may return {members: [...]} or bare [...].  A successful
    # but structurally empty response remains a legitimate empty roster.
    members_raw = data.get("members", data) if isinstance(data, dict) else data
    if not isinstance(members_raw, list):
        return []
    return [
        GroupMember(
            uid=m.get("uid", ""),
            name=m.get("name", ""),
            role=m.get("role"),
            robot=m.get("robot"),
        )
        for m in members_raw
    ]


async def get_group_info(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    group_no: str,
) -> GroupInfo:
    """
    Get information about a group.

    Args:
        group_no: Group ID (channel_id).

    Returns:
        GroupInfo with group_no, name, and extra fields.

    Raises:
        RuntimeError: If the API call fails.
    """
    data = await get_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}",
    )
    if not isinstance(data, dict):
        raise RuntimeError("Octo group info returned an invalid response")
    known_keys = {"group_no", "name"}
    return GroupInfo(
        group_no=data.get("group_no", group_no),
        name=data.get("name", ""),
        extra={k: v for k, v in data.items() if k not in known_keys},
    )


async def fetch_user_info(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    uid: str,
) -> dict[str, str] | None:
    """
    Fetch user info by UID.

    Returns:
        Dict with uid, name, avatar keys, or None if unavailable.
    """
    url = f"{api_url.rstrip('/')}/v1/bot/user/info"
    headers = {"Authorization": f"Bearer {bot_token}"}
    try:
        async with session.get(
            url,
            headers=headers,
            params={"uid": uid},
            timeout=aiohttp.ClientTimeout(total=5),
            allow_redirects=False,
        ) as resp:
            if resp.status == 404:
                return None
            if not 200 <= resp.status < 300:
                logger.error("octo: fetch_user_info failed (HTTP %d)", resp.status)
                return None
            data = await resp.json()
            if data and data.get("name"):
                return {
                    "uid": data.get("uid", uid),
                    "name": data["name"],
                    "avatar": data.get("avatar", ""),
                }
            return None
    except Exception as exc:
        logger.error("octo: fetch_user_info failed (%s)", type(exc).__name__)
        return None


# ─── GROUP.md API ─────────────────────────────────────────────────────────────


async def get_group_md(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    group_no: str,
) -> dict[str, Any] | None:
    """
    Fetch GROUP.md content for a group.

    Returns:
        Dict with content, version, updated_at, updated_by, or None on 404.
    """
    path = f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/md"
    url = f"{api_url.rstrip('/')}{path}"
    headers = {"Authorization": f"Bearer {bot_token}"}
    async with session.get(
        url,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
        allow_redirects=False,
    ) as resp:
        if resp.status == 404:
            return None
        if not 200 <= resp.status < 300:
            raise _response_error(path, resp)
        return await resp.json()


async def update_group_md(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    group_no: str,
    content: str,
) -> dict[str, Any] | None:
    """
    Update GROUP.md content for a group.

    Returns:
        Parsed response dict, or None for a successful empty response.

    Raises:
        OctoApiError: The server returned a non-2xx response.
    """
    return await put_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/md",
        {"content": content},
    )


# ─── Bot DELETE / generic helpers ────────────────────────────────────────────


async def delete_json(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """DELETE on a Octo endpoint, raising on non-2xx."""
    url = f"{api_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json",
    }
    kwargs: dict[str, Any] = {
        "headers": headers,
        "timeout": DEFAULT_TIMEOUT,
        "allow_redirects": False,
    }
    if payload is not None:
        kwargs["data"] = json.dumps(payload)
    async with session.delete(url, **kwargs) as resp:
        if not 200 <= resp.status < 300:
            raise _response_error(path, resp)
        text = await resp.text()
        return json.loads(text) if text else None


async def put_json(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """PUT JSON on a Octo endpoint, raising on non-2xx."""
    url = f"{api_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json",
    }
    async with session.put(
        url,
        data=json.dumps(payload),
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
        allow_redirects=False,
    ) as resp:
        if not 200 <= resp.status < 300:
            raise _response_error(path, resp)
        text = await resp.text()
        return json.loads(text) if text else None


# ─── Space-level operations ──────────────────────────────────────────────────


async def search_space_members(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    keyword: str | None = None,
    space_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fuzzy-search members of the bot's space (org).

    Returns ``[{uid, name, robot}, ...]`` (empty list when no matches).
    """
    qs: list[str] = []
    if keyword:
        qs.append(f"keyword={quote(keyword)}")
    if space_id:
        qs.append(f"space_id={quote(space_id)}")
    qs.append(f"limit={int(max(1, min(limit, 500)))}")
    path = "/v1/bot/space/members?" + "&".join(qs)
    result = await get_json(session, api_url, bot_token, path)
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        members = result.get("members")
        return members if isinstance(members, list) else []
    return []


# ─── Group create / update / membership ──────────────────────────────────────


async def create_group(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    members: list[str],
    creator: str,
    name: str | None = None,
    space_id: str | None = None,
) -> dict[str, Any]:
    """Create a new group with the given members.

    *creator* is the uid that becomes the group owner.
    """
    payload: dict[str, Any] = {"members": members, "creator": creator}
    if name is not None:
        payload["name"] = name
    if space_id is not None:
        payload["space_id"] = space_id
    result = await post_json(
        session, api_url, bot_token, "/v1/bot/createGroup", payload
    )
    return result or {}


async def update_group(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    name: str | None = None,
    notice: str | None = None,
) -> None:
    """Update a group's metadata (name/notice).

    Raises ``ValueError`` rather than reporting a successful no-op when the
    caller supplies no mutable fields.
    """
    payload: dict[str, Any] = {}
    if name is not None:
        payload["name"] = name
    if notice is not None:
        payload["notice"] = notice
    if not payload:
        raise ValueError("update_group requires at least one of: name, notice")
    await put_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/info",
        payload,
    )


async def add_group_members(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    members: list[str],
) -> dict[str, Any]:
    result = await post_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/members/add",
        {"members": members},
    )
    return result or {}


async def remove_group_members(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    members: list[str],
) -> dict[str, Any]:
    result = await post_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/members/remove",
        {"members": members},
    )
    return result or {}


# ─── Threads ─────────────────────────────────────────────────────────────────


async def create_thread(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    name: str,
    source_message_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name}
    if source_message_id is not None:
        payload["source_message_id"] = source_message_id
    result = await post_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads",
        payload,
    )
    return result or {}


async def list_threads(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
) -> list[dict[str, Any]]:
    result = await get_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads",
    )
    if not result:
        return []
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []
    threads = result.get("threads")
    return threads if isinstance(threads, list) else []


async def get_thread(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    short_id: str,
) -> dict[str, Any]:
    result = await get_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads/{_api_path_segment(short_id, 'short_id')}",
    )
    return result or {}


async def delete_thread(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    short_id: str,
) -> None:
    await delete_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads/{_api_path_segment(short_id, 'short_id')}",
    )


async def list_thread_members(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    short_id: str,
) -> list[dict[str, Any]]:
    result = await get_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads/{_api_path_segment(short_id, 'short_id')}/members",
    )
    if not result:
        return []
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []
    members = result.get("members")
    return members if isinstance(members, list) else []


async def join_thread(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    short_id: str,
) -> None:
    await post_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads/{_api_path_segment(short_id, 'short_id')}/join",
        {},
    )


async def leave_thread(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    short_id: str,
) -> None:
    await post_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads/{_api_path_segment(short_id, 'short_id')}/leave",
        {},
    )


async def get_thread_md(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    short_id: str,
) -> dict[str, Any]:
    """Read THREAD.md. Returns {} on 404."""
    try:
        result = await get_json(
            session,
            api_url,
            bot_token,
            f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads/{_api_path_segment(short_id, 'short_id')}/md",
        )
    except OctoApiError as e:
        if e.status == 404:
            return {"content": "", "version": 0, "updated_at": None, "updated_by": ""}
        raise
    return result or {}


async def update_thread_md(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    group_no: str,
    short_id: str,
    content: str,
) -> dict[str, Any]:
    result = await put_json(
        session,
        api_url,
        bot_token,
        f"/v1/bot/groups/{_api_path_segment(group_no, 'group_no')}/threads/{_api_path_segment(short_id, 'short_id')}/md",
        {"content": content},
    )
    return result or {}


# ─── Voice context ───────────────────────────────────────────────────────────


async def get_voice_context(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
) -> dict[str, Any]:
    """Bot owner's personal voice correction context.

    Returns ``{has_context, context, updated_at}`` (empty when none set).
    """
    try:
        result = await get_json(session, api_url, bot_token, "/v1/bot/voice/context")
    except OctoApiError as e:
        if e.status == 404:
            return {"has_context": False, "context": "", "updated_at": None}
        raise
    return result or {"has_context": False, "context": "", "updated_at": None}


async def update_voice_context(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    content: str,
) -> None:
    # Server expects {"context": "..."} (NOT "content").
    await put_json(
        session, api_url, bot_token, "/v1/bot/voice/context", {"context": content}
    )


async def delete_voice_context(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
) -> None:
    await delete_json(session, api_url, bot_token, "/v1/bot/voice/context")
