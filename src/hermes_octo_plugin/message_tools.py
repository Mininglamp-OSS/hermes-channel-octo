"""Safe current-conversation RichText, media, profile, and card-edit tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import uuid
from typing import Any

import aiohttp

from . import api, cards
from .card_tools import (
    DISPLAY_BLOCK_SCHEMA,
    TrustedOctoRoute,
    _new_guarded_http_session,
    _resolve_adapter,
    _get_card_profile,
    _trusted_route,
)
from .types import (
    CARD_PROFILE_V1,
    RICH_TEXT_BLOCK_IMAGE,
    RICH_TEXT_BLOCK_TEXT,
    RICH_TEXT_IMAGE_PLACEHOLDER,
    CardProfileManifest,
    MessageType,
    RichTextBlock,
    SendMessageResult,
)

logger = logging.getLogger(__name__)

_MAX_MEDIA_BYTES = api.MAX_OUTBOUND_MEDIA_BYTES
_MAX_BLOCKS = 50
_MAX_TEXT_CHARS = 20_000
_MAX_SOURCE_CHARS = 4_096
_MAX_FILENAME_BYTES = 255

_TEXT_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "const": RICH_TEXT_BLOCK_TEXT},
        "text": {"type": "string", "minLength": 1, "maxLength": _MAX_TEXT_CHARS},
    },
    "required": ["type", "text"],
}
_IMAGE_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "const": RICH_TEXT_BLOCK_IMAGE},
        "url": {"type": "string", "minLength": 1, "maxLength": _MAX_SOURCE_CHARS},
    },
    "required": ["type", "url"],
}

RICH_TEXT_TOOL_SCHEMA = {
    "name": "octo_send_rich_text",
    "description": (
        "Send controlled text/image RichText blocks to the current trusted Octo "
        "conversation. Remote images are downloaded and re-uploaded first."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "blocks": {
                "type": "array",
                "minItems": 1,
                "maxItems": _MAX_BLOCKS,
                "items": {"oneOf": [_TEXT_BLOCK_SCHEMA, _IMAGE_BLOCK_SCHEMA]},
            },
            "reply_to_message_id": {"type": "string", "minLength": 1, "maxLength": 64},
        },
        "required": ["blocks"],
    },
}


def _media_tool_schema(name: str, label: str, *, metadata: tuple[str, ...]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "source": {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_SOURCE_CHARS,
            "description": "An http/https URL, local file path, or file:// URL.",
        },
        "caption": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
        "reply_to_message_id": {"type": "string", "minLength": 1, "maxLength": 64},
    }
    if name == "octo_send_file":
        properties["file_name"] = {"type": "string", "minLength": 1, "maxLength": 255}
    for field in metadata:
        properties[field] = {"type": "integer", "minimum": 1, "maximum": 2_147_483_647}
    return {
        "name": name,
        "description": (
            f"Verify, upload, and send {label} to the current trusted Octo conversation."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": ["source"],
        },
    }


IMAGE_TOOL_SCHEMA = _media_tool_schema(
    "octo_send_image", "an image", metadata=("width", "height")
)
FILE_TOOL_SCHEMA = _media_tool_schema("octo_send_file", "a file", metadata=())
VOICE_TOOL_SCHEMA = _media_tool_schema(
    "octo_send_voice", "a voice clip", metadata=("duration",)
)
VIDEO_TOOL_SCHEMA = _media_tool_schema(
    "octo_send_video", "a video", metadata=("width", "height", "duration")
)
CARD_PROFILE_TOOL_SCHEMA = {
    "name": "octo_card_profile",
    "description": "Read normalized Type-17 capabilities for the current Octo bot.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    },
}
EDIT_CARD_TOOL_SCHEMA = {
    "name": "octo_edit_card",
    "description": (
        "Render controlled display blocks and terminally edit a live interactive "
        "Type-17 card registered in the current trusted Octo conversation."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message_id": {"type": "string", "minLength": 1, "maxLength": 64},
            "card_seq": {"type": "integer", "minimum": 1},
            "title": {"type": "string", "maxLength": 2_000},
            "blocks": {
                "type": "array",
                "maxItems": 50,
                "items": DISPLAY_BLOCK_SCHEMA,
            },
            "final": {"type": "boolean"},
        },
        "required": ["message_id", "card_seq", "blocks"],
    },
}

MESSAGE_TOOL_SCHEMAS = (
    RICH_TEXT_TOOL_SCHEMA,
    IMAGE_TOOL_SCHEMA,
    FILE_TOOL_SCHEMA,
    VOICE_TOOL_SCHEMA,
    VIDEO_TOOL_SCHEMA,
    CARD_PROFILE_TOOL_SCHEMA,
    EDIT_CARD_TOOL_SCHEMA,
)


def _ok(**data: object) -> str:
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


def _receipt_fields(result: SendMessageResult) -> dict[str, object]:
    fields: dict[str, object] = {"message_id": result.message_id}
    if result.message_seq is not None:
        fields["message_seq"] = result.message_seq
    if result.client_msg_no is not None:
        fields["client_msg_no"] = result.client_msg_no
    return fields

def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _context() -> tuple[Any, TrustedOctoRoute] | str:
    adapter = _resolve_adapter()
    if adapter is None:
        return _error("Octo adapter is not running in this process")
    if not adapter._api_url or not adapter._bot_token:
        return _error("Octo adapter is not configured")
    route = _trusted_route(adapter, require_session_key=False)
    if route is None:
        return _error("trusted Octo session route is unavailable")
    return adapter, route


def _safe_filename(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if (
        not candidate
        or candidate != Path(candidate).name
        or any(ord(char) < 32 or ord(char) == 127 for char in candidate)
        or len(candidate.encode("utf-8")) > _MAX_FILENAME_BYTES
    ):
        return None
    return candidate


def _positive_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 2_147_483_647:
        raise ValueError(f"{field} must be a positive integer")
    return value




async def _read_media_source(
    source: object,
    *,
    media_kind: str,
) -> tuple[bytes, str, str]:
    if not isinstance(source, str) or not source or len(source) > _MAX_SOURCE_CHARS:
        raise ValueError("source must be a bounded URL or local path")
    if source.startswith(("http://", "https://")):
        async with aiohttp.ClientSession() as download_session:
            file_data, content_type, filename = await api.download_file(
                download_session,
                source,
                max_size=_MAX_MEDIA_BYTES,
                enforce_host_safety=False,
            )
    else:
        file_data, filename = await asyncio.to_thread(
            api.read_local_media,
            source,
            max_size=_MAX_MEDIA_BYTES,
        )
        content_type = api.infer_content_type(filename)
    safe_name = _safe_filename(filename)
    if safe_name is None:
        raise ValueError("media filename is invalid")
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    expected_prefix = {
        "image": "image/",
        "voice": "audio/",
        "video": "video/",
    }.get(media_kind)
    if expected_prefix and not normalized_type.startswith(expected_prefix):
        raise ValueError(f"source is not valid {media_kind} media")
    return file_data, normalized_type or "application/octet-stream", safe_name


async def _upload_media(
    session: Any,
    adapter: Any,
    source: object,
    *,
    media_kind: str,
) -> tuple[str, bytes, str, str]:
    file_data, content_type, filename = await _read_media_source(
        source, media_kind=media_kind
    )
    uploaded_url = await api.upload_and_get_url(
        session,
        adapter._api_url,
        adapter._bot_token,
        filename,
        file_data,
        content_type,
    )
    return uploaded_url, file_data, content_type, filename


async def octo_send_rich_text_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    context = _context()
    if isinstance(context, str):
        return context
    adapter, route = context
    raw_blocks = args.get("blocks")
    if not isinstance(raw_blocks, list) or not 0 < len(raw_blocks) <= _MAX_BLOCKS:
        return _error("blocks must contain 1-50 controlled RichText blocks")
    try:
        async with _new_guarded_http_session(adapter._api_url) as session:
            blocks: list[RichTextBlock] = []
            plain_parts: list[str] = []
            failed_images = 0
            for raw_block in raw_blocks:
                if not isinstance(raw_block, dict):
                    raise ValueError("RichText blocks must be objects")
                block_type = raw_block.get("type")
                if block_type == RICH_TEXT_BLOCK_TEXT:
                    text = raw_block.get("text")
                    if not isinstance(text, str) or not text or len(text) > _MAX_TEXT_CHARS:
                        raise ValueError("RichText text must be non-empty and bounded")
                    blocks.append(RichTextBlock(type=RICH_TEXT_BLOCK_TEXT, text=text))
                    plain_parts.append(text)
                    continue
                if block_type != RICH_TEXT_BLOCK_IMAGE:
                    raise ValueError("unsupported RichText block type")
                try:
                    uploaded_url, data, content_type, filename = await _upload_media(
                        session, adapter, raw_block.get("url"), media_kind="image"
                    )
                    dimensions = api.parse_image_dimensions(data, content_type)
                    if dimensions is None:
                        raise ValueError(
                            "RichText image dimensions could not be verified"
                        )
                    width, height = dimensions
                    blocks.append(
                        RichTextBlock(
                            type=RICH_TEXT_BLOCK_IMAGE,
                            url=uploaded_url,
                            width=width,
                            height=height,
                            size=len(data),
                            name=filename,
                        )
                    )
                    plain_parts.append(RICH_TEXT_IMAGE_PLACEHOLDER)
                except Exception as exc:
                    failed_images += 1
                    logger.warning(
                        "Octo RichText image omitted after %s", type(exc).__name__
                    )
            if not blocks:
                return _error("all RichText image blocks failed")
            client_msg_no = str(uuid.uuid4())
            send_result = await api.send_rich_text_message(
                session,
                adapter._api_url,
                adapter._bot_token,
                channel_id=route.channel_id,
                channel_type=route.channel_type,
                blocks=blocks,
                plain="".join(plain_parts),
                reply_msg_id=args.get("reply_to_message_id") or None,
                client_msg_no=client_msg_no,
                on_behalf_of=adapter.on_behalf_of,
            )
        result = {
            "sent": True,
            "mode": "rich-text",
            **_receipt_fields(send_result),
        }
        if failed_images:
            result["failed_images"] = failed_images
        return _ok(**result)
    except (PermissionError, ValueError) as exc:
        return _error(str(exc))
    except Exception as exc:
        logger.error("Octo RichText tool failed (%s)", type(exc).__name__)
        return _error("Octo RichText delivery failed")


async def _send_media(
    args: dict[str, Any],
    *,
    media_kind: str,
    message_type: MessageType,
) -> str:
    context = _context()
    if isinstance(context, str):
        return context
    adapter, route = context
    try:
        width = _positive_integer(args.get("width"), "width")
        height = _positive_integer(args.get("height"), "height")
        duration = _positive_integer(args.get("duration"), "duration")
        caption = args.get("caption")
        if caption is not None and (
            not isinstance(caption, str) or len(caption) > _MAX_TEXT_CHARS
        ):
            raise ValueError("caption must be bounded text")
        async with _new_guarded_http_session(adapter._api_url) as session:
            uploaded_url, data, content_type, detected_name = await _upload_media(
                session, adapter, args.get("source"), media_kind=media_kind
            )
            if media_kind == "image":
                dimensions = api.parse_image_dimensions(data, content_type)
                if dimensions is not None:
                    width, height = dimensions
            requested_name = args.get("file_name")
            if requested_name is not None:
                detected_name = _safe_filename(requested_name) or ""
                if not detected_name:
                    raise ValueError("file_name is invalid")
            media_client_msg_no = str(uuid.uuid4())
            send_result = await api.send_media_message(
                session,
                adapter._api_url,
                adapter._bot_token,
                channel_id=route.channel_id,
                channel_type=route.channel_type,
                msg_type=message_type,
                url=uploaded_url,
                name=detected_name,
                size=len(data),
                width=width,
                height=height,
                duration=duration,
                reply_msg_id=args.get("reply_to_message_id") or None,
                client_msg_no=media_client_msg_no,
                on_behalf_of=adapter.on_behalf_of,
            )
            if caption:
                caption_client_msg_no = str(uuid.uuid4())
                await api.send_message(
                    session,
                    adapter._api_url,
                    adapter._bot_token,
                    channel_id=route.channel_id,
                    channel_type=route.channel_type,
                    content=caption,
                    reply_msg_id=args.get("reply_to_message_id") or None,
                    client_msg_no=caption_client_msg_no,
                    on_behalf_of=adapter.on_behalf_of,
                )
        return _ok(
            sent=True,
            mode=media_kind,
            **_receipt_fields(send_result),
        )
    except (PermissionError, ValueError) as exc:
        return _error(str(exc))
    except Exception as exc:
        logger.error("Octo %s tool failed (%s)", media_kind, type(exc).__name__)
        return _error(f"Octo {media_kind} delivery failed")


async def octo_send_image_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    return await _send_media(args, media_kind="image", message_type=MessageType.Image)


async def octo_send_file_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    return await _send_media(args, media_kind="file", message_type=MessageType.File)


async def octo_send_voice_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    return await _send_media(args, media_kind="voice", message_type=MessageType.Voice)


async def octo_send_video_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    return await _send_media(args, media_kind="video", message_type=MessageType.Video)


def _optional_sorted(value: frozenset[str] | None) -> list[str] | None:
    return sorted(value) if value is not None else None


async def octo_card_profile_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    if args:
        return _error("octo_card_profile accepts no arguments")
    context = _context()
    if isinstance(context, str):
        return context
    adapter, _route = context
    try:
        async with _new_guarded_http_session(adapter._api_url) as session:
            manifest = await _get_card_profile(adapter, session)
        capabilities = cards.derive_card_capabilities(manifest)
        return _ok(
            available=manifest.available,
            enabled=manifest.enabled,
            profiles=list(manifest.profiles) if manifest.profiles is not None else None,
            card_version=manifest.card_version,
            elements=list(manifest.elements) if manifest.elements is not None else None,
            inputs=list(manifest.inputs) if manifest.inputs is not None else None,
            actions=list(manifest.actions) if manifest.actions is not None else None,
            limits=dict(manifest.limits),
            capabilities={
                "authoritative": capabilities.authoritative,
                "profiles": _optional_sorted(capabilities.profiles),
                "elements": _optional_sorted(capabilities.elements),
                "inputs": _optional_sorted(capabilities.inputs),
                "actions": _optional_sorted(capabilities.actions),
                "max_nodes": capabilities.max_nodes,
                "max_depth": capabilities.max_depth,
                "max_payload_bytes": capabilities.max_payload_bytes,
                "max_input_text_bytes": capabilities.max_input_text_bytes,
                "max_inputs_bytes": capabilities.max_inputs_bytes,
            },
        )
    except Exception as exc:
        logger.error("Octo card profile tool failed (%s)", type(exc).__name__)
        return _error("Octo card profile lookup failed")


def _edit_profile_enabled(manifest: CardProfileManifest) -> bool:
    configured = os.getenv("OCTO_CARD_MESSAGE_ENABLED") == "1"
    if not cards.card_delivery_enabled(manifest, configured_enabled=configured):
        return False
    if manifest.available:
        return (
            manifest.profiles is not None
            and CARD_PROFILE_V1 in manifest.profiles
            and manifest.card_version == cards.CARD_VERSION
        )
    return True


async def octo_edit_card_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    context = _context()
    if isinstance(context, str):
        return context
    adapter, route = context
    message_id = args.get("message_id")
    card_seq = args.get("card_seq")
    if not isinstance(message_id, str) or not message_id or len(message_id) > 64:
        return _error("message_id must be a bounded identifier")
    if isinstance(card_seq, bool) or not isinstance(card_seq, int) or card_seq <= 0:
        return _error("card_seq must be a positive integer")
    raw_blocks = args.get("blocks")
    if not isinstance(raw_blocks, list):
        return _error("blocks must be a controlled display block array")
    registry = adapter._card_sessions
    if not registry.claim_edit(
        message_id=message_id,
        card_seq=card_seq,
        session_key=route.session_key,
        channel_id=route.channel_id,
        channel_type=route.channel_type,
        requester_uid=route.requester_uid,
    ):
        return _error("card edit does not match a live trusted card session")
    claim_id = -card_seq
    try:
        async with _new_guarded_http_session(adapter._api_url) as session:
            manifest = await _get_card_profile(adapter, session)
            if not _edit_profile_enabled(manifest):
                registry.release(message_id, claim_id)
                return _error("Type-17 card editing is unavailable")
            capabilities = cards.derive_card_capabilities(manifest)
            rendered = cards.build_display_card(
                title=args.get("title"),
                blocks=raw_blocks,
                capabilities=capabilities,
            )
            await api.edit_card_message(
                session,
                adapter._api_url,
                adapter._bot_token,
                channel_id=route.channel_id,
                channel_type=route.channel_type,
                message_id=message_id,
                card=rendered.card,
                card_seq=card_seq,
                plain=rendered.plain,
                transient=not bool(args.get("final", True)),
                profile=CARD_PROFILE_V1,
            )
    except (cards.CardLimitError, TypeError, ValueError) as exc:
        registry.release(message_id, claim_id)
        return _error(str(exc))
    except Exception as exc:
        registry.release(message_id, claim_id)
        logger.error("Octo edit-card tool failed (%s)", type(exc).__name__)
        return _error("Octo card edit failed")
    registry.complete(message_id, claim_id)
    return _ok(edited=True, message_id=message_id, card_seq=card_seq)


MESSAGE_TOOL_HANDLERS = {
    RICH_TEXT_TOOL_SCHEMA["name"]: octo_send_rich_text_handler,
    IMAGE_TOOL_SCHEMA["name"]: octo_send_image_handler,
    FILE_TOOL_SCHEMA["name"]: octo_send_file_handler,
    VOICE_TOOL_SCHEMA["name"]: octo_send_voice_handler,
    VIDEO_TOOL_SCHEMA["name"]: octo_send_video_handler,
    CARD_PROFILE_TOOL_SCHEMA["name"]: octo_card_profile_handler,
    EDIT_CARD_TOOL_SCHEMA["name"]: octo_edit_card_handler,
}
