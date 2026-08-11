"""Safe current-conversation RichText, media, profile, and card-edit tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any



from . import api, cards
from .mention import _utf16_length, parse_structured_mentions
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
    MentionEntity,
    MessageType,
    RichTextBlock,
    SendMessageResult,
)

logger = logging.getLogger(__name__)

_MAX_MEDIA_BYTES = api.MAX_OUTBOUND_MEDIA_BYTES
_MAX_BLOCKS = 50
_MAX_TEXT_CHARS = 20_000
_MAX_SOURCE_CHARS = 4_096

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
            "title": {"type": "string", "maxLength": 2_000},
            "blocks": {
                "type": "array",
                "maxItems": 50,
                "items": DISPLAY_BLOCK_SCHEMA,
            },
            "final": {"type": "boolean"},
        },
        "required": ["message_id", "blocks"],
    },
}

MESSAGE_TOOL_SCHEMAS = (
    RICH_TEXT_TOOL_SCHEMA,
    IMAGE_TOOL_SCHEMA,
    FILE_TOOL_SCHEMA,
    VOICE_TOOL_SCHEMA,
    VIDEO_TOOL_SCHEMA,
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
    return api.safe_media_filename(value)


def _positive_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 2_147_483_647:
        raise ValueError(f"{field} must be a positive integer")
    return value




async def _read_media_source(
    session: Any,
    source: object,
    *,
    media_kind: str,
) -> tuple[bytes, str, str]:
    if not isinstance(source, str) or not source or len(source) > _MAX_SOURCE_CHARS:
        raise ValueError("source must be a bounded URL or local path")
    if source.startswith(("http://", "https://")):
        file_data, content_type, filename = await api.download_file(
            session,
            source,
            max_size=_MAX_MEDIA_BYTES,
            policy=getattr(session, "transport_policy", None),
        )
    else:
        local_source = source
        if source.startswith("file://"):
            from urllib.parse import unquote, urlparse

            parsed = urlparse(source)
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError("local media source is not authorized")
            local_source = unquote(parsed.path)
        file_data, filename = await asyncio.to_thread(
            api.read_authorized_local_media,
            local_source,
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
        session, source, media_kind=media_kind
    )
    uploaded_url = await api.upload_and_get_url(
        session,
        adapter._api_url,
        adapter._bot_token,
        filename,
        file_data,
        content_type,
        policy=getattr(session, "transport_policy", None),
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
        async with _new_guarded_http_session(
            adapter._api_url,
            adapter._cdn_url,
        ) as session:
            blocks: list[RichTextBlock] = []
            plain_parts: list[str] = []
            failed_images = 0
            mention_uids: list[str] = []
            seen_mention_uids: set[str] = set()
            mention_entities: list[MentionEntity] = []
            plain_utf16_length = 0
            message_mention_allowlist: set[str] | None = None
            filtered_mentions = 0
            for raw_block in raw_blocks:
                if not isinstance(raw_block, dict):
                    raise ValueError("RichText blocks must be objects")
                block_type = raw_block.get("type")
                if block_type == RICH_TEXT_BLOCK_TEXT:
                    text = raw_block.get("text")
                    if not isinstance(text, str) or not text or len(text) > _MAX_TEXT_CHARS:
                        raise ValueError("RichText text must be non-empty and bounded")
                    structured_mention_count = len(parse_structured_mentions(text))
                    if (
                        structured_mention_count
                        and message_mention_allowlist is None
                    ):
                        message_mention_allowlist = (
                            await adapter._mention_uid_allowlist(
                                route.chat_id,
                                route.channel_type,
                                http_session=session,
                            )
                        )
                    converted_text, block_entities, block_uids = (
                        await adapter._prepare_outbound_mentions(
                            text,
                            route.chat_id,
                            route.channel_type,
                            mention_uid_allowlist=message_mention_allowlist,
                            log_filtered=False,
                        )
                    )
                    filtered_mentions += structured_mention_count - len(
                        block_uids or ()
                    )
                    blocks.append(
                        RichTextBlock(type=RICH_TEXT_BLOCK_TEXT, text=converted_text)
                    )
                    plain_parts.append(converted_text)
                    if block_entities:
                        mention_entities.extend(
                            MentionEntity(
                                uid=entity.uid,
                                offset=plain_utf16_length + entity.offset,
                                length=entity.length,
                            )
                            for entity in block_entities
                        )
                    if block_uids:
                        for uid in block_uids:
                            if uid not in seen_mention_uids:
                                seen_mention_uids.add(uid)
                                mention_uids.append(uid)
                    plain_utf16_length += _utf16_length(converted_text)
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
                    plain_utf16_length += _utf16_length(RICH_TEXT_IMAGE_PLACEHOLDER)
                except Exception as exc:
                    failed_images += 1
                    logger.warning(
                        "Octo RichText image omitted after %s", type(exc).__name__
                    )
            if filtered_mentions:
                logger.warning(
                    "Octo RichText filtered %d unverified mention(s)",
                    filtered_mentions,
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
                mention_uids=mention_uids or None,
                mention_entities=mention_entities or None,
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
        caption_text = caption
        caption_entities: list[Any] | None = None
        caption_uids: list[str] | None = None
        requested_name = args.get("file_name")
        validated_name: str | None = None
        if requested_name is not None:
            validated_name = _safe_filename(requested_name)
            if not validated_name:
                raise ValueError("file_name is invalid")
        async with _new_guarded_http_session(
            adapter._api_url,
            adapter._cdn_url,
        ) as session:
            if caption:
                caption_text, caption_entities, caption_uids = (
                    await adapter._prepare_outbound_mentions(
                        caption,
                        route.chat_id,
                        route.channel_type,
                        http_session=session,
                    )
                )
            uploaded_url, data, content_type, detected_name = await _upload_media(
                session, adapter, args.get("source"), media_kind=media_kind
            )
            if media_kind == "image":
                dimensions = api.parse_image_dimensions(data, content_type)
                if dimensions is not None:
                    width, height = dimensions
            if validated_name is not None:
                detected_name = validated_name
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
                    content=caption_text,
                    reply_msg_id=args.get("reply_to_message_id") or None,
                    mention_uids=caption_uids,
                    mention_entities=caption_entities,
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
    if not isinstance(message_id, str) or not message_id or len(message_id) > 64:
        return _error("message_id must be a bounded identifier")
    raw_blocks = args.get("blocks")
    if not isinstance(raw_blocks, list):
        return _error("blocks must be a controlled display block array")
    final = bool(args.get("final", True))
    registry = adapter._card_sessions
    card_seq = registry.claim_edit(
        message_id=message_id,
        session_key=route.session_key,
        channel_id=route.channel_id,
        channel_type=route.channel_type,
        requester_uid=route.requester_uid,
    )
    if card_seq is None:
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
                transient=not final,
            )
    except (cards.CardLimitError, TypeError, ValueError) as exc:
        registry.release(message_id, claim_id)
        return _error(str(exc))
    except Exception as exc:
        registry.release(message_id, claim_id)
        logger.error("Octo edit-card tool failed (%s)", type(exc).__name__)
        return _error("Octo card edit failed")
    if final:
        registry.complete(message_id, claim_id)
    else:
        registry.release_edit(message_id, card_seq)
    return _ok(edited=True, message_id=message_id, card_seq=card_seq)


MESSAGE_TOOL_HANDLERS = {
    RICH_TEXT_TOOL_SCHEMA["name"]: octo_send_rich_text_handler,
    IMAGE_TOOL_SCHEMA["name"]: octo_send_image_handler,
    FILE_TOOL_SCHEMA["name"]: octo_send_file_handler,
    VOICE_TOOL_SCHEMA["name"]: octo_send_voice_handler,
    VIDEO_TOOL_SCHEMA["name"]: octo_send_video_handler,
    EDIT_CARD_TOOL_SCHEMA["name"]: octo_edit_card_handler,
}
