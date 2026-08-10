"""LLM-callable card tools bound to the current trusted Octo conversation."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

from . import api, cards
from .agent_tools import (
    _new_guarded_http_session,
    _resolve_adapter,
    _valid_target_channel_id,
)
from .types import (
    CARD_PROFILE_V1,
    CARD_PROFILE_V2,
    CardProfileManifest,
    ChannelType,
    SendMessageResult,
)

logger = logging.getLogger(__name__)

_TRUSTED_FIELDS = frozenset(
    {
        "channel_id",
        "channel_type",
        "target",
        "requester_uid",
        "session_key",
        "binding_id",
    }
)

_DISPLAY_TEXT_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "const": "text"},
        "text": {"type": "string", "maxLength": 65536},
    },
    "required": ["type", "text"],
}
_DISPLAY_HEADING_BLOCK_SCHEMA = {
    **_DISPLAY_TEXT_BLOCK_SCHEMA,
    "properties": {
        **_DISPLAY_TEXT_BLOCK_SCHEMA["properties"],
        "type": {"type": "string", "const": "heading"},
    },
}
_DISPLAY_SECTION_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "const": "section"},
        "title": {"type": "string", "maxLength": 65536},
        "text": {"type": "string", "maxLength": 65536},
    },
    "required": ["type"],
}
_DISPLAY_FACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string", "maxLength": 65536},
        "value": {"type": "string", "maxLength": 65536},
    },
    "required": ["label", "value"],
}
_DISPLAY_FACTS_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "const": "facts"},
        "items": {
            "type": "array",
            "items": _DISPLAY_FACT_SCHEMA,
            "maxItems": 50,
        },
    },
    "required": ["type", "items"],
}
_DISPLAY_IMAGE_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "const": "image"},
        "url": {"type": "string", "maxLength": 2048},
        "alt": {"type": "string", "maxLength": 65536},
    },
    "required": ["type", "url", "alt"],
}
_DISPLAY_ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string", "maxLength": 200},
        "url": {"type": "string", "maxLength": 2048},
    },
    "required": ["label", "url"],
}
_DISPLAY_ACTIONS_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "const": "actions"},
        "items": {
            "type": "array",
            "items": _DISPLAY_ACTION_SCHEMA,
            "minItems": 1,
            "maxItems": 6,
        },
    },
    "required": ["type", "items"],
}
_DISPLAY_BLOCK_SCHEMA = {
    "oneOf": [
        _DISPLAY_HEADING_BLOCK_SCHEMA,
        _DISPLAY_TEXT_BLOCK_SCHEMA,
        _DISPLAY_SECTION_BLOCK_SCHEMA,
        _DISPLAY_FACTS_BLOCK_SCHEMA,
        _DISPLAY_IMAGE_BLOCK_SCHEMA,
        _DISPLAY_ACTIONS_BLOCK_SCHEMA,
    ]
}

# Reused by the safe current-conversation edit-card tool. This schema permits
# only renderer-owned block variants, never an arbitrary Adaptive Card tree.
DISPLAY_BLOCK_SCHEMA = _DISPLAY_BLOCK_SCHEMA

DISPLAY_CARD_TOOL_SCHEMA = {
    "name": "octo_send_display_card",
    "description": (
        "Send a controlled display card to the current Octo conversation. "
        "The destination and requester identity come only from the trusted session. "
        "Falls back to plain text when Type-17 cards are unavailable."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "maxLength": 65536},
            "blocks": {
                "type": "array",
                "items": _DISPLAY_BLOCK_SCHEMA,
                "maxItems": 100,
            },
        },
        "required": ["blocks"],
    },
}

_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "maxLength": 64},
        "kind": {
            "type": "string",
            "enum": ["text", "number", "date", "time", "toggle", "choice"],
        },
        "label": {"type": "string", "maxLength": 64},
        "placeholder": {"type": "string", "maxLength": 64},
        "choices": {
            "type": "array",
            "maxItems": 128,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string", "maxLength": 64},
                    "value": {"type": "string", "maxLength": 64},
                },
                "required": ["title", "value"],
            },
        },
    },
    "required": ["id", "kind"],
}

_BUTTON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "maxLength": 64},
        "label": {"type": "string", "maxLength": 64},
        "style": {"type": "string", "enum": ["positive", "destructive"]},
        "data": {"type": "object", "maxProperties": 50},
    },
    "required": ["id", "label"],
}

INTERACTIVE_CARD_TOOL_SCHEMA = {
    "name": "octo_send_interactive_card",
    "description": (
        "Send a controlled form card to the current Octo conversation. "
        "Action binding, destination, and requester identity are generated from "
        "the trusted session; unsupported deployments receive plain text instead."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "maxLength": 200},
            "text": {"type": "string", "maxLength": 2000},
            "inputs": {"type": "array", "items": _INPUT_SCHEMA, "maxItems": 5},
            "buttons": {
                "type": "array",
                "items": _BUTTON_SCHEMA,
                "minItems": 1,
                "maxItems": 6,
            },
        },
        "required": ["title", "buttons"],
    },
}


@dataclass(frozen=True)
class TrustedOctoRoute:
    """Task-local current-conversation route, never supplied by the model."""

    channel_id: str
    chat_id: str
    channel_type: ChannelType
    requester_uid: str
    session_key: str


def _ok(**values: object) -> str:
    return json.dumps({"ok": True, **values}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _trusted_route(adapter: Any, *, require_session_key: bool) -> TrustedOctoRoute | None:
    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
        requester_uid = get_session_env("HERMES_SESSION_USER_ID", "").strip()
        session_key = get_session_env("HERMES_SESSION_KEY", "").strip()
    except Exception:
        return None
    if platform != "octo" or not chat_id or not requester_uid:
        return None
    if require_session_key and not session_key:
        return None
    try:
        channel_type = adapter._resolve_channel_type(chat_id)
        channel_id = adapter._outbound_channel_id(chat_id, channel_type)
    except Exception:
        return None
    if not _valid_target_channel_id(channel_id, channel_type):
        return None
    return TrustedOctoRoute(
        chat_id=chat_id,
        channel_id=channel_id,
        channel_type=channel_type,
        requester_uid=requester_uid,
        session_key=session_key,
    )


async def _get_card_profile(adapter: Any, session: Any) -> CardProfileManifest:
    """Fetch once per short adapter-local TTL without caching failures."""
    cache = adapter._card_profile_cache
    cached = cache.get()
    if cached is not None:
        return cached
    manifest = await api.get_card_profile(
        session,
        adapter._api_url,
        adapter._bot_token,
    )
    cache.put(manifest)
    return manifest


def _profile_enabled(manifest: CardProfileManifest, profile: str) -> bool:
    configured_enabled = os.getenv("OCTO_CARD_MESSAGE_ENABLED") == "1"
    if not cards.card_delivery_enabled(
        manifest,
        configured_enabled=configured_enabled,
    ):
        return False
    if profile == CARD_PROFILE_V2 and not manifest.available:
        return False
    if manifest.available and (
        manifest.profiles is None
        or profile not in manifest.profiles
        or manifest.card_version != cards.CARD_VERSION
    ):
        return False
    return True


def _receipt_fields(result: SendMessageResult) -> dict[str, object]:
    fields: dict[str, object] = {"message_id": result.message_id}
    if result.message_seq is not None:
        fields["message_seq"] = result.message_seq
    if result.client_msg_no is not None:
        fields["client_msg_no"] = result.client_msg_no
    return fields


async def _send_plain(
    session: Any,
    adapter: Any,
    route: TrustedOctoRoute,
    plain: str,
) -> str:
    client_msg_no = str(uuid.uuid4())
    result = await api.send_message(
        session,
        adapter._api_url,
        adapter._bot_token,
        channel_id=route.channel_id,
        channel_type=route.channel_type,
        content=plain,
        client_msg_no=client_msg_no,
        on_behalf_of=adapter.on_behalf_of,
    )
    return _ok(mode="plain", **_receipt_fields(result))


def _reject_trusted_fields(args: dict[str, Any]) -> str | None:
    if _TRUSTED_FIELDS.intersection(args):
        return _error("trusted session fields cannot be supplied")
    return None


async def octo_send_display_card_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    """Render and send a display card to the trusted current Octo conversation."""
    rejected = _reject_trusted_fields(args)
    if rejected is not None:
        return rejected
    adapter = _resolve_adapter()
    if adapter is None:
        return _error("Octo adapter is not connected")
    route = _trusted_route(adapter, require_session_key=False)
    if route is None:
        return _error("trusted Octo conversation context is unavailable")
    try:
        rendered = cards.build_display_card(
            title=args.get("title"),
            blocks=args.get("blocks", ()),
        )
    except (TypeError, ValueError):
        return _error("invalid display card request")

    try:
        async with _new_guarded_http_session(adapter._api_url) as session:
            if adapter.on_behalf_of is not None:
                return await _send_plain(session, adapter, route, rendered.plain)
            try:
                manifest = await _get_card_profile(adapter, session)
            except Exception:
                logger.warning("[Octo] card profile lookup failed; using plain fallback")
                return await _send_plain(session, adapter, route, rendered.plain)
            if not _profile_enabled(manifest, CARD_PROFILE_V1):
                return await _send_plain(session, adapter, route, rendered.plain)
            try:
                rendered = cards.build_display_card(
                    title=args.get("title"),
                    blocks=args.get("blocks", ()),
                    capabilities=cards.derive_card_capabilities(manifest),
                )
            except (cards.CardLimitError, TypeError, ValueError):
                return await _send_plain(session, adapter, route, rendered.plain)
            result = await api.send_card_message(
                session,
                adapter._api_url,
                adapter._bot_token,
                channel_id=route.channel_id,
                channel_type=route.channel_type,
                card=rendered.card,
                plain=rendered.plain,
                profile=CARD_PROFILE_V1,
            )
    except Exception:
        logger.exception("[Octo] display card delivery failed")
        return _error("Octo card delivery failed")
    return _ok(mode="card", **_receipt_fields(result))


async def octo_send_interactive_card_handler(
    args: dict[str, Any],
    **_kwargs: Any,
) -> str:
    """Render and send a session-bound form card to the current conversation."""
    rejected = _reject_trusted_fields(args)
    if rejected is not None:
        return rejected
    adapter = _resolve_adapter()
    if adapter is None:
        return _error("Octo adapter is not connected")
    route = _trusted_route(adapter, require_session_key=True)
    if route is None:
        return _error("trusted Octo session context is unavailable")
    binding_id = str(uuid.uuid4())
    try:
        rendered = cards.build_interactive_card(
            title=args.get("title"),
            text=args.get("text"),
            inputs=args.get("inputs", ()),
            buttons=args.get("buttons", ()),
            binding_id=binding_id,
        )
    except (TypeError, ValueError):
        return _error("invalid interactive card request")

    try:
        async with _new_guarded_http_session(adapter._api_url) as session:
            if adapter.on_behalf_of is not None:
                return await _send_plain(session, adapter, route, rendered.plain)
            try:
                manifest = await _get_card_profile(adapter, session)
            except Exception:
                logger.warning("[Octo] card profile lookup failed; using plain fallback")
                return await _send_plain(session, adapter, route, rendered.plain)
            if not _profile_enabled(manifest, CARD_PROFILE_V2):
                return await _send_plain(session, adapter, route, rendered.plain)
            capabilities = cards.derive_card_capabilities(manifest)
            try:
                rendered = cards.build_interactive_card(
                    title=args.get("title"),
                    text=args.get("text"),
                    inputs=args.get("inputs", ()),
                    buttons=args.get("buttons", ()),
                    binding_id=binding_id,
                    capabilities=capabilities,
                )
            except (cards.CardLimitError, TypeError, ValueError):
                return await _send_plain(session, adapter, route, rendered.plain)
            result = await api.send_card_message(
                session,
                adapter._api_url,
                adapter._bot_token,
                channel_id=route.channel_id,
                channel_type=route.channel_type,
                card=rendered.card,
                plain=rendered.plain,
                profile=CARD_PROFILE_V2,
            )
            from .card_sessions import CardSession

            adapter._register_card_session(
                CardSession(
                    message_id=result.message_id,
                    binding_id=binding_id,
                    session_key=route.session_key,
                    chat_id=route.chat_id,
                    channel_id=route.channel_id,
                    channel_type=route.channel_type,
                    requester_uid=route.requester_uid,
                    card=rendered.card,
                    plain=rendered.plain,
                    action_labels=rendered.action_labels,
                    input_ids=rendered.input_ids,
                    max_input_text_bytes=capabilities.max_input_text_bytes,
                    max_inputs_bytes=capabilities.max_inputs_bytes,
                )
            )
    except Exception:
        logger.exception("[Octo] interactive card delivery failed")
        return _error("Octo card delivery failed")
    return _ok(
        mode="card",
        binding_id=binding_id,
        **_receipt_fields(result),
    )
