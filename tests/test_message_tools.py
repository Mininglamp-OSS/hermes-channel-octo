"""Current-conversation RichText, media, profile, and card-edit tool contracts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from hermes_octo_plugin import message_tools
from hermes_octo_plugin.card_tools import TrustedOctoRoute
from hermes_octo_plugin.types import (
    CardProfileManifest,
    ChannelType,
    MessageType,
    SendMessageResult,
)


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


_ROUTE = TrustedOctoRoute(
    chat_id="group-1",
    channel_id="group-1",
    channel_type=ChannelType.Group,
    requester_uid="user-1",
    session_key="octo:group-1:user-1",
)
_ADAPTER = SimpleNamespace(
    _api_url="https://octo.invalid",
    _bot_token="test-token",
    _cdn_url="https://cdn.octo.invalid/assets",
    on_behalf_of="grantor-1",
    _card_profile_cache=message_tools.cards.CardProfileCache(),
)
_CARD_SESSIONS = MagicMock()
_CARD_SESSIONS.claim_edit.return_value = True
_CARD_SESSIONS.release_edit.return_value = None
_CARD_SESSIONS.complete.return_value = None
_ADAPTER._card_sessions = _CARD_SESSIONS
_MANIFEST = CardProfileManifest(
    available=True,
    enabled=True,
    profiles=("octo/v1", "octo/v2"),
    card_version="1.5",
    elements=("TextBlock",),
    inputs=(),
    actions=("Action.OpenUrl",),
    limits={"max_nodes": 20, "max_depth": 8, "max_payload_bytes": 65536},
)


@contextmanager
def _tool_context():
    with (
        patch.object(message_tools, "_resolve_adapter", return_value=_ADAPTER),
        patch.object(message_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(
            message_tools,
            "_new_guarded_http_session",
            return_value=_Session(),
        ),
    ):
        yield


def test_message_tool_schemas_expose_no_model_controlled_route_or_identity():
    schemas = message_tools.MESSAGE_TOOL_SCHEMAS
    assert {schema["name"] for schema in schemas} == {
        "octo_send_rich_text",
        "octo_send_image",
        "octo_send_file",
        "octo_send_voice",
        "octo_send_video",
        "octo_card_profile",
        "octo_edit_card",
    }
    for schema in schemas:
        properties = schema["parameters"]["properties"]
        assert {
            "target",
            "channel_id",
            "channel_type",
            "requester_uid",
            "session_key",
        }.isdisjoint(properties)
        assert schema["parameters"]["additionalProperties"] is False

    rich_blocks = message_tools.RICH_TEXT_TOOL_SCHEMA["parameters"]["properties"][
        "blocks"
    ]["items"]
    assert {variant["properties"]["type"]["const"] for variant in rich_blocks["oneOf"]} == {
        "text",
        "image",
    }


@pytest.mark.asyncio
async def test_rich_text_tool_verifies_uploads_and_derives_plain_fallback():
    download = AsyncMock(return_value=(b"png", "image/png", "safe.png"))
    upload = AsyncMock(return_value="https://cdn.example/safe.png")
    send = AsyncMock(
        return_value=SendMessageResult(
            message_id="rich-1",
            message_seq=8,
            client_msg_no="rich-client-1",
        )
    )
    with (
        _tool_context(),
        patch.object(message_tools.api, "download_file", download),
        patch.object(message_tools.api, "parse_image_dimensions", return_value=(12, 10)),
        patch.object(message_tools.api, "upload_and_get_url", upload),
        patch.object(message_tools.api, "send_rich_text_message", send),
    ):
        result = json.loads(
            await message_tools.octo_send_rich_text_handler(
                {
                    "blocks": [
                        {"type": "text", "text": "Review"},
                        {"type": "image", "url": "https://public.example/a.png"},
                    ],
                    "reply_to_message_id": "reply-1",
                }
            )
        )

    assert result == {
        "ok": True,
        "data": {
            "sent": True,
            "mode": "rich-text",
            "message_id": "rich-1",
            "message_seq": 8,
            "client_msg_no": "rich-client-1",
        },
    }
    download.assert_awaited_once()
    upload.assert_awaited_once()
    kwargs = send.await_args.kwargs
    assert kwargs["channel_id"] == "group-1"
    assert kwargs["channel_type"] == ChannelType.Group
    assert [block.to_dict() for block in kwargs["blocks"]] == [
        {"type": "text", "text": "Review"},
        {
            "type": "image",
            "url": "https://cdn.example/safe.png",
            "width": 12,
            "height": 10,
            "size": 3,
            "name": "safe.png",
        },
    ]
    assert kwargs["plain"] == "Review[图片]"
    assert kwargs["reply_msg_id"] == "reply-1"
    assert str(UUID(kwargs["client_msg_no"])) == kwargs["client_msg_no"]
    assert kwargs["on_behalf_of"] == "grantor-1"

@pytest.mark.asyncio
async def test_rich_text_tool_delivers_surviving_blocks_after_one_image_fails():
    upload = AsyncMock(
        side_effect=[
            RuntimeError("signed URL token=secret"),
            ("https://cdn.example/good.png", b"png", "image/png", "good.png"),
        ]
    )
    send = AsyncMock(
        return_value=SendMessageResult(
            message_id="rich-2",
            client_msg_no="rich-client-2",
        )
    )
    with (
        _tool_context(),
        patch.object(message_tools, "_upload_media", upload),
        patch.object(message_tools.api, "parse_image_dimensions", return_value=(12, 10)),
        patch.object(message_tools.api, "send_rich_text_message", send),
    ):
        result = json.loads(
            await message_tools.octo_send_rich_text_handler(
                {
                    "blocks": [
                        {"type": "text", "text": "Before"},
                        {"type": "image", "url": "https://public.example/bad.png"},
                        {"type": "image", "url": "https://public.example/good.png"},
                        {"type": "text", "text": "After"},
                    ]
                }
            )
        )

    assert result == {
        "ok": True,
        "data": {
            "sent": True,
            "mode": "rich-text",
            "message_id": "rich-2",
            "client_msg_no": "rich-client-2",
            "failed_images": 1,
        },
    }
    assert [block.to_dict() for block in send.await_args.kwargs["blocks"]] == [
        {"type": "text", "text": "Before"},
        {
            "type": "image",
            "url": "https://cdn.example/good.png",
            "width": 12,
            "height": 10,
            "size": 3,
            "name": "good.png",
        },
        {"type": "text", "text": "After"},
    ]
    assert send.await_args.kwargs["plain"] == "Before[图片]After"
    assert "secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_file_tool_accepts_a_local_path(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("local media", encoding="utf-8")
    upload = AsyncMock(return_value="https://cdn.example/report.txt")
    send = AsyncMock(
        return_value=SendMessageResult(
            message_id="media-1",
            message_seq=9,
            client_msg_no="media-client-1",
        )
    )
    with (
        _tool_context(),
        patch.object(message_tools.api, "upload_and_get_url", upload),
        patch.object(message_tools.api, "send_media_message", send),
    ):
        result = json.loads(
            await message_tools.octo_send_file_handler({"source": str(source)})
        )

    assert result == {
        "ok": True,
        "data": {
            "sent": True,
            "mode": "file",
            "message_id": "media-1",
            "message_seq": 9,
            "client_msg_no": "media-client-1",
        },
    }
    assert upload.await_args.args[3:] == (
        "report.txt",
        b"local media",
        "text/plain",
    )
    assert send.await_args.kwargs["msg_type"] == MessageType.File
    client_msg_no = send.await_args.kwargs["client_msg_no"]
    assert str(UUID(client_msg_no)) == client_msg_no
    assert send.await_args.kwargs["on_behalf_of"] == "grantor-1"

@pytest.mark.asyncio
async def test_remote_media_uses_guarded_session_and_keeps_host_safety_enabled() -> None:
    adapter = SimpleNamespace(**vars(_ADAPTER))
    session_factory = MagicMock(return_value=_Session())
    download = AsyncMock(return_value=(b"data", "application/octet-stream", "a.bin"))
    upload = AsyncMock(return_value="https://cdn.octo.invalid/a.bin")
    send = AsyncMock(return_value=SendMessageResult(message_id="media-safe"))
    with (
        patch.object(message_tools, "_resolve_adapter", return_value=adapter),
        patch.object(message_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(message_tools, "_new_guarded_http_session", session_factory),
        patch.object(message_tools.api, "download_file", download),
        patch.object(message_tools.api, "upload_and_get_url", upload),
        patch.object(message_tools.api, "send_media_message", send),
    ):
        result = json.loads(
            await message_tools.octo_send_file_handler(
                {"source": "https://files.example/report.bin"}
            )
        )

    assert result["ok"] is True
    session_factory.assert_called_once_with(
        "https://octo.invalid",
        "https://cdn.octo.invalid/assets",
    )
    assert download.await_args.args[0].__class__ is _Session
    assert "enforce_host_safety" not in download.await_args.kwargs




@pytest.mark.asyncio
async def test_image_tool_routes_verified_remote_media_to_current_conversation():
    download = AsyncMock(return_value=(b"image", "image/png", "image.png"))
    upload = AsyncMock(return_value="https://cdn.example/image.png")
    send = AsyncMock(
        return_value=SendMessageResult(
            message_id="media-2",
            client_msg_no="media-client-2",
        )
    )
    caption = AsyncMock()
    with (
        _tool_context(),
        patch.object(message_tools.api, "download_file", download),
        patch.object(message_tools.api, "parse_image_dimensions", return_value=(40, 30)),
        patch.object(message_tools.api, "upload_and_get_url", upload),
        patch.object(message_tools.api, "send_media_message", send),
        patch.object(message_tools.api, "send_message", caption),
    ):
        result = json.loads(
            await message_tools.octo_send_image_handler(
                {
                    "source": "https://public.example/image.png",
                    "caption": "See this",
                }
            )
        )

    assert result == {
        "ok": True,
        "data": {
            "sent": True,
            "mode": "image",
            "message_id": "media-2",
            "client_msg_no": "media-client-2",
        },
    }
    assert "enforce_host_safety" not in download.await_args.kwargs
    send.assert_awaited_once_with(
        ANY,
        "https://octo.invalid",
        "test-token",
        channel_id="group-1",
        channel_type=ChannelType.Group,
        msg_type=MessageType.Image,
        url="https://cdn.example/image.png",
        name="image.png",
        size=5,
        width=40,
        height=30,
        duration=None,
        reply_msg_id=None,
        client_msg_no=ANY,
        on_behalf_of="grantor-1",
    )
    caption.assert_awaited_once()
    media_client_msg_no = send.await_args.kwargs["client_msg_no"]
    assert str(UUID(media_client_msg_no)) == media_client_msg_no
    caption_client_msg_no = caption.await_args.kwargs["client_msg_no"]
    assert str(UUID(caption_client_msg_no)) == caption_client_msg_no
    assert caption.await_args.kwargs["on_behalf_of"] == "grantor-1"


@pytest.mark.asyncio
async def test_card_profile_tool_returns_normalized_negotiated_capabilities():
    with (
        _tool_context(),
        patch.object(message_tools.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
    ):
        result = json.loads(await message_tools.octo_card_profile_handler({}))

    assert result["ok"] is True
    assert result["data"]["available"] is True
    assert result["data"]["enabled"] is True
    assert result["data"]["profiles"] == ["octo/v1", "octo/v2"]
    assert result["data"]["card_version"] == "1.5"
    assert result["data"]["capabilities"]["max_nodes"] == 20


@pytest.mark.asyncio
async def test_edit_card_tool_renders_controlled_blocks_and_edits_current_channel():
    _CARD_SESSIONS.reset_mock()
    _CARD_SESSIONS.claim_edit.return_value = True
    edit = AsyncMock(return_value={"ok": True})
    with (
        _tool_context(),
        patch.object(message_tools.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
        patch.object(message_tools.api, "edit_card_message", edit),
    ):
        result = json.loads(
            await message_tools.octo_edit_card_handler(
                {
                    "message_id": "card-1",
                    "card_seq": 2,
                    "title": "Updated",
                    "blocks": [{"type": "text", "text": "Done"}],
                    "final": True,
                }
            )
        )

    assert result == {
        "ok": True,
        "data": {"edited": True, "message_id": "card-1", "card_seq": 2},
    }
    kwargs = edit.await_args.kwargs
    assert kwargs["channel_id"] == "group-1"
    assert kwargs["channel_type"] == ChannelType.Group
    assert kwargs["message_id"] == "card-1"
    assert kwargs["card_seq"] == 2
    assert kwargs["transient"] is False
    assert kwargs["plain"] == "Updated\nDone"
    _CARD_SESSIONS.claim_edit.assert_called_once_with(
        message_id="card-1",
        card_seq=2,
        session_key="octo:group-1:user-1",
        channel_id="group-1",
        channel_type=ChannelType.Group,
        requester_uid="user-1",
    )
    _CARD_SESSIONS.complete.assert_called_once_with("card-1", -2)


@pytest.mark.asyncio
async def test_edit_card_tool_fails_closed_without_a_matching_registered_session():
    sessions = MagicMock()
    sessions.claim_edit.return_value = False
    adapter = SimpleNamespace(
        _api_url="https://octo.invalid",
        _bot_token="test-token",
        _card_sessions=sessions,
    )
    edit = AsyncMock()
    with (
        patch.object(message_tools, "_resolve_adapter", return_value=adapter),
        patch.object(message_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(
            message_tools,
            "_new_guarded_http_session",
            return_value=_Session(),
        ),
        patch.object(message_tools.api, "edit_card_message", edit),
    ):
        result = json.loads(
            await message_tools.octo_edit_card_handler(
                {
                    "message_id": "unregistered",
                    "card_seq": 1,
                    "blocks": [{"type": "text", "text": "forged"}],
                }
            )
        )

    assert result == {
        "error": "card edit does not match a live trusted card session"
    }
    edit.assert_not_awaited()
