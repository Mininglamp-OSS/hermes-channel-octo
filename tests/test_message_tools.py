"""Current-conversation RichText, media, profile, and card-edit tool contracts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch
from uuid import UUID

import pytest

from hermes_octo_plugin import message_tools
from hermes_octo_plugin.card_sessions import CardSession, CardSessionRegistry
from hermes_octo_plugin.card_tools import TrustedOctoRoute
from hermes_octo_plugin.types import (
    CardProfileManifest,
    ChannelType,
    GroupMember,
    MentionEntity,
    MessageType,
    SendMessageResult,
)
from tests.conftest import make_bare_adapter

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
    _prepare_outbound_mentions=AsyncMock(
        side_effect=lambda content, *_args, **_kwargs: (content, None, None)
    ),
)
def _editable_session(
    message_id: str = "card-1",
    *,
    kind: str = "interactive",
    clarify: object | None = None,
) -> CardSession:
    return CardSession(
        message_id=message_id,
        binding_id="binding-1",
        session_key=_ROUTE.session_key,
        chat_id=_ROUTE.chat_id,
        channel_id=_ROUTE.channel_id,
        channel_type=_ROUTE.channel_type,
        requester_uid=_ROUTE.requester_uid,
        card={},
        plain="Editable",
        action_labels={},
        input_ids=(),
        clarify=clarify,
        kind=kind,
    )


_ADAPTER._card_sessions = CardSessionRegistry()
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
def _tool_context(
    adapter: SimpleNamespace = _ADAPTER,
    session: _Session | None = None,
):
    with (
        patch.object(message_tools, "_resolve_adapter", return_value=adapter),
        patch.object(message_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(
            message_tools,
            "_new_guarded_http_session",
            return_value=session or _Session(),
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
async def test_rich_text_tool_converts_structured_mentions_with_plain_offsets():
    session = _Session()
    prepare_mentions = AsyncMock(
        return_value=(
            "@Alice report",
            [MentionEntity(uid="u1", offset=0, length=6)],
            ["u1"],
        )
    )
    mention_uid_allowlist = AsyncMock(return_value={"u1"})
    adapter = SimpleNamespace(
        _api_url=_ADAPTER._api_url,
        _bot_token=_ADAPTER._bot_token,
        _cdn_url=_ADAPTER._cdn_url,
        on_behalf_of=_ADAPTER.on_behalf_of,
        _prepare_outbound_mentions=prepare_mentions,
        _mention_uid_allowlist=mention_uid_allowlist,
    )
    upload = AsyncMock(
        return_value=("https://cdn.example/a.png", b"png", "image/png", "a.png")
    )
    send = AsyncMock(return_value=SendMessageResult(message_id="rich-mention"))

    with (
        _tool_context(adapter, session),
        patch.object(message_tools, "_upload_media", upload),
        patch.object(message_tools.api, "parse_image_dimensions", return_value=(12, 10)),
        patch.object(message_tools.api, "send_rich_text_message", send),
    ):
        result = json.loads(
            await message_tools.octo_send_rich_text_handler(
                {
                    "blocks": [
                        {"type": "image", "url": "https://public.example/a.png"},
                        {"type": "text", "text": "@[u1:Alice] report"},
                    ]
                }
            )
        )

    assert result["ok"] is True
    mention_uid_allowlist.assert_awaited_once_with(
        _ROUTE.chat_id,
        _ROUTE.channel_type,
        http_session=session,
    )
    prepare_mentions.assert_awaited_once_with(
        "@[u1:Alice] report",
        _ROUTE.chat_id,
        _ROUTE.channel_type,
        mention_uid_allowlist={"u1"},
        log_filtered=False,
    )
    kwargs = send.await_args.kwargs
    assert kwargs["plain"] == "[图片]@Alice report"
    assert kwargs["blocks"][1].text == "@Alice report"
    assert kwargs["mention_uids"] == ["u1"]
    assert [
        (entity.uid, entity.offset, entity.length)
        for entity in kwargs["mention_entities"]
    ] == [("u1", 4, 6)]


@pytest.mark.asyncio
async def test_rich_text_tool_deduplicates_mention_uids_across_text_blocks():
    session = _Session()
    prepare_mentions = AsyncMock(
        side_effect=[
            (
                "@Alice one",
                [MentionEntity(uid="u1", offset=0, length=6)],
                ["u1"],
            ),
            (
                "@Alice two",
                [MentionEntity(uid="u1", offset=0, length=6)],
                ["u1"],
            ),
        ]
    )
    mention_uid_allowlist = AsyncMock(return_value={"u1"})
    adapter = SimpleNamespace(
        _api_url=_ADAPTER._api_url,
        _bot_token=_ADAPTER._bot_token,
        _cdn_url=_ADAPTER._cdn_url,
        on_behalf_of=_ADAPTER.on_behalf_of,
        _prepare_outbound_mentions=prepare_mentions,
        _mention_uid_allowlist=mention_uid_allowlist,
    )
    send = AsyncMock(return_value=SendMessageResult(message_id="rich-mention"))

    with (
        _tool_context(adapter, session),
        patch.object(message_tools.api, "send_rich_text_message", send),
    ):
        result = json.loads(
            await message_tools.octo_send_rich_text_handler(
                {
                    "blocks": [
                        {"type": "text", "text": "@[u1:Alice] one"},
                        {"type": "text", "text": "@[u1:Alice] two"},
                    ]
                }
            )
        )

    assert result["ok"] is True
    mention_uid_allowlist.assert_awaited_once_with(
        _ROUTE.chat_id,
        _ROUTE.channel_type,
        http_session=session,
    )
    assert prepare_mentions.await_args_list == [
        call(
            "@[u1:Alice] one",
            _ROUTE.chat_id,
            _ROUTE.channel_type,
            mention_uid_allowlist={"u1"},
            log_filtered=False,
        ),
        call(
            "@[u1:Alice] two",
            _ROUTE.chat_id,
            _ROUTE.channel_type,
            mention_uid_allowlist={"u1"},
            log_filtered=False,
        ),
    ]
    kwargs = send.await_args.kwargs
    assert [block.text for block in kwargs["blocks"]] == ["@Alice one", "@Alice two"]
    assert kwargs["mention_uids"] == ["u1"]
    assert [
        (entity.uid, entity.offset, entity.length)
        for entity in kwargs["mention_entities"]
    ] == [("u1", 0, 6), ("u1", 10, 6)]

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
async def test_file_tool_rejects_invalid_name_before_upload_side_effect():
    upload = AsyncMock(
        return_value=(
            "https://cdn.example/report.bin",
            b"data",
            "application/octet-stream",
            "report.bin",
        )
    )
    send = AsyncMock(return_value=SendMessageResult(message_id="file-1"))

    with (
        _tool_context(),
        patch.object(message_tools, "_upload_media", upload),
        patch.object(message_tools.api, "send_media_message", send),
    ):
        result = json.loads(
            await message_tools.octo_send_file_handler(
                {
                    "source": "https://public.example/report.bin",
                    "file_name": "../secret.bin",
                }
            )
        )

    assert result == {"error": "file_name is invalid"}
    upload.assert_not_awaited()
    send.assert_not_awaited()


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
        patch.object(message_tools.api, "authorize_local_media_path", return_value=str(source)),
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
async def test_file_tool_rejects_local_path_before_read_when_hermes_denies(tmp_path):
    source = tmp_path / "denied.txt"
    source.write_text("must not be read", encoding="utf-8")
    read_local = MagicMock(side_effect=AssertionError("denied path was read"))
    with (
        _tool_context(),
        patch.object(message_tools.api, "authorize_local_media_path", return_value=None),
        patch.object(message_tools.api, "read_local_media", read_local),
    ):
        result = json.loads(
            await message_tools.octo_send_file_handler({"source": str(source)})
        )

    assert result == {"error": "local media source is not authorized"}
    read_local.assert_not_called()


@pytest.mark.asyncio
async def test_file_tool_reads_only_the_path_authorized_by_hermes(tmp_path):
    requested = tmp_path / "requested.txt"
    authorized = tmp_path / "authorized.txt"
    requested.write_text("requested", encoding="utf-8")
    authorized.write_text("authorized", encoding="utf-8")
    upload = AsyncMock(return_value="https://cdn.example/authorized.txt")
    send = AsyncMock(return_value=SendMessageResult(message_id="media-authorized"))
    with (
        _tool_context(),
        patch.object(
            message_tools.api,
            "authorize_local_media_path",
            return_value=str(authorized),
        ) as validate,
        patch.object(message_tools.api, "upload_and_get_url", upload),
        patch.object(message_tools.api, "send_media_message", send),
    ):
        result = json.loads(
            await message_tools.octo_send_file_handler({"source": str(requested)})
        )

    assert result["ok"] is True
    validate.assert_called_once_with(str(requested))
    assert upload.await_args.args[3:] == (
        "authorized.txt",
        b"authorized",
        "text/plain",
    )
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


@pytest.mark.asyncio
async def test_media_tool_converts_caption_mentions_for_current_route():
    tool_session = _Session()
    roster = AsyncMock(
        return_value=[GroupMember(uid="u1", name="Alice", robot=False)]
    )
    adapter = make_bare_adapter()
    adapter._api_url = _ADAPTER._api_url
    adapter._bot_token = _ADAPTER._bot_token
    adapter._cdn_url = _ADAPTER._cdn_url
    adapter._on_behalf_of = _ADAPTER.on_behalf_of
    adapter._http_session = object()
    download = AsyncMock(
        return_value=(b"document", "application/octet-stream", "report.bin")
    )
    upload = AsyncMock(return_value="https://cdn.example/report.bin")
    send_media = AsyncMock(
        return_value=SendMessageResult(message_id="media-caption")
    )
    send_caption = AsyncMock()
    with (
        _tool_context(adapter, tool_session),
        patch.object(message_tools.api, "get_group_members", roster),
        patch.object(message_tools.api, "download_file", download),
        patch.object(message_tools.api, "upload_and_get_url", upload),
        patch.object(message_tools.api, "send_media_message", send_media),
        patch.object(message_tools.api, "send_message", send_caption),
    ):
        result = json.loads(
            await message_tools.octo_send_file_handler(
                {
                    "source": "https://public.example/report.bin",
                    "caption": "@[u1:Alice] report",
                }
            )
        )

    assert result["ok"] is True
    roster.assert_awaited_once_with(
        tool_session,
        _ADAPTER._api_url,
        _ADAPTER._bot_token,
        _ROUTE.chat_id,
    )
    kwargs = send_caption.await_args.kwargs
    assert kwargs["content"] == "@Alice report"
    assert kwargs["mention_uids"] == ["u1"]
    assert [entity.uid for entity in kwargs["mention_entities"]] == ["u1"]


@pytest.mark.asyncio
async def test_edit_card_tool_releases_transient_edits_and_completes_final_edit():
    adapter = SimpleNamespace(
        _api_url="https://octo.invalid",
        _bot_token="test-token",
        _card_profile_cache=message_tools.cards.CardProfileCache(),
        _card_sessions=CardSessionRegistry(),
    )
    adapter._card_sessions.register(_editable_session())
    edit = AsyncMock()
    with (
        _tool_context(adapter),
        patch.object(
            message_tools.api,
            "get_card_profile",
            AsyncMock(return_value=_MANIFEST),
        ),
        patch.object(message_tools.api, "edit_card_message", edit),
    ):
        transient = json.loads(
            await message_tools.octo_edit_card_handler(
                {
                    "message_id": "card-1",
                    "title": "Updated",
                    "blocks": [{"type": "text", "text": "First"}],
                    "final": False,
                }
            )
        )
        final = json.loads(
            await message_tools.octo_edit_card_handler(
                {
                    "message_id": "card-1",
                    "title": "Updated",
                    "blocks": [{"type": "text", "text": "Final"}],
                    "final": True,
                }
            )
        )
        after_final = json.loads(
            await message_tools.octo_edit_card_handler(
                {
                    "message_id": "card-1",
                    "blocks": [{"type": "text", "text": "Forged"}],
                }
            )
        )

    assert transient == {
        "ok": True,
        "data": {"edited": True, "message_id": "card-1", "card_seq": 1},
    }
    assert final == {
        "ok": True,
        "data": {"edited": True, "message_id": "card-1", "card_seq": 2},
    }
    assert after_final == {
        "error": "card edit does not match a live trusted card session"
    }
    assert [
        (call.kwargs["card_seq"], call.kwargs["transient"], call.kwargs["plain"])
        for call in edit.await_args_list
    ] == [
        (1, True, "Updated\nFirst"),
        (2, False, "Updated\nFinal"),
    ]


@pytest.mark.asyncio
async def test_edit_card_tool_releases_failed_edit_claim_for_a_later_edit():
    adapter = SimpleNamespace(
        _api_url="https://octo.invalid",
        _bot_token="test-token",
        _card_profile_cache=message_tools.cards.CardProfileCache(),
        _card_sessions=CardSessionRegistry(),
    )
    adapter._card_sessions.register(_editable_session())
    edit = AsyncMock(side_effect=[RuntimeError("network down"), None])
    with (
        _tool_context(adapter),
        patch.object(
            message_tools.api,
            "get_card_profile",
            AsyncMock(return_value=_MANIFEST),
        ),
        patch.object(message_tools.api, "edit_card_message", edit),
    ):
        failed = json.loads(
            await message_tools.octo_edit_card_handler(
                {
                    "message_id": "card-1",
                    "blocks": [{"type": "text", "text": "First"}],
                }
            )
        )
        retried = json.loads(
            await message_tools.octo_edit_card_handler(
                {
                    "message_id": "card-1",
                    "blocks": [{"type": "text", "text": "Second"}],
                    "final": True,
                }
            )
        )

    assert failed == {"error": "Octo card edit failed"}
    assert retried == {
        "ok": True,
        "data": {"edited": True, "message_id": "card-1", "card_seq": 2},
    }
    assert [call.kwargs["card_seq"] for call in edit.await_args_list] == [1, 2]

@pytest.mark.asyncio
async def test_edit_card_tool_fails_closed_without_a_matching_registered_session():
    adapter = SimpleNamespace(
        _api_url="https://octo.invalid",
        _bot_token="test-token",
        _card_sessions=CardSessionRegistry(),
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
                    "blocks": [{"type": "text", "text": "forged"}],
                }
            )
        )

    assert result == {
        "error": "card edit does not match a live trusted card session"
    }
    edit.assert_not_awaited()
