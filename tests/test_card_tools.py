"""Current-conversation Octo card tool contracts."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from hermes_octo_plugin import card_tools, cards
from hermes_octo_plugin.types import CardProfileManifest, ChannelType, SendMessageResult


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _Adapter:
    _api_url = "https://api.example.invalid"
    _bot_token = "test-token"
    on_behalf_of: str | None = None
    def __init__(self) -> None:
        self.sessions = []
        self._card_profile_cache = cards.CardProfileCache()

    def _register_card_session(self, session) -> None:
        self.sessions.append(session)


    def _resolve_channel_type(self, chat_id: str) -> ChannelType:
        assert chat_id == "group-1"
        return ChannelType.Group

    def _outbound_channel_id(
        self,
        chat_id: str,
        channel_type: ChannelType,
    ) -> str:
        assert channel_type == ChannelType.Group
        return chat_id


def _session_value(name: str, default: str = "") -> str:
    values = {
        "HERMES_SESSION_PLATFORM": "octo",
        "HERMES_SESSION_CHAT_ID": "group-1",
        "HERMES_SESSION_USER_ID": "user-1",
        "HERMES_SESSION_KEY": "octo:group-1:user-1",
    }
    return values.get(name, default)


@pytest.mark.asyncio
async def test_card_profile_probe_reuses_the_adapter_short_lived_cache() -> None:
    adapter = _Adapter()
    adapter._card_profile_cache = cards.CardProfileCache()
    manifest = CardProfileManifest(available=True, enabled=True)
    probe = AsyncMock(return_value=manifest)

    with patch.object(card_tools.api, "get_card_profile", probe):
        assert await card_tools._get_card_profile(adapter, _Session()) is manifest
        assert await card_tools._get_card_profile(adapter, _Session()) is manifest

    probe.assert_awaited_once()


@pytest.mark.asyncio
async def test_display_tool_sends_negotiated_card_only_to_trusted_conversation() -> None:
    manifest = CardProfileManifest(
        available=True,
        enabled=True,
        profiles=("octo/v1", "octo/v2"),
        card_version="1.5",
        elements=("TextBlock",),
    )
    with (
        patch.object(card_tools, "_resolve_adapter", return_value=_Adapter()),
        patch.object(card_tools, "_new_guarded_http_session", return_value=_Session()),
        patch("gateway.session_context.get_session_env", side_effect=_session_value),
        patch.object(card_tools.api, "get_card_profile", AsyncMock(return_value=manifest)),
        patch.object(
            card_tools.api,
            "send_card_message",
            AsyncMock(
                return_value=SendMessageResult(
                    message_id="card-1",
                    message_seq=7,
                    client_msg_no="card-client-1",
                )
            ),
        ) as send_card,
    ):
        result = json.loads(
            await card_tools.octo_send_display_card_handler(
                {
                    "title": "Review",
                    "blocks": [{"type": "text", "text": "Ready"}],
                }
            )
        )

    assert result == {
        "ok": True,
        "mode": "card",
        "message_id": "card-1",
        "message_seq": 7,
        "client_msg_no": "card-client-1",
    }
    kwargs = send_card.await_args.kwargs
    assert kwargs["channel_id"] == "group-1"
    assert kwargs["channel_type"] == ChannelType.Group
    assert kwargs["profile"] == "octo/v1"
    assert kwargs["plain"] == "Review\nReady"


@pytest.mark.asyncio
async def test_display_tool_falls_back_to_plain_text_in_same_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OCTO_CARD_MESSAGE_ENABLED", raising=False)
    with (
        patch.object(card_tools, "_resolve_adapter", return_value=_Adapter()),
        patch.object(card_tools, "_new_guarded_http_session", return_value=_Session()),
        patch("gateway.session_context.get_session_env", side_effect=_session_value),
        patch.object(
            card_tools.api,
            "get_card_profile",
            AsyncMock(return_value=CardProfileManifest(available=False, enabled=False)),
        ),
        patch.object(card_tools.api, "send_card_message", AsyncMock()) as send_card,
        patch.object(
            card_tools.api,
            "send_message",
            AsyncMock(return_value=SendMessageResult(message_id="text-1")),
        ) as send_text,
    ):
        result = json.loads(
            await card_tools.octo_send_display_card_handler(
                {"blocks": [{"type": "text", "text": "Fallback"}]}
            )
        )

    assert result == {"ok": True, "mode": "plain", "message_id": "text-1"}
    send_card.assert_not_awaited()
    assert send_text.await_args.kwargs["channel_id"] == "group-1"
    assert send_text.await_args.kwargs["content"] == "Fallback"


@pytest.mark.asyncio
async def test_display_tool_uses_plain_on_behalf_of_delivery() -> None:
    adapter = _Adapter()
    adapter.on_behalf_of = "grantor-1"
    send_text = AsyncMock(
        return_value=SendMessageResult(
            message_id="text-obo",
            message_seq=8,
            client_msg_no="text-client-obo",
        )
    )
    get_profile = AsyncMock()
    with (
        patch.object(card_tools, "_resolve_adapter", return_value=adapter),
        patch.object(card_tools, "_new_guarded_http_session", return_value=_Session()),
        patch("gateway.session_context.get_session_env", side_effect=_session_value),
        patch.object(card_tools.api, "get_card_profile", get_profile),
        patch.object(card_tools.api, "send_card_message", AsyncMock()) as send_card,
        patch.object(card_tools.api, "send_message", send_text),
    ):
        result = json.loads(
            await card_tools.octo_send_display_card_handler(
                {"blocks": [{"type": "text", "text": "Persona fallback"}]}
            )
        )

    assert result == {
        "ok": True,
        "mode": "plain",
        "message_id": "text-obo",
        "message_seq": 8,
        "client_msg_no": "text-client-obo",
    }
    get_profile.assert_not_awaited()
    send_card.assert_not_awaited()
    assert send_text.await_args.kwargs["on_behalf_of"] == "grantor-1"
    client_msg_no = send_text.await_args.kwargs["client_msg_no"]
    assert str(UUID(client_msg_no)) == client_msg_no


@pytest.mark.asyncio
async def test_interactive_tool_binds_submit_to_trusted_session() -> None:
    manifest = CardProfileManifest(
        available=True,
        enabled=True,
        profiles=("octo/v2",),
        card_version="1.5",
        elements=("TextBlock",),
        inputs=("Input.Text",),
        actions=("Action.Submit",),
    )
    adapter = _Adapter()
    with (
        patch.object(card_tools, "_resolve_adapter", return_value=adapter),
        patch.object(card_tools, "_new_guarded_http_session", return_value=_Session()),
        patch("gateway.session_context.get_session_env", side_effect=_session_value),
        patch.object(card_tools.api, "get_card_profile", AsyncMock(return_value=manifest)),
        patch.object(
            card_tools.api,
            "send_card_message",
            AsyncMock(
                return_value=SendMessageResult(
                    message_id="card-2",
                    message_seq=9,
                    client_msg_no="card-client-2",
                )
            ),
        ) as send_card,
    ):
        result = json.loads(
            await card_tools.octo_send_interactive_card_handler(
                {
                    "title": "Approve",
                    "inputs": [{"id": "note", "kind": "text", "label": "Note"}],
                    "buttons": [
                        {
                            "id": "approve",
                            "label": "Approve",
                            "data": {"decision": "approve", "_octo_binding": "forged"},
                        }
                    ],
                }
            )
        )

    assert result["ok"] is True
    assert result["mode"] == "card"
    assert result["message_id"] == "card-2"
    assert result["message_seq"] == 9
    assert result["client_msg_no"] == "card-client-2"
    sent_card = send_card.await_args.kwargs["card"]
    binding = sent_card["actions"][0]["data"]["_octo_binding"]
    assert binding != "forged"
    assert result["binding_id"] == binding
    assert send_card.await_args.kwargs["profile"] == "octo/v2"
    assert len(adapter.sessions) == 1
    registered = adapter.sessions[0]
    assert registered.message_id == "card-2"
    assert registered.binding_id == binding
    assert registered.session_key == "octo:group-1:user-1"
    assert registered.requester_uid == "user-1"


@pytest.mark.asyncio
async def test_interactive_tool_falls_back_when_v2_manifest_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCTO_CARD_MESSAGE_ENABLED", "1")
    adapter = _Adapter()
    send_card = AsyncMock(return_value=SendMessageResult(message_id="card-legacy"))
    send_text = AsyncMock(return_value=SendMessageResult(message_id="text-2"))
    with (
        patch.object(card_tools, "_resolve_adapter", return_value=adapter),
        patch.object(card_tools, "_new_guarded_http_session", return_value=_Session()),
        patch("gateway.session_context.get_session_env", side_effect=_session_value),
        patch.object(
            card_tools.api,
            "get_card_profile",
            AsyncMock(return_value=CardProfileManifest(available=False, enabled=False)),
        ),
        patch.object(card_tools.api, "send_card_message", send_card),
        patch.object(card_tools.api, "send_message", send_text),
    ):
        result = json.loads(
            await card_tools.octo_send_interactive_card_handler(
                {
                    "title": "Approve",
                    "buttons": [{"id": "approve", "label": "Approve"}],
                }
            )
        )

    assert result == {"ok": True, "mode": "plain", "message_id": "text-2"}
    send_card.assert_not_awaited()
    send_text.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler,args",
    [
        (
            card_tools.octo_send_display_card_handler,
            {"blocks": [{"type": "text", "text": "x"}], "channel_id": "other"},
        ),
        (
            card_tools.octo_send_interactive_card_handler,
            {
                "title": "x",
                "buttons": [{"id": "ok", "label": "OK"}],
                "requester_uid": "other",
            },
        ),
    ],
)
async def test_card_tools_reject_model_controlled_route_or_identity(handler, args) -> None:
    with patch.object(card_tools, "_resolve_adapter", return_value=_Adapter()):
        result = json.loads(await handler(args))

    assert result == {"ok": False, "error": "trusted session fields cannot be supplied"}


def test_card_tool_schemas_expose_no_destination_or_identity_fields() -> None:
    for schema in (
        card_tools.DISPLAY_CARD_TOOL_SCHEMA,
        card_tools.INTERACTIVE_CARD_TOOL_SCHEMA,
    ):
        parameters = schema["parameters"]
        assert parameters["additionalProperties"] is False
        assert {
            "channel_id",
            "channel_type",
            "target",
            "requester_uid",
            "session_key",
        }.isdisjoint(parameters["properties"])



def test_display_tool_schema_exposes_only_controlled_block_variants() -> None:
    item_schema = card_tools.DISPLAY_CARD_TOOL_SCHEMA["parameters"][
        "properties"
    ]["blocks"]["items"]
    block_types = {
        variant["properties"]["type"]["const"]
        for variant in item_schema["oneOf"]
    }

    assert block_types == {
        "heading",
        "text",
        "section",
        "facts",
        "image",
        "actions",
    }