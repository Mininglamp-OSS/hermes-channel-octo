"""Standalone sender contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from hermes_octo_plugin import adapter as adapter_module
from hermes_octo_plugin import api
from hermes_octo_plugin.adapter import _standalone_send
from hermes_octo_plugin.types import GroupMember, SendMessageResult


class _Session:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *_args):
        return None


@pytest.mark.asyncio
async def test_standalone_sender_returns_real_server_message_identity():
    config = SimpleNamespace(
        extra={
            "api_url": "https://api.example.invalid",
            "bot_token": "test-token",
            "on_behalf_of": "grantor-1",
        },
        token="",
    )
    with (
        patch.object(
            adapter_module,
            "_new_guarded_http_session",
            return_value=_Session(),
        ) as guarded_session,
        patch.object(
            api,
            "send_message",
            AsyncMock(
                return_value=SendMessageResult(
                    message_id="9223372036854775807",
                    message_seq=4,
                    client_msg_no="client-1",
                )
            ),
        ) as send_message,
    ):
        result = await _standalone_send(config, "group-1", "hello")

    assert result == {
        "success": True,
        "message_id": "9223372036854775807",
        "message_seq": 4,
        "client_msg_no": "client-1",
    }
    kwargs = send_message.await_args.kwargs
    assert kwargs["on_behalf_of"] == "grantor-1"
    assert str(UUID(kwargs["client_msg_no"])) == kwargs["client_msg_no"]
    guarded_session.assert_called_once_with("https://api.example.invalid")



@pytest.mark.asyncio
async def test_standalone_sender_filters_mentions_with_authoritative_roster():
    config = SimpleNamespace(
        extra={
            "api_url": "https://api.example.invalid",
            "bot_token": "test-token",
        },
        token="",
    )
    members = [GroupMember(uid="member-1", name="Member", robot=False)]
    send_message = AsyncMock(
        return_value=SendMessageResult(message_id="message-1")
    )
    with (
        patch.object(
            adapter_module,
            "_new_guarded_http_session",
            return_value=_Session(),
        ),
        patch.object(
            api,
            "get_group_members",
            AsyncMock(return_value=members),
        ) as get_members,
        patch.object(api, "send_message", send_message),
    ):
        result = await _standalone_send(
            config,
            "group-1",
            "@[member-1:Member] and @[outsider:Eve]",
        )

    assert result["success"] is True
    get_members.assert_awaited_once()
    kwargs = send_message.await_args.kwargs
    assert kwargs["content"] == "@Member and @Eve"
    assert kwargs["mention_uids"] == ["member-1"]
    assert [entity.uid for entity in kwargs["mention_entities"]] == ["member-1"]


@pytest.mark.asyncio
async def test_standalone_sender_sends_inert_mention_when_roster_is_unavailable():
    config = SimpleNamespace(
        extra={
            "api_url": "https://api.example.invalid",
            "bot_token": "test-token",
        },
        token="",
    )
    send_message = AsyncMock(
        return_value=SendMessageResult(message_id="standalone-message")
    )
    with (
        patch.object(
            adapter_module,
            "_new_guarded_http_session",
            return_value=_Session(),
        ),
        patch.object(
            api,
            "get_group_members",
            AsyncMock(side_effect=RuntimeError("offline")),
        ),
        patch.object(api, "send_message", send_message),
    ):
        result = await _standalone_send(
            config,
            "group-1",
            "@[member-1:Member]",
        )

    assert result["success"] is True
    kwargs = send_message.await_args.kwargs
    assert kwargs["content"] == "@Member"
    assert kwargs["mention_uids"] == []
    assert kwargs["mention_entities"] == []


@pytest.mark.asyncio
async def test_standalone_sender_sends_inert_mention_for_unusable_roster_uids():
    config = SimpleNamespace(
        extra={
            "api_url": "https://api.example.invalid",
            "bot_token": "test-token",
        },
        token="",
    )
    send_message = AsyncMock(
        return_value=SendMessageResult(message_id="standalone-message")
    )
    with (
        patch.object(
            adapter_module,
            "_new_guarded_http_session",
            return_value=_Session(),
        ),
        patch.object(
            api,
            "get_group_members",
            AsyncMock(
                return_value=[GroupMember(uid="/", name="invalid", robot=False)]
            ),
        ),
        patch.object(api, "send_message", send_message),
    ):
        result = await _standalone_send(
            config,
            "group-1",
            "@[member-1:Member]",
        )

    assert result["success"] is True
    kwargs = send_message.await_args.kwargs
    assert kwargs["content"] == "@Member"
    assert kwargs["mention_uids"] == []
    assert kwargs["mention_entities"] == []