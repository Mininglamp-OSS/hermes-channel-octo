"""Final-only Octo text delivery contracts.

Octo's Bot API exposes ``message/edit`` for persisted message-extra updates, but
that endpoint is not a dependable client-visible text streaming transport.
Hermes must therefore buffer model deltas and send one authoritative text
message when the turn completes.
"""

from __future__ import annotations
from typing import Any, cast

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin.adapter import MAX_MESSAGE_LENGTH, OctoAdapter
from hermes_octo_plugin.types import (
    ChannelType,
    GroupMember,
    MentionEntity,
    SendMessageResult,
)
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    adapter = make_bare_adapter()
    adapter._http_session = cast(Any, object())  # truthy test transport
    adapter._api_url = "https://example.test"
    adapter._bot_token = "tok"
    adapter.truncate_message = (
        lambda content, max_length=MAX_MESSAGE_LENGTH, len_fn=None: [content]
    )
    return adapter


def test_octo_disables_gateway_edit_streaming() -> None:
    """The gateway must choose its send-final-only path for Octo."""
    assert OctoAdapter.SUPPORTS_MESSAGE_EDITING is False


@pytest.mark.asyncio
async def test_gateway_final_response_does_not_quote_trigger_message() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group
    complete = "先执行工具，再给出完整最终答案。"

    with (
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(
                return_value=SendMessageResult(
                    message_id="server-final",
                    message_seq=42,
                )
            ),
        ) as send_message,
        patch(
            "hermes_octo_plugin.adapter.api.edit_message",
            new=AsyncMock(),
        ) as edit_message,
    ):
        result = await adapter.send(
            "chatA",
            complete,
            reply_to="inbound-1",
            metadata={"notify": True},
        )

    assert result.success is True
    assert result.message_id == "server-final"
    send_message.assert_awaited_once()
    send_call = send_message.await_args
    assert send_call is not None
    assert send_call.kwargs["content"] == complete
    assert send_call.kwargs["reply_msg_id"] is None
    edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_reply_still_quotes_target_message() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(return_value=SendMessageResult(message_id="server-reply")),
    ) as send_message:
        result = await adapter.send("chatA", "明确回复", reply_to="message-42")

    assert result.success is True
    send_call = send_message.await_args
    assert send_call is not None
    assert send_call.kwargs["reply_msg_id"] == "message-42"


@pytest.mark.asyncio
async def test_complete_send_uses_fresh_group_roster_for_mentions() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group
    adapter._group_member_rosters["chatA"] = {"stale-member": "过期成员"}
    adapter._group_robot_map["chatA"] = {"stale-member": False}
    adapter._group_cache_timestamps["chatA"] = 2**63
    complete = "结果：@[member-1:成员]，尾部完整。"
    get_members = AsyncMock(
        return_value=[GroupMember(uid="member-1", name="成员", robot=False)]
    )

    with (
        patch(
            "hermes_octo_plugin.adapter.api.get_group_members",
            new=get_members,
        ),
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-final")),
        ) as send_message,
    ):
        result = await adapter.send("chatA", complete)

    assert result.success is True
    get_members.assert_awaited_once_with(
        adapter._http_session,
        "https://example.test",
        "tok",
        "chatA",
    )
    send_call = send_message.await_args
    assert send_call is not None
    kwargs = send_call.kwargs
    assert kwargs["content"] == "结果：@成员，尾部完整。"
    assert kwargs["mention_uids"] == ["member-1"]
    entities = cast(list[MentionEntity], kwargs["mention_entities"])
    assert [(entity.uid, entity.offset, entity.length) for entity in entities] == [
        ("member-1", 3, 3)
    ]



@pytest.mark.asyncio
async def test_unknown_group_roster_is_fetched_before_sending_mentions() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group
    complete = "结果：@[member-1:成员]，尾部完整。"
    get_members = AsyncMock(
        return_value=[GroupMember(uid="member-1", name="成员", robot=False)]
    )

    with (
        patch(
            "hermes_octo_plugin.adapter.api.get_group_members",
            new=get_members,
        ),
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-final")),
        ) as send_message,
    ):
        result = await adapter.send("chatA", complete)

    assert result.success is True
    get_members.assert_awaited_once()
    kwargs = send_message.await_args.kwargs
    assert kwargs["content"] == "结果：@成员，尾部完整。"
    assert kwargs["mention_uids"] == ["member-1"]


@pytest.mark.asyncio
async def test_unknown_group_roster_failure_sends_inert_mention_and_reply() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group

    with (
        patch(
            "hermes_octo_plugin.adapter.api.get_group_members",
            new=AsyncMock(side_effect=RuntimeError("offline")),
        ),
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-final")),
        ) as send_message,
    ):
        result = await adapter.send("chatA", "@[member-1:成员]")

    assert result.success is True
    kwargs = send_message.await_args.kwargs
    assert kwargs["content"] == "@成员"
    assert kwargs["mention_uids"] == []
    assert kwargs["mention_entities"] == []


@pytest.mark.asyncio
async def test_later_chunk_roster_failure_sends_complete_inert_message() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group
    adapter.truncate_message = MagicMock(
        return_value=["first chunk", "@[member-1:成员]"]
    )

    with (
        patch(
            "hermes_octo_plugin.adapter.api.get_group_members",
            new=AsyncMock(side_effect=RuntimeError("offline")),
        ),
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(
                side_effect=[
                    SendMessageResult(message_id="server-first"),
                    SendMessageResult(message_id="server-second"),
                ]
            ),
        ) as send_message,
    ):
        result = await adapter.send(
            "chatA",
            "first chunk@[member-1:成员]",
        )

    assert result.success is True
    assert send_message.await_count == 2
    first, second = send_message.await_args_list
    assert first.kwargs["content"] == "first chunk"
    assert first.kwargs["mention_uids"] is None
    assert first.kwargs["mention_entities"] is None
    assert second.kwargs["content"] == "@成员"
    assert second.kwargs["mention_uids"] == []
    assert second.kwargs["mention_entities"] == []


@pytest.mark.asyncio
async def test_split_structured_mention_sends_no_raw_fragments() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group
    adapter.truncate_message = MagicMock(
        return_value=["prefix @[member-1", ":成员] suffix"]
    )
    get_members = AsyncMock()

    with (
        patch(
            "hermes_octo_plugin.adapter.api.get_group_members",
            new=get_members,
        ),
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(),
        ) as send_message,
    ):
        result = await adapter.send(
            "chatA",
            "prefix @[member-1:成员] suffix",
        )

    assert result.success is False
    assert "mention" in (result.error or "")
    get_members.assert_not_awaited()
    send_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("roster_uid", ["", " ", "/", 123])
async def test_unusable_group_roster_sends_inert_mention(
    roster_uid: Any,
) -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group

    with (
        patch(
            "hermes_octo_plugin.adapter.api.get_group_members",
            new=AsyncMock(
                return_value=[
                    GroupMember(uid=roster_uid, name="成员", robot=False)
                ]
            ),
        ),
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-final")),
        ) as send_message,
    ):
        result = await adapter.send("chatA", "@[member-1:成员]")

    assert result.success is True
    kwargs = send_message.await_args.kwargs
    assert kwargs["content"] == "@成员"
    assert kwargs["mention_uids"] == []
    assert kwargs["mention_entities"] == []


@pytest.mark.asyncio
async def test_dm_structured_mention_is_inert_without_roster_lookup() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["user-1"] = ChannelType.DM
    get_members = AsyncMock()

    with (
        patch(
            "hermes_octo_plugin.adapter.api.get_group_members",
            new=get_members,
        ),
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-final")),
        ) as send_message,
    ):
        result = await adapter.send("user-1", "@[arbitrary-uid:Alice]")

    assert result.success is True
    get_members.assert_not_awaited()
    kwargs = send_message.await_args.kwargs
    assert kwargs["content"] == "@Alice"
    assert kwargs["mention_uids"] == []
    assert kwargs["mention_entities"] == []

@pytest.mark.asyncio
async def test_send_returns_the_real_server_message_id() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.DM

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(return_value=SendMessageResult(message_id="9223372036854775807")),
    ):
        result = await adapter.send("chatA", "完整回答")

    assert result.success is True
    assert result.message_id == "9223372036854775807"


@pytest.mark.asyncio
async def test_thread_send_never_mutates_membership_implicitly() -> None:
    adapter = _make_adapter()
    thread_id = "group-1____thread-1"
    adapter._chat_kind[thread_id] = ChannelType.CommunityTopic

    with (
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
        ) as send_message,
        patch(
            "hermes_octo_plugin.adapter.api.join_thread",
            new=AsyncMock(),
        ) as join_thread,
    ):
        result = await adapter.send(thread_id, "hello")

    assert result.success is True
    join_thread.assert_not_awaited()
    send_call = send_message.await_args
    assert send_call is not None
    assert send_call.kwargs["channel_id"] == thread_id
    assert send_call.kwargs["channel_type"] == ChannelType.CommunityTopic


@pytest.mark.asyncio
async def test_send_failure_is_reported() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(side_effect=RuntimeError("upstream unavailable")),
    ):
        result = await adapter.send("chatA", "完整回答")

    assert result.success is False
    assert "upstream unavailable" in (result.error or "")


@pytest.mark.asyncio
async def test_long_final_response_uses_normal_chunking() -> None:
    adapter = _make_adapter()
    adapter._chat_kind["chatA"] = ChannelType.Group
    adapter.truncate_message = (
        lambda content, max_length=MAX_MESSAGE_LENGTH, len_fn=None: [
            "first",
            "second",
        ]
    )
    send_message = AsyncMock(
        side_effect=[
            SendMessageResult(message_id="server-1"),
            SendMessageResult(message_id="server-2"),
        ]
    )

    with patch("hermes_octo_plugin.adapter.api.send_message", new=send_message):
        result = await adapter.send("chatA", "x" * (MAX_MESSAGE_LENGTH + 1))

    assert [call.kwargs["content"] for call in send_message.await_args_list] == [
        "first",
        "second",
    ]
    assert result.message_id == "server-2"


def test_strip_hermes_cursor_helper() -> None:
    adapter = _make_adapter()
    assert adapter._strip_hermes_cursor("hello ▉") == "hello"
    assert adapter._strip_hermes_cursor("hello") == "hello"
