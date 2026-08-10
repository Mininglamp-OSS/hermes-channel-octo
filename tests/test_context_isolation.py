"""Regression tests for group, thread, and Space context boundaries."""

from __future__ import annotations

import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from types import SimpleNamespace
import time
from uuid import UUID

import pytest

from hermes_octo_plugin import api
from hermes_octo_plugin.adapter import CACHE_MAX_AGE_S
from hermes_octo_plugin.types import ChannelType, GroupMember, SendMessageResult
from tests.conftest import make_bare_adapter


@pytest.mark.asyncio
async def test_thread_member_refresh_uses_parent_roster_and_keeps_it_scoped():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    members = [GroupMember(uid="u1", name="Alice")]

    with patch.object(api, "get_group_members", AsyncMock(return_value=members)) as get_members:
        refreshed = await adapter._refresh_group_member_cache(
            "group-1____thread-1", force=True
        )

    assert refreshed is True
    get_members.assert_awaited_once_with(
        ANY, "https://api.example.invalid", "test-token", "group-1"
    )
    assert adapter._group_member_rosters == {"group-1": {"u1": "Alice"}}


@pytest.mark.asyncio
async def test_inflight_member_refresh_cannot_restore_evicted_group_scope():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._delete_md_from_disk = MagicMock()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_members(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return [GroupMember(uid="u1", name="Alice", robot=False)]

    with patch.object(api, "get_group_members", new=blocked_members):
        task = asyncio.create_task(
            adapter._refresh_group_member_cache("group-1", force=True)
        )
        await entered.wait()
        await adapter._evict_group_scope("group-1")
        release.set()
        refreshed = await task

    assert refreshed is False
    assert "group-1" not in adapter._group_member_rosters
    assert "group-1" not in adapter._group_robot_map
    assert "group-1" not in adapter._group_cache_timestamps
    assert adapter.find_shared_groups("u1") == []


@pytest.mark.asyncio
async def test_thread_history_fetches_the_composite_channel_as_a_topic():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"

    with patch.object(api, "get_channel_messages", AsyncMock(return_value=[])) as get_history:
        await adapter._build_history_context("group-1____thread-1", "bot-1")

    get_history.assert_awaited_once_with(
        ANY,
        "https://api.example.invalid",
        "test-token",
        channel_id="group-1____thread-1",
        channel_type=ChannelType.CommunityTopic,
        limit=adapter._history_limit,
    )


@pytest.mark.asyncio
async def test_parent_group_history_keeps_group_channel_type():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"

    with patch.object(api, "get_channel_messages", AsyncMock(return_value=[])) as get_history:
        await adapter._build_history_context("group-1", "bot-1")

    assert get_history.await_args.kwargs["channel_id"] == "group-1"
    assert get_history.await_args.kwargs["channel_type"] == ChannelType.Group


@pytest.mark.asyncio
async def test_group_history_uses_the_parent_roster_not_global_name_cache():
    adapter = make_bare_adapter()
    adapter._group_member_rosters = {"group-1": {"u1": "Alice"}}
    adapter._uid_to_name = {"u1": "Leaked Name"}
    adapter._group_histories["group-1"] = [
        {"sender": "u1", "body": "message", "mention": None, "timestamp": 1}
        for _ in range(adapter._history_limit)
    ]

    context = await adapter._build_history_context("group-1", "bot-1")

    assert "Alice(u1)" in context
    assert "Leaked Name" not in context


@pytest.mark.asyncio
async def test_space_qualified_dm_sessions_stay_isolated_but_reply_to_bare_uid():
    adapter = make_bare_adapter()
    adapter._robot_id = "bot-1"
    adapter._aes_key = b"key"
    adapter._aes_iv = b"iv"
    adapter.handle_message = AsyncMock()
    adapter.build_source = MagicMock(
        side_effect=lambda **kwargs: SimpleNamespace(chat_id=kwargs["chat_id"])
    )

    def recv(channel_id: str):
        return SimpleNamespace(
            message_id="message-1",
            message_seq=1,
            from_uid="user-1",
            channel_id=channel_id,
            channel_type=ChannelType.DM,
            timestamp=1,
            encrypted_payload=b"ignored",
        )

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=b'{"type": 1, "content": "hi"}'):
        await adapter._handle_recv(recv("s14_user-1"))
        await adapter._handle_recv(recv("s27_user-1"))

    first_event, second_event = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert first_event.source.chat_id == "s14_user-1"
    assert second_event.source.chat_id == "s27_user-1"
    assert first_event.source.chat_id != second_event.source.chat_id

    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._on_behalf_of = "grantor-1"
    with patch.object(
        api,
        "send_message",
        AsyncMock(return_value=SendMessageResult(message_id="server-1")),
    ) as send_message:
        result = await adapter.send(
            "s14_user-1", "reply", metadata={"no_stream": True}
        )

    assert result.success is True
    assert send_message.await_args.kwargs["channel_id"] == "user-1"
    assert send_message.await_args.kwargs["channel_type"] == ChannelType.DM
    assert send_message.await_args.kwargs["on_behalf_of"] == "grantor-1"
    client_msg_no = send_message.await_args.kwargs["client_msg_no"]
    assert str(UUID(client_msg_no)) == client_msg_no


@pytest.mark.asyncio
async def test_space_qualified_dm_receipts_and_typing_use_bare_wire_uid():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._space_dm_targets = {"s14_user-1": "user-1"}

    with (
        patch.object(api, "send_read_receipt", AsyncMock()) as receipt,
        patch.object(api, "send_typing", AsyncMock()) as typing,
    ):
        await adapter._send_read_receipt_safe(
            "s14_user-1", ChannelType.DM, ["message-1"]
        )
        await adapter._send_typing_safe("s14_user-1", ChannelType.DM)

    assert receipt.await_args.args[3] == "user-1"
    assert typing.await_args.kwargs["channel_id"] == "user-1"


def test_cache_eviction_removes_only_stale_scoped_group_and_space_maps():
    adapter = make_bare_adapter()
    adapter._group_member_rosters = {
        "group-stale": {"u1": "Alice"},
        "group-fresh": {"u2": "Bob"},
    }
    adapter._space_dm_targets = {
        "s14_user-1": "user-1",
        "s27_user-1": "user-1",
    }
    adapter._known_group_ids = {"group-stale", "group-fresh"}
    adapter._group_names = {
        "group-stale": "Stale Secret Group",
        "group-fresh": "Fresh Group",
    }
    adapter._user_group_index = {
        "u1": {"group-stale", "group-fresh"},
        "u2": {"group-stale"},
    }
    now = time.monotonic()
    adapter._cache_activity = {
        "group-stale": now - CACHE_MAX_AGE_S - 1,
        "group-fresh": now,
        "s14_user-1": now - CACHE_MAX_AGE_S - 1,
        "s27_user-1": now,
    }

    adapter._cleanup_caches()

    assert "group-stale" not in adapter._group_member_rosters
    assert adapter._group_member_rosters["group-fresh"] == {"u2": "Bob"}
    assert "s14_user-1" not in adapter._space_dm_targets
    assert adapter._space_dm_targets["s27_user-1"] == "user-1"
    assert adapter._known_group_ids == {"group-fresh"}
    assert "group-stale" not in adapter._group_names
    assert adapter.find_shared_groups("u1") == [
        {"group_no": "group-fresh", "name": "Fresh Group"}
    ]
    assert adapter.find_shared_groups("u2") == []


@pytest.mark.asyncio
async def test_group_mention_context_uses_current_group_name_map_not_global_map():
    adapter = make_bare_adapter()
    adapter._robot_id = "bot-1"
    adapter._aes_key = b"key"
    adapter._aes_iv = b"iv"
    adapter._group_member_rosters = {"group-1": {"right-uid": "Bob"}}
    adapter._member_map = {"Bob": "wrong-uid"}
    adapter.handle_message = AsyncMock()
    adapter.build_source = MagicMock(
        side_effect=lambda **kwargs: SimpleNamespace(chat_id=kwargs["chat_id"])
    )
    recv = SimpleNamespace(
        message_id="message-1",
        message_seq=1,
        from_uid="human-1",
        channel_id="group-1",
        channel_type=ChannelType.Group,
        timestamp=1,
        encrypted_payload=b"ignored",
    )

    with patch(
        "hermes_octo_plugin.adapter.aes_decrypt",
        return_value=b'{"type": 1, "content": "@Bob hi", "mention": {"uids": ["bot-1"]}}',
    ):
        await adapter._handle_recv(recv)

    event = adapter.handle_message.await_args.args[0]
    assert event.text == "@[right-uid:Bob] hi"
    assert "wrong-uid" not in event.text


@pytest.mark.asyncio
async def test_startup_prefetch_populates_the_same_scoped_roster_as_refresh():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._hydrate_md_cache_from_disk = MagicMock()
    adapter._write_md_to_disk = MagicMock()

    with (
        patch.object(api, "fetch_bot_groups", AsyncMock(return_value=[{"group_no": "group-1"}])),
        patch.object(api, "get_group_md", AsyncMock(return_value=None)),
        patch.object(
            api,
            "get_group_members",
            AsyncMock(return_value=[GroupMember(uid="u1", name="Alice")]),
        ),
    ):
        await adapter._prefetch_groups_and_members()

    assert adapter._group_member_rosters == {"group-1": {"u1": "Alice"}}


@pytest.mark.asyncio
async def test_inflight_prefetch_cannot_restore_evicted_group_scope():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._hydrate_md_cache_from_disk = MagicMock()
    adapter._write_md_to_disk = MagicMock()
    adapter._delete_md_from_disk = MagicMock()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_md(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return {"content": "stale", "version": 9}

    get_members = AsyncMock(
        return_value=[GroupMember(uid="u1", name="Alice", robot=False)]
    )
    with (
        patch.object(
            api,
            "fetch_bot_groups",
            AsyncMock(return_value=[{"group_no": "group-1"}]),
        ),
        patch.object(api, "get_group_md", new=blocked_md),
        patch.object(api, "get_group_members", get_members),
    ):
        task = asyncio.create_task(adapter._prefetch_groups_and_members())
        await entered.wait()
        await adapter._evict_group_scope("group-1")
        release.set()
        await task

    assert "group-1" not in adapter._known_group_ids
    assert "group-1" not in adapter._group_md_cache
    assert "group-1" not in adapter._group_md_checked
    assert "group-1" not in adapter._group_member_rosters
    assert "group-1" not in adapter._group_names
    assert adapter.find_shared_groups("u1") == []
    get_members.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_prefetch_evicts_membership_facts_absent_from_server_snapshot():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._hydrate_md_cache_from_disk = MagicMock()
    adapter._write_md_to_disk = MagicMock()
    adapter._delete_md_from_disk = MagicMock()
    adapter._known_group_ids = {"group-left"}
    adapter._group_member_rosters = {"group-left": {"u1": "Alice"}}
    adapter._group_names = {"group-left": "Old Secret"}
    adapter._user_group_index = {"u1": {"group-left"}}
    adapter._group_histories = {
        "group-left": [{"body": "secret"}],
        "group-left____t1": [{"body": "thread secret"}],
    }
    adapter._group_md_cache = {
        "group-left": {"content": "secret", "version": 1},
        "group-left____t1": {"content": "thread secret", "version": 1},
    }
    adapter._group_md_checked = {"group-left", "group-left____t1"}

    with (
        patch.object(
            api,
            "fetch_bot_groups",
            AsyncMock(return_value=[{"group_no": "group-current"}]),
        ),
        patch.object(api, "get_group_md", AsyncMock(return_value=None)),
        patch.object(
            api,
            "get_group_members",
            AsyncMock(return_value=[GroupMember(uid="u2", name="Bob")]),
        ),
    ):
        await adapter._prefetch_groups_and_members()

    assert adapter._known_group_ids == {"group-current"}
    assert "group-left" not in adapter._group_member_rosters
    assert "group-left" not in adapter._group_names
    assert adapter.find_shared_groups("u1") == []
    assert not any(key.startswith("group-left") for key in adapter._group_histories)
    assert not any(key.startswith("group-left") for key in adapter._group_md_cache)
    assert not any(key.startswith("group-left") for key in adapter._group_md_checked)
