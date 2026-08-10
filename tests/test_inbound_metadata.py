"""Inbound metadata gates: broadcasts, robots, cards, and unknown types."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import MessageType as HermesMessageType
from hermes_octo_plugin import api
from hermes_octo_plugin.types import ChannelType
from tests.conftest import make_bare_adapter


def _group_recv(payload: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        message_id="message-1",
        message_seq=1,
        from_uid="human-1",
        channel_id="group-1",
        channel_type=ChannelType.Group,
        timestamp=1,
        encrypted_payload=payload,
    )


def _inbound_adapter():
    adapter = make_bare_adapter()
    adapter._robot_id = "bot-1"
    adapter._aes_key = b"key"
    adapter._aes_iv = b"iv"
    adapter.handle_message = AsyncMock()
    adapter.build_source = MagicMock(
        side_effect=lambda **kwargs: SimpleNamespace(chat_id=kwargs["chat_id"])
    )
    return adapter


@pytest.mark.asyncio
async def test_human_broadcast_does_not_activate_the_bot():
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "notice", "mention": {"all": 1, "humans": 1}}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(_group_recv(raw))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ignore_legacy_all", "expected_calls"),
    [(False, 1), (True, 0)],
)
async def test_legacy_all_activation_honors_ignore_setting(
    ignore_legacy_all: bool,
    expected_calls: int,
):
    adapter = _inbound_adapter()
    adapter._ignore_mention_all = ignore_legacy_all
    raw = b'{"type": 1, "content": "legacy broadcast", "mention": {"all": 1}}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(_group_recv(raw))

    assert adapter.handle_message.await_count == expected_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mention",
    [
        '{"ais": 1}',
        '{"all": 1, "humans": 1, "ais": 1}',
        '{"all": 1, "humans": 1, "uids": ["bot-1"]}',
    ],
)
async def test_ai_broadcast_or_explicit_bot_uid_activates_the_bot(mention):
    adapter = _inbound_adapter()
    raw = ('{"type": 1, "content": "notice", "mention": ' + mention + "}").encode()

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(_group_recv(raw))

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("robot_value", [True, None])
async def test_relaxed_mention_gate_fails_closed_for_robot_or_unknown_sender(robot_value):
    adapter = _inbound_adapter()
    adapter._require_mention = False
    adapter._group_robot_map = {"group-1": {"human-1": robot_value}}
    raw = b'{"type": 1, "content": "ordinary group message"}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(_group_recv(raw))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_relaxed_mention_gate_allows_only_a_confirmed_human_sender():
    adapter = _inbound_adapter()
    adapter._require_mention = False
    adapter._group_robot_map = {"group-1": {"human-1": False}}
    raw = b'{"type": 1, "content": "ordinary group message"}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(_group_recv(raw))

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_relaxed_mention_gate_fails_closed_after_member_refresh_failure():
    adapter = _inbound_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._require_mention = False
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._uid_to_name = {"human-1": "Alice"}
    adapter._group_md_checked.add("group-1")
    # This was a confirmed human before the cache expired. A failed refresh
    # must not keep that stale authorization decision alive.
    adapter._group_robot_map = {"group-1": {"human-1": False}}
    raw = b'{"type": 1, "content": "ordinary group message"}'

    with (
        patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw),
        patch.object(
            api,
            "get_group_members",
            AsyncMock(side_effect=RuntimeError("member API unavailable")),
        ),
    ):
        await adapter._handle_recv(_group_recv(raw))

    adapter.handle_message.assert_not_awaited()
    assert "group-1" not in adapter._group_robot_map


@pytest.mark.asyncio
async def test_gif_is_delivered_as_visual_media():
    adapter = _inbound_adapter()
    raw = (
        b'{"type": 3, "url": "https://media.example/animated.gif", '
        b'"mention": {"uids": ["bot-1"]}}'
    )

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(_group_recv(raw))

    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == HermesMessageType.PHOTO
    assert event.media_urls == ["https://media.example/animated.gif"]
    assert event.media_types == ["image/gif"]
