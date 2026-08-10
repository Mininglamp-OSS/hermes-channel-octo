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
@pytest.mark.parametrize("raw", [b"[]", b"null", b'"text"'])
async def test_non_object_payload_is_ignored(raw: bytes) -> None:
    adapter = _inbound_adapter()

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
async def test_ignore_mention_all_disables_ai_broadcast_but_not_direct_uid():
    adapter = _inbound_adapter()
    adapter._ignore_mention_all = True
    ai_broadcast = (
        b'{"type": 1, "content": "notice", '
        b'"mention": {"all": true, "ais": true}}'
    )
    direct = (
        b'{"type": 1, "content": "direct", '
        b'"mention": {"all": true, "ais": true, "uids": ["bot-1"]}}'
    )

    with patch(
        "hermes_octo_plugin.adapter.aes_decrypt",
        side_effect=[ai_broadcast, direct],
    ):
        await adapter._handle_recv(_group_recv(ai_broadcast))
        await adapter._handle_recv(_group_recv(direct))

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "direct"


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
async def test_eligible_inbound_rolls_progress_before_gateway_busy_dispatch():
    adapter = _inbound_adapter()
    order: list[str] = []
    adapter.handle_message = AsyncMock(side_effect=lambda _event: order.append("gateway"))
    raw = (
        b'{"type": 1, "content": "follow up", '
        b'"mention": {"uids": ["bot-1"]}}'
    )

    with (
        patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw),
        patch(
            "hermes_octo_plugin.card_progress.on_octo_inbound_message",
            side_effect=lambda **_kwargs: order.append("progress"),
        ) as rollover,
    ):
        await adapter._handle_recv(_group_recv(raw))

    rollover.assert_called_once_with(
        chat_id="group-1",
        requester_uid="human-1",
    )
    assert order == ["progress", "gateway"]


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
    adapter._download_inbound_media_to_local = AsyncMock(
        return_value="/tmp/octo-media/animated.gif"
    )
    raw = (
        b'{"type": 3, "url": "https://media.example/animated.gif", '
        b'"mention": {"uids": ["bot-1"]}}'
    )

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(_group_recv(raw))

    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == HermesMessageType.PHOTO
    assert event.media_urls == ["/tmp/octo-media/animated.gif"]
    assert event.media_types == ["image/gif"]
    adapter._download_inbound_media_to_local.assert_awaited_once_with(
        "https://media.example/animated.gif",
        "image/gif",
    )
