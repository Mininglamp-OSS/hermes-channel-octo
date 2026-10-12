"""Inbound gate: Octo system accounts must never reach the agent.

``botfather`` / ``notification`` / ``fileHelper`` / ``____system`` / ``u_10000``
are platform plumbing. A message from one of them used to start an agent turn
whose reply looped back through the same account (unbounded model calls).

Structured events are still allowed through, because system accounts emit those
legitimately: the guard sits *after* the structured-event block, so such frames
are handled and return before it is reached.

Every negative case below has a positive control next to it, so a test cannot
pass by accident — each negative assertion is paired with a real sender/shape
that must still be delivered.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin.types import ChannelType
from tests.conftest import make_bare_adapter

SYSTEM_UIDS = ["botfather", "notification", "fileHelper", "____system", "u_10000"]
HUMAN_UID = "a" * 32


def _recv(from_uid: str, payload: bytes, *, channel_id: str, channel_type: ChannelType):
    return SimpleNamespace(
        message_id="message-1",
        message_seq=1,
        from_uid=from_uid,
        channel_id=channel_id,
        channel_type=channel_type,
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
@pytest.mark.parametrize("uid", SYSTEM_UIDS)
async def test_system_account_dm_never_reaches_the_agent(uid):
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hello"}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv(uid, raw, channel_id=uid, channel_type=ChannelType.DM)
        )

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uid",
    [
        "s14_botfather",
        "s3_notification",
        "s14_fileHelper",
        "s27_u_10000",
        "s14_____system",
    ],
)
async def test_space_prefixed_system_account_dm_never_reaches_the_agent(uid):
    """Octo carries Space membership inline (``s14_botfather``).

    The raw wire uid is not in the denylist, so a guard that compares it
    un-normalized fails open here — exactly the shape the loop from #41
    arrives in on a Space-scoped deployment.
    """
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hello"}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv(uid, raw, channel_id=uid, channel_type=ChannelType.DM)
        )

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_space_prefixed_human_dm_still_reaches_the_agent():
    """Positive control for the Space-prefix case: normalizing must not
    swallow real senders who happen to carry a Space prefix."""
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hello"}'
    human = f"s14_{HUMAN_UID}"

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv(human, raw, channel_id=human, channel_type=ChannelType.DM)
        )

    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_system_account_group_message_never_reaches_the_agent():
    """The system account @-mentions the bot, so nothing but the system-account
    gate can keep this frame away from the agent (a non-mentioning group message
    is already stopped by the pre-existing mention gate)."""
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hello", "mention": {"uids": ["bot-1"]}}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv("botfather", raw, channel_id="group-1", channel_type=ChannelType.Group)
        )

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_human_group_message_mentioning_the_bot_still_reaches_the_agent():
    """Positive control for the group case above: the same frame from a human
    sender is delivered, so that assertion is about the sender, not the shape."""
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hello", "mention": {"uids": ["bot-1"]}}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv(HUMAN_UID, raw, channel_id="group-1", channel_type=ChannelType.Group)
        )

    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_system_account_group_message_is_not_recorded_in_history():
    """Pins the side effect: dropping the frame means it never reaches the
    mention gate, which is what writes non-@ group messages into history."""
    adapter = _inbound_adapter()
    adapter._record_history_entry = MagicMock()
    raw = b'{"type": 1, "content": "hello"}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv("botfather", raw, channel_id="group-1", channel_type=ChannelType.Group)
        )

    adapter._record_history_entry.assert_not_called()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_human_non_mention_group_message_is_recorded_in_history():
    """Positive control for the history assertion: a human sending the same
    non-mentioning group message does reach `_record_history_entry`, so the
    assertion above can actually fail."""
    adapter = _inbound_adapter()
    adapter._record_history_entry = MagicMock()
    raw = b'{"type": 1, "content": "hello"}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv(HUMAN_UID, raw, channel_id="group-1", channel_type=ChannelType.Group)
        )

    adapter._record_history_entry.assert_called_once()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["group_invite", "friend_request", "x"])
async def test_system_account_dm_with_unhandled_event_type_never_reaches_the_agent(
    event_type,
):
    """Only ``group_md_updated`` / ``group_md_deleted`` are handled downstream.
    Any other ``event.type`` carrying content used to pass an event-shaped
    exemption and fall through to the agent (no mention gate on a DM)."""
    adapter = _inbound_adapter()
    raw = (
        '{"type": 1, "content": "You were invited", "event": {"type": "'
        + event_type
        + '"}}'
    ).encode()

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv("notification", raw, channel_id="notification", channel_type=ChannelType.DM)
        )

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["group_md_updated", "group_md_deleted"])
async def test_system_account_handled_event_without_channel_id_never_reaches_the_agent(
    event_type,
):
    """Now structural: the handler requires ``msg.channel_id``, so a handled
    event type with an empty channel id misses it — and still never reaches the
    agent, because the guard is downstream of the handler."""
    adapter = _inbound_adapter()
    raw = (
        '{"type": 1, "content": "hi", "event": {"type": "' + event_type + '"}}'
    ).encode()

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv("botfather", raw, channel_id="", channel_type=ChannelType.DM)
        )

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_human_dm_with_event_still_reaches_the_agent():
    """Positive control for the two cases above: an unknown event type from a
    human is delivered, so those assertions are about the sender."""
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hi", "event": {"type": "group_invite"}}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv(HUMAN_UID, raw, channel_id=HUMAN_UID, channel_type=ChannelType.DM)
        )

    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_system_account_structured_event_still_flows():
    adapter = _inbound_adapter()
    group_md = MagicMock()
    adapter._handle_group_md_event = group_md
    raw = b'{"type": 1, "event": {"type": "group_md_deleted"}}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv("botfather", raw, channel_id="group-1", channel_type=ChannelType.Group)
        )

    assert group_md.call_count == 1
    assert adapter.handle_message.await_count == 0
