"""Inbound gate: Octo system accounts must never reach the agent.

``botfather`` / ``notification`` / ``____system`` / ``u_10000`` are platform
plumbing. A DM from one of them used to start an agent turn whose reply looped
back through the same account (unbounded model calls). Structured events are
still allowed through, because system accounts emit those legitimately.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin.types import ChannelType
from tests.conftest import make_bare_adapter


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
@pytest.mark.parametrize("uid", ["botfather", "notification", "____system", "u_10000"])
async def test_system_account_dm_never_reaches_the_agent(uid):
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hello"}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv(uid, raw, channel_id=uid, channel_type=ChannelType.DM)
        )

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_system_account_group_message_never_reaches_the_agent():
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hello"}'

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv("botfather", raw, channel_id="group-1", channel_type=ChannelType.Group)
        )

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_human_dm_still_reaches_the_agent():
    adapter = _inbound_adapter()
    raw = b'{"type": 1, "content": "hello"}'
    human = "a" * 32

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(
            _recv(human, raw, channel_id=human, channel_type=ChannelType.DM)
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
