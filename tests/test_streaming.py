"""Unit tests for the cross-segment coalescing streaming model.

Octo has no real edit-in-place API, so hermes' segment-by-segment
streaming sends would otherwise produce one bubble per LLM segment —
broken markdown at segment boundaries. We buffer EVERYTHING for a chat
across segments and only flush after STREAM_FLUSH_DELAY_S of true idle,
coalescing the response into one clean bubble.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from hermes_octo_plugin.adapter import (
    OctoAdapter,
    STREAM_FLUSH_DELAY_S,
)
from hermes_octo_plugin.types import ChannelType
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    a = make_bare_adapter()
    a._http_session = object()  # truthy
    a._api_url = "https://example.test"
    a._bot_token = "tok"
    a.truncate_message = lambda content, max_len: [content]
    return a


# ─── send buffers (cursor or no cursor) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_send_opens_buffer_for_first_segment():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch("hermes_octo_plugin.adapter.api.send_message", new=AsyncMock(side_effect=AssertionError("no early write"))):
        r = await a.send("chatA", "查到了 ▉")

    assert r.success
    state = a._active_streams["chatA"]
    assert state["current_segment"] == "查到了"
    assert state["segments"] == []
    state["flush_task"].cancel()


@pytest.mark.asyncio
async def test_send_with_reply_to_still_buffers():
    """reply_to MUST NOT bypass the buffer — the consumer's first-frame
    send always passes _initial_reply_to_id, so a reply_to opt-out would
    silently defeat coalescing for every streaming response."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.DM

    with patch("hermes_octo_plugin.adapter.api.send_message", new=AsyncMock(side_effect=AssertionError("no early write"))):
        r = await a.send("chatA", "hi ▉", reply_to="parent-123")

    assert r.success
    assert "chatA" in a._active_streams
    state = a._active_streams["chatA"]
    assert state["current_segment"] == "hi"
    state["flush_task"].cancel()


@pytest.mark.asyncio
async def test_send_with_no_stream_metadata_uses_normal_path():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.DM
    captured: list = []

    async def fake_msg(_s, _u, _t, *, channel_id, channel_type, content, **kw):
        captured.append(content)

    with patch("hermes_octo_plugin.adapter.api.send_message", new=fake_msg):
        await a.send("chatA", "hi", metadata={"no_stream": True})

    assert captured == ["hi"]
    assert "chatA" not in a._active_streams


# ─── second send appends as new segment, doesn't drop prior ─────────────────


@pytest.mark.asyncio
async def test_second_send_closes_prior_segment():
    """A new send() (e.g. next-segment first-frame) must NOT drop the
    prior in-progress segment — close it into segments[] first."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch("hermes_octo_plugin.adapter.api.send_message", new=AsyncMock(side_effect=AssertionError("no early write"))):
        await a.send("chatA", "**Headers:** ▉")
        await a.send("chatA", "- `Authorization: ...` ▉")

    state = a._active_streams["chatA"]
    assert state["segments"] == ["**Headers:**"]
    assert state["current_segment"] == "- `Authorization: ...`"
    state["flush_task"].cancel()


# ─── edit_message ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_edit_message_updates_current_segment():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch("hermes_octo_plugin.adapter.api.send_message", new=AsyncMock(side_effect=AssertionError("no early write"))):
        r = await a.send("chatA", "查 ▉")
        await a.edit_message("chatA", r.message_id, "查到了。 ▉")
        await a.edit_message("chatA", r.message_id, "查到了。最终内容 ▉")

    state = a._active_streams["chatA"]
    assert state["current_segment"] == "查到了。最终内容"
    assert state["segments"] == []  # nothing finalized yet
    state["flush_task"].cancel()


@pytest.mark.asyncio
async def test_finalize_closes_current_segment_but_does_not_flush():
    """finalize=True closes the current segment into segments[] but the
    actual octo write happens only after STREAM_FLUSH_DELAY_S idle."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch("hermes_octo_plugin.adapter.api.send_message", new=AsyncMock(side_effect=AssertionError("no early write"))):
        r = await a.send("chatA", "seg1 ▉")
        await a.edit_message("chatA", r.message_id, "seg1 complete", finalize=True)

    state = a._active_streams["chatA"]
    assert state["segments"] == ["seg1 complete"]
    assert state["current_segment"] == ""
    state["flush_task"].cancel()


@pytest.mark.asyncio
async def test_edit_message_returns_failure_when_no_buffer():
    a = _make_adapter()
    r = await a.edit_message("chatA", "buf-???", "anything")
    assert r.success is False


# ─── multi-segment coalescing (the main behaviour) ──────────────────────────


@pytest.mark.asyncio
async def test_two_segments_coalesce_into_one_send_normal_call():
    """The behaviour the buffer exists to provide: a response split across
    two segments (e.g. by a tool call) lands as ONE octo message, not two.
    """
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    sent: list = []

    async def fake_msg(_s, _u, _t, *, channel_id, channel_type, content, **kw):
        sent.append(content)

    # Block the flush task from firing until we say so — real production
    # uses sleep(3s); a stubbed-out sleep would fire as soon as the test
    # yields between operations, hiding the coalescing.
    async def never_sleep(_):
        await asyncio.Event().wait()  # blocks forever

    with patch("hermes_octo_plugin.adapter.api.send_message", new=fake_msg), \
         patch("hermes_octo_plugin.adapter.asyncio.sleep", new=never_sleep):
        # Segment 1 streaming + finalize
        r1 = await a.send("chatA", "**Headers:** ▉")
        await a.edit_message("chatA", r1.message_id, "**Headers:**\n```bash", finalize=True)

        # Segment 2 first frame — must JOIN, not start a new bubble.
        r2 = await a.send("chatA", "curl -X POST ▉")
        assert r2.message_id == r1.message_id

        await a.edit_message("chatA", r2.message_id, "curl -X POST /v1/bot/register", finalize=True)

        # Manually trigger the flush (production: watchdog after 3s of silence)
        await a._close_active_stream("chatA")

    # ONE outbound message containing both segments concatenated
    assert len(sent) == 1
    assert sent[0] == "**Headers:**\n```bash" + "curl -X POST /v1/bot/register"


@pytest.mark.asyncio
async def test_commentary_coalesces_with_response_if_close_in_time():
    """One-shot commentary (cursor-free) shares the same buffer when sent
    inside the coalescing window. Trade-off documented on the class."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.DM
    sent: list = []

    async def fake_msg(_s, _u, _t, *, channel_id, channel_type, content, **kw):
        sent.append(content)

    async def never_sleep(_):
        await asyncio.Event().wait()

    with patch("hermes_octo_plugin.adapter.api.send_message", new=fake_msg), \
         patch("hermes_octo_plugin.adapter.asyncio.sleep", new=never_sleep):
        # Tool indicator (no cursor) — opens buffer
        await a.send("chatA", "📚 skill_view: octo-bot-api")
        # Response segment within the coalescing window
        r2 = await a.send("chatA", "查到了 ▉")
        await a.edit_message("chatA", r2.message_id, "查到了，结果是...", finalize=True)
        await a._close_active_stream("chatA")

    assert len(sent) == 1
    assert sent[0] == "📚 skill_view: octo-bot-api" + "查到了，结果是..."


# ─── idle flush watchdog actually fires ─────────────────────────────────────


@pytest.mark.asyncio
async def test_idle_flush_delivers_buffered_content():
    """When NO further activity arrives, the watchdog flushes after the
    idle delay. Verified by letting the patched sleep return immediately."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.DM
    sent: list = []

    async def fake_msg(_s, _u, _t, *, channel_id, channel_type, content, **kw):
        sent.append(content)

    real_sleep = asyncio.sleep

    async def short_sleep(_):
        await real_sleep(0)

    with patch("hermes_octo_plugin.adapter.api.send_message", new=fake_msg), \
         patch("hermes_octo_plugin.adapter.asyncio.sleep", new=short_sleep):
        await a.send("chatA", "abandoned ▉")
        # Yield several times to let the watchdog fire its (patched) sleep
        for _ in range(5):
            await real_sleep(0)

    assert sent == ["abandoned"]
    assert "chatA" not in a._active_streams


# ─── cursor strip helper ────────────────────────────────────────────────────


def test_strip_hermes_cursor_helper():
    a = _make_adapter()
    assert a._strip_hermes_cursor("hello ▉") == "hello"
    assert a._strip_hermes_cursor("hello") == "hello"
    assert a._strip_hermes_cursor("PO▉ST") == "POST"
    assert a._strip_hermes_cursor("") == ""


# ─── joined_buffer helper ───────────────────────────────────────────────────


def test_joined_buffer_concatenates_segments_and_current():
    a = _make_adapter()
    state = {"segments": ["one", "two"], "current_segment": "three"}
    assert a._joined_buffer(state) == "onetwothree"


def test_joined_buffer_handles_empty_current():
    a = _make_adapter()
    state = {"segments": ["one"], "current_segment": ""}
    assert a._joined_buffer(state) == "one"


def test_joined_buffer_handles_empty_state():
    a = _make_adapter()
    state = {"segments": [], "current_segment": ""}
    assert a._joined_buffer(state) == ""
