"""Unit tests for the server-backed cross-segment streaming model.

The first Hermes frame creates one real Octo message. Later frames edit that
same message so segment boundaries do not split markdown across bubbles. An
idle watchdog performs the protocol-level finalize.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin.adapter import (
    MAX_MESSAGE_LENGTH,
    OctoAdapter,
    STREAM_FLUSH_DELAY_S,
)
from hermes_octo_plugin.types import ChannelType, SendMessageResult
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    a = make_bare_adapter()
    a._http_session = object()  # truthy
    a._api_url = "https://example.test"
    a._bot_token = "tok"
    a.truncate_message = lambda content, max_len: [content]
    return a


# ─── server-acknowledged streaming lifecycle ─────────────────────────────────


@pytest.mark.asyncio
async def test_first_stream_send_waits_for_server_identity_before_success():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(
            return_value=SendMessageResult(message_id="server-1", message_seq=1)
        ),
    ) as send_message:
        result = await a.send("chatA", "partial ▉")

    assert result.success is True
    assert result.message_id == "server-1"
    assert a._active_streams["chatA"]["message_id"] == "server-1"
    send_message.assert_awaited_once()
    a._active_streams["chatA"]["flush_task"].cancel()


@pytest.mark.asyncio
async def test_first_stream_send_failure_is_reported_without_buffering():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(side_effect=RuntimeError("upstream unavailable")),
    ):
        result = await a.send("chatA", "partial ▉")

    assert result.success is False
    assert "chatA" not in a._active_streams


@pytest.mark.asyncio
async def test_follow_on_segment_edit_failure_preserves_last_confirmed_state():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
    ):
        first = await a.send("chatA", "confirmed ▉")

    with patch(
        "hermes_octo_plugin.adapter.api.edit_message",
        new=AsyncMock(side_effect=RuntimeError("edit rejected")),
    ) as edit_message:
        second = await a.send("chatA", "unconfirmed ▉")

    assert first.message_id == "server-1"
    assert second.success is False
    assert a._joined_buffer(a._active_streams["chatA"]) == "confirmed"
    edit_message.assert_awaited_once()
    a._active_streams["chatA"]["flush_task"].cancel()


@pytest.mark.asyncio
async def test_idle_finalize_failure_keeps_stream_retryable():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group
    a._active_streams["chatA"] = {
        "message_id": "server-1",
        "channel_type": ChannelType.Group,
        "segments": ["confirmed"],
        "current_segment": "",
        "flush_task": None,
    }

    with (
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=AsyncMock()),
        patch(
            "hermes_octo_plugin.adapter.api.edit_message",
            new=AsyncMock(side_effect=RuntimeError("finalize rejected")),
        ),
    ):
        await a._close_stream_after_idle("chatA", "server-1")

    assert "chatA" in a._active_streams
    assert a._joined_buffer(a._active_streams["chatA"]) == "confirmed"


@pytest.mark.asyncio
async def test_concurrent_first_frames_create_only_one_server_message():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    create_calls = 0

    async def delayed_send(*_args, **_kwargs):
        nonlocal create_calls
        create_calls += 1
        first_entered.set()
        await release_first.wait()
        return SendMessageResult(message_id=f"server-{create_calls}")

    with (
        patch("hermes_octo_plugin.adapter.api.send_message", new=delayed_send),
        patch(
            "hermes_octo_plugin.adapter.api.edit_message",
            new=AsyncMock(return_value={"accepted": True}),
        ),
    ):
        first = asyncio.create_task(a.send("chatA", "one"))
        await first_entered.wait()
        second = asyncio.create_task(a.send("chatA", "two"))
        await asyncio.sleep(0)
        assert create_calls == 1
        release_first.set()
        first_result, second_result = await asyncio.gather(first, second)

    assert first_result.message_id == second_result.message_id == "server-1"
    assert a._joined_buffer(a._active_streams["chatA"]) == "onetwo"
    a._active_streams["chatA"]["flush_task"].cancel()


@pytest.mark.asyncio
async def test_idle_finalize_and_new_send_are_linearized():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group
    a._active_streams["chatA"] = {
        "message_id": "server-1",
        "channel_type": ChannelType.Group,
        "segments": ["old answer"],
        "current_segment": "",
        "flush_task": None,
    }
    finalize_entered = asyncio.Event()
    release_finalize = asyncio.Event()

    async def controlled_edit(*_args, finalize, **_kwargs):
        if finalize:
            finalize_entered.set()
            await release_finalize.wait()
        return {"accepted": True}

    with (
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=AsyncMock()),
        patch("hermes_octo_plugin.adapter.api.edit_message", new=controlled_edit),
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-2")),
        ) as send_message,
    ):
        close_task = asyncio.create_task(
            a._close_stream_after_idle("chatA", "server-1")
        )
        await finalize_entered.wait()
        send_task = asyncio.create_task(a.send("chatA", "new answer"))
        await asyncio.sleep(0)
        assert not send_task.done()
        release_finalize.set()
        await close_task
        result = await send_task

    assert result.message_id == "server-2"
    send_message.assert_awaited_once()
    assert a._joined_buffer(a._active_streams["chatA"]) == "new answer"
    a._active_streams["chatA"]["flush_task"].cancel()


@pytest.mark.asyncio
async def test_failed_idle_finalize_retries_and_closes_stream():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group
    a._active_streams["chatA"] = {
        "message_id": "server-1",
        "channel_type": ChannelType.Group,
        "segments": ["confirmed"],
        "current_segment": "",
        "flush_task": None,
    }
    attempts = 0
    real_sleep = asyncio.sleep

    async def flaky_edit(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient finalize failure")
        return {"accepted": True}

    async def short_sleep(_delay):
        await real_sleep(0)

    with (
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=short_sleep),
        patch("hermes_octo_plugin.adapter.api.edit_message", new=flaky_edit),
    ):
        await a._close_stream_after_idle("chatA", "server-1")
        for _ in range(10):
            if "chatA" not in a._active_streams:
                break
            await real_sleep(0)

    assert attempts == 2
    assert "chatA" not in a._active_streams


@pytest.mark.asyncio
async def test_oversized_first_stream_frame_fails_before_server_io():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group
    a.truncate_message = lambda content, max_len: [
        content[:max_len], content[max_len:]
    ]

    with patch(
        "hermes_octo_plugin.adapter.api.send_message", new=AsyncMock()
    ) as send_message:
        result = await a.send("chatA", "x" * (MAX_MESSAGE_LENGTH + 1))

    assert result.success is False
    assert "single-message limit" in (result.error or "")
    send_message.assert_not_awaited()
    assert "chatA" not in a._active_streams


# ─── send buffers (cursor or no cursor) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_send_opens_server_backed_stream_for_first_segment():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
    ):
        result = await a.send("chatA", "查到了 ▉")

    assert result.success
    assert result.message_id == "server-1"
    state = a._active_streams["chatA"]
    assert state["current_segment"] == "查到了"
    assert state["segments"] == []
    state["flush_task"].cancel()


@pytest.mark.asyncio
async def test_streamed_send_exposes_only_the_real_server_identity():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
    ):
        result = await a.send("chatA", "partial ▉")

    assert result.success is True
    assert result.message_id == "server-1"
    a._active_streams["chatA"]["flush_task"].cancel()


@pytest.mark.asyncio
async def test_send_with_reply_to_still_buffers():
    """reply_to MUST NOT bypass the buffer — the consumer's first-frame
    send always passes _initial_reply_to_id, so a reply_to opt-out would
    silently defeat coalescing for every streaming response."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.DM

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
    ) as send_message:
        result = await a.send("chatA", "hi ▉", reply_to="parent-123")

    assert result.success
    assert result.message_id == "server-1"
    assert "chatA" in a._active_streams
    assert a._active_streams["chatA"]["current_segment"] == "hi"
    assert send_message.await_args is not None
    assert send_message.await_args.kwargs["reply_msg_id"] is None
    a._active_streams["chatA"]["flush_task"].cancel()


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


@pytest.mark.asyncio
async def test_direct_send_returns_the_real_server_message_id():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.DM

    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(
            return_value=SendMessageResult(
                message_id="9223372036854775807", message_seq=8
            )
        ),
    ):
        result = await a.send("chatA", "hi", metadata={"no_stream": True})

    assert result.success is True
    assert result.message_id == "9223372036854775807"


@pytest.mark.asyncio
async def test_direct_thread_send_never_mutates_membership_implicitly():
    a = _make_adapter()
    thread_id = "group-1____thread-1"
    a._chat_kind[thread_id] = ChannelType.CommunityTopic

    with (
        patch(
            "hermes_octo_plugin.adapter.api.join_thread",
            new=AsyncMock(),
        ) as join_thread,
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
        ) as send_message,
    ):
        result = await a.send(thread_id, "hello", metadata={"no_stream": True})

    assert result.success is True
    join_thread.assert_not_awaited()
    send_message.assert_awaited_once()


# ─── second send appends as new segment, doesn't drop prior ─────────────────


@pytest.mark.asyncio
async def test_second_send_closes_prior_segment():
    """A new send() (e.g. next-segment first-frame) must NOT drop the
    prior in-progress segment — close it into segments[] first."""
    a = _make_adapter()
    a._on_behalf_of = "grantor-1"
    a._chat_kind["chatA"] = ChannelType.Group

    with (
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
        ),
        patch(
            "hermes_octo_plugin.adapter.api.edit_message",
            new=AsyncMock(return_value={"accepted": True}),
        ) as edit_message,
    ):
        await a.send("chatA", "**Headers:** ▉")
        result = await a.send("chatA", "- `Authorization: ...` ▉")

    assert result.success is True
    assert result.message_id == "server-1"
    state = a._active_streams["chatA"]
    assert state["segments"] == ["**Headers:**"]
    assert state["current_segment"] == "- `Authorization: ...`"
    assert edit_message.await_args is not None
    assert edit_message.await_args.kwargs["content"] == (
        "**Headers:**- `Authorization: ...`"
    )
    assert edit_message.await_args.kwargs["on_behalf_of"] == "grantor-1"
    state["flush_task"].cancel()


# ─── edit_message ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_edit_message_updates_current_segment():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with (
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
        ),
        patch(
            "hermes_octo_plugin.adapter.api.edit_message",
            new=AsyncMock(return_value={"accepted": True}),
        ) as edit_message,
    ):
        result = await a.send("chatA", "查 ▉")
        await a.edit_message("chatA", result.message_id, "查到了。 ▉")
        await a.edit_message("chatA", result.message_id, "查到了。最终内容 ▉")

    state = a._active_streams["chatA"]
    assert state["current_segment"] == "查到了。最终内容"
    assert state["segments"] == []
    assert edit_message.await_count == 2
    state["flush_task"].cancel()


@pytest.mark.asyncio
async def test_finalize_closes_current_segment_but_does_not_flush():
    """finalize=True closes the current segment into segments[] but the
    actual octo write happens only after STREAM_FLUSH_DELAY_S idle."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with (
        patch(
            "hermes_octo_plugin.adapter.api.send_message",
            new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
        ),
        patch(
            "hermes_octo_plugin.adapter.api.edit_message",
            new=AsyncMock(return_value={"accepted": True}),
        ) as edit_message,
    ):
        result = await a.send("chatA", "seg1 ▉")
        final = await a.edit_message(
            "chatA", result.message_id, "seg1 complete", finalize=True
        )

    assert final.success is True
    state = a._active_streams["chatA"]
    assert state["segments"] == ["seg1 complete"]
    assert state["current_segment"] == ""
    assert edit_message.await_args is not None
    assert edit_message.await_args.kwargs["finalize"] is False
    state["flush_task"].cancel()


@pytest.mark.asyncio
async def test_edit_message_returns_failure_when_no_buffer():
    a = _make_adapter()
    r = await a.edit_message("chatA", "buf-???", "anything")
    assert r.success is False


@pytest.mark.asyncio
async def test_edit_server_message_uses_native_endpoint_and_accepts_finalize():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.edit_message",
        new=AsyncMock(return_value={"accepted": True}),
    ) as edit:
        result = await a.edit_message(
            "chatA", "9223372036854775807", "updated", finalize=True
        )

    assert result.success is True
    assert result.message_id == "9223372036854775807"
    assert edit.await_args.kwargs == {
        "channel_id": "chatA",
        "channel_type": ChannelType.Group,
        "message_id": "9223372036854775807",
        "content": "updated",
        "finalize": True,
        "on_behalf_of": None,
    }


@pytest.mark.asyncio
async def test_native_edit_failure_reports_failed_send_result():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group

    with patch(
        "hermes_octo_plugin.adapter.api.edit_message",
        new=AsyncMock(side_effect=RuntimeError("backend rejected edit")),
    ) as edit:
        result = await a.edit_message("chatA", "server-id", "updated")

    edit.assert_awaited_once()
    assert result.success is False
    assert result.message_id is None


# ─── multi-segment coalescing (the main behaviour) ──────────────────────────


@pytest.mark.asyncio
async def test_two_segments_update_one_server_message():
    """Two Hermes segments create one Octo message and edit it in place."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group
    created: list[str] = []
    edits: list[tuple[str, bool]] = []

    async def fake_msg(_s, _u, _t, *, content, **_kw):
        created.append(content)
        return SendMessageResult(message_id="server-1")

    async def fake_edit(_s, _u, _t, *, content, finalize, **_kw):
        edits.append((content, finalize))
        return {"accepted": True}

    async def never_sleep(_delay):
        await asyncio.Event().wait()

    with (
        patch("hermes_octo_plugin.adapter.api.send_message", new=fake_msg),
        patch("hermes_octo_plugin.adapter.api.edit_message", new=fake_edit),
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=never_sleep),
    ):
        first = await a.send("chatA", "**Headers:** ▉")
        await a.edit_message(
            "chatA", first.message_id, "**Headers:**\n```bash", finalize=True
        )
        second = await a.send("chatA", "curl -X POST ▉")
        assert second.message_id == first.message_id == "server-1"
        await a.edit_message(
            "chatA",
            second.message_id,
            "curl -X POST /v1/bot/register",
            finalize=True,
        )
        await a._close_active_stream("chatA")

    final_body = "**Headers:**\n```bash" + "curl -X POST /v1/bot/register"
    assert created == ["**Headers:**"]
    assert edits[-1] == (final_body, True)
    assert all(body.startswith("**Headers:**") for body, _finalize in edits)


@pytest.mark.asyncio
async def test_commentary_and_response_update_one_server_message():
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.DM
    created: list[str] = []
    edits: list[tuple[str, bool]] = []

    async def fake_msg(_s, _u, _t, *, content, **_kw):
        created.append(content)
        return SendMessageResult(message_id="server-1")

    async def fake_edit(_s, _u, _t, *, content, finalize, **_kw):
        edits.append((content, finalize))
        return {"accepted": True}

    async def never_sleep(_delay):
        await asyncio.Event().wait()

    with (
        patch("hermes_octo_plugin.adapter.api.send_message", new=fake_msg),
        patch("hermes_octo_plugin.adapter.api.edit_message", new=fake_edit),
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=never_sleep),
    ):
        await a.send("chatA", "📚 skill_view: octo-bot-api")
        response = await a.send("chatA", "查到了 ▉")
        await a.edit_message(
            "chatA", response.message_id, "查到了，结果是...", finalize=True
        )
        await a._close_active_stream("chatA")

    final_body = "📚 skill_view: octo-bot-api" + "查到了，结果是..."
    assert created == ["📚 skill_view: octo-bot-api"]
    assert edits[-1] == (final_body, True)


# ─── idle flush watchdog actually fires ─────────────────────────────────────


@pytest.mark.asyncio
async def test_idle_flush_delivers_buffered_content():
    """When NO further activity arrives, the watchdog flushes after the
    idle delay. Verified by letting the patched sleep return immediately."""
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.DM
    created: list[str] = []
    finalized: list[tuple[str, bool]] = []

    async def fake_msg(_s, _u, _t, *, content, **_kw):
        created.append(content)
        return SendMessageResult(message_id="server-1")

    async def fake_edit(_s, _u, _t, *, content, finalize, **_kw):
        finalized.append((content, finalize))
        return {"accepted": True}

    real_sleep = asyncio.sleep

    async def short_sleep(_delay):
        await real_sleep(0)

    with (
        patch("hermes_octo_plugin.adapter.api.send_message", new=fake_msg),
        patch("hermes_octo_plugin.adapter.api.edit_message", new=fake_edit),
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=short_sleep),
    ):
        result = await a.send("chatA", "abandoned ▉")
        assert result.message_id == "server-1"
        for _ in range(5):
            await real_sleep(0)

    assert created == ["abandoned"]
    assert finalized == [("abandoned", True)]
    assert "chatA" not in a._active_streams


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["send", "edit"])
async def test_cancelled_follow_on_edit_rearms_idle_finalize_watchdog(operation: str):
    a = _make_adapter()
    a._chat_kind["chatA"] = ChannelType.Group
    with patch(
        "hermes_octo_plugin.adapter.api.send_message",
        new=AsyncMock(return_value=SendMessageResult(message_id="server-1")),
    ):
        first = await a.send("chatA", "confirmed")

    edit_entered = asyncio.Event()
    release_edit = asyncio.Event()

    async def blocked_edit(*_args, **_kwargs):
        edit_entered.set()
        await release_edit.wait()
        return {"accepted": True}

    with patch("hermes_octo_plugin.adapter.api.edit_message", new=blocked_edit):
        if operation == "send":
            task = asyncio.create_task(a.send("chatA", "replacement"))
        else:
            task = asyncio.create_task(
                a.edit_message("chatA", first.message_id, "replacement")
            )
        await edit_entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    state = a._active_streams["chatA"]
    assert a._joined_buffer(state) == "confirmed"
    assert state["flush_task"] is not None
    assert not state["flush_task"].done()
    state["flush_task"].cancel()


@pytest.mark.asyncio
async def test_disconnect_waits_for_first_frame_then_clears_committed_stream():
    a = _make_adapter()
    a._mark_disconnected = MagicMock()
    a._chat_kind["chatA"] = ChannelType.Group
    a._http_session = MagicMock()
    a._http_session.close = AsyncMock()
    send_entered = asyncio.Event()
    release_send = asyncio.Event()

    async def blocked_send(*_args, **_kwargs):
        send_entered.set()
        await release_send.wait()
        return SendMessageResult(message_id="server-race")

    with (
        patch("hermes_octo_plugin.adapter.api.send_message", new=blocked_send),
        patch(
            "hermes_octo_plugin.adapter.api.edit_message",
            new=AsyncMock(return_value={"accepted": True}),
        ),
    ):
        send_task = asyncio.create_task(a.send("chatA", "first"))
        await send_entered.wait()
        disconnect_task = asyncio.create_task(a.disconnect())
        await asyncio.sleep(0)
        disconnected_before_send_returned = disconnect_task.done()
        release_send.set()
        send_result, _ = await asyncio.gather(send_task, disconnect_task)

    assert disconnected_before_send_returned is False
    assert send_result.message_id == "server-race"
    assert a._http_session is None
    assert a._active_streams == {}


@pytest.mark.asyncio
async def test_cancelled_disconnect_clears_every_stream_and_watchdog():
    a = _make_adapter()
    a._mark_disconnected = MagicMock()
    a._http_session = MagicMock()
    a._http_session.close = AsyncMock()
    watchdogs = []
    for chat_id in ("group-1", "group-1____thread-1"):
        watchdog = asyncio.create_task(asyncio.Event().wait())
        watchdogs.append(watchdog)
        a._active_streams[chat_id] = {
            "message_id": f"server-{chat_id}",
            "channel_type": ChannelType.Group,
            "segments": ["confirmed"],
            "current_segment": "",
            "flush_task": watchdog,
        }

    close_entered = asyncio.Event()

    async def blocked_close(chat_id: str):
        del chat_id
        close_entered.set()
        await asyncio.Event().wait()

    a._close_active_stream = blocked_close  # type: ignore[method-assign]
    disconnect_task = asyncio.create_task(a.disconnect())
    await close_entered.wait()
    disconnect_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await disconnect_task

    assert a._active_streams == {}
    assert all(task.done() for task in watchdogs)


@pytest.mark.asyncio
async def test_group_scope_eviction_waits_for_inflight_stream_transition():
    a = _make_adapter()
    a._delete_md_from_disk = MagicMock()
    chat_id = "group-1____thread-1"
    state = {
        "message_id": "server-1",
        "channel_type": ChannelType.CommunityTopic,
        "segments": ["old"],
        "current_segment": "",
        "flush_task": None,
    }
    a._active_streams[chat_id] = state
    edit_entered = asyncio.Event()
    release_edit = asyncio.Event()

    async def blocked_edit(*_args, **_kwargs):
        edit_entered.set()
        await release_edit.wait()
        return {"accepted": True}

    with patch("hermes_octo_plugin.adapter.api.edit_message", new=blocked_edit):
        edit_task = asyncio.create_task(
            a.edit_message(chat_id, "server-1", "replacement")
        )
        await edit_entered.wait()
        eviction_result = a._evict_group_scope("group-1")
        eviction_task = (
            asyncio.create_task(eviction_result)
            if asyncio.iscoroutine(eviction_result)
            else None
        )
        try:
            assert a._active_streams.get(chat_id) is state
        finally:
            release_edit.set()
            await edit_task
            if eviction_task is not None:
                await eviction_task

    assert chat_id not in a._active_streams


@pytest.mark.asyncio
async def test_group_scope_eviction_waits_for_inflight_first_frame():
    a = _make_adapter()
    a._delete_md_from_disk = MagicMock()
    chat_id = "group-1____thread-1"
    send_entered = asyncio.Event()
    release_send = asyncio.Event()

    async def blocked_send(*_args, **_kwargs):
        send_entered.set()
        await release_send.wait()
        return SendMessageResult(message_id="server-race")

    with patch("hermes_octo_plugin.adapter.api.send_message", new=blocked_send):
        send_task = asyncio.create_task(a.send(chat_id, "first"))
        await send_entered.wait()
        eviction_task = asyncio.create_task(a._evict_group_scope("group-1"))
        await asyncio.sleep(0)
        eviction_finished_before_send = eviction_task.done()
        release_send.set()
        await asyncio.gather(send_task, eviction_task)

    assert eviction_finished_before_send is False
    assert chat_id not in a._active_streams


def test_idle_stream_lock_is_weakly_reclaimed():
    a = _make_adapter()

    lock = a._stream_lock_for("chat-1")
    assert a._stream_lock_for("chat-1") is lock
    del lock

    assert "chat-1" not in a._stream_locks


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
