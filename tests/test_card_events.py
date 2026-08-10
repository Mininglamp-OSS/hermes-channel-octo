"""Durable card-action polling, validation, and dispatch contracts."""

from __future__ import annotations
import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key

from hermes_octo_plugin import card_events
from hermes_octo_plugin.adapter import OctoAdapter
from hermes_octo_plugin.types import ChannelType


def _event(event_id: int = 11, **overrides):
    data = {
        "message_id": "message-1",
        "channel_id": "group-1",
        "channel_type": 2,
        "action_id": "approve",
        "operator_uid": "user-1",
        "inputs": {"note": "ok", "count": 2, "enabled": False},
        "data": {"_octo_binding": "binding-1", "decision": "approve"},
    }
    data.update(overrides)
    return {"event_id": event_id, "event_type": "card_action", "event_data": data}


class _MemoryCursor:
    def __init__(self, initial: int = 0, operations: list[str] | None = None) -> None:
        self.value = initial
        self.saved: list[int] = []
        self.operations = operations

    async def load(self) -> int:
        return self.value

    async def save(self, event_id: int) -> None:
        self.value = event_id
        self.saved.append(event_id)
        if self.operations is not None:
            self.operations.append(f"save:{event_id}")


def _session(**overrides):
    values = {
        "message_id": "message-1",
        "binding_id": "binding-1",
        "session_key": "session-1",
        "chat_id": "group-1",
        "channel_id": "group-1",
        "channel_type": ChannelType.Group,
        "requester_uid": "user-1",
        "card": {"type": "AdaptiveCard", "version": "1.5", "body": []},
        "plain": "Approval",
        "action_labels": {"approve": "Approve"},
        "input_ids": ("note", "count", "enabled"),
        "max_input_text_bytes": 4096,
        "max_inputs_bytes": 16384,
    }
    values.update(overrides)
    return card_events.CardSession(**values)


@pytest.mark.asyncio
async def test_file_cursor_store_is_atomic_durable_and_monotonic(tmp_path) -> None:
    store = card_events.FileEventCursorStore(owner_id="bot-1", base_dir=tmp_path)

    assert await store.load() == 0
    await store.save(12)
    assert await store.load() == 12
    assert store.path.read_text(encoding="utf-8") == '{"event_id":12}\n'
    assert not list(store.path.parent.glob("*.tmp"))

    with pytest.raises(ValueError, match="invalid event cursor"):
        await store.save(-1)
    with pytest.raises(ValueError, match="move backwards"):
        await store.save(11)

    case_sensitive = card_events.FileEventCursorStore(
        owner_id="Bot-Case",
        base_dir=tmp_path,
    )
    assert case_sensitive.path.parent.name == "Bot-Case"

    opaque_id = "机器人/../bot id:α"
    opaque = card_events.FileEventCursorStore(
        owner_id=opaque_id,
        base_dir=tmp_path,
    )
    same_opaque = card_events.FileEventCursorStore(
        owner_id=opaque_id,
        base_dir=tmp_path,
    )
    assert opaque.path.parent.parent == tmp_path
    assert opaque.path.parent.name == same_opaque.path.parent.name
    assert opaque.path.parent.name != opaque_id

    with pytest.raises(ValueError, match="invalid Octo owner id"):
        card_events.FileEventCursorStore(owner_id="   ", base_dir=tmp_path)


@pytest.mark.parametrize("bad_id", [True, -1, 1.5, 2**53])
def test_card_action_parser_rejects_unsafe_event_ids(bad_id) -> None:
    assert card_events.parse_card_action(_event(bad_id)) is None


def test_card_action_parser_normalizes_scalar_inputs_without_trusting_objects() -> None:
    raw = _event(
        inputs={
            "note": "ok",
            "count": 2,
            "enabled": False,
            "nested": {"bad": True},
            "list": ["bad"],
        }
    )
    action = card_events.parse_card_action(raw)

    assert action is not None
    assert action.inputs == {"note": "ok", "count": "2", "enabled": "false"}


@pytest.mark.parametrize(
    "raw",
    [
        _event(channel_type=2.9),
        _event(inputs={"note": 10**1000}),
        _event(inputs={"note": "\ud800"}),
        _event(inputs={f"k{index}": "v" for index in range(129)}),
        _event(data={f"k{index}": "v" for index in range(129)}),
    ],
)
def test_card_action_parser_is_total_and_rejects_oversized_or_noncanonical_values(
    raw,
) -> None:
    assert card_events.parse_card_action(raw) is None

def test_registry_claims_card_edits_only_for_the_exact_live_session() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())
    exact = {
        "message_id": "message-1",
        "card_seq": 2,
        "session_key": "session-1",
        "channel_id": "group-1",
        "channel_type": ChannelType.Group,
        "requester_uid": "user-1",
    }

    for field, forged in (
        ("session_key", "other-session"),
        ("channel_id", "other-channel"),
        ("channel_type", ChannelType.DM),
        ("requester_uid", "other-user"),
    ):
        assert registry.claim_edit(**{**exact, field: forged}) is False

    assert registry.claim_edit(**exact) is True
    assert registry.claim("message-1", 99).status == "duplicate"
    registry.complete("message-1", -2)
    assert registry.claim_edit(**exact) is False


@pytest.mark.asyncio
async def test_poller_does_not_ack_unowned_card_actions() -> None:
    cursor = _MemoryCursor(10)
    callback = AsyncMock(side_effect=["missing", "ignored", "duplicate", "completed"])
    ack = AsyncMock()
    with (
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(return_value=[_event(11), _event(12), _event(13), _event(14)]),
        ),
        patch.object(card_events.api, "ack_bot_event", ack),
    ):
        poller = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=callback,
            wait_seconds=0,
        )
        await poller.initialize()
        await poller.poll_once()

    assert cursor.saved == [11, 12, 13, 14]
    assert [call.kwargs["event_id"] for call in ack.await_args_list] == [13, 14]


@pytest.mark.asyncio
async def test_duplicate_is_owned_only_after_exact_action_validation() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())
    valid = card_events.parse_card_action(_event())
    forged = card_events.parse_card_action(_event(12, operator_uid="other-user"))
    assert valid is not None and forged is not None
    assert await card_events.handle_card_action(
        registry, valid, AsyncMock(return_value=True)
    ) == "completed"

    assert await card_events.handle_card_action(
        registry, forged, AsyncMock(return_value=True)
    ) == "ignored"



@pytest.mark.asyncio
async def test_poller_orders_dispatch_save_ack_and_preserves_cursor_on_failure() -> None:
    operations: list[str] = []
    cursor = _MemoryCursor(10, operations)

    async def dispatch(action) -> str:
        operations.append(f"dispatch:{action.event_id}")
        return "completed"

    async def ack(*_args, event_id: int, **_kwargs) -> None:
        operations.append(f"ack:{event_id}")

    with (
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(return_value=[_event(12), _event(11)]),
        ),
        patch.object(card_events.api, "ack_bot_event", side_effect=ack),
    ):
        poller = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=dispatch,
            interval_seconds=2,
            wait_seconds=0,
        )
        await poller.initialize()
        delay = await poller.poll_once()

    assert operations == [
        "dispatch:11",
        "save:11",
        "ack:11",
        "dispatch:12",
        "save:12",
        "ack:12",
    ]
    assert poller.cursor == 12
    assert delay == 2

    failed_cursor = _MemoryCursor(20)
    with (
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(return_value=[_event(21)]),
        ),
        patch.object(card_events.api, "ack_bot_event", AsyncMock()) as failed_ack,
    ):
        failed = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=failed_cursor,
            on_card_action=AsyncMock(side_effect=RuntimeError("dispatch failed")),
            interval_seconds=2,
            wait_seconds=25,
        )
        await failed.initialize()
        assert await failed.poll_once() == 2

    assert failed_cursor.saved == []
    failed_ack.assert_not_awaited()
    assert failed.cursor == 20


@pytest.mark.asyncio
async def test_long_poll_immediate_empty_and_errors_are_paced_not_hot_loop() -> None:
    poller = card_events.EventPoller(
        session=object(),
        api_url="https://api.example.invalid",
        bot_token="test-token",
        cursor_store=_MemoryCursor(),
        on_card_action=AsyncMock(),
        interval_seconds=2,
        wait_seconds=25,
    )
    await poller.initialize()

    with patch.object(card_events.api, "fetch_bot_events", AsyncMock(return_value=[])):
        assert await poller.poll_once() == 2
    with patch.object(
        card_events.api,
        "fetch_bot_events",
        AsyncMock(side_effect=RuntimeError("down")),
    ):
        assert await poller.poll_once() == 2
        assert await poller.poll_once() == 4



@pytest.mark.asyncio
async def test_backoff_saturates_without_large_exponent_overflow() -> None:
    poller = card_events.EventPoller(
        session=object(),
        api_url="https://api.example.invalid",
        bot_token="test-token",
        cursor_store=_MemoryCursor(),
        on_card_action=AsyncMock(),
        interval_seconds=2,
        wait_seconds=0,
    )
    poller._consecutive_errors = 10_000
    with patch.object(
        card_events.api,
        "fetch_bot_events",
        AsyncMock(side_effect=RuntimeError("still down")),
    ):
        assert await poller.poll_once() == card_events.MAX_EVENT_BACKOFF_SECONDS


@pytest.mark.asyncio
async def test_ack_is_retried_after_durable_cursor_save() -> None:
    operations: list[str] = []
    cursor = _MemoryCursor(10, operations)

    async def dispatch(_action) -> str:
        operations.append("dispatch")
        return "completed"

    async def ack(*_args, **_kwargs) -> None:
        operations.append("ack")
        if operations.count("ack") == 1:
            raise RuntimeError("transient")

    with (
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(return_value=[_event(11)]),
        ),
        patch.object(card_events.api, "ack_bot_event", side_effect=ack),
        patch.object(card_events.asyncio, "sleep", AsyncMock()),
    ):
        poller = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=dispatch,
            wait_seconds=0,
        )
        await poller.initialize()
        await poller.poll_once()

    assert operations == ["dispatch", "save:11", "ack", "ack"]
    assert poller.cursor == 11

@pytest.mark.asyncio
async def test_registry_blocks_replay_binding_channel_operator_and_input_mismatch() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())
    dispatch = AsyncMock(return_value=True)

    valid = card_events.parse_card_action(_event())
    assert valid is not None
    assert await card_events.handle_card_action(registry, valid, dispatch) == "completed"
    assert await card_events.handle_card_action(registry, valid, dispatch) == "duplicate"
    assert dispatch.await_count == 1

    for raw in (
        _event(20, data={"_octo_binding": "forged"}),
        _event(21, channel_id="other"),
        _event(22, operator_uid="other"),
        _event(23, action_id="forged"),
        _event(24, inputs={"unknown": "value"}),
    ):
        registry.register(_session())
        action = card_events.parse_card_action(raw)
        assert action is not None
        assert await card_events.handle_card_action(registry, action, dispatch) == "ignored"
    assert dispatch.await_count == 1


@pytest.mark.asyncio
async def test_registry_accepts_dm_action_with_server_bot_channel_view() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(
        _session(
            chat_id="user-1",
            channel_id="user-1",
            channel_type=ChannelType.DM,
        )
    )
    action = card_events.parse_card_action(
        _event(channel_id="bot-1", channel_type=1)
    )
    assert action is not None

    assert (
        await card_events.handle_card_action(
            registry,
            action,
            AsyncMock(return_value=True),
        )
        == "completed"
    )


@pytest.mark.asyncio
async def test_dispatch_failures_retry_then_dead_letter() -> None:
    registry = card_events.CardSessionRegistry(max_dispatch_attempts=3)
    registry.register(_session())
    action = card_events.parse_card_action(_event())
    assert action is not None
    dispatch = AsyncMock(side_effect=RuntimeError("agent down"))

    with pytest.raises(RuntimeError, match="agent down"):
        await card_events.handle_card_action(registry, action, dispatch)
    with pytest.raises(RuntimeError, match="agent down"):
        await card_events.handle_card_action(registry, action, dispatch)
    assert await card_events.handle_card_action(registry, action, dispatch) == "dead_letter"
    assert await card_events.handle_card_action(registry, action, dispatch) == "duplicate"
    assert dispatch.await_count == 3

@pytest.mark.asyncio
async def test_action_status_edits_are_processing_then_terminal() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())
    action = card_events.parse_card_action(_event())
    assert action is not None
    update = AsyncMock()

    assert await card_events.handle_card_action(
        registry,
        action,
        AsyncMock(return_value=True),
        update_status=update,
    ) == "completed"

    assert [call.args[2] for call in update.await_args_list] == [
        "processing",
        "completed",
    ]
    assert update.await_args_list[0].kwargs == {"transient": True}
    assert update.await_args_list[1].kwargs == {"transient": False}


def test_action_status_renderer_freezes_inputs_and_removes_actions() -> None:
    rendered = card_events.render_card_action_status(
        _session(
            card={
                "type": "AdaptiveCard",
                "version": "1.5",
                "body": [
                    {"type": "Input.Text", "id": "note", "label": "Note"},
                    {"type": "TextBlock", "text": "Review", "wrap": True},
                ],
                "actions": [{"type": "Action.Submit", "title": "Approve"}],
            }
        ),
        _event_action := card_events.parse_card_action(
            _event(inputs={"note": "[click](https://evil.example/path?token=x)"})
        ),
        "completed",
    )
    assert _event_action is not None
    assert "actions" not in rendered.card
    assert all(not node["type"].startswith("Input.") for node in rendered.card["body"])
    visible = "\n".join(node.get("text", "") for node in rendered.card["body"])
    assert "token=x" in visible
    assert "https://evil.example/path" in visible
    assert "Completed" in rendered.plain





@pytest.mark.asyncio
async def test_cancelled_dispatch_releases_the_card_claim_for_replay() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())
    action = card_events.parse_card_action(_event())
    assert action is not None

    with pytest.raises(asyncio.CancelledError):
        await card_events.handle_card_action(
            registry,
            action,
            AsyncMock(side_effect=asyncio.CancelledError),
        )

    assert registry.claim("message-1", action.event_id).status == "claimed"


@pytest.mark.asyncio
async def test_cancelled_processing_status_edit_releases_claim_for_replay() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())
    action = card_events.parse_card_action(_event())
    assert action is not None

    with pytest.raises(asyncio.CancelledError):
        await card_events.handle_card_action(
            registry,
            action,
            AsyncMock(return_value=True),
            update_status=AsyncMock(side_effect=asyncio.CancelledError),
        )

    assert registry.claim("message-1", action.event_id).status == "claimed"

@pytest.mark.asyncio
async def test_dispatch_bridge_uses_public_handle_message_and_exact_session() -> None:
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="group-1",
        chat_type="group",
        user_id="user-1",
        user_name="user-1",
    )
    session_key = build_session_key(source)
    session = _session(session_key=session_key)
    action = card_events.parse_card_action(_event())
    assert action is not None
    adapter = SimpleNamespace(
        config=SimpleNamespace(extra={}),
        _message_handler=object(),
        build_source=MagicMock(return_value=source),
        handle_message=AsyncMock(),
    )

    assert await card_events.dispatch_card_action_event(adapter, session, action) is True
    adapter.handle_message.assert_awaited_once()
    dispatched = adapter.handle_message.await_args.args[0]
    assert dispatched.message_id == "card_action:11"
    assert dispatched.source == source
    assert dispatched.text == (
        '[Octo card action]\naction_id="approve"\n'
        'inputs={"count":"2","enabled":"false","note":"ok"}\n'
        'data={"decision":"approve"}'
    )

    adapter.handle_message.reset_mock()
    mismatched = _session(session_key="another-session")
    assert await card_events.dispatch_card_action_event(adapter, mismatched, action) is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_adapter_owns_poller_registry_and_card_action_dispatch() -> None:
    adapter = object.__new__(OctoAdapter)
    adapter._http_session = object()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._registration = SimpleNamespace(owner_uid="owner-1", robot_id="bot-1")
    adapter._event_task = None
    adapter._event_poller = None
    adapter._card_sessions = card_events.CardSessionRegistry()
    adapter._disconnecting = False
    adapter._event_poll_interval_s = 3.5
    adapter._event_poll_wait_s = 12
    adapter._event_poll_limit = 80
    task = MagicMock()
    task.done.return_value = False
    poller = MagicMock()
    poller.start.return_value = task

    with (
        patch.object(card_events, "FileEventCursorStore") as cursor_store,
        patch.object(card_events, "EventPoller", return_value=poller) as poller_type,
    ):
        adapter._start_card_event_poller()

    cursor_store.assert_called_once_with(owner_id="bot-1")
    assert poller_type.call_args.kwargs["session"] is adapter._http_session
    assert poller_type.call_args.kwargs["on_card_action"] == adapter._handle_card_action_event
    assert poller_type.call_args.kwargs["interval_seconds"] == 3.5
    assert poller_type.call_args.kwargs["wait_seconds"] == 12
    assert poller_type.call_args.kwargs["limit"] == 80
    assert adapter._event_task is task

    registered = _session()
    adapter._register_card_session(registered)
    action = card_events.parse_card_action(_event())
    assert action is not None
    with patch.object(
        card_events,
        "handle_card_action",
        AsyncMock(return_value="completed"),
    ) as handle:
        await adapter._handle_card_action_event(action)
    assert handle.await_args.args[0] is adapter._card_sessions
    assert handle.await_args.args[1] is action

    adapter._stop_card_event_poller()
    poller.stop.assert_called_once()
    assert adapter._event_task is None
    assert adapter._event_poller is None
    assert adapter._card_sessions.claim("message-1", 99).status == "claimed"
