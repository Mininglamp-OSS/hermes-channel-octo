"""Durable card-action polling, validation, and dispatch contracts."""

from __future__ import annotations
import asyncio
import json
import logging

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
        self.pending_ack_event_id: int | None = None
        self.saved: list[int] = []
        self.pending_saved: list[int | None] = []
        self.operations = operations

    async def load(self) -> int:
        return self.value

    async def load_pending_ack(self) -> int | None:
        return self.pending_ack_event_id

    async def save(
        self,
        event_id: int,
        *,
        pending_ack_event_id: int | None = None,
    ) -> None:
        self.value = event_id
        self.pending_ack_event_id = pending_ack_event_id
        self.saved.append(event_id)
        self.pending_saved.append(pending_ack_event_id)
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

    await store.save(13, pending_ack_event_id=13)
    reopened = card_events.FileEventCursorStore(owner_id="bot-1", base_dir=tmp_path)
    assert await reopened.load() == 13
    assert await reopened.load_pending_ack() == 13
    assert (
        store.path.read_text(encoding="utf-8")
        == '{"event_id":13,"pending_ack_event_id":13}\n'
    )
    await reopened.save(13)
    assert await reopened.load_pending_ack() is None
    assert store.path.read_text(encoding="utf-8") == '{"event_id":13}\n'

    with pytest.raises(ValueError, match="invalid pending ack"):
        await store.save(13, pending_ack_event_id=14)

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

def test_registry_claims_only_ordinary_interactive_card_edits() -> None:
    registry = card_events.CardSessionRegistry()
    exact = {
        "session_key": "session-1",
        "channel_id": "group-1",
        "channel_type": ChannelType.Group,
        "requester_uid": "user-1",
    }
    registry.register(_session())
    registry.register(_session(message_id="clarify-1", clarify=object()))
    registry.register(_session(message_id="reasoning-1", kind="reasoning"))

    for field, forged in (
        ("session_key", "other-session"),
        ("channel_id", "other-channel"),
        ("channel_type", ChannelType.DM),
        ("requester_uid", "other-user"),
    ):
        assert registry.claim_edit(
            message_id="message-1",
            **{**exact, field: forged},
        ) is None

    assert registry.claim_edit(message_id="clarify-1", **exact) is None
    assert registry.claim_edit(message_id="reasoning-1", **exact) is None
    assert registry.claim_edit(message_id="message-1", **exact) == 1
    registry.release_edit("message-1", 1)
    assert registry.claim_edit(message_id="message-1", **exact) == 2
    registry.complete("message-1", -2)
    assert registry.claim_edit(message_id="message-1", **exact) is None


def test_registry_registration_rejects_invalid_ids_and_unsafe_capacity_eviction() -> None:
    registry = card_events.CardSessionRegistry(max_sessions=1)

    with pytest.raises(ValueError, match="message_id"):
        registry.register(_session(message_id=""))

    registry.register(_session())
    with pytest.raises(ValueError, match="capacity"):
        registry.register(_session(message_id="message-2"))
    assert registry.peek("message-1") == _session()

    assert registry.claim("message-1", 1).status == "claimed"
    registry.complete("message-1", 1)
    registry.register(_session(message_id="message-2"))
    assert registry.peek("message-1") is None
    assert registry.peek("message-2") == _session(message_id="message-2")



def test_registry_reregistration_rejects_pending_message_id_collision() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())

    with pytest.raises(ValueError, match="already active"):
        registry.register(_session(plain="Replacement"))

    assert registry.peek("message-1") == _session()

def test_action_echo_is_escaped_once() -> None:
    frozen = card_events._freeze_action_node(
        {"type": "Input.Text", "id": "note", "label": "Note"},
        {"note": "a[b"},
    )

    assert frozen is not None
    assert frozen["text"] == r"Note: a\[b"



def test_registry_refreshes_only_same_reasoning_session_identity() -> None:
    registry = card_events.CardSessionRegistry()
    original = _session(
        kind="reasoning",
        action_labels={"reasoning_stop": "停止"},
    )
    registry.register(original)

    refreshed = _session(
        kind="reasoning",
        action_labels={"reasoning_retry": "重试"},
    )
    registry.refresh_reasoning(refreshed)
    assert registry.peek("message-1") == refreshed

    with pytest.raises(ValueError, match="identity mismatch"):
        registry.refresh_reasoning(
            _session(
                kind="reasoning",
                binding_id="forged-binding",
                action_labels={"reasoning_retry": "重试"},
            )
        )
    assert registry.peek("message-1") == refreshed

def test_registry_reregistration_rejects_active_message_id_collision() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())
    assert registry.claim("message-1", 99).status == "claimed"

    with pytest.raises(ValueError, match="already active"):
        registry.register(_session(plain="Replacement"))

    assert registry.peek("message-1") == _session()
    assert registry.claim("message-1", 100).status == "duplicate"

def test_default_card_session_ttl_covers_the_default_clarify_window() -> None:
    with patch(
        "hermes_octo_plugin.card_sessions.time.monotonic",
        side_effect=[0.0, 0.0, 3601.0],
    ):
        registry = card_events.CardSessionRegistry()
        registry.register(_session())
        claim = registry.claim("message-1", 1)

    assert claim.status == "claimed"


@pytest.mark.asyncio
async def test_poller_does_not_ack_unowned_card_actions() -> None:
    cursor = _MemoryCursor(10)
    callback = AsyncMock(
        side_effect=["missing", "ignored", "duplicate", "completed", "unsupported"]
    )
    ack = AsyncMock()
    with (
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(
                return_value=[
                    _event(11),
                    _event(12),
                    _event(13),
                    _event(14),
                    _event(15),
                ]
            ),
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

    assert cursor.saved == [11, 12, 13, 13, 14, 14, 15, 15]
    assert cursor.pending_saved == [None, None, 13, None, 14, None, 15, None]
    assert [call.kwargs["event_id"] for call in ack.await_args_list] == [13, 14, 15]


@pytest.mark.asyncio
async def test_poller_logs_each_parse_rejection_once_without_event_payload_values(
    caplog,
) -> None:
    warning_counts_at_cursor_advance: list[int] = []

    class _Cursor(_MemoryCursor):
        async def save(self, event_id, *, pending_ack_event_id=None) -> None:
            warning_counts_at_cursor_advance.append(
                sum(
                    record.getMessage().startswith("Octo card action rejected")
                    for record in caplog.records
                )
            )
            await super().save(
                event_id,
                pending_ack_event_id=pending_ack_event_id,
            )

    cursor = _Cursor(10)
    rejected = _event(11)
    rejected["event_type"] = "unexpected"
    rejected["event_data"]["operator_uid"] = "operator-secret"
    rejected["event_data"]["inputs"] = {"note": "input-secret"}
    with (
        caplog.at_level(logging.WARNING, logger="hermes_octo_plugin.card_events"),
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(return_value=[rejected, rejected]),
        ),
    ):
        poller = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=AsyncMock(),
            wait_seconds=0,
        )
        await poller.initialize()
        await poller.poll_once()

    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Octo card action rejected")
    ] == [
        "Octo card action rejected "
        "event_id=11 message_id=message-1 action_id=approve reason=parse_invalid"
    ]
    assert cursor.saved == [11, 11]
    assert warning_counts_at_cursor_advance == [1, 1]
    assert "operator-secret" not in caplog.text
    assert "input-secret" not in caplog.text


@pytest.mark.asyncio
async def test_poller_bounds_unique_rejection_warnings_per_batch(caplog) -> None:
    cursor = _MemoryCursor(10)
    rejected = []
    for event_id in range(11, 11 + card_events._MAX_REJECTION_LOGS + 1):
        event = _event(event_id)
        event["event_type"] = "unexpected"
        rejected.append(event)
    with (
        caplog.at_level(logging.WARNING, logger="hermes_octo_plugin.card_events"),
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(return_value=rejected),
        ),
    ):
        poller = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=AsyncMock(),
            wait_seconds=0,
        )
        await poller.initialize()
        await poller.poll_once()

    assert sum(
        record.getMessage().startswith("Octo card action rejected")
        for record in caplog.records
    ) == card_events._MAX_REJECTION_LOGS
    assert cursor.saved == list(range(11, 11 + card_events._MAX_REJECTION_LOGS + 1))


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



def test_card_status_rerender_neutralizes_all_model_authored_markdown() -> None:
    session = _session(
        card={
            "type": "AdaptiveCard",
            "body": [
                {
                    "type": "Input.Text",
                    "id": "answer",
                    "label": r"![label](http://10.0.0.5/pixel)",
                },
            ],
            "actions": [],
        },
        action_labels={"submit": r"![action](http://10.0.0.5/pixel)"},
    )
    action = card_events.CardAction(
        event_id=1,
        message_id=session.message_id,
        channel_id=session.channel_id,
        channel_type=session.channel_type,
        operator_uid=session.requester_uid,
        action_id="submit",
        inputs={"answer": r"\[typed](http://10.0.0.5/pixel)"},
        data={"_octo_binding": session.binding_id},
    )

    rendered = card_events.render_card_action_status(session, action, "completed")
    serialized = json.dumps(rendered.card, ensure_ascii=False)

    assert "![label]" not in serialized
    assert "![action]" not in serialized
    assert r"\\\[typed" in serialized

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
        "save:11",
        "dispatch:12",
        "save:12",
        "ack:12",
        "save:12",
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
async def test_poller_backoff_warnings_are_bounded_and_do_not_leak_errors(
    caplog,
) -> None:
    poller = card_events.EventPoller(
        session=object(),
        api_url="https://api.example.invalid",
        bot_token="test-token",
        cursor_store=_MemoryCursor(),
        on_card_action=AsyncMock(),
        interval_seconds=8,
        wait_seconds=0,
    )
    failure = RuntimeError("https://private.example/upload?signature=secret")

    with (
        caplog.at_level(logging.WARNING, logger="hermes_octo_plugin.card_events"),
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(side_effect=failure),
        ),
    ):
        assert [await poller.poll_once() for _ in range(4)] == [8, 16, 30, 30]

    assert [record.getMessage() for record in caplog.records] == [
        "Octo event polling failed (RuntimeError); retrying in 8.0 seconds",
        "Octo event polling failed (RuntimeError); retrying in 16.0 seconds",
        "Octo event polling failed (RuntimeError); retrying in 30.0 seconds",
    ]
    assert "https://private.example/upload?signature=secret" not in caplog.text

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

    assert operations == ["dispatch", "save:11", "ack", "ack", "save:11"]
    assert poller.cursor == 11


@pytest.mark.asyncio
async def test_failed_ack_is_persisted_and_retried_without_redispatch() -> None:
    cursor = _MemoryCursor(10)
    dispatch = AsyncMock(return_value="completed")
    ack = AsyncMock(side_effect=RuntimeError("ack unavailable"))
    fetch = AsyncMock(return_value=[_event(11)])

    with (
        patch.object(card_events.api, "fetch_bot_events", fetch),
        patch.object(card_events.api, "ack_bot_event", ack),
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
        await poller.poll_once()

    assert dispatch.await_count == 1
    assert fetch.await_count == 1
    assert cursor.value == 11
    assert cursor.pending_ack_event_id == 11
    assert ack.await_count == 6

    with (
        patch.object(card_events.api, "fetch_bot_events", AsyncMock(return_value=[])),
        patch.object(card_events.api, "ack_bot_event", AsyncMock()) as recovered_ack,
    ):
        restarted = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=dispatch,
            wait_seconds=0,
        )
        await restarted.initialize()
        await restarted.poll_once()

    recovered_ack.assert_awaited_once()
    assert cursor.pending_ack_event_id is None
    assert dispatch.await_count == 1


@pytest.mark.asyncio
async def test_restart_abandons_stuck_ack_and_resumes_fetching(caplog) -> None:
    operations: list[str] = []
    cursor = _MemoryCursor(11, operations)
    cursor.pending_ack_event_id = 11
    ack = AsyncMock(side_effect=RuntimeError("ack permanently rejected"))

    async def fetch_events(
        *_args,
        since_event_id: int,
        **_kwargs,
    ) -> list[dict]:
        operations.append(f"fetch:{since_event_id}")
        return [_event(11), _event(12)]

    async def dispatch(action) -> str:
        operations.append(f"dispatch:{action.event_id}")
        return "missing"

    fetch = AsyncMock(side_effect=fetch_events)
    with (
        caplog.at_level(logging.ERROR, logger="hermes_octo_plugin.card_events"),
        patch.object(card_events.api, "fetch_bot_events", fetch),
        patch.object(card_events.api, "ack_bot_event", ack),
        patch.object(card_events.asyncio, "sleep", AsyncMock()),
    ):
        restarted = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=dispatch,
            wait_seconds=0,
        )
        await restarted.initialize()
        for _ in range(3):
            await restarted.poll_once()

    assert ack.await_count == 9
    fetch.assert_awaited_once()
    assert operations == [
        "save:11",
        "fetch:11",
        "dispatch:12",
        "save:12",
    ]
    assert cursor.value == 12
    assert cursor.pending_ack_event_id is None
    assert cursor.pending_saved == [None, None]
    assert "abandoning pending ack" in caplog.text


@pytest.mark.asyncio
async def test_ack_abandon_save_failure_preserves_pending_state(caplog) -> None:
    class _FailingClearCursor(_MemoryCursor):
        async def save(
            self,
            event_id: int,
            *,
            pending_ack_event_id: int | None = None,
        ) -> None:
            if event_id == 11 and pending_ack_event_id is None:
                raise RuntimeError("cursor disk unavailable")
            await super().save(
                event_id,
                pending_ack_event_id=pending_ack_event_id,
            )

    cursor = _FailingClearCursor(11)
    cursor.pending_ack_event_id = 11
    ack = AsyncMock(side_effect=RuntimeError("ack permanently rejected"))
    fetch = AsyncMock(return_value=[])

    with (
        caplog.at_level(logging.ERROR, logger="hermes_octo_plugin.card_events"),
        patch.object(card_events.api, "fetch_bot_events", fetch),
        patch.object(card_events.api, "ack_bot_event", ack),
        patch.object(card_events.asyncio, "sleep", AsyncMock()),
    ):
        restarted = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=AsyncMock(),
            wait_seconds=0,
        )
        await restarted.initialize()
        for _ in range(3):
            await restarted.poll_once()

    assert ack.await_count == 9
    assert restarted._pending_ack_event_id == 11
    assert cursor.pending_ack_event_id == 11
    fetch.assert_not_awaited()
    assert "abandoning pending ack" not in caplog.text


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
        registry.discard("message-1")
        registry.register(_session())
        action = card_events.parse_card_action(raw)
        assert action is not None
        assert await card_events.handle_card_action(registry, action, dispatch) == "ignored"
    assert dispatch.await_count == 1


@pytest.mark.asyncio
async def test_registry_accepts_only_bound_dm_channel_views() -> None:
    registry = card_events.CardSessionRegistry()
    session = _session(
        chat_id="user-1",
        channel_id="user-1",
        channel_type=ChannelType.DM,
        action_channel_ids=("user-1", "bot-1"),
    )
    registry.register(session)
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

    registry.register(session)
    forged = card_events.parse_card_action(
        _event(event_id=19, channel_id="attacker", channel_type=1)
    )
    assert forged is not None
    dispatch = AsyncMock(return_value=True)
    assert (
        await card_events.handle_card_action(registry, forged, dispatch)
        == "ignored"
    )
    dispatch.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_ignored_dispatch_restores_a_terminal_retryable_card_state() -> None:
    registry = card_events.CardSessionRegistry()
    registry.register(_session())
    action = card_events.parse_card_action(_event())
    assert action is not None
    update = AsyncMock()

    assert await card_events.handle_card_action(
        registry,
        action,
        AsyncMock(return_value="ignored"),
        update_status=update,
    ) == "ignored"

    assert [call.args[2] for call in update.await_args_list] == [
        "processing",
        "failed",
    ]
    assert update.await_args_list[-1].kwargs == {"transient": False}
    assert registry.claim("message-1", action.event_id).status == "claimed"


@pytest.mark.asyncio
async def test_exhausted_ack_persists_cursor_and_pending_ack() -> None:
    cursor = _MemoryCursor(10)
    ack = AsyncMock(side_effect=RuntimeError("ack down"))
    with (
        patch.object(
            card_events.api,
            "fetch_bot_events",
            AsyncMock(return_value=[_event(11)]),
        ),
        patch.object(card_events.api, "ack_bot_event", ack),
        patch.object(card_events.asyncio, "sleep", AsyncMock()),
    ):
        poller = card_events.EventPoller(
            session=object(),
            api_url="https://api.example.invalid",
            bot_token="test-token",
            cursor_store=cursor,
            on_card_action=AsyncMock(return_value="completed"),
            wait_seconds=0,
        )
        await poller.initialize()
        await poller.poll_once()

    assert ack.await_count == 3
    assert cursor.saved == [11]
    assert cursor.pending_saved == [11]
    assert cursor.pending_ack_event_id == 11
    assert poller.cursor == 11


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
async def test_dispatch_bridge_preserves_a_non_default_session_profile() -> None:
    source = SimpleNamespace(profile="work")
    session = _session(session_key="agent:work:octo:group:group-1:user-1")
    action = card_events.parse_card_action(_event())
    assert action is not None
    adapter = SimpleNamespace(
        config=SimpleNamespace(extra={}),
        _message_handler=object(),
        build_source=MagicMock(return_value=source),
        handle_message=AsyncMock(),
    )

    def profiled_session_key(
        _source,
        group_sessions_per_user: bool = True,
        thread_sessions_per_user: bool = False,
        *,
        profile: str | None = None,
    ) -> str:
        del group_sessions_per_user, thread_sessions_per_user
        return (
            "agent:work:octo:group:group-1:user-1"
            if profile == "work"
            else "agent:main:octo:group:group-1:user-1"
        )

    with patch.object(
        card_events,
        "build_session_key",
        profiled_session_key,
    ):
        dispatched = await card_events.dispatch_card_action_event(
            adapter,
            session,
            action,
        )

    assert dispatched is True
    adapter.handle_message.assert_awaited_once()


@pytest.mark.parametrize(
    ("action_id", "label"),
    [("reasoning_stop", "停止"), ("reasoning_retry", "重试")],
)
@pytest.mark.asyncio
async def test_adapter_consumes_owned_registry_reasoning_control_without_user_turn(
    action_id: str,
    label: str,
) -> None:
    adapter = object.__new__(OctoAdapter)
    adapter._card_sessions = card_events.CardSessionRegistry()
    adapter.handle_message = AsyncMock()
    reasoning_id = "session-1:turn-1:1"
    adapter._card_sessions.register(
        _session(
            binding_id=reasoning_id,
            action_labels={action_id: label},
            input_ids=(),
            kind="reasoning",
        )
    )
    action = card_events.parse_card_action(
        _event(
            action_id=action_id,
            inputs={},
            data={"reasoningId": reasoning_id},
        )
    )
    assert action is not None

    assert await adapter._handle_card_action_event(action) == "unsupported"
    assert await adapter._handle_card_action_event(action) == "duplicate"
    adapter.handle_message.assert_not_awaited()

@pytest.mark.asyncio
async def test_registry_reasoning_control_accepts_the_owned_dm_channel_alias() -> None:
    adapter = object.__new__(OctoAdapter)
    adapter._card_sessions = card_events.CardSessionRegistry()
    adapter.handle_message = AsyncMock()
    reasoning_id = "session-1:turn-1:1"
    adapter._card_sessions.register(
        _session(
            binding_id=reasoning_id,
            channel_id="user-1",
            channel_type=ChannelType.DM,
            action_channel_ids=("user-1", "bot-1"),
            action_labels={"reasoning_stop": "停止"},
            input_ids=(),
            kind="reasoning",
        )
    )
    action = card_events.parse_card_action(
        _event(
            channel_id="bot-1",
            channel_type=1,
            action_id="reasoning_stop",
            inputs={},
            data={"reasoningId": reasoning_id},
        )
    )
    assert action is not None

    assert await adapter._handle_card_action_event(action) == "unsupported"
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
