"""Hermes 0.20 native clarify delivery and resolution contracts."""

from __future__ import annotations
import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from tools import clarify_gateway

from hermes_octo_plugin import adapter as adapter_module
from hermes_octo_plugin import api, card_events, card_progress, card_tools
from hermes_octo_plugin.adapter import OctoAdapter
from hermes_octo_plugin.card_tools import TrustedOctoRoute
from hermes_octo_plugin.types import CardProfileManifest, ChannelType, SendMessageResult
from tests.conftest import make_bare_adapter


_ROUTE = TrustedOctoRoute(
    channel_id="group-1",
    chat_id="group-1",
    channel_type=ChannelType.Group,
    requester_uid="user-1",
    session_key="octo:group:group-1:user-1",
)
_MANIFEST = CardProfileManifest(
    available=True,
    enabled=True,
    profiles=("octo/v1", "octo/v2"),
    card_version="1.5",
    elements=("TextBlock", "Container", "ActionSet", "RichTextBlock"),
    inputs=("Input.ChoiceSet",),
    actions=("Action.Submit",),
    limits={"max_actions": 8, "max_inputs": 4},
)


def _bare_clarify_adapter(*, native: bool) -> OctoAdapter:
    adapter = make_bare_adapter()
    adapter._native_clarify_enabled = native
    adapter._http_session = object()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._on_behalf_of = ""
    return adapter

def _card_nodes(value: object, node_type: str) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    if isinstance(value, dict):
        if value.get("type") == node_type:
            matches.append(value)
        for child in value.values():
            matches.extend(_card_nodes(child, node_type))
    elif isinstance(value, list):
        for child in value:
            matches.extend(_card_nodes(child, node_type))
    return matches


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.14.9", False),
        ("0.19.7", False),
        ("0.20.0", True),
        ("0.20.4", True),
        ("0.21.0", True),
        ("1.0.0", True),
        ("not-a-version", False),
        ("0.21.0rc1", True),
    ],
)
def test_constructor_enables_native_clarify_from_hermes_020(
    version: str,
    expected: bool,
) -> None:
    with patch.object(adapter_module, "_package_version", return_value=version):
        adapter = OctoAdapter(SimpleNamespace(extra={}))

    assert adapter._native_clarify_enabled is expected


@pytest.mark.asyncio
async def test_legacy_clarify_versions_delegate_to_base_text_fallback() -> None:
    adapter = _bare_clarify_adapter(native=False)
    expected = SendResult(success=True, message_id="text-1")
    fallback = AsyncMock(return_value=expected)

    with patch.object(BasePlatformAdapter, "send_clarify", fallback):
        result = await OctoAdapter.send_clarify(
            adapter,
            "group-1",
            "Which option?",
            ["A", "B"],
            clarify_id="clarify-legacy",
            session_key=_ROUTE.session_key,
        )

    assert result is expected
    fallback.assert_awaited_once_with(
        "group-1",
        "Which option?",
        ["A", "B"],
        clarify_id="clarify-legacy",
        session_key=_ROUTE.session_key,
        metadata=None,
    )


@pytest.mark.asyncio
async def test_native_clarify_waits_for_scheduled_progress_card() -> None:
    adapter = _bare_clarify_adapter(native=True)
    adapter._gateway_loop = asyncio.get_running_loop()
    events: list[str] = []
    progress_sent = asyncio.Event()

    async def send_progress(**_kwargs) -> None:
        await asyncio.sleep(0)
        events.append("progress")
        progress_sent.set()

    async def send_clarify_card(*_args, **_kwargs) -> SendMessageResult:
        assert progress_sent.is_set()
        events.append("clarify")
        return SendMessageResult(message_id="clarify-message")

    clarify_id = "clarify-ordered"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False
    wait_for_progress = AsyncMock(side_effect=send_progress)
    try:
        with (
            patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
            patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
            patch.object(api, "send_card_message", side_effect=send_clarify_card),
            patch.object(
                card_progress,
                "wait_for_initial_delivery",
                wait_for_progress,
            ),
        ):
            result = await OctoAdapter.send_clarify(
                adapter,
                _ROUTE.chat_id,
                "Which option?",
                ["A", "B"],
                clarify_id=clarify_id,
                session_key=_ROUTE.session_key,
            )
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert result.success is True
    assert events == ["progress", "clarify"]
    wait_for_progress.assert_awaited_once_with(
        adapter=adapter,
        session_key=_ROUTE.session_key,
    )

@pytest.mark.asyncio
async def test_hermes_020_single_choice_clarify_sends_bound_type17_card() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-native-single"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False
    send_card = AsyncMock(
        return_value=SendMessageResult(
            message_id="card-message-1",
            message_seq=7,
            client_msg_no="client-1",
        )
    )
    fallback = AsyncMock(return_value=SendResult(success=False))
    try:
        with (
            patch.object(BasePlatformAdapter, "send_clarify", fallback),
            patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
            patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
            patch.object(api, "send_card_message", send_card),
        ):
            result = await OctoAdapter.send_clarify(
                adapter,
                "group-1",
                "Which option?",
                ["A", "B"],
                clarify_id=clarify_id,
                session_key=_ROUTE.session_key,
            )
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert result.success is True
    assert result.message_id == "card-message-1"
    fallback.assert_not_awaited()
    assert send_card.await_count == 1
    kwargs = send_card.await_args.kwargs
    assert kwargs["client_msg_no"]
    card = kwargs["card"]
    actions = _card_nodes(card, "Action.Submit")
    assert [action["title"] for action in actions] == ["A", "B", "其他"]
    assert all("choice" not in action["data"] for action in actions)
    assert all("clarify_id" not in action["data"] for action in actions)
    assert all("session_key" not in action["data"] for action in actions)
    assert all(action["data"]["_octo_binding"] for action in actions)

    claimed = adapter._card_sessions.claim("card-message-1", 11)
    assert claimed.status == "claimed"
    assert claimed.session is not None
    assert claimed.session.clarify is not None
    assert claimed.session.clarify.clarify_id == clarify_id
    assert claimed.session.clarify.action_choices == (
        ("clarify_choice_0", "A"),
        ("clarify_choice_1", "B"),
    )


@pytest.mark.asyncio
async def test_hermes_020_multi_select_clarify_uses_choiceset_contract() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-native-multi"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose several",
        ["A", "B", "C"],
    )
    entry.multi_select = True
    send_card = AsyncMock(
        return_value=SendMessageResult(message_id="card-message-2")
    )
    try:
        with (
            patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
            patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
            patch.object(api, "send_card_message", send_card),
        ):
            result = await OctoAdapter.send_clarify(
                adapter,
                "group-1",
                "Choose several",
                ["A", "B", "C"],
                clarify_id=clarify_id,
                session_key=_ROUTE.session_key,
            )
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert result.success is True
    card = send_card.await_args.kwargs["card"]
    choice_sets = _card_nodes(card, "Input.ChoiceSet")
    assert len(choice_sets) == 1
    assert choice_sets[0]["isMultiSelect"] is True
    assert [choice["value"] for choice in choice_sets[0]["choices"]] == [
        "clarify_choice_0",
        "clarify_choice_1",
        "clarify_choice_2",
    ]


def test_multi_clarify_completed_card_shows_canonical_choices() -> None:
    session = _clarify_session(
        clarify_id="clarify-result-multi",
        multi_select=True,
    )
    action = _clarify_action(
        "clarify_confirm",
        inputs={"clarify_choices": "clarify_choice_2,clarify_choice_0"},
    )

    rendered = card_events.render_card_action_status(session, action, "completed")

    visible = "\n".join(
        node.get("text", "")
        for node in _card_nodes(rendered.card, "TextBlock")
    )
    assert "需要确认" in visible
    assert "Choose several" in visible
    assert "已选择" in visible
    assert "A、C" in visible
    assert "已提交" in visible
    assert "clarify_choice_" not in visible
    assert _ROUTE.requester_uid not in visible
    assert "clarify_choice_" not in rendered.plain
    assert _ROUTE.requester_uid not in rendered.plain
    assert not _card_nodes(rendered.card, "Input.ChoiceSet")
    assert not _card_nodes(rendered.card, "Action.Submit")


def test_single_clarify_completed_card_shows_canonical_choice() -> None:
    rendered = card_events.render_card_action_status(
        _clarify_session(clarify_id="clarify-result-single"),
        _clarify_action("clarify_choice_1"),
        "completed",
    )

    assert "已选择\nB" in rendered.plain
    assert rendered.plain.endswith("已提交")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("processing", "正在提交…"),
        ("awaiting_text", "请直接发送文字回复"),
        ("expired", "该确认已失效或已处理"),
        ("failed", "提交失败，请重试"),
    ],
)
def test_clarify_card_uses_localized_status_without_internal_identity(
    status: str,
    expected: str,
) -> None:
    rendered = card_events.render_card_action_status(
        _clarify_session(clarify_id=f"clarify-result-{status}"),
        _clarify_action("clarify_other"),
        status,
    )

    assert expected in rendered.plain
    assert _ROUTE.requester_uid not in rendered.plain
    assert "已选择" not in rendered.plain

def _clarify_session(
    *,
    clarify_id: str,
    multi_select: bool = False,
) -> card_events.CardSession:
    clarify = card_events.ClarifySession(
        clarify_id=clarify_id,
        multi_select=multi_select,
        question="Choose several" if multi_select else "Choose",
        choices=("A", "B", "C"),
        action_choices=(
            ("clarify_choice_0", "A"),
            ("clarify_choice_1", "B"),
            ("clarify_choice_2", "C"),
        ),
        input_id="clarify_choices" if multi_select else None,
        confirm_action_id="clarify_confirm" if multi_select else None,
        other_action_id="clarify_other",
    )
    return card_events.CardSession(
        message_id="card-message",
        binding_id="binding-1",
        session_key=_ROUTE.session_key,
        chat_id=_ROUTE.chat_id,
        channel_id=_ROUTE.channel_id,
        channel_type=_ROUTE.channel_type,
        requester_uid=_ROUTE.requester_uid,
        card={"type": "AdaptiveCard", "version": "1.5", "body": []},
        plain="Clarify",
        action_labels={
            "clarify_choice_0": "A",
            "clarify_choice_1": "B",
            "clarify_choice_2": "C",
            "clarify_confirm": "提交",
            "clarify_other": "其他",
        },
        input_ids=("clarify_choices",) if multi_select else (),
        clarify=clarify,
    )


def _clarify_action(
    action_id: str,
    *,
    inputs: dict[str, str] | None = None,
    event_id: int = 17,
) -> card_events.CardAction:
    return card_events.CardAction(
        event_id=event_id,
        message_id="card-message",
        channel_id=_ROUTE.channel_id,
        channel_type=_ROUTE.channel_type,
        action_id=action_id,
        inputs=inputs or {},
        operator_uid=_ROUTE.requester_uid,
        data={"_octo_binding": "binding-1"},
    )


@pytest.mark.asyncio
async def test_single_clarify_action_resolves_gateway_primitive_without_message_turn() -> None:
    clarify_id = "clarify-resolve-single"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose",
        ["A", "B", "C"],
    )
    try:
        status = await card_events.dispatch_clarify_action(
            _clarify_session(clarify_id=clarify_id),
            _clarify_action("clarify_choice_1"),
        )
        assert status == "completed"
        assert entry.response == "B"
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)


@pytest.mark.asyncio
async def test_multi_clarify_action_resolves_canonical_json_in_choice_order() -> None:
    clarify_id = "clarify-resolve-multi"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose several",
        ["A", "B", "C"],
    )
    entry.multi_select = True
    try:
        status = await card_events.dispatch_clarify_action(
            _clarify_session(clarify_id=clarify_id, multi_select=True),
            _clarify_action(
                "clarify_confirm",
                inputs={"clarify_choices": "clarify_choice_2,clarify_choice_0"},
            ),
        )
        assert status == "completed"
        assert entry.response == '[\"A\",\"C\"]'
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)


@pytest.mark.asyncio
async def test_invalid_multi_submit_releases_card_for_a_later_valid_action() -> None:
    registry = card_events.CardSessionRegistry()
    clarify_id = "clarify-invalid-multi"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose several",
        ["A", "B", "C"],
    )
    entry.multi_select = True
    registry.register(
        _clarify_session(clarify_id=clarify_id, multi_select=True)
    )
    try:
        invalid = await card_events.handle_card_action(
            registry,
            _clarify_action(
                "clarify_confirm",
                inputs={"clarify_choices": ""},
            ),
            lambda session, claimed: card_events.dispatch_clarify_action(
                session,
                claimed,
            ),
        )
        valid = await card_events.handle_card_action(
            registry,
            _clarify_action(
                "clarify_confirm",
                inputs={"clarify_choices": "clarify_choice_1"},
                event_id=18,
            ),
            lambda session, claimed: card_events.dispatch_clarify_action(
                session,
                claimed,
            ),
        )
        response = entry.response
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert invalid == "ignored"
    assert valid == "completed"
    assert response == '[\"B\"]'


@pytest.mark.asyncio
async def test_clarify_other_switches_same_request_to_text_capture() -> None:
    clarify_id = "clarify-other"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose",
        ["A", "B", "C"],
    )
    entry.multi_select = False
    try:
        status = await card_events.dispatch_clarify_action(
            _clarify_session(clarify_id=clarify_id),
            _clarify_action("clarify_other"),
        )
        assert status == "awaiting_text"
        assert entry.awaiting_text is True
        assert entry.response is None
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)


@pytest.mark.asyncio
async def test_stale_clarify_click_is_consumed_as_expired() -> None:
    status = await card_events.dispatch_clarify_action(
        _clarify_session(clarify_id="already-gone"),
        _clarify_action("clarify_choice_0"),
    )
    assert status == "expired"



@pytest.mark.asyncio
async def test_reused_clarify_id_cannot_resolve_a_replacement_entry() -> None:
    clarify_id = "clarify-reused-id"
    clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose",
        ["A", "B", "C"],
    )
    clarify_gateway.clear_session(_ROUTE.session_key)
    replacement_session = f"{_ROUTE.session_key}:replacement"
    replacement = clarify_gateway.register(
        clarify_id,
        replacement_session,
        "Different question",
        ["X", "Y"],
    )
    replacement.multi_select = False
    try:
        status = await card_events.dispatch_clarify_action(
            _clarify_session(clarify_id=clarify_id),
            _clarify_action("clarify_choice_0"),
        )
        response = replacement.response
    finally:
        clarify_gateway.clear_session(replacement_session)

    assert status == "expired"
    assert response is None


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError(),
        api.OctoApiError("/v1/bot/sendMessage", status=409),
    ],
    ids=["timeout", "conflict"],
)
@pytest.mark.asyncio
async def test_ambiguous_card_send_failure_retries_once_without_text_fallback(
    failure: BaseException,
) -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-ambiguous-send"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False
    send_card = AsyncMock(side_effect=[failure, failure])
    fallback = AsyncMock(return_value=SendResult(success=True, message_id="text-1"))
    try:
        with (
            patch.object(BasePlatformAdapter, "send_clarify", fallback),
            patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
            patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
            patch.object(api, "send_card_message", send_card),
        ):
            result = await OctoAdapter.send_clarify(
                adapter,
                _ROUTE.chat_id,
                "Which option?",
                ["A", "B"],
                clarify_id=clarify_id,
                session_key=_ROUTE.session_key,
            )
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert result.success is False
    assert result.retryable is True
    fallback.assert_not_awaited()
    assert send_card.await_count == 2
    first = send_card.await_args_list[0].kwargs["client_msg_no"]
    second = send_card.await_args_list[1].kwargs["client_msg_no"]
    assert first == second


@pytest.mark.asyncio
async def test_definitive_card_rejection_uses_base_text_fallback() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-rejected-send"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False
    expected = SendResult(success=True, message_id="text-fallback")
    fallback = AsyncMock(return_value=expected)
    send_card = AsyncMock(
        side_effect=api.OctoApiError(
            "/v1/bot/sendMessage",
            status=400,
        )
    )
    try:
        with (
            patch.object(BasePlatformAdapter, "send_clarify", fallback),
            patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
            patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
            patch.object(api, "send_card_message", send_card),
        ):
            result = await OctoAdapter.send_clarify(
                adapter,
                _ROUTE.chat_id,
                "Which option?",
                ["A", "B"],
                clarify_id=clarify_id,
                session_key=_ROUTE.session_key,
            )
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert result is expected
    assert send_card.await_count == 1
    fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleared_clarify_during_post_is_not_registered_as_live_card() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-cleared-during-post"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False

    async def send_then_clear(*_args, **_kwargs) -> SendMessageResult:
        clarify_gateway.clear_session(_ROUTE.session_key)
        return SendMessageResult(message_id="stale-card-message")

    with (
        patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
        patch.object(api, "send_card_message", side_effect=send_then_clear),
    ):
        result = await OctoAdapter.send_clarify(
            adapter,
            _ROUTE.chat_id,
            "Which option?",
            ["A", "B"],
            clarify_id=clarify_id,
            session_key=_ROUTE.session_key,
        )

    assert result.success is False
    assert result.message_id == "stale-card-message"
    assert adapter._card_sessions.claim("stale-card-message", 1).status == "missing"


@pytest.mark.asyncio
async def test_clarify_action_adapter_path_never_injects_message_event() -> None:
    adapter = _bare_clarify_adapter(native=True)
    adapter._http_session = object()
    clarify_id = "clarify-adapter-dispatch"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose",
        ["A", "B", "C"],
    )
    entry.multi_select = False
    session = _clarify_session(clarify_id=clarify_id)
    adapter._card_sessions.register(session)
    action = _clarify_action("clarify_choice_0")
    normal_dispatch = AsyncMock(return_value=True)
    try:
        with (
            patch.object(card_events, "dispatch_card_action_event", normal_dispatch),
            patch.object(api, "edit_card_message", AsyncMock(return_value={})),
        ):
            status = await adapter._handle_card_action_event(action)
            response = entry.response
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert status == "completed"
    assert response == "A"
    normal_dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_clarify_action_replay_does_not_resolve_twice() -> None:
    registry = card_events.CardSessionRegistry()
    clarify_id = "clarify-replay"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose",
        ["A", "B", "C"],
    )
    entry.multi_select = False
    registry.register(_clarify_session(clarify_id=clarify_id))
    action = _clarify_action("clarify_choice_0")
    try:
        first = await card_events.handle_card_action(
            registry,
            action,
            lambda session, claimed: card_events.dispatch_clarify_action(
                session,
                claimed,
            ),
        )
        second = await card_events.handle_card_action(
            registry,
            action,
            lambda session, claimed: card_events.dispatch_clarify_action(
                session,
                claimed,
            ),
        )
        response = entry.response
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert first == "completed"
    assert second == "duplicate"
    assert response == "A"
