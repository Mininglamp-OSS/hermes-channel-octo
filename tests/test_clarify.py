"""Hermes 0.20 native clarify delivery and resolution contracts."""

from __future__ import annotations
import asyncio
import threading

from unittest.mock import AsyncMock, patch

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from tools import clarify_gateway

from hermes_octo_plugin import api, card_events, card_progress, card_tools, clarify
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
        ("0.20.1.dev2", False),
        ("0.20.4rc1", False),
        ("0.21.0rc1", False),
        ("0.21.0.dev1", False),
        ("0.21.0", False),
        ("0.22.0", False),
        ("1.0.0", False),
        ("not-a-version", False),
    ],
)
def test_constructor_enables_native_clarify_only_for_stable_hermes_020(
    version: str,
    expected: bool,
) -> None:
    with patch.object(clarify, "package_version", return_value=version):
        actual = clarify.native_clarify_supported()

    assert actual is expected


def test_unstable_native_clarify_gate_logs_its_decision_once(caplog) -> None:
    version = "0.20.999.dev999"
    with (
        patch.object(clarify, "package_version", return_value=version),
        caplog.at_level("INFO"),
    ):
        assert clarify.native_clarify_supported() is False
        assert clarify.native_clarify_supported() is False

    assert sum(
        version in record.message and "native clarify disabled" in record.message
        for record in caplog.records
    ) == 1


def test_missing_packaging_only_disables_native_clarify() -> None:
    real_import = __import__

    def import_without_packaging(name, *args, **kwargs):
        if name == "packaging.version":
            raise ModuleNotFoundError("packaging unavailable")
        return real_import(name, *args, **kwargs)

    with (
        patch.object(clarify, "package_version", return_value="0.20.0"),
        patch("builtins.__import__", side_effect=import_without_packaging),
    ):
        assert clarify.native_clarify_supported() is False


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
async def test_native_clarify_host_integration_failure_uses_text_fallback() -> None:
    adapter = _bare_clarify_adapter(native=True)
    expected = SendResult(success=True, message_id="text-fallback")
    fallback = AsyncMock(return_value=expected)

    with (
        patch.object(BasePlatformAdapter, "send_clarify", fallback),
        patch(
            "hermes_octo_plugin.adapter._deliver_clarify",
            AsyncMock(side_effect=RuntimeError("host integration changed")),
        ),
    ):
        result = await OctoAdapter.send_clarify(
            adapter,
            "group-1",
            "Which option?",
            ["A", "B"],
            clarify_id="clarify-host-failure",
            session_key=_ROUTE.session_key,
        )

    assert result is expected
    fallback.assert_awaited_once()


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
    wait_for_progress.assert_awaited_once()
    kwargs = wait_for_progress.await_args.kwargs
    assert kwargs["adapter"] is adapter
    assert kwargs["session_key"] == _ROUTE.session_key
    assert 0 < kwargs["timeout"] <= 5.0


@pytest.mark.asyncio
async def test_native_clarify_limits_progress_wait_to_five_seconds() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-progress-budget"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False
    wait_for_progress = AsyncMock()
    try:
        with (
            patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
            patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
            patch.object(
                card_progress,
                "wait_for_initial_delivery",
                wait_for_progress,
            ),
            patch.object(
                api,
                "send_card_message",
                AsyncMock(return_value=SendMessageResult(message_id="clarify-budget")),
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
    timeout = wait_for_progress.await_args.kwargs["timeout"]
    assert 0 < timeout <= 5.0


@pytest.mark.asyncio
async def test_cleared_clarify_while_waiting_for_progress_is_never_posted() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-cleared-before-post"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False

    async def clear_pending(**_kwargs) -> None:
        clarify_gateway.clear_session(_ROUTE.session_key)

    send_card = AsyncMock()
    fallback = AsyncMock()
    with (
        patch.object(BasePlatformAdapter, "send_clarify", fallback),
        patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
        patch.object(card_progress, "wait_for_initial_delivery", side_effect=clear_pending),
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

    assert result.success is False
    assert result.error == "Hermes clarify is no longer pending"
    send_card.assert_not_awaited()
    fallback.assert_not_awaited()



@pytest.mark.parametrize("phase", ["profile", "progress"])
@pytest.mark.asyncio
async def test_native_clarify_timeout_returns_failure_without_text_fallback(
    phase: str,
) -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = f"clarify-{phase}-timeout"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False
    get_profile = AsyncMock(return_value=_MANIFEST)
    wait_for_progress = AsyncMock()
    if phase == "profile":
        get_profile.side_effect = TimeoutError
    else:
        wait_for_progress.side_effect = TimeoutError
    send_card = AsyncMock()
    fallback = AsyncMock()
    try:
        with (
            patch.object(BasePlatformAdapter, "send_clarify", fallback),
            patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
            patch.object(api, "get_card_profile", get_profile),
            patch.object(
                card_progress,
                "wait_for_initial_delivery",
                wait_for_progress,
            ),
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
    assert result.error == "Octo clarify card delivery timed out"
    send_card.assert_not_awaited()
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleared_clarify_during_profile_lookup_is_never_posted_or_fallbacked() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-cleared-during-profile"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False
    async def profile_then_clear(*_args, **_kwargs) -> CardProfileManifest:
        clarify_gateway.clear_session(_ROUTE.session_key)
        return CardProfileManifest(available=True, enabled=False)

    send_card = AsyncMock()
    fallback = AsyncMock()
    with (
        patch.object(BasePlatformAdapter, "send_clarify", fallback),
        patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(api, "get_card_profile", side_effect=profile_then_clear),
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

    assert result.success is False
    assert result.error == "Hermes clarify is no longer pending"
    send_card.assert_not_awaited()
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_failure_after_clarify_cancellation_never_sends_text_fallback() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-profile-failed-after-cancel"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False

    async def clear_then_fail(*_args, **_kwargs) -> CardProfileManifest:
        clarify_gateway.clear_session(_ROUTE.session_key)
        raise RuntimeError("profile unavailable")

    fallback = AsyncMock()
    with (
        patch.object(BasePlatformAdapter, "send_clarify", fallback),
        patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(api, "get_card_profile", side_effect=clear_then_fail),
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
    assert result.error == "Hermes clarify is no longer pending"
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_clarify_cleared_during_post_keeps_sent_card_session_owned() -> None:
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
        return SendMessageResult(message_id="clarify-orphan-guard")

    fallback = AsyncMock()
    with (
        patch.object(BasePlatformAdapter, "send_clarify", fallback),
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
    assert result.message_id == "clarify-orphan-guard"
    claim = adapter._card_sessions.claim("clarify-orphan-guard", 1)
    assert claim.status == "claimed"
    assert claim.session is not None
    assert claim.session.clarify is not None
    assert claim.session.clarify.clarify_id == clarify_id
    fallback.assert_not_awaited()

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
async def test_binding_failure_retires_the_sent_clarify_card() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-binding-failure"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False
    edit_card = AsyncMock()
    try:
        with (
            patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
            patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
            patch.object(
                api,
                "send_card_message",
                AsyncMock(return_value=SendMessageResult(message_id="orphan-card")),
            ),
            patch.object(api, "edit_card_message", edit_card),
            patch.object(
                adapter,
                "_register_card_session",
                side_effect=RuntimeError("registry unavailable"),
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

    assert result.success is False
    assert result.message_id == "orphan-card"
    edit_card.assert_awaited_once()
    kwargs = edit_card.await_args.kwargs
    assert kwargs["message_id"] == "orphan-card"
    assert kwargs["card_seq"] == 1
    assert kwargs["transient"] is False
    assert _card_nodes(kwargs["card"], "Action.Submit") == []
    assert "不可用" in kwargs["plain"]


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
    entry: object | None = None,
    multi_select: bool = False,
) -> card_events.CardSession:
    clarify = card_events.ClarifySession(
        clarify_id=clarify_id,
        entry=entry if entry is not None else object(),
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
        card={
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [],
            **(
                {
                    "actions": [{
                        "type": "Action.Submit",
                        "id": "clarify_confirm",
                        "title": "提交",
                        "data": {"_octo_binding": "binding-1"},
                    }]
                }
                if multi_select
                else {}
            ),
        },
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
            _clarify_session(clarify_id=clarify_id, entry=entry),
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
            _clarify_session(
                clarify_id=clarify_id,
                entry=entry,
                multi_select=True,
            ),
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
async def test_invalid_multi_submit_keeps_card_retryable_for_a_later_valid_action() -> None:
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
        _clarify_session(
            clarify_id=clarify_id,
            entry=entry,
            multi_select=True,
        )
    )
    updates: list[tuple[str, bool, cards.CardRenderResult]] = []

    async def update_status(session, action, status, *, transient):
        updates.append((
            status,
            transient,
            card_events.render_card_action_status(session, action, status),
        ))

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
            update_status=update_status,
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

    assert invalid == "invalid"
    assert [(status, transient) for status, transient, _ in updates] == [
        ("invalid", False)
    ]
    invalid_card = updates[0][2]
    assert "请选择至少一个选项" in invalid_card.plain
    assert _card_nodes(invalid_card.card, "Action.Submit")
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
            _clarify_session(clarify_id=clarify_id, entry=entry),
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
async def test_reused_clarify_id_cannot_resolve_a_same_signature_replacement() -> None:
    clarify_id = "clarify-reused-id"
    original = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose",
        ["A", "B", "C"],
    )
    original.multi_select = False
    old_card = _clarify_session(clarify_id=clarify_id, entry=original)
    clarify_gateway.clear_session(_ROUTE.session_key)
    replacement = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose",
        ["A", "B", "C"],
    )
    replacement.multi_select = False
    try:
        status = await card_events.dispatch_clarify_action(
            old_card,
            _clarify_action("clarify_choice_0"),
        )
        response = replacement.response
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert status == "expired"
    assert response is None


@pytest.mark.asyncio
async def test_reused_clarify_id_gets_a_new_delivery_id_per_occurrence() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-reused-delivery"
    send_card = AsyncMock(
        side_effect=[
            SendMessageResult(message_id="card-1"),
            SendMessageResult(message_id="card-2"),
        ]
    )
    with (
        patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
        patch.object(api, "send_card_message", send_card),
    ):
        for _ in range(2):
            entry = clarify_gateway.register(
                clarify_id,
                _ROUTE.session_key,
                "Which option?",
                ["A", "B"],
            )
            entry.multi_select = False
            try:
                result = await OctoAdapter.send_clarify(
                    adapter,
                    _ROUTE.chat_id,
                    "Which option?",
                    ["A", "B"],
                    clarify_id=clarify_id,
                    session_key=_ROUTE.session_key,
                )
                assert result.success is True
            finally:
                clarify_gateway.clear_session(_ROUTE.session_key)

    assert send_card.await_count == 2
    first = send_card.await_args_list[0].kwargs["client_msg_no"]
    second = send_card.await_args_list[1].kwargs["client_msg_no"]
    assert first != second


@pytest.mark.asyncio
async def test_reused_clarify_id_cannot_race_between_validation_and_resolution() -> None:
    clarify_id = "clarify-reused-during-dispatch"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Choose",
        ["A", "B", "C"],
    )
    replacement_box: list[object] = []
    inner_lock = threading.RLock()

    def replace_after_first_full_release() -> None:
        clarify_gateway.clear_session(_ROUTE.session_key)
        replacement_box.append(
            clarify_gateway.register(
                clarify_id,
                _ROUTE.session_key,
                "Replacement",
                ["X", "Y"],
            )
        )

    class ReplaceOnRelease:
        def __init__(self) -> None:
            self.depth = 0
            self.fired = False

        def __enter__(self):
            inner_lock.acquire()
            self.depth += 1
            return self

        def __exit__(self, *_args):
            self.depth -= 1
            inner_lock.release()
            if self.depth == 0 and not self.fired:
                self.fired = True
                replace_after_first_full_release()

    try:
        with patch.object(clarify_gateway, "_lock", ReplaceOnRelease()):
            status = await card_events.dispatch_clarify_action(
                _clarify_session(clarify_id=clarify_id, entry=entry),
                _clarify_action("clarify_choice_1"),
            )
        replacement = replacement_box[0]
        response = replacement.response
    finally:
        clarify_gateway.clear_session(_ROUTE.session_key)

    assert status == "completed"
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
async def test_definitive_rejection_after_cancellation_never_sends_text_fallback() -> None:
    adapter = _bare_clarify_adapter(native=True)
    clarify_id = "clarify-rejected-after-cancel"
    entry = clarify_gateway.register(
        clarify_id,
        _ROUTE.session_key,
        "Which option?",
        ["A", "B"],
    )
    entry.multi_select = False

    async def clear_then_reject(*_args, **_kwargs) -> SendMessageResult:
        clarify_gateway.clear_session(_ROUTE.session_key)
        raise api.OctoApiError("/v1/bot/sendMessage", status=400)

    fallback = AsyncMock()
    with (
        patch.object(BasePlatformAdapter, "send_clarify", fallback),
        patch.object(card_tools, "_trusted_route", return_value=_ROUTE),
        patch.object(api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
        patch.object(api, "send_card_message", side_effect=clear_then_reject),
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
    assert result.error == "Hermes clarify is no longer pending"
    fallback.assert_not_awaited()




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
    session = _clarify_session(clarify_id=clarify_id, entry=entry)
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
    registry.register(_clarify_session(clarify_id=clarify_id, entry=entry))
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
