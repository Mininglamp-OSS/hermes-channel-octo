"""Progress-card lifecycle and concurrency contracts."""

from __future__ import annotations
import asyncio

from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin import card_progress, cards, types
from hermes_octo_plugin.card_tools import TrustedOctoRoute
from hermes_octo_plugin.types import CardProfileManifest, ChannelType, SendMessageResult
from hermes_octo_plugin.api import OctoApiError


class _Adapter:
    _api_url = "https://api.example.invalid"
    _bot_token = "test-token"
    _http_session = object()
    _disconnecting = False
    on_behalf_of: str | None = None
    progress_card_renderer = "local"

    def __init__(self) -> None:
        self.pending: list[Callable[[], Coroutine[Any, Any, None]]] = []
        self._card_profile_cache = cards.CardProfileCache()

    def _schedule_card_progress(
        self,
        factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> bool:
        self.pending.append(factory)
        return True

    async def run_next(self) -> None:
        await self.pending.pop(0)()


_ROUTE = TrustedOctoRoute(
    chat_id="group-1",
    channel_id="group-1",
    channel_type=ChannelType.Group,
    requester_uid="user-1",
    session_key="octo:group-1:user-1",
)
_MANIFEST = CardProfileManifest(
    available=True,
    enabled=True,
    profiles=("octo/v1",),
    card_version="1.5",
    elements=("TextBlock",),
)


def test_progress_renderer_uses_allowlisted_bounded_summaries_without_dlp() -> None:
    rendered = cards.build_progress_card(
        phase="running",
        tools=[
            {
                "tool_call_id": "call-1",
                "tool_name": "read",
                "args": {
                    "path": "/Users/example/project/config.py",
                    "token": "AKIA1234567890ABCDEF",
                },
                "status": "running",
            },
            {
                "tool_call_id": "call-2",
                "tool_name": "mcp__untrusted__tool-token-AKIA1234567890ABCDEF",
                "args": {"secret": "hidden"},
                "status": "failed",
                "error": "Authorization: Bearer hidden",
            },
        ],
    )

    assert "读取文件" in rendered.plain
    assert "扩展工具" in rendered.plain
    assert "AKIA1234567890ABCDEF" not in rendered.plain
    assert "Authorization: Bearer hidden" in rendered.plain
    assert "reasoning" not in rendered.plain.lower()
    assert "🤖" not in rendered.plain


def test_current_hermes_tools_have_chinese_labels_and_bounded_parameter_summaries() -> None:
    cases = [
        (
            "read_file",
            {"path": "/work/project/src/cards.py", "offset": 420, "limit": 141},
            "读取文件",
            "…/src/cards.py · 第 420–560 行",
        ),
        (
            "search_files",
            {"pattern": "reasoning", "path": "/work/project/src"},
            "搜索文件",
            "reasoning · …/project/src",
        ),
        (
            "terminal",
            {
                "command": (
                    "HERMES_HOME=/Users/example/.hermes/profiles/xiao_ai "
                    "uv run pytest -q tests/test_card_progress.py"
                )
            },
            "运行命令",
            "uv run pytest",
        ),
        (
            "browser_navigate",
            {"url": "https://user:password@docs.example.com/private?q=secret"},
            "打开网页",
            "https://user:password@docs.example.com/private?q=secret",
        ),
        (
            "tool_search",
            {"query": "calendar integration"},
            "查找工具",
            "calendar integration",
        ),
        (
            "tool_describe",
            {"name": "browser_navigate"},
            "读取工具说明",
            "browser_navigate",
        ),
        (
            "skill_view",
            {"name": "frontend-design"},
            "读取技能",
            "frontend-design",
        ),
        (
            "tool_call",
            {
                "name": "read_file",
                "arguments": {"path": "/work/project/src/cards.py"},
            },
            "读取文件",
            "…/src/cards.py",
        ),
        (
            "lcm_inspect",
            {"query": "private conversation text"},
            "检查上下文",
            "",
        ),
        (
            "browser_type",
            {"text": "Bearer secret-value", "ref": "password"},
            "填写表单",
            "",
        ),
    ]

    for tool_name, params, label, summary in cases:
        assert cards.localized_tool_label(tool_name, params) == label
        assert cards.summarize_tool_params(tool_name, params) == summary


@pytest.mark.asyncio
async def test_progress_lifecycle_sends_then_edits_transient_and_final() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(return_value=SendMessageResult(message_id="progress-1"))
    edit = AsyncMock(return_value={})
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(card_progress.api, "send_card_message", send),
        patch.object(card_progress.api, "edit_card_message", edit),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        assert adapter.pending == []

        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-a",
            tool_name="read",
            args={"path": "/tmp/a.py"},
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-b",
            tool_name="bash",
            args={"command": "pytest -q"},
        )
        assert len(adapter.pending) == 1
        controller.tool_finished(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-b",
            tool_name="bash",
            status="error",
            error="secret-token-AKIA1234567890ABCDEF",
        )
        controller.tool_finished(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-a",
            tool_name="read",
            status="ok",
        )
        await adapter.run_next()

        controller.complete(session_id="session-1", turn_id="turn-1")
        await adapter.run_next()

    assert send.await_count == 1
    assert edit.await_count == 1
    sent = send.await_args.kwargs
    final = edit.await_args.kwargs
    assert "● 读取文件 · /tmp/a.py · 已完成" in sent["plain"]
    assert "○ 运行命令 · pytest · 失败" in sent["plain"]
    assert "secret-token-AKIA1234567890ABCDEF" in sent["plain"]
    assert final["card_seq"] == 1
    assert final["transient"] is False
    assert final["plain"].startswith("处理进度 · 已完成")
    assert controller.state_count == 0




@pytest.mark.asyncio
async def test_follow_up_rolls_old_card_to_stopped_and_sends_new_card_at_bottom() -> (
    None
):
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(
        side_effect=[
            SendMessageResult(message_id="progress-1"),
            SendMessageResult(message_id="progress-2"),
        ]
    )
    edit = AsyncMock(return_value={})
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(card_progress.api, "send_card_message", send),
        patch.object(card_progress.api, "edit_card_message", edit),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-1",
            tool_name="read",
            args={"path": "/tmp/first.py"},
        )
        await adapter.run_next()

        controller.rollover_for_inbound(
            chat_id="group-1",
            requester_uid="user-1",
        )
        controller.rollover_for_inbound(
            chat_id="group-1",
            requester_uid="user-1",
        )
        assert len(adapter.pending) == 1
        controller.tool_started(
            session_id="session-1",
            turn_id="steer-turn-2",
            tool_call_id="call-2",
            tool_name="terminal",
            args={"command": "pytest -q"},
        )
        assert len(adapter.pending) == 2

        await adapter.run_next()
        await adapter.run_next()
        controller.tool_finished(
            session_id="session-1",
            turn_id="steer-turn-2",
            tool_call_id="call-2",
            tool_name="terminal",
            status="ok",
        )
        await adapter.run_next()
        controller.complete(session_id="session-1", turn_id="steer-turn-2")
        await adapter.run_next()

    assert send.await_count == 2
    assert [
        call.kwargs["client_msg_no"] for call in send.await_args_list
    ] == [
        "card-progress:turn-1:0",
        "card-progress:turn-1:1",
    ]
    old_final = edit.await_args_list[0].kwargs
    assert old_final["message_id"] == "progress-1"
    assert old_final["transient"] is False
    assert old_final["plain"].startswith("处理进度 · 已停止")
    second_send = send.await_args_list[1].kwargs
    assert "运行命令" in second_send["plain"]
    assert "读取文件" not in second_send["plain"]
    new_final = edit.await_args_list[-1].kwargs
    assert new_final["message_id"] == "progress-2"
    assert new_final["transient"] is False
    assert new_final["plain"].startswith("处理进度 · 已完成")
    assert controller.state_count == 0


@pytest.mark.asyncio
async def test_follow_up_drops_an_old_card_that_was_never_sent() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(return_value=SendMessageResult(message_id="progress-2"))
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(card_progress.api, "send_card_message", send),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-1",
            tool_name="read",
            args={"path": "/tmp/first.py"},
        )
        controller.rollover_for_inbound(
            chat_id="group-1",
            requester_uid="user-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-2",
            tool_name="terminal",
            args={"command": "pytest -q"},
        )

        await adapter.run_next()
        await adapter.run_next()

    send.assert_awaited_once()
    assert send.await_args.kwargs["client_msg_no"] == "card-progress:turn-1:1"
    assert "运行命令" in send.await_args.kwargs["plain"]
    assert "读取文件" not in send.await_args.kwargs["plain"]




def test_turn_identity_fallback_fails_closed_when_session_is_ambiguous() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    for turn_id in ("turn-a", "turn-b"):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id=turn_id,
        )

    controller.tool_started(
        session_id="session-1",
        turn_id="steer-turn",
        tool_call_id="call-1",
        tool_name="terminal",
        args={"command": "pytest -q"},
    )
    assert adapter.pending == []

    controller.tool_started(
        session_id="session-1",
        turn_id="turn-a",
        tool_call_id="call-2",
        tool_name="terminal",
        args={"command": "pytest -q"},
    )
    assert len(adapter.pending) == 1


def test_progress_is_disabled_for_on_behalf_of_delivery() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    adapter.on_behalf_of = "grantor-1"

    controller.begin(
        adapter=adapter,
        route=_ROUTE,
        session_id="session-1",
        turn_id="turn-1",
    )

    assert controller.state_count == 0
    assert adapter.pending == []

@pytest.mark.asyncio
async def test_progress_card_is_finalized_after_answer_stream_delivery() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    events: list[str] = []
    send_card = AsyncMock(
        side_effect=lambda *args, **kwargs: (
            events.append("card-send")
            or SendMessageResult(message_id="progress-1")
        )
    )
    edit_card = AsyncMock(
        side_effect=lambda *args, **kwargs: events.append("card-final") or {}
    )
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(card_progress.api, "send_card_message", send_card),
        patch.object(card_progress.api, "edit_card_message", edit_card),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-1",
            tool_name="read",
            args={"path": "/tmp/input.py"},
        )
        await adapter.run_next()
        events.append("answer-stream-finalized")
        controller.complete(session_id="session-1", turn_id="turn-1")
        await adapter.run_next()

    assert events == ["card-send", "answer-stream-finalized", "card-final"]

@pytest.mark.asyncio
async def test_progress_stays_silent_for_pure_model_turn_and_card_authoring_tools() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(return_value=SendMessageResult(message_id="unexpected"))
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(card_progress.api, "send_card_message", send),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.model_started(
            session_id="session-1", turn_id="turn-1", model_call_id="model-1"
        )
        controller.model_finished(
            session_id="session-1", turn_id="turn-1", model_call_id="model-1"
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="card-1",
            tool_name="octo_send_display_card",
            args={"title": "Result"},
        )
        controller.complete(session_id="session-1", turn_id="turn-1")
        while adapter.pending:
            await adapter.run_next()

    send.assert_not_awaited()






@pytest.mark.asyncio
async def test_initial_delivery_barrier_ignores_other_session_progress() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    other_route = TrustedOctoRoute(
        chat_id="group-2",
        channel_id="group-2",
        channel_type=ChannelType.Group,
        requester_uid="user-2",
        session_key="octo:group-2:user-2",
    )
    send = AsyncMock(
        side_effect=[
            SendMessageResult(message_id="progress-1"),
            SendMessageResult(message_id="progress-2"),
        ]
    )
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(card_progress.api, "send_card_message", send),
    ):
        for session_id, turn_id, route in (
            ("session-1", "turn-1", _ROUTE),
            ("session-2", "turn-2", other_route),
        ):
            controller.begin(
                adapter=adapter,
                route=route,
                session_id=session_id,
                turn_id=turn_id,
            )
            controller.tool_started(
                session_id=session_id,
                turn_id=turn_id,
                tool_call_id=f"clarify-{turn_id}",
                tool_name="clarify",
                args={"question": "Choose"},
            )

        waiting = asyncio.create_task(
            controller.wait_for_initial_delivery(
                adapter=adapter,
                session_key=_ROUTE.session_key,
            )
        )
        await asyncio.sleep(0)
        await adapter.run_next()
        await asyncio.wait_for(waiting, timeout=0.1)

    assert len(adapter.pending) == 1
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelling_initial_delivery_wait_does_not_cancel_progress() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(return_value=SendMessageResult(message_id="progress-1"))
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(card_progress.api, "send_card_message", send),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="clarify-1",
            tool_name="clarify",
            args={"question": "Choose"},
        )
        waiting = asyncio.create_task(
            controller.wait_for_initial_delivery(
                adapter=adapter,
                session_key=_ROUTE.session_key,
            )
        )
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await adapter.run_next()

    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_progress_turns_are_isolated_and_session_cancel_drops_pending_work() -> (
    None
):
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(
        side_effect=[
            SendMessageResult(message_id="progress-1"),
            SendMessageResult(message_id="progress-2"),
        ]
    )
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(card_progress.api, "send_card_message", send),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-2",
            turn_id="turn-2",
        )
        controller.tool_started(
            session_id="session-2",
            turn_id="turn-2",
            tool_call_id="call-2",
            tool_name="read",
            args={"path": "/tmp/two.py"},
        )
        assert controller.state_count == 2
        controller.cancel_session("session-1")
        assert controller.state_count == 1
        await adapter.run_next()

    assert send.await_count == 1
    assert send.await_args.kwargs["client_msg_no"] == "card-progress:turn-2:0"


@pytest.mark.asyncio
async def test_progress_delivery_failure_is_fail_soft() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    with (
        patch.object(
            card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)
        ),
        patch.object(
            card_progress.api,
            "send_card_message",
            AsyncMock(side_effect=RuntimeError("network down")),
        ),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-1",
            tool_name="read",
            args={"path": "/tmp/input.py"},
        )
        await adapter.run_next()

    assert controller.state_count == 0


def test_session_end_finalizes_the_exact_turn_instead_of_cancelling_it() -> None:
    controller = MagicMock()
    with patch.object(card_progress, "_CONTROLLER", controller):
        card_progress.on_session_end(
            session_id="session-1",
            task_id="task-1",
            turn_id="turn-1",
            completed=False,
            failed=False,
            interrupted=False,
            platform="octo",
        )

    controller.complete.assert_called_once_with(
        session_id="session-1",
        turn_id="turn-1",
        failed=True,
        terminal_phase="error",
    )
    controller.cancel_session.assert_not_called()

def test_session_end_preserves_stopped_error_and_incomplete_terminal_states() -> None:
    controller = MagicMock()
    with patch.object(card_progress, "_CONTROLLER", controller):
        for kwargs, expected in (
            (
                {"completed": False, "failed": False, "interrupted": True},
                "stopped",
            ),
            (
                {"completed": False, "failed": True, "interrupted": False},
                "error",
            ),
            (
                {"completed": False, "failed": False, "interrupted": False},
                "error",
            ),
            (
                {"completed": True, "failed": False, "interrupted": False},
                "completed",
            ),
        ):
            card_progress.on_session_end(
                session_id="session-1",
                turn_id="turn-1",
                platform="octo",
                **kwargs,
            )

    assert [
        call.kwargs["terminal_phase"]
        for call in controller.complete.call_args_list
    ] == ["stopped", "error", "error", "completed"]


@pytest.mark.asyncio
async def test_missing_tool_call_ids_pair_with_last_running_same_name() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(return_value=SendMessageResult(message_id="progress-1"))
    with (
        patch.object(
            card_progress.api,
            "get_card_profile",
            AsyncMock(return_value=_MANIFEST),
        ),
        patch.object(card_progress.api, "send_card_message", send),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="",
            tool_name="read",
            args={"path": "/tmp/first.txt"},
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="",
            tool_name="read",
            args={"path": "/tmp/second.txt"},
        )
        controller.tool_finished(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="",
            tool_name="read",
            status="ok",
            result={"match_count": 2},
        )
        controller.complete(session_id="session-1", turn_id="turn-1")
        await adapter.run_next()

    plain = send.await_args.kwargs["plain"]
    assert plain.index("first.txt") < plain.index("second.txt")
    assert "first.txt · 2 项结果" not in plain
    assert "second.txt · 2 项结果" in plain


def test_fallback_progress_matches_openclaw_group_window_and_terminal_collapse() -> None:
    tools = [
        {
            "tool_name": "read",
            "status": "complete",
            "summary": f"/tmp/{index}.txt",
            "duration_ms": 100,
        }
        for index in range(14)
    ]
    rendered = cards.build_progress_card(
        phase="completed",
        tools=tools,
        elapsed_ms=2_000,
        capabilities=cards.CardCapabilities(
            available=True,
            enabled=True,
            elements=frozenset(
                {"TextBlock", "Container", "ColumnSet", "ActionSet", "RichTextBlock"}
            ),
            actions=frozenset({"Action.ToggleVisibility"}),
        ),
    )

    assert rendered.plain.startswith("处理进度 · 已完成 · 14 个步骤 · 2.0s")
    assert "已隐藏前 2 个步骤" in rendered.plain
    assert "读取文件 × 12 · 总计 1.2s · 最近：/tmp/13.txt" in rendered.plain
    detail = rendered.card["body"][1]
    assert detail["id"] == "timeline_detail"
    assert detail["isVisible"] is False
    toggle_json = str(rendered.card["body"][0])
    assert "收起执行详情" in toggle_json
    assert "展开执行详情" in toggle_json


def test_progress_renderer_uses_reasoning_only_when_public_thought_is_visible() -> None:
    tools = [
        {
            "tool_name": "__thinking__",
            "status": "complete",
            "thought": "Inspect the lifecycle.",
            "duration_ms": 300,
        },
        {
            "tool_name": "read",
            "status": "complete",
            "summary": "/tmp/input.py",
            "result_summary": "7 项结果",
            "duration_ms": 200,
        },
    ]
    fallback = cards.build_agent_progress_card(
        phase="completed",
        tools=tools,
        elapsed_ms=500,
        reasoning_id="session:turn",
        reasoning_visible=False,
    )
    reasoning = cards.build_agent_progress_card(
        phase="completed",
        tools=tools,
        elapsed_ms=500,
        reasoning_id="session:turn",
        reasoning_visible=True,
    )

    assert "Inspect the lifecycle." not in fallback.plain
    assert "7 项结果" in fallback.plain
    assert "Inspect the lifecycle." in reasoning.plain


@pytest.mark.asyncio
async def test_registry_edit_retries_same_sequence_and_recovers_after_exhaustion() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    adapter.progress_card_renderer = "registry"
    manifest = CardProfileManifest(
        available=True,
        enabled=True,
        templating=types.CardTemplatingCapability(
            supported=True,
            wire="template-ref/v1",
            templates=(
                types.CardTemplateCapability(
                    id="ai.reasoning-process",
                    version="0.3.0",
                    views=(
                        types.CardTemplateViewCapability(
                            name="active", wire_profile="octo/v2", states=("reasoning", "answering")
                        ),
                        types.CardTemplateViewCapability(
                            name="error", wire_profile="octo/v2", states=("error",)
                        ),
                        types.CardTemplateViewCapability(
                            name="result", wire_profile="octo/v1", states=("completed", "stopped")
                        ),
                    ),
                ),
            ),
        ),
    )
    send_template = AsyncMock(return_value=SendMessageResult(message_id="reasoning-1"))
    edit_template = AsyncMock(
        side_effect=[
            OctoApiError("/v1/bot/message/edit", status=503),
            OctoApiError("/v1/bot/message/edit", status=503),
            OctoApiError("/v1/bot/message/edit", status=503),
            {},
        ]
    )
    with (
        patch.object(card_progress.api, "get_card_profile", AsyncMock(return_value=manifest)),
        patch.object(card_progress, "send_template_card_message", send_template),
        patch.object(card_progress, "edit_template_card_message", edit_template),
        patch.object(card_progress.asyncio, "sleep", AsyncMock()),
    ):
        controller.begin(
            adapter=adapter, route=_ROUTE, session_id="session-1", turn_id="turn-1"
        )
        controller.tool_started(
            session_id="session-1", turn_id="turn-1", tool_call_id="call-1",
            tool_name="read", args={"path": "/tmp/input.py"},
        )
        await adapter.run_next()
        controller.tool_finished(
            session_id="session-1", turn_id="turn-1", tool_call_id="call-1",
            tool_name="read", status="ok", result={"match_count": 1},
        )
        await adapter.run_next()
        controller.complete(session_id="session-1", turn_id="turn-1")
        await adapter.run_next()

    assert [call.kwargs["card_seq"] for call in edit_template.await_args_list] == [1, 1, 1, 2]
    assert edit_template.await_args_list[-1].kwargs["transient"] is False
    assert controller.state_count == 0


def test_post_llm_waits_for_authoritative_session_end_status() -> None:
    controller = MagicMock()
    with patch.object(card_progress, "_CONTROLLER", controller):
        card_progress.on_post_llm_call(
            session_id="session-1",
            turn_id="turn-1",
            platform="octo",
        )
        controller.complete.assert_not_called()
        card_progress.on_session_end(
            session_id="session-1",
            turn_id="turn-1",
            completed=False,
            failed=True,
            interrupted=False,
            platform="octo",
        )

    controller.complete.assert_called_once_with(
        session_id="session-1",
        turn_id="turn-1",
        failed=True,
        terminal_phase="error",
    )


@pytest.mark.asyncio
async def test_failed_turn_finalizes_as_stopped_and_closes_running_tools() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(return_value=SendMessageResult(message_id="progress-1"))
    edit = AsyncMock(return_value={})
    with (
        patch.object(
            card_progress.api,
            "get_card_profile",
            AsyncMock(return_value=_MANIFEST),
        ),
        patch.object(card_progress.api, "send_card_message", send),
        patch.object(card_progress.api, "edit_card_message", edit),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-1",
            tool_name="read",
            args={"path": "/tmp/input.txt"},
        )
        controller.complete(
            session_id="session-1",
            turn_id="turn-1",
            failed=True,
        )
        await adapter.run_next()

    edit.assert_not_awaited()
    final = send.await_args.kwargs
    assert final["plain"].startswith("处理进度 · 已停止")
    assert "失败" in final["plain"]
    assert controller.state_count == 0


@pytest.mark.asyncio
async def test_progress_keeps_rendering_after_more_than_32_tool_calls() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(return_value=SendMessageResult(message_id="progress-1"))
    edit = AsyncMock(return_value={})
    with (
        patch.object(
            card_progress.api,
            "get_card_profile",
            AsyncMock(return_value=_MANIFEST),
        ),
        patch.object(
            card_progress.api,
            "send_card_message",
            send,
        ),
        patch.object(card_progress.api, "edit_card_message", edit),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        for index in range(33):
            controller.tool_started(
                session_id="session-1",
                turn_id="turn-1",
                tool_call_id=f"call-{index}",
                tool_name="read",
                args={"path": f"/tmp/{index}.txt"},
            )
        controller.complete(session_id="session-1", turn_id="turn-1")
        await adapter.run_next()

    send.assert_awaited_once()
    edit.assert_not_awaited()
    assert controller.state_count == 0


def test_reasoning_process_data_matches_openclaw_summary_contract() -> None:
    data = cards.build_reasoning_process_data(
        phase="completed",
        tools=[
            {
                "tool_name": "__thinking__",
                "status": "complete",
                "thought": "Inspect the relevant implementation.",
                "duration_ms": 1_200,
            },
            {
                "tool_name": "read",
                "status": "complete",
                "summary": "…/src/cards.py",
                "result_summary": "42 results",
                "duration_ms": 300,
            },
            {
                "tool_name": "bash",
                "status": "failed",
                "summary": "pytest",
                "error": "command failed",
                "duration_ms": 800,
            },
        ],
        elapsed_ms=12_000,
        reasoning_id="session-1:turn-1",
    )

    assert data["state"] == "completed"
    assert data["statusLabel"] == "已完成"
    assert data["traceExpanded"] is False
    assert data["traceCollapsed"] is True
    assert data["timerText"] == "12.0s · 1 个阶段 · 2 次工具调用"
    assert data["phases"] == [
        {
            "thought": "Inspect the relevant implementation.",
            "actions": [
                {
                    "tool": "读取文件",
                    "detail": "…/src/cards.py · 42 项结果",
                    "statusGlyph": "●",
                    "statusTone": "Good",
                },
                {
                    "tool": "运行命令",
                    "detail": "pytest · command failed",
                    "statusGlyph": "○",
                    "statusTone": "Attention",
                },
            ],
        }
    ]


def test_reasoning_process_renderer_uses_collapsed_terminal_trace() -> None:
    rendered = cards.build_reasoning_process_card(
        phase="completed",
        tools=[
            {
                "tool_name": "__thinking__",
                "status": "complete",
                "thought": "Check the implementation.",
            },
            {
                "tool_name": "read",
                "status": "complete",
                "summary": "…/src/cards.py",
                "result_summary": "completed",
            },
        ],
        elapsed_ms=2_000,
        reasoning_id="session-1:turn-1",
        capabilities=cards.CardCapabilities(
            available=True,
            enabled=True,
            elements=frozenset({"TextBlock", "Container", "ColumnSet", "ActionSet"}),
            actions=frozenset({"Action.ToggleVisibility"}),
        ),
    )

    body = rendered.card["body"]
    assert rendered.card["metadata"] == {"octo_layout": "agent_progress_v1"}
    assert body[0]["id"] == "octo-execution-trace-header"
    assert body[0]["style"] == "emphasis"
    assert body[1]["id"] == "trace_panel"
    assert body[1]["isVisible"] is False
    assert body[2]["id"] == "collapsed_panel"
    assert body[2]["isVisible"] is True
    toggle = body[3]["items"][0]["actions"][0]
    assert toggle == {
        "type": "Action.ToggleVisibility",
        "id": "reasoning_toggle",
        "title": "展开/收起执行详情",
        "targetElements": ["trace_panel", "collapsed_panel"],
    }
    assert rendered.plain.startswith("处理进度 · 已完成 · 2.0s · 1 个阶段 · 1 次工具调用")
    assert "Check the implementation." in rendered.plain
    assert "读取文件 · …/src/cards.py · 已完成" in rendered.plain
    assert "✦" not in str(rendered.card)
    assert "Reasoning" not in str(rendered.card)


def test_reasoning_summary_preserves_public_thought_but_not_raw_tool_output() -> None:
    assert (
        cards.sanitize_reasoning_thought("Authorization: Bearer abcdefghijklmnop")
        == "Authorization: Bearer abcdefghijklmnop"
    )
    assert (
        cards.summarize_tool_result(
            "bash",
            {
                "details": {"exitCode": 0},
                "content": [{"type": "text", "text": "Bearer raw-secret"}],
            },
        )
        == "退出码 0"
    )
    assert (
        cards.summarize_tool_result(
            "read",
            {"content": "complete file contents"},
        )
        == "已完成"
    )


def test_api_hooks_capture_only_public_reasoning_summary() -> None:
    controller = MagicMock()
    with (
        patch.object(card_progress, "_CONTROLLER", controller),
        patch.object(
            card_progress,
            "_reasoning_summaries_enabled",
            return_value=True,
        ),
    ):
        card_progress.on_pre_api_request(
            platform="octo",
            session_id="session-1",
            turn_id="turn-1",
            api_request_id="api-1",
        )
        card_progress.on_post_api_request(
            platform="octo",
            session_id="session-1",
            turn_id="turn-1",
            api_request_id="api-1",
            api_duration=1.25,
            assistant_message={
                "reasoning": "raw hidden chain of thought",
                "reasoning_content": "raw provider scratchpad",
                "reasoning_details": [
                    {
                        "type": "reasoning.summary",
                        "summary": "Inspecting the card lifecycle.",
                    }
                ],
                "codex_reasoning_items": [
                    {
                        "type": "reasoning",
                        "encrypted_content": "opaque-hidden-state",
                        "text": "raw hidden Codex reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "Checking the negotiated renderer.",
                            }
                        ],
                    }
                ],
                "content": None,
                "tool_calls": [{"id": "call-1"}],
            },
        )

    controller.model_started.assert_called_once_with(
        session_id="session-1",
        turn_id="turn-1",
        model_call_id="api-1",
    )
    controller.model_finished.assert_called_once_with(
        session_id="session-1",
        turn_id="turn-1",
        model_call_id="api-1",
        thought="Inspecting the card lifecycle. Checking the negotiated renderer.",
        duration_ms=1_250,
        answering=False,
        failed=False,
    )


@pytest.mark.asyncio
async def test_progress_defaults_to_local_type17_when_registry_is_compatible() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    manifest = CardProfileManifest(
        available=True,
        enabled=True,
        profiles=("octo/v1", "octo/v2"),
        card_version="1.5",
        elements=("TextBlock", "Container", "ColumnSet", "ActionSet"),
        actions=("Action.ToggleVisibility",),
        templating=types.CardTemplatingCapability(
            supported=True,
            wire="template-ref/v1",
            templates=(
                types.CardTemplateCapability(
                    id="ai.reasoning-process",
                    version="0.3.0",
                    views=(
                        types.CardTemplateViewCapability(
                            name="active",
                            wire_profile="octo/v2",
                            states=("reasoning", "answering"),
                        ),
                        types.CardTemplateViewCapability(
                            name="error",
                            wire_profile="octo/v2",
                            states=("error",),
                        ),
                        types.CardTemplateViewCapability(
                            name="result",
                            wire_profile="octo/v1",
                            states=("completed", "stopped"),
                        ),
                    ),
                ),
            ),
        ),
    )
    send_card = AsyncMock(
        return_value=SendMessageResult(message_id="progress-1")
    )
    edit_card = AsyncMock(return_value={})
    send_template = AsyncMock()
    edit_template = AsyncMock()
    with (
        patch.object(
            card_progress.api,
            "get_card_profile",
            AsyncMock(return_value=manifest),
        ),
        patch.object(card_progress.api, "send_card_message", send_card),
        patch.object(card_progress.api, "edit_card_message", edit_card),
        patch.object(
            card_progress,
            "send_template_card_message",
            send_template,
        ),
        patch.object(
            card_progress,
            "edit_template_card_message",
            edit_template,
        ),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        controller.tool_started(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-1",
            tool_name="read_file",
            args={"path": "/work/src/cards.py", "offset": 10, "limit": 20},
        )
        await adapter.run_next()
        controller.tool_finished(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-1",
            tool_name="read_file",
            status="ok",
            duration_ms=200,
            result={"match_count": 7, "content": "must not be rendered"},
        )
        await adapter.run_next()
        controller.complete(session_id="session-1", turn_id="turn-1")
        await adapter.run_next()

    send_template.assert_not_awaited()
    edit_template.assert_not_awaited()
    assert send_card.await_args.kwargs["profile"] == "octo/v1"
    assert send_card.await_args.kwargs["card"]["metadata"] == {
        "octo_layout": "agent_progress_v1"
    }
    final = edit_card.await_args_list[-1].kwargs
    assert final["transient"] is False
    assert final["plain"].startswith("处理进度 · 已完成")
    assert "读取文件" in final["plain"]
    assert "7 项结果" in final["plain"]


@pytest.mark.parametrize("action", ["reasoning_stop", "reasoning_retry"])
def test_registry_reasoning_template_rejects_unhandled_submit_actions(
    action: str,
) -> None:
    templating = types.CardTemplatingCapability(
        supported=True,
        wire="template-ref/v1",
        templates=(
            types.CardTemplateCapability(
                id="ai.reasoning-process",
                version="0.3.0",
                views=(
                    types.CardTemplateViewCapability(
                        name="active",
                        wire_profile="octo/v2",
                        states=("reasoning", "answering"),
                        submit_actions=(action,) if action == "reasoning_stop" else (),
                    ),
                    types.CardTemplateViewCapability(
                        name="error",
                        wire_profile="octo/v2",
                        states=("error",),
                        submit_actions=(action,) if action == "reasoning_retry" else (),
                    ),
                    types.CardTemplateViewCapability(
                        name="result",
                        wire_profile="octo/v1",
                        states=("completed", "stopped"),
                    ),
                ),
            ),
        ),
    )

    assert cards.select_reasoning_process_template(templating) is None
