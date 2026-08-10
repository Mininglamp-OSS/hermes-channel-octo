"""Progress-card lifecycle and concurrency contracts."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin import card_progress, cards
from hermes_octo_plugin.card_tools import TrustedOctoRoute
from hermes_octo_plugin.types import CardProfileManifest, ChannelType, SendMessageResult


class _Adapter:
    _api_url = "https://api.example.invalid"
    _bot_token = "test-token"
    _http_session = object()
    _disconnecting = False
    on_behalf_of: str | None = None

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


def test_progress_renderer_uses_only_safe_bounded_summaries() -> None:
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

    assert "config.py" in rendered.plain
    assert "MCP tool" in rendered.plain
    assert "AKIA1234567890ABCDEF" not in rendered.plain
    assert "hidden" not in rendered.plain
    assert "reasoning" not in rendered.plain.lower()


@pytest.mark.asyncio
async def test_progress_lifecycle_sends_then_edits_transient_and_final() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(return_value=SendMessageResult(message_id="progress-1"))
    edit = AsyncMock(return_value={})
    with (
        patch.object(card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
        patch.object(card_progress.api, "send_card_message", send),
        patch.object(card_progress.api, "edit_card_message", edit),
    ):
        controller.begin(
            adapter=adapter,
            route=_ROUTE,
            session_id="session-1",
            turn_id="turn-1",
        )
        await adapter.run_next()

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
    assert edit.await_count == 2
    transient = edit.await_args_list[0].kwargs
    final = edit.await_args_list[1].kwargs
    assert transient["card_seq"] == 1
    assert transient["transient"] is True
    assert "read (/tmp/a.py): complete" in transient["plain"]
    assert "bash (pytest): failed" in transient["plain"]
    assert "AKIA1234567890ABCDEF" not in transient["plain"]
    assert final["card_seq"] == 2
    assert final["transient"] is False
    assert final["plain"].startswith("Completed")
    assert controller.state_count == 0


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
async def test_progress_turns_are_isolated_and_session_cancel_drops_pending_work() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    send = AsyncMock(
        side_effect=[
            SendMessageResult(message_id="progress-1"),
            SendMessageResult(message_id="progress-2"),
        ]
    )
    with (
        patch.object(card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
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
        assert controller.state_count == 2
        controller.cancel_session("session-1")
        assert controller.state_count == 1
        await adapter.run_next()
        await adapter.run_next()

    assert send.await_count == 1
    assert send.await_args.kwargs["client_msg_no"].endswith("turn-2")


@pytest.mark.asyncio
async def test_progress_delivery_failure_is_fail_soft() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
    with (
        patch.object(card_progress.api, "get_card_profile", AsyncMock(return_value=_MANIFEST)),
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
    )
    controller.cancel_session.assert_not_called()


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

    final = edit.await_args.kwargs
    assert final["transient"] is False
    assert final["plain"].startswith("Stopped")
    assert "failed" in final["plain"]
    assert controller.state_count == 0


@pytest.mark.asyncio
async def test_progress_keeps_rendering_after_more_than_32_tool_calls() -> None:
    controller = card_progress.CardProgressController()
    adapter = _Adapter()
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
            AsyncMock(return_value=SendMessageResult(message_id="progress-1")),
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

    edit.assert_awaited_once()
    assert controller.state_count == 0
