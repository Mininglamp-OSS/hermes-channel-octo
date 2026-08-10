"""Fail-soft Octo progress cards driven by Hermes lifecycle hooks."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import api, cards
from .agent_tools import _resolve_adapter
from .api import edit_template_card_message, send_template_card_message
from .card_tools import (
    TrustedOctoRoute,
    _get_card_profile,
    _profile_enabled,
    _trusted_route,
)
from .types import CARD_PROFILE_V1

logger = logging.getLogger(__name__)
_MAX_PROGRESS_TOOL_ENTRIES = 32
_CARD_AUTHORING_TOOLS = frozenset({
    "octo_send_display_card",
    "octo_send_interactive_card",
})



_ProgressKey = tuple[str, str, int]


@dataclass
class _ProgressTool:
    tool_call_id: str
    tool_name: str
    label: str
    summary: str
    status: str = "running"
    error: str = ""
    thought: str = ""
    result_summary: str = ""
    duration_ms: int | None = None
    started_at: float = field(default_factory=time.monotonic)


@dataclass
class _ProgressTurn:
    adapter: Any
    route: TrustedOctoRoute
    session_id: str
    turn_id: str
    segment_no: int
    tools: OrderedDict[str, _ProgressTool] = field(default_factory=OrderedDict)
    fallback_ids: dict[str, list[str]] = field(default_factory=dict)
    revision: int = 0
    delivered_revision: int = -1
    message_id: str | None = None
    card_seq: int = 0
    scheduled: bool = False
    send_started: bool = False
    final: bool = False
    final_phase: str = "completed"
    phase: str = "thinking"
    started_at: float = field(default_factory=time.monotonic)
    delivery_mode: str | None = None
    template_ref: dict[str, str] | None = None
    capabilities: cards.CardCapabilities | None = None


class CardProgressController:
    """Thread-safe turn state with serialized network drains on the gateway loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[_ProgressKey, _ProgressTurn] = {}

    @property
    def state_count(self) -> int:
        with self._lock:
            return len(self._states)

    def begin(
        self,
        *,
        adapter: Any,
        route: TrustedOctoRoute,
        session_id: str,
        turn_id: str,
    ) -> None:
        if not session_id or not turn_id or adapter.on_behalf_of is not None:
            return
        with self._lock:
            matching = [
                key
                for key in self._states
                if key[0] == session_id and key[1] == turn_id
            ]
            if any(not self._states[key].final for key in matching):
                return
            segment_no = max((key[2] for key in matching), default=-1) + 1
            key = (session_id, turn_id, segment_no)
            self._states[key] = _ProgressTurn(
                adapter=adapter,
                route=route,
                session_id=session_id,
                turn_id=turn_id,
                segment_no=segment_no,
            )
            self._schedule_locked(key)

    def model_started(
        self,
        *,
        session_id: str,
        turn_id: str,
        model_call_id: str,
    ) -> None:
        with self._lock:
            key = self._find_key_locked(session_id, turn_id)
            state = self._states.get(key) if key is not None else None
            if state is None or state.final:
                return
            assert key is not None
            call_id = model_call_id or f"model-{len(state.tools) + 1}"
            step_id = f"__thinking__:{call_id}"
            state.tools[step_id] = _ProgressTool(
                tool_call_id=step_id,
                tool_name="__thinking__",
                summary="",
                label=cards.localized_tool_label("__thinking__"),
            )
            state.phase = "thinking"
            state.revision += 1

    def model_finished(
        self,
        *,
        session_id: str,
        turn_id: str,
        model_call_id: str,
        thought: str = "",
        duration_ms: int | None = None,
        answering: bool = False,
        failed: bool = False,
    ) -> None:
        with self._lock:
            key = self._find_key_locked(session_id, turn_id)
            state = self._states.get(key) if key is not None else None
            if state is None or state.final:
                return
            assert key is not None
            step_id = f"__thinking__:{model_call_id}" if model_call_id else ""
            tool = state.tools.get(step_id)
            if tool is None:
                tool = next(
                    (
                        candidate
                        for candidate in reversed(state.tools.values())
                        if candidate.tool_name == "__thinking__"
                        and candidate.status == "running"
                    ),
                    None,
                )
            if tool is None:
                return
            tool.status = "failed" if failed else "complete"
            tool.thought = thought
            if (
                isinstance(duration_ms, int)
                and not isinstance(duration_ms, bool)
                and duration_ms >= 0
            ):
                tool.duration_ms = duration_ms
            else:
                tool.duration_ms = max(
                    0,
                    int((time.monotonic() - tool.started_at) * 1_000),
                )
            state.phase = "answering" if answering else state.phase
            state.revision += 1
            self._schedule_locked(key)

    def tool_started(
        self,
        *,
        session_id: str,
        turn_id: str,
        tool_call_id: str,
        tool_name: str,
        args: object,
    ) -> None:
        with self._lock:
            key = self._find_key_locked(session_id, turn_id)
            state = self._states.get(key) if key is not None else None
            if state is None or state.final:
                return
            assert key is not None
            if tool_name in _CARD_AUTHORING_TOOLS:
                has_real_tool = any(
                    candidate.tool_name != "__thinking__"
                    for candidate in state.tools.values()
                )
                for step_id, candidate in list(state.tools.items()):
                    if candidate.tool_name != "__thinking__":
                        continue
                    candidate.status = "complete"
                    candidate.duration_ms = max(
                        0,
                        int((time.monotonic() - candidate.started_at) * 1_000),
                    )
                    if not has_real_tool:
                        state.tools.pop(step_id, None)
                state.revision += 1
                return
            key_label = cards.safe_tool_label(tool_name)
            display_label = cards.localized_tool_label(tool_name, args)
            summary = cards.summarize_tool_params(tool_name, args)
            call_id = tool_call_id or f"{key_label}:{len(state.tools) + 1}"
            state.tools[call_id] = _ProgressTool(
                tool_call_id=call_id,
                tool_name=tool_name or key_label,
                summary=summary,
                label=display_label,
            )
            state.phase = "tool"
            if not tool_call_id:
                state.fallback_ids.setdefault(key_label, []).append(call_id)
            state.revision += 1
            self._schedule_locked(key)

    def tool_finished(
        self,
        *,
        session_id: str,
        turn_id: str,
        tool_call_id: str,
        tool_name: str,
        status: str,
        error: object = "",
        duration_ms: int | None = None,
        result: object = None,
    ) -> None:
        with self._lock:
            key = self._find_key_locked(session_id, turn_id)
            state = self._states.get(key) if key is not None else None
            if state is None or state.final:
                return
            assert key is not None
            call_id = tool_call_id
            if not call_id:
                label = cards.safe_tool_label(tool_name)
                pending = state.fallback_ids.get(label)
                if pending:
                    call_id = pending.pop()
                    if not pending:
                        state.fallback_ids.pop(label, None)
            tool = state.tools.get(call_id)
            if tool is None:
                return
            tool.status = (
                "failed"
                if status in {"cancelled", "error", "blocked", "failed"}
                else "complete"
            )
            tool.error = (
                cards.sanitize_error_text(error) if tool.status == "failed" else ""
            )
            tool.result_summary = (
                ""
                if tool.status == "failed"
                else cards.summarize_tool_result(tool.tool_name, result)
            )
            if (
                isinstance(duration_ms, int)
                and not isinstance(duration_ms, bool)
                and duration_ms >= 0
            ):
                tool.duration_ms = duration_ms
            else:
                tool.duration_ms = max(
                    0,
                    int((time.monotonic() - tool.started_at) * 1_000),
                )
            state.revision += 1
            self._schedule_locked(key)

    def complete(
        self,
        *,
        session_id: str,
        turn_id: str,
        failed: bool = False,
        terminal_phase: str | None = None,
    ) -> None:
        with self._lock:
            key = self._find_key_locked(session_id, turn_id)
            state = self._states.get(key) if key is not None else None
            if state is None:
                return
            assert key is not None
            state.final = True
            state.final_phase = terminal_phase or ("stopped" if failed else "completed")
            state.phase = state.final_phase
            now = time.monotonic()
            for tool in state.tools.values():
                if tool.status == "running":
                    tool.status = "failed" if failed or state.final_phase != "completed" else "complete"
                    tool.duration_ms = max(
                        0,
                        int((now - tool.started_at) * 1_000),
                    )
            state.revision += 1
            self._schedule_locked(key)

    def rollover_for_inbound(
        self,
        *,
        chat_id: str,
        requester_uid: str,
    ) -> None:
        """Stop the visible segment so the next tool starts a card at the bottom."""
        if not chat_id or not requester_uid:
            return
        with self._lock:
            candidates = [
                (key, state)
                for key, state in self._states.items()
                if not state.final
                and state.route.chat_id == chat_id
                and state.route.requester_uid == requester_uid
            ]
            if len(candidates) != 1:
                return
            old_key, state = candidates[0]
            has_real_tool = any(
                tool.tool_name != "__thinking__" for tool in state.tools.values()
            )
            if not has_real_tool and state.message_id is None:
                return
            next_segment = state.segment_no + 1
            new_key = (state.session_id, state.turn_id, next_segment)
            self._states[new_key] = _ProgressTurn(
                adapter=state.adapter,
                route=state.route,
                session_id=state.session_id,
                turn_id=state.turn_id,
                segment_no=next_segment,
                delivery_mode=state.delivery_mode,
                template_ref=state.template_ref,
                capabilities=state.capabilities,
            )
            if state.message_id is None and not state.send_started:
                self._states.pop(old_key, None)
                return
            state.final = True
            state.final_phase = "stopped"
            state.phase = "stopped"
            now = time.monotonic()
            for tool in state.tools.values():
                if tool.status == "running":
                    tool.status = "failed"
                    tool.duration_ms = max(
                        0,
                        int((now - tool.started_at) * 1_000),
                    )
            state.revision += 1
            self._schedule_locked(old_key)

    def cancel_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            for key in [key for key in self._states if key[0] == session_id]:
                self._states.pop(key, None)

    def cancel_adapter(self, adapter: object) -> None:
        with self._lock:
            for key in [
                key for key, state in self._states.items() if state.adapter is adapter
            ]:
                self._states.pop(key, None)

    def cancel_all(self) -> None:
        with self._lock:
            self._states.clear()

    def _find_key_locked(
        self,
        session_id: str,
        turn_id: str,
    ) -> _ProgressKey | None:
        if not session_id:
            return None
        candidates = [
            key
            for key, state in self._states.items()
            if key[0] == session_id and not state.final
        ]
        if turn_id:
            exact = [key for key in candidates if key[1] == turn_id]
            if exact:
                return max(exact, key=lambda key: key[2])
        logical_turns = {(key[0], key[1]) for key in candidates}
        if len(logical_turns) != 1:
            return None
        return max(candidates, key=lambda key: key[2], default=None)

    def _schedule_locked(self, key: _ProgressKey) -> None:
        state = self._states.get(key)
        if state is None or state.scheduled:
            return
        has_real_tool = any(
            tool.tool_name != "__thinking__" for tool in state.tools.values()
        )
        if not has_real_tool:
            if state.final:
                self._states.pop(key, None)
            return
        state.scheduled = True
        try:
            scheduled = state.adapter._schedule_card_progress(lambda: self._drain(key))
        except Exception:
            scheduled = False
        if not scheduled:
            state.scheduled = False
            self._states.pop(key, None)

    @staticmethod
    def _tool_snapshot(state: _ProgressTurn) -> list[dict[str, object]]:
        return [
            {
                "tool_call_id": tool.tool_call_id,
                "tool_name": tool.tool_name,
                "label": tool.label,
                "summary": tool.summary,
                "status": tool.status,
                "error": tool.error,
                "thought": tool.thought,
                "result_summary": tool.result_summary,
                "duration_ms": tool.duration_ms,
            }
            for tool in list(state.tools.values())[-_MAX_PROGRESS_TOOL_ENTRIES:]
        ]

    async def _drain(self, key: _ProgressKey) -> None:
        while True:
            with self._lock:
                state = self._states.get(key)
                if state is None:
                    return
                adapter = state.adapter
                if adapter._disconnecting or adapter._http_session is None:
                    self._states.pop(key, None)
                    return
                revision = state.revision
                message_id = state.message_id
                final = state.final
                phase = state.final_phase if final else state.phase
                tools = self._tool_snapshot(state)
                next_seq = state.card_seq + 1
                delivery_mode = state.delivery_mode
                template_ref = state.template_ref
                capabilities = state.capabilities
                elapsed_ms = max(
                    0,
                    int((time.monotonic() - state.started_at) * 1_000),
                )
                reasoning_id = (
                    f"{state.session_id}:{state.turn_id}:{state.segment_no}"
                )
            try:
                if delivery_mode is None:
                    manifest = await _get_card_profile(
                        adapter,
                        adapter._http_session,
                    )
                    selected = (
                        cards.select_reasoning_process_template(
                            manifest.templating
                        )
                        if adapter.progress_card_renderer == "registry"
                        else None
                    )
                    enabled = (
                        manifest.available and manifest.enabled
                        if selected is not None
                        else _profile_enabled(manifest, CARD_PROFILE_V1)
                    )
                    if not enabled:
                        with self._lock:
                            self._states.pop(key, None)
                        return
                    with self._lock:
                        current = self._states.get(key)
                        if current is None or current is not state:
                            return
                        current.template_ref = selected
                        current.delivery_mode = (
                            "registry" if selected is not None else "card"
                        )
                        current.capabilities = cards.derive_card_capabilities(manifest)
                    continue

                if not tools:
                    with self._lock:
                        current = self._states.get(key)
                        if current is None or current is not state:
                            return
                        current.delivered_revision = revision
                        current.scheduled = False
                        if current.final:
                            self._states.pop(key, None)
                    return

                wire_data = (
                    cards.build_reasoning_process_wire_data(
                        phase=phase,
                        tools=tools,
                        elapsed_ms=elapsed_ms,
                        reasoning_id=reasoning_id,
                    )
                    if delivery_mode == "registry"
                    else None
                )
                if delivery_mode == "registry" and wire_data is None:
                    with self._lock:
                        current = self._states.get(key)
                        if current is None or current is not state:
                            return
                        current.delivered_revision = revision
                        current.scheduled = False
                        if current.final:
                            self._states.pop(key, None)
                    return

                if message_id is None:
                    with self._lock:
                        current = self._states.get(key)
                        if current is None or current is not state:
                            return
                        current.send_started = True
                    if delivery_mode == "registry":
                        assert template_ref is not None
                        assert wire_data is not None
                        result = await send_template_card_message(
                            adapter._http_session,
                            adapter._api_url,
                            adapter._bot_token,
                            channel_id=state.route.channel_id,
                            channel_type=state.route.channel_type,
                            template_ref=template_ref,
                            state=str(wire_data["state"]),
                            data=wire_data,
                            client_msg_no=(
                                f"card-progress:{state.turn_id}:{state.segment_no}"
                            )[:128],
                        )
                    else:
                        rendered = cards.build_agent_progress_card(
                            phase=phase,
                            tools=tools,
                            elapsed_ms=elapsed_ms,
                            reasoning_id=reasoning_id,
                            reasoning_visible=True,
                            capabilities=capabilities,
                        )
                        result = await api.send_card_message(
                            adapter._http_session,
                            adapter._api_url,
                            adapter._bot_token,
                            channel_id=state.route.channel_id,
                            channel_type=state.route.channel_type,
                            card=rendered.card,
                            plain=rendered.plain,
                            client_msg_no=(
                                f"card-progress:{state.turn_id}:{state.segment_no}"
                            )[:128],
                            profile=CARD_PROFILE_V1,
                        )
                    with self._lock:
                        current = self._states.get(key)
                        if current is None or current is not state:
                            return
                        current.send_started = False
                        current.message_id = result.message_id
                        current.delivered_revision = revision
                        if current.revision == revision:
                            current.scheduled = False
                            if current.final:
                                self._states.pop(key, None)
                            return
                    continue

                if delivery_mode == "registry":
                    assert template_ref is not None
                    assert wire_data is not None
                    delivered = False
                    for attempt in range(3):
                        try:
                            await edit_template_card_message(
                                adapter._http_session,
                                adapter._api_url,
                                adapter._bot_token,
                                channel_id=state.route.channel_id,
                                channel_type=state.route.channel_type,
                                message_id=message_id,
                                template_ref=template_ref,
                                state=str(wire_data["state"]),
                                data=wire_data,
                                card_seq=next_seq,
                                transient=not final,
                            )
                            delivered = True
                            break
                        except api.OctoApiError:
                            if attempt < 2:
                                await asyncio.sleep(0.1 * (2**attempt))
                    if not delivered:
                        logger.warning("[Octo] progress card edit retries exhausted")
                        with self._lock:
                            current = self._states.get(key)
                            if current is not None and current is state:
                                current.card_seq = next_seq
                                current.delivered_revision = revision
                                current.scheduled = False
                        return
                else:
                    rendered = cards.build_agent_progress_card(
                        phase=phase,
                        tools=tools,
                        elapsed_ms=elapsed_ms,
                        reasoning_id=reasoning_id,
                        reasoning_visible=True,
                        capabilities=capabilities,
                    )
                    await api.edit_card_message(
                        adapter._http_session,
                        adapter._api_url,
                        adapter._bot_token,
                        channel_id=state.route.channel_id,
                        channel_type=state.route.channel_type,
                        message_id=message_id,
                        card=rendered.card,
                        card_seq=next_seq,
                        plain=rendered.plain,
                        transient=not final,
                        profile=CARD_PROFILE_V1,
                    )
            except Exception:
                logger.warning("[Octo] progress card update failed", exc_info=True)
                with self._lock:
                    self._states.pop(key, None)
                return

            with self._lock:
                current = self._states.get(key)
                if current is None or current is not state:
                    return
                current.card_seq = next_seq
                current.delivered_revision = revision
                if current.revision == revision:
                    current.scheduled = False
                    if current.final:
                        self._states.pop(key, None)
                    return


_CONTROLLER = CardProgressController()


def _turn_identity(kwargs: dict[str, Any]) -> tuple[str, str]:
    session_id = str(kwargs.get("session_id") or "")
    turn_id = str(kwargs.get("turn_id") or kwargs.get("task_id") or "")
    return session_id, turn_id


def _reasoning_summaries_enabled() -> bool:
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:
        return False
    try:
        from gateway.display_config import resolve_display_setting

        return bool(
            resolve_display_setting(
                config,
                "octo",
                "show_reasoning",
                False,
            )
        )
    except Exception:
        display = config.get("display")
        if not isinstance(display, Mapping):
            return False
        platforms = display.get("platforms")
        if isinstance(platforms, Mapping):
            octo = platforms.get("octo")
            if isinstance(octo, Mapping) and isinstance(
                octo.get("show_reasoning"),
                bool,
            ):
                return bool(octo["show_reasoning"])
        return bool(display.get("show_reasoning") is True)


def _message_field(message: object, name: str) -> object:
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def _summary_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Mapping):
            text = item.get("text") or item.get("summary")
            if isinstance(text, str):
                parts.append(text)
    return " ".join(parts)


def _public_reasoning_summary(message: object) -> str:
    explicit = _message_field(message, "reasoning_summary")
    candidates = [_summary_text(explicit)]
    for field_name in ("reasoning_details", "codex_reasoning_items"):
        details = _message_field(message, field_name)
        if not isinstance(details, list):
            continue
        for item in details:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type") or "").lower()
            if field_name == "codex_reasoning_items":
                if item_type == "reasoning":
                    candidates.append(_summary_text(item.get("summary")))
                continue
            if "summary" in item_type:
                candidates.append(
                    _summary_text(item.get("summary") or item.get("text"))
                )
    joined = " ".join(candidate.strip() for candidate in candidates if candidate)
    return cards.sanitize_reasoning_thought(joined) if joined else ""


def on_octo_inbound_message(*, chat_id: str, requester_uid: str) -> None:
    """Roll progress before Hermes' base adapter diverts a busy follow-up."""
    _CONTROLLER.rollover_for_inbound(
        chat_id=chat_id,
        requester_uid=requester_uid,
    )


def _assistant_is_answering(message: object) -> bool:
    content = _message_field(message, "content")
    tool_calls = _message_field(message, "tool_calls")
    return bool(content) and not tool_calls


def on_pre_llm_call(**kwargs: Any) -> None:
    if str(kwargs.get("platform") or "").strip().lower() != "octo":
        return
    adapter = _resolve_adapter()
    if adapter is None:
        return
    route = _trusted_route(adapter, require_session_key=False)
    if route is None:
        return
    session_id, turn_id = _turn_identity(kwargs)
    _CONTROLLER.begin(
        adapter=adapter,
        route=route,
        session_id=session_id,
        turn_id=turn_id,
    )


def on_pre_api_request(**kwargs: Any) -> None:
    if str(kwargs.get("platform") or "").strip().lower() != "octo":
        return
    session_id, turn_id = _turn_identity(kwargs)
    _CONTROLLER.model_started(
        session_id=session_id,
        turn_id=turn_id,
        model_call_id=str(kwargs.get("api_request_id") or ""),
    )


def _api_duration_ms(value: object) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return int(value * 1_000)


def on_post_api_request(**kwargs: Any) -> None:
    if str(kwargs.get("platform") or "").strip().lower() != "octo":
        return
    session_id, turn_id = _turn_identity(kwargs)
    message = kwargs.get("assistant_message")
    _CONTROLLER.model_finished(
        session_id=session_id,
        turn_id=turn_id,
        model_call_id=str(kwargs.get("api_request_id") or ""),
        thought=(
            _public_reasoning_summary(message) if _reasoning_summaries_enabled() else ""
        ),
        duration_ms=_api_duration_ms(kwargs.get("api_duration")),
        answering=_assistant_is_answering(message),
        failed=False,
    )


def on_pre_tool_call(**kwargs: Any) -> None:
    session_id, turn_id = _turn_identity(kwargs)
    _CONTROLLER.tool_started(
        session_id=session_id,
        turn_id=turn_id,
        tool_call_id=str(kwargs.get("tool_call_id") or ""),
        tool_name=str(kwargs.get("tool_name") or ""),
        args=kwargs.get("args"),
    )


def on_post_tool_call(**kwargs: Any) -> None:
    session_id, turn_id = _turn_identity(kwargs)
    _CONTROLLER.tool_finished(
        session_id=session_id,
        turn_id=turn_id,
        tool_call_id=str(kwargs.get("tool_call_id") or ""),
        tool_name=str(kwargs.get("tool_name") or ""),
        status=str(kwargs.get("status") or "ok"),
        error=kwargs.get("error_message") or kwargs.get("error_type") or "",
        duration_ms=kwargs.get("duration_ms"),
        result=kwargs.get("result"),
    )


def on_post_llm_call(**_kwargs: Any) -> None:
    """Wait for ``on_session_end``, which carries authoritative turn status."""


def on_session_end(**kwargs: Any) -> None:
    session_id, turn_id = _turn_identity(kwargs)
    if session_id and turn_id:
        if kwargs.get("completed"):
            terminal_phase = "completed"
        elif kwargs.get("interrupted"):
            terminal_phase = "stopped"
        else:
            terminal_phase = "error"
        _CONTROLLER.complete(
            session_id=session_id,
            turn_id=turn_id,
            failed=terminal_phase != "completed",
            terminal_phase=terminal_phase,
        )
    elif session_id:
        _CONTROLLER.cancel_session(session_id)


def cancel_adapter_progress(adapter: object) -> None:
    _CONTROLLER.cancel_adapter(adapter)
