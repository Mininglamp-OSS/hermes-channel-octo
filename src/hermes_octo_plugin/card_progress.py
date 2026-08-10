"""Fail-soft Octo progress cards driven by Hermes lifecycle hooks."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from . import api, cards
from .agent_tools import _resolve_adapter
from .card_tools import (
    TrustedOctoRoute,
    _get_card_profile,
    _profile_enabled,
    _trusted_route,
)
from .types import CARD_PROFILE_V1

logger = logging.getLogger(__name__)
_MAX_PROGRESS_TOOL_ENTRIES = 32


@dataclass
class _ProgressTool:
    tool_call_id: str
    tool_name: str
    summary: str
    status: str = "running"
    error: str = ""


@dataclass
class _ProgressTurn:
    adapter: Any
    route: TrustedOctoRoute
    session_id: str
    turn_id: str
    tools: OrderedDict[str, _ProgressTool] = field(default_factory=OrderedDict)
    fallback_ids: dict[str, list[str]] = field(default_factory=dict)
    revision: int = 0
    delivered_revision: int = -1
    message_id: str | None = None
    card_seq: int = 0
    scheduled: bool = False
    final: bool = False
    final_phase: str = "completed"


class CardProgressController:
    """Thread-safe turn state with serialized network drains on the gateway loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[tuple[str, str], _ProgressTurn] = {}

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
        key = (session_id, turn_id)
        with self._lock:
            self._states[key] = _ProgressTurn(
                adapter=adapter,
                route=route,
                session_id=session_id,
                turn_id=turn_id,
            )
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
            label = cards.safe_tool_label(tool_name)
            summary = cards.summarize_tool_params(tool_name, args)
            call_id = tool_call_id or f"{label}:{len(state.tools) + 1}"
            state.tools[call_id] = _ProgressTool(
                tool_call_id=call_id,
                tool_name=label,
                summary=summary,
            )
            if not tool_call_id:
                state.fallback_ids.setdefault(label, []).append(call_id)
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
    ) -> None:
        with self._lock:
            key = self._find_key_locked(session_id, turn_id)
            state = self._states.get(key) if key is not None else None
            if state is None or state.final:
                return
            call_id = tool_call_id
            if not call_id:
                label = cards.safe_tool_label(tool_name)
                pending = state.fallback_ids.get(label)
                if pending:
                    call_id = pending.pop(0)
                    if not pending:
                        state.fallback_ids.pop(label, None)
            tool = state.tools.get(call_id)
            if tool is None:
                return
            tool.status = "failed" if status in {"error", "blocked", "failed"} else "complete"
            tool.error = cards.sanitize_error_text(error) if tool.status == "failed" else ""
            state.revision += 1
            self._schedule_locked(key)

    def complete(
        self,
        *,
        session_id: str,
        turn_id: str,
        failed: bool = False,
    ) -> None:
        with self._lock:
            key = self._find_key_locked(session_id, turn_id)
            state = self._states.get(key) if key is not None else None
            if state is None:
                return
            state.final = True
            state.final_phase = "failed" if failed else "completed"
            if failed:
                for tool in state.tools.values():
                    if tool.status == "running":
                        tool.status = "failed"
            state.revision += 1
            self._schedule_locked(key)

    def cancel_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            for key in [key for key in self._states if key[0] == session_id]:
                self._states.pop(key, None)

    def cancel_adapter(self, adapter: object) -> None:
        with self._lock:
            for key in [
                key
                for key, state in self._states.items()
                if state.adapter is adapter
            ]:
                self._states.pop(key, None)

    def cancel_all(self) -> None:
        with self._lock:
            self._states.clear()

    def _find_key_locked(
        self,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, str] | None:
        if not session_id:
            return None
        if turn_id:
            key = (session_id, turn_id)
            return key if key in self._states else None
        candidates = [key for key in self._states if key[0] == session_id]
        return candidates[0] if len(candidates) == 1 else None

    def _schedule_locked(self, key: tuple[str, str]) -> None:
        state = self._states.get(key)
        if state is None or state.scheduled:
            return
        state.scheduled = True
        try:
            scheduled = state.adapter._schedule_card_progress(
                lambda: self._drain(key)
            )
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
                "summary": tool.summary,
                "status": tool.status,
                "error": tool.error,
            }
            for tool in list(state.tools.values())[-_MAX_PROGRESS_TOOL_ENTRIES:]
        ]

    async def _drain(self, key: tuple[str, str]) -> None:
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
                final_phase = state.final_phase
                tools = self._tool_snapshot(state)
                next_seq = state.card_seq + 1
            try:
                if message_id is None:
                    manifest = await _get_card_profile(
                        adapter,
                        adapter._http_session,
                    )
                    if not _profile_enabled(manifest, CARD_PROFILE_V1):
                        with self._lock:
                            self._states.pop(key, None)
                        return
                    rendered = cards.build_progress_card(
                        phase="starting",
                        capabilities=cards.derive_card_capabilities(manifest),
                    )
                    result = await api.send_card_message(
                        adapter._http_session,
                        adapter._api_url,
                        adapter._bot_token,
                        channel_id=state.route.channel_id,
                        channel_type=state.route.channel_type,
                        card=rendered.card,
                        plain=rendered.plain,
                        client_msg_no=f"card-progress:{state.turn_id}"[:128],
                        profile=CARD_PROFILE_V1,
                    )
                    with self._lock:
                        current = self._states.get(key)
                        if current is not state:
                            return
                        current.message_id = result.message_id
                        current.delivered_revision = revision
                        if current.revision == revision and not current.final:
                            current.scheduled = False
                            return
                    continue

                rendered = cards.build_progress_card(
                    phase=final_phase if final else "running",
                    tools=tools,
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
                if current is not state:
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
    )


def on_post_llm_call(**_kwargs: Any) -> None:
    """Wait for ``on_session_end``, which carries authoritative turn status."""


def on_session_end(**kwargs: Any) -> None:
    session_id, turn_id = _turn_identity(kwargs)
    if session_id and turn_id:
        _CONTROLLER.complete(
            session_id=session_id,
            turn_id=turn_id,
            failed=bool(
                kwargs.get("failed")
                or kwargs.get("interrupted")
                or kwargs.get("completed") is False
            ),
        )
    elif session_id:
        _CONTROLLER.cancel_session(session_id)


def cancel_adapter_progress(adapter: object) -> None:
    _CONTROLLER.cancel_adapter(adapter)
