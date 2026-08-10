"""HTTP Bot heartbeat coexisting with the WebSocket heartbeat."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin import api
from hermes_octo_plugin.adapter import HTTP_HEARTBEAT_INTERVAL_S
from tests.conftest import make_bare_adapter


@pytest.mark.asyncio
async def test_http_heartbeat_uses_independent_30_second_cadence():
    adapter = make_bare_adapter()
    adapter._connected = True
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    delays: list[float] = []

    async def one_tick(delay):
        delays.append(delay)

    async def stop_after_heartbeat(*_args):
        adapter._connected = False

    with (
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=one_tick),
        patch(
            "hermes_octo_plugin.adapter.api.send_heartbeat",
            AsyncMock(side_effect=stop_after_heartbeat),
        ) as heartbeat,
    ):
        await adapter._http_heartbeat_loop()

    assert delays == [HTTP_HEARTBEAT_INTERVAL_S]
    assert HTTP_HEARTBEAT_INTERVAL_S == 30.0
    heartbeat.assert_awaited_once()


@pytest.mark.asyncio
async def test_ws_heartbeat_keeps_ping_cadence_independent():
    adapter = make_bare_adapter()
    adapter._connected = True
    adapter._ws = MagicMock()
    adapter._ws.send = AsyncMock()
    delays: list[float] = []

    async def one_tick(delay):
        delays.append(delay)
        adapter._connected = False

    with (
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=one_tick),
        patch("hermes_octo_plugin.adapter.encode_ping_packet", return_value=b"ping"),
    ):
        await adapter._heartbeat_loop()

    assert delays == [adapter._heartbeat_interval_s]
    adapter._ws.send.assert_awaited_once_with(b"ping")


@pytest.mark.asyncio
async def test_ws_ping_is_not_blocked_by_a_slow_optional_http_heartbeat():
    adapter = make_bare_adapter()
    adapter._connected = True
    adapter._ping_max_retry = 100
    adapter._ws = MagicMock()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._http_heartbeat_task = None

    order: list[str] = []
    heartbeat_started = asyncio.Event()
    release_http = asyncio.Event()
    release_loop = asyncio.Event()
    ticks = 0

    async def one_tick(delay):
        nonlocal ticks
        if delay == HTTP_HEARTBEAT_INTERVAL_S:
            return
        ticks += 1
        if ticks > 1:
            await release_loop.wait()

    async def record_ping(_packet):
        order.append("ping")

    async def slow_heartbeat(*_args):
        order.append("http")
        heartbeat_started.set()
        await release_http.wait()

    adapter._ws.send = AsyncMock(side_effect=record_ping)
    ws_loop_task: asyncio.Task[None] | None = None
    http_loop_task: asyncio.Task[None] | None = None
    try:
        with (
            patch("hermes_octo_plugin.adapter.asyncio.sleep", new=one_tick),
            patch("hermes_octo_plugin.adapter.api.send_heartbeat", slow_heartbeat),
            patch("hermes_octo_plugin.adapter.encode_ping_packet", return_value=b"ping"),
        ):
            ws_loop_task = asyncio.create_task(adapter._heartbeat_loop())
            http_loop_task = asyncio.create_task(adapter._http_heartbeat_loop())
            await heartbeat_started.wait()
            assert order[0] == "ping"
            adapter._connected = False
            release_http.set()
            release_loop.set()
            await asyncio.gather(ws_loop_task, http_loop_task)
    finally:
        adapter._connected = False
        release_http.set()
        release_loop.set()
        for task in (ws_loop_task, http_loop_task):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (ws_loop_task, http_loop_task) if task),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_reconnect_replaces_a_live_heartbeat_task_before_starting_another():
    adapter = make_bare_adapter()
    blocker = asyncio.Event()

    async def wait_forever():
        await blocker.wait()

    old_task = asyncio.create_task(wait_forever())
    adapter._heartbeat_task = old_task
    adapter._heartbeat_loop = wait_forever
    try:
        await adapter._start_heartbeat_task()

        assert old_task.cancelled() or old_task.done()
        assert adapter._heartbeat_task is not old_task
        assert not adapter._heartbeat_task.done()
    finally:
        adapter._heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await adapter._heartbeat_task


@pytest.mark.asyncio
async def test_reconnect_replaces_a_live_receive_loop_without_retiring_new_connection():
    adapter = make_bare_adapter()
    blocker = asyncio.Event()
    adapter._connected = True
    adapter._need_reconnect = True
    adapter._ws = SimpleNamespace(recv=AsyncMock(side_effect=blocker.wait))
    old_task = asyncio.create_task(adapter._receive_loop())
    adapter._recv_task = old_task
    await asyncio.sleep(0)

    try:
        with patch.object(adapter, "_spawn_reconnect_task") as reconnect:
            await adapter._start_receive_task()
            await asyncio.sleep(0)

        assert old_task.cancelled() or old_task.done()
        assert adapter._connected is True
        assert adapter._recv_task is not old_task
        assert not adapter._recv_task.done()
        reconnect.assert_not_called()
    finally:
        adapter._recv_task.cancel()
        await adapter._recv_task


@pytest.mark.asyncio
async def test_heartbeat_404_disables_only_http_compatibility_probe():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    failure = api.OctoApiError("/v1/bot/heartbeat", status=404)

    with patch(
        "hermes_octo_plugin.adapter.api.send_heartbeat",
        AsyncMock(side_effect=failure),
    ) as heartbeat:
        await adapter._send_http_heartbeat_safe()
        await adapter._send_http_heartbeat_safe()

    assert adapter._http_heartbeat_disabled is True
    heartbeat.assert_awaited_once()
