"""Unit tests for P1-1 reconnect hardening — dedup + cooldown + stagger."""

from __future__ import annotations

import asyncio
import logging

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin.adapter import (
    OctoAdapter,
    RECONNECT_STAGGER_MAX_S,
    TOKEN_REFRESH_COOLDOWN_S,
)
from hermes_octo_plugin.protocol import PacketType
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    """Construct a bare adapter without going through __init__ (which needs
    a hermes PlatformConfig). Set only the fields _schedule_reconnect /
    _do_connect read so tests stay isolated."""
    a = make_bare_adapter()
    a._need_reconnect = True
    a._api_url = "https://example.test"
    a._bot_token = "tok"
    # Reconnect tests mock every API/WS operation and do not exercise aiohttp
    # ownership.  Seed a non-network session so direct private _do_connect()
    # calls cannot allocate a real ClientSession that only disconnect() owns.
    a._http_session = MagicMock()
    return a


# ─── Reconnect dedup ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_reentry_closes_previous_owned_http_session():
    a = _make_adapter()
    a._http_session = None
    a._reconnect_task = None
    a._heartbeat_task = None
    a._http_heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._ws = None
    a._mark_disconnected = MagicMock()
    first = MagicMock()
    first.close = AsyncMock()
    second = MagicMock()
    second.close = AsyncMock()
    a._do_connect = AsyncMock(return_value=True)  # type: ignore[method-assign]

    with patch(
        "hermes_octo_plugin.adapter.aiohttp.ClientSession",
        side_effect=[first, second],
    ):
        assert await a.connect() is True
        assert await a.connect() is True

    first.close.assert_awaited_once()
    assert a._http_session is second


@pytest.mark.asyncio
async def test_concurrent_connect_calls_never_overlap_handshakes():
    a = _make_adapter()
    a._http_session = None
    a._ws = None
    a._mark_disconnected = MagicMock()
    first_session = MagicMock()
    first_session.close = AsyncMock()
    second_session = MagicMock()
    second_session.close = AsyncMock()
    a._new_http_session = MagicMock(  # type: ignore[method-assign]
        side_effect=[first_session, second_session]
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def delayed_connect():
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered.set()
        await release.wait()
        active -= 1
        return True

    a._do_connect = delayed_connect  # type: ignore[method-assign]
    first = asyncio.create_task(a.connect())
    await entered.wait()
    second = asyncio.create_task(a.connect())
    await asyncio.sleep(0)
    assert max_active == 1
    release.set()
    assert await first is True
    assert await second is True
    assert max_active == 1
    first_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cold_connect_failure_closes_partial_transport_resources(caplog):
    a = _make_adapter()
    a._http_session = None
    a._reconnect_task = None
    a._heartbeat_task = None
    a._http_heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._ws = None
    a._mark_disconnected = MagicMock()
    session = MagicMock()
    session.close = AsyncMock()
    a._do_connect = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("Authorization=Bearer secret-connect-token")
    )

    with patch(
        "hermes_octo_plugin.adapter.aiohttp.ClientSession",
        return_value=session,
    ):
        assert await a.connect() is False

    session.close.assert_awaited_once()
    assert a._http_session is None
    assert "secret-connect-token" not in caplog.text
    assert "RuntimeError" in caplog.text



@pytest.mark.asyncio
async def test_repeated_connect_cancellation_cannot_interrupt_session_close():
    a = _make_adapter()
    a._http_session = None
    a._ws = None
    a._reconnect_task = None
    a._heartbeat_task = None
    a._http_heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._prefetch_task = None
    connect_entered = asyncio.Event()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    close_completed = asyncio.Event()
    close_cancelled = False

    session = MagicMock()

    async def protected_close():
        nonlocal close_cancelled
        close_entered.set()
        try:
            await release_close.wait()
            close_completed.set()
        except asyncio.CancelledError:
            close_cancelled = True
            raise

    session.close = protected_close
    a._new_http_session = MagicMock(return_value=session)  # type: ignore[method-assign]

    async def blocked_connect() -> bool:
        connect_entered.set()
        await asyncio.Event().wait()
        return True

    a._do_connect = blocked_connect  # type: ignore[method-assign]
    task = asyncio.create_task(a.connect())
    await connect_entered.wait()
    task.cancel()
    await close_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert close_cancelled is False
    assert close_completed.is_set()
    assert a._http_session is None


@pytest.mark.asyncio
async def test_reconnect_dedup_runs_once():
    """Concurrent _schedule_reconnect calls collapse to a single attempt."""
    a = _make_adapter()
    call_count = 0
    real_sleep = asyncio.sleep

    async def fake_do_connect():
        nonlocal call_count
        call_count += 1
        a._reconnect_attempts = 0

    a._do_connect = fake_do_connect  # type: ignore[method-assign]

    # Replace ONLY the backoff sleep (the long one inside _schedule_reconnect)
    # with a real tiny yield. The patch hits the symbol the adapter module
    # uses, so we keep a captured reference to the real sleep to avoid the
    # patch recursing into itself.
    async def short_sleep(_delay):
        await real_sleep(0)

    with patch("hermes_octo_plugin.adapter.asyncio.sleep", new=short_sleep):
        await asyncio.gather(
            a._schedule_reconnect(),
            a._schedule_reconnect(),
            a._schedule_reconnect(),
        )

    assert call_count == 1, "second/third concurrent reconnects must be deduped"
    assert a._reconnect_in_progress is False


@pytest.mark.asyncio
async def test_reconnect_dedup_clears_after_success():
    """A second reconnect attempted AFTER the first succeeded should run."""
    a = _make_adapter()
    call_count = 0

    async def fake_do_connect():
        nonlocal call_count
        call_count += 1

    a._do_connect = fake_do_connect  # type: ignore[method-assign]

    with patch("hermes_octo_plugin.adapter.asyncio.sleep", new=AsyncMock()):
        await a._schedule_reconnect()
        await a._schedule_reconnect()

    assert call_count == 2


@pytest.mark.asyncio
async def test_reconnect_skipped_when_need_reconnect_false():
    """If the adapter is being torn down, no reconnect is attempted."""
    a = _make_adapter()
    a._need_reconnect = False
    a._do_connect = AsyncMock()  # type: ignore[method-assign]

    with patch("hermes_octo_plugin.adapter.asyncio.sleep", new=AsyncMock()):
        await a._schedule_reconnect()

    a._do_connect.assert_not_called()
    assert a._reconnect_in_progress is False


@pytest.mark.asyncio
async def test_disconnect_cancels_inflight_reconnect_before_it_can_resurrect_adapter():
    a = _make_adapter()
    entered_connect = asyncio.Event()
    release_connect = asyncio.Event()
    a._heartbeat_task = None
    a._http_heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._ws = None
    a._mark_disconnected = MagicMock()

    session = MagicMock()
    session.close = AsyncMock()
    a._http_session = session

    async def delayed_connect():
        entered_connect.set()
        await release_connect.wait()
        a._connected = True
        return True

    a._do_connect = delayed_connect  # type: ignore[method-assign]

    with patch("hermes_octo_plugin.adapter.asyncio.sleep", new=AsyncMock()):
        reconnect_task = asyncio.create_task(a._schedule_reconnect())
        a._reconnect_task = reconnect_task
        await entered_connect.wait()
        try:
            await a.disconnect()
            assert reconnect_task.done()
            assert a._connected is False
            session.close.assert_awaited_once()
        finally:
            release_connect.set()
            if not reconnect_task.done():
                reconnect_task.cancel()
            await asyncio.gather(reconnect_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_while_draining_reconnect_still_finishes_disconnect():
    a = _make_adapter()
    a._connected = True
    a._heartbeat_task = None
    a._http_heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._prefetch_task = None
    a._ws = None
    a._mark_disconnected = MagicMock()
    session = MagicMock()
    session.close = AsyncMock()
    a._http_session = session
    draining = asyncio.Event()
    release = asyncio.Event()

    async def reconnect_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            draining.set()
            await release.wait()

    reconnect_task = asyncio.create_task(reconnect_cleanup())
    a._reconnect_task = reconnect_task
    disconnect_task = asyncio.create_task(a.disconnect())
    await draining.wait()

    disconnect_task.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await disconnect_task

    assert reconnect_task.done()
    assert a._connected is False
    assert a._http_session is None
    session.close.assert_awaited_once()
    a._mark_disconnected.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_stops_card_event_poller() -> None:
    a = _make_adapter()
    a._reconnect_task = None
    a._heartbeat_task = None
    a._http_heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._prefetch_task = None
    a._ws = None
    a._http_session = None
    a._mark_disconnected = MagicMock()
    poller = MagicMock()
    a._event_poller = poller
    a._event_task = None

    await a.disconnect()

    poller.stop.assert_called()


@pytest.mark.asyncio
async def test_disconnect_cancels_prefetch_before_closing_session():
    a = _make_adapter()
    a._reconnect_task = None
    a._heartbeat_task = None
    a._http_heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._ws = None
    a._mark_disconnected = MagicMock()
    session = MagicMock()
    session.close = AsyncMock()
    a._http_session = session
    started = asyncio.Event()

    async def prefetch():
        started.set()
        await asyncio.Event().wait()

    a._prefetch_task = asyncio.create_task(prefetch())
    await started.wait()
    await a.disconnect()

    assert a._prefetch_task is None
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_disconnect_cancellation_cannot_cancel_transport_cleanup():
    a = _make_adapter()
    a._reconnect_task = None
    a._heartbeat_task = None
    a._http_heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._prefetch_task = None
    a._ws = None
    a._mark_disconnected = MagicMock()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    close_completed = asyncio.Event()
    close_cancelled = False

    session = MagicMock()

    async def protected_close():
        nonlocal close_cancelled
        close_entered.set()
        try:
            await release_close.wait()
            close_completed.set()
        except asyncio.CancelledError:
            close_cancelled = True
            raise

    session.close = protected_close
    a._http_session = session

    task = asyncio.create_task(a.disconnect())
    await close_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert close_cancelled is False
    assert close_completed.is_set()
    a._mark_disconnected.assert_called_once()


@pytest.mark.asyncio
async def test_reconnect_reschedules_on_connect_failure(caplog):
    """When _do_connect raises, a fresh reconnect task is spawned."""
    a = _make_adapter()
    calls = 0

    async def flaky_connect():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise RuntimeError("SessionToken=secret-reconnect-token")

    a._do_connect = flaky_connect  # type: ignore[method-assign]

    spawned: list = []

    real_create_task = asyncio.create_task

    def capture_create_task(coro):
        # Capture so the test can await the rescheduled attempt
        task = real_create_task(coro)
        spawned.append(task)
        return task

    with (
        patch("hermes_octo_plugin.adapter.asyncio.sleep", new=AsyncMock()),
        patch(
            "hermes_octo_plugin.adapter.asyncio.create_task", new=capture_create_task
        ),
    ):
        await a._schedule_reconnect()
        # Drain the rescheduled task that was create_task'd.
        for t in spawned:
            await t

    assert calls >= 2, "failure should trigger a retry"
    assert "secret-reconnect-token" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_do_connect_prefers_configured_websocket_url():
    adapter = _make_adapter()
    adapter._ws_url = "wss://override.example/socket"
    registration = MagicMock(
        robot_id="bot",
        owner_uid="owner",
        im_token="token",
        ws_url="wss://server.example/socket",
    )
    guarded_socket = MagicMock()
    open_socket = AsyncMock(return_value=guarded_socket)
    connect = AsyncMock(side_effect=RuntimeError("stop after URL selection"))

    with (
        patch(
            "hermes_octo_plugin.adapter.api.register_bot",
            AsyncMock(return_value=registration),
        ),
        patch(
            "hermes_octo_plugin.adapter._open_guarded_websocket_socket",
            open_socket,
            create=True,
        ),
        patch("hermes_octo_plugin.adapter.websockets.connect", connect),
    ):
        with pytest.raises(RuntimeError, match="URL selection"):
            await adapter._do_connect()

    assert connect.await_args.args[0] == "wss://override.example/socket"
    open_socket.assert_awaited_once_with("wss://override.example/socket")
    assert connect.await_args.kwargs["sock"] is guarded_socket
    assert connect.await_args.kwargs["proxy"] is None
    guarded_socket.close.assert_called_once()


@pytest.mark.asyncio
async def test_registration_info_log_omits_robot_and_owner_ids(caplog):
    adapter = _make_adapter()
    robot_id = "stable-robot-id"
    owner_id = "stable-owner-id"
    registration = MagicMock(
        robot_id=robot_id,
        owner_uid=owner_id,
        im_token="token",
        ws_url="wss://server.example/socket",
    )
    caplog.set_level(logging.INFO, logger="hermes_octo_plugin.adapter")

    with (
        patch(
            "hermes_octo_plugin.adapter.api.register_bot",
            AsyncMock(return_value=registration),
        ),
        patch(
            "hermes_octo_plugin.adapter._open_guarded_websocket_socket",
            AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "hermes_octo_plugin.adapter.websockets.connect",
            AsyncMock(side_effect=RuntimeError("stop after registration")),
        ),
        pytest.raises(RuntimeError, match="stop after registration"),
    ):
        await adapter._do_connect()

    assert "Bot registered" in caplog.text
    assert robot_id not in caplog.text
    assert owner_id not in caplog.text


@pytest.mark.asyncio
async def test_guarded_socket_closes_when_cancellation_precedes_websocket_handoff():
    adapter = _make_adapter()
    registration = MagicMock(
        robot_id="bot",
        owner_uid="owner",
        im_token="token",
        ws_url="wss://server.example/socket",
    )
    guarded_socket = MagicMock()
    opened = asyncio.Event()

    async def open_socket(_url: str):
        asyncio.get_running_loop().call_soon(opened.set)
        return guarded_socket

    async def pre_handoff_connect(*_args, **_kwargs):
        await asyncio.Event().wait()

    with (
        patch(
            "hermes_octo_plugin.adapter.api.register_bot",
            AsyncMock(return_value=registration),
        ),
        patch(
            "hermes_octo_plugin.adapter._open_guarded_websocket_socket",
            open_socket,
        ),
        patch("hermes_octo_plugin.adapter.websockets.connect", pre_handoff_connect),
    ):
        task = asyncio.create_task(adapter._do_connect())
        await opened.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert adapter._ws is None
    guarded_socket.close.assert_called_once()


@pytest.mark.asyncio
async def test_successful_websocket_handoff_does_not_close_guarded_socket():
    adapter = _make_adapter()
    adapter._cache_cleanup_task = MagicMock()
    adapter._prefetch_task = MagicMock()
    registration = MagicMock(
        robot_id="bot",
        owner_uid="owner",
        im_token="token",
        ws_url="wss://server.example/socket",
    )
    guarded_socket = MagicMock()
    websocket = MagicMock()
    websocket.send = AsyncMock()
    websocket.recv = AsyncMock(return_value=b"connack")
    connack = MagicMock(
        reason_code=1,
        server_key="c2VydmVyLXB1YmxpYy1rZXk=",
        salt="0123456789abcdef",
        server_version=4,
    )

    with (
        patch(
            "hermes_octo_plugin.adapter.api.register_bot",
            AsyncMock(return_value=registration),
        ),
        patch(
            "hermes_octo_plugin.adapter._open_guarded_websocket_socket",
            AsyncMock(return_value=guarded_socket),
        ),
        patch(
            "hermes_octo_plugin.adapter.websockets.connect",
            AsyncMock(return_value=websocket),
        ),
        patch(
            "hermes_octo_plugin.adapter.generate_keypair",
            return_value=(MagicMock(), b"client-public-key"),
        ),
        patch(
            "hermes_octo_plugin.adapter.compute_shared_secret",
            return_value=b"shared-secret",
        ),
        patch(
            "hermes_octo_plugin.adapter.derive_aes_key",
            return_value=b"derived-aes-key",
        ),
        patch(
            "hermes_octo_plugin.adapter.try_unpack_one",
            return_value=(b"connack", bytearray()),
        ),
        patch(
            "hermes_octo_plugin.adapter.decode_packet",
            return_value=(PacketType.CONNACK, connack),
        ),
        patch.object(adapter, "_start_heartbeat_task", AsyncMock()),
        patch.object(adapter, "_start_receive_task", AsyncMock()),
        patch.object(adapter, "_start_card_event_poller"),
        patch.object(adapter, "_mark_connected"),
    ):
        assert await adapter._do_connect() is True

    assert adapter._ws is websocket
    guarded_socket.close.assert_not_called()


# ─── Token refresh cooldown ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_token_refresh_skipped_within_cooldown():
    """When two failures happen back-to-back, only the first triggers a forced
    token refresh; subsequent attempts within TOKEN_REFRESH_COOLDOWN_S reuse
    the cached token (force_refresh=False)."""
    a = _make_adapter()
    refresh_calls: list[bool] = []

    async def fake_register_bot(_session, _api_url, _bot_token, *, force_refresh=False):
        refresh_calls.append(force_refresh)
        m = MagicMock()
        m.robot_id = "bot"
        m.owner_uid = "owner"
        m.im_token = "imtok"
        m.ws_url = "wss://example"
        return m

    # First attempt: reconnect_attempts increases from 0 to 1 inside
    # _schedule_reconnect, so the force_refresh decision in _do_connect sees
    # attempts>0 and (cooldown elapsed) → forces refresh. To exercise this
    # cleanly, set attempts=1 directly and trigger _do_connect logic.
    a._reconnect_attempts = 1
    # Use a monotonic-relative value so cooldown check works on systems where
    # time.monotonic() may be < TOKEN_REFRESH_COOLDOWN_S after process start
    # (e.g. fresh CI containers).
    a._last_token_refresh = time.monotonic() - (TOKEN_REFRESH_COOLDOWN_S + 10)

    with (
        patch("hermes_octo_plugin.adapter.api.register_bot", new=fake_register_bot),
        patch.object(a, "_ws", None),
    ):
        # Stop _do_connect at the registration call — anything beyond would
        # need a real WS. We only care about the force_refresh decision.
        try:
            await a._do_connect()
        except Exception:
            pass

        # Second attempt < cooldown: should NOT force refresh
        a._reconnect_attempts = 2
        try:
            await a._do_connect()
        except Exception:
            pass

    assert refresh_calls[0] is True, "first failure-driven attempt should force refresh"
    assert refresh_calls[1] is False, "second attempt within cooldown should NOT force"


@pytest.mark.asyncio
async def test_token_refresh_resumes_after_cooldown():
    """When elapsed > TOKEN_REFRESH_COOLDOWN_S, force_refresh fires again."""
    a = _make_adapter()
    refresh_calls: list[bool] = []

    async def fake_register_bot(_session, _api_url, _bot_token, *, force_refresh=False):
        refresh_calls.append(force_refresh)
        m = MagicMock()
        m.robot_id = "bot"
        m.owner_uid = "owner"
        m.im_token = "t"
        m.ws_url = "wss://e"
        return m

    a._reconnect_attempts = 1
    # Pretend last refresh happened well past the cooldown.
    a._last_token_refresh = time.monotonic() - (TOKEN_REFRESH_COOLDOWN_S + 10)

    with patch("hermes_octo_plugin.adapter.api.register_bot", new=fake_register_bot):
        try:
            await a._do_connect()
        except Exception:
            pass

    assert refresh_calls == [True]


# ─── Stagger ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconnect_stagger_adds_random_offset():
    """Captured sleep delay should always include up to RECONNECT_STAGGER_MAX_S
    of extra random offset on top of the exponential backoff."""
    a = _make_adapter()
    a._do_connect = AsyncMock()  # type: ignore[method-assign]
    captured: list[float] = []

    async def capture_sleep(delay):
        captured.append(delay)

    with patch("hermes_octo_plugin.adapter.asyncio.sleep", new=capture_sleep):
        # Pin random to a deterministic value so we can predict the upper bound
        with patch("hermes_octo_plugin.adapter.random.random", return_value=1.0):
            await a._schedule_reconnect()

    assert len(captured) == 1
    # Base exponential: 3.0 * 2^0 = 3.0; jitter factor 0.75 + 1.0*0.5 = 1.25
    # Stagger: 1.0 * RECONNECT_STAGGER_MAX_S = RECONNECT_STAGGER_MAX_S
    expected_max = 3.0 * 1.25 + RECONNECT_STAGGER_MAX_S
    assert captured[0] == pytest.approx(expected_max, rel=0.01)
