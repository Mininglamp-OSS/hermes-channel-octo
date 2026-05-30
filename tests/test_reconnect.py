"""Unit tests for P1-1 reconnect hardening — dedup + cooldown + stagger."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin.adapter import (
    OctoAdapter,
    RECONNECT_STAGGER_MAX_S,
    TOKEN_REFRESH_COOLDOWN_S,
)
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    """Construct a bare adapter without going through __init__ (which needs
    a hermes PlatformConfig). Set only the fields _schedule_reconnect /
    _do_connect read so tests stay isolated."""
    a = make_bare_adapter()
    a._need_reconnect = True
    a._api_url = "https://example.test"
    a._bot_token = "tok"
    return a


# ─── Reconnect dedup ─────────────────────────────────────────────────────────


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
async def test_reconnect_reschedules_on_connect_failure():
    """When _do_connect raises, a fresh reconnect task is spawned."""
    a = _make_adapter()
    calls = 0

    async def flaky_connect():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise RuntimeError("simulated failure")

    a._do_connect = flaky_connect  # type: ignore[method-assign]

    spawned: list = []

    real_create_task = asyncio.create_task

    def capture_create_task(coro):
        # Capture so the test can await the rescheduled attempt
        task = real_create_task(coro)
        spawned.append(task)
        return task

    with patch("hermes_octo_plugin.adapter.asyncio.sleep", new=AsyncMock()), \
         patch("hermes_octo_plugin.adapter.asyncio.create_task", new=capture_create_task):
        await a._schedule_reconnect()
        # Drain the rescheduled task that was create_task'd.
        for t in spawned:
            await t

    assert calls >= 2, "failure should trigger a retry"


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
    a._last_token_refresh = 0.0  # never refreshed

    with patch("hermes_octo_plugin.adapter.api.register_bot", new=fake_register_bot), \
         patch.object(a, "_ws", None):
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
        m.robot_id = "bot"; m.owner_uid = "owner"; m.im_token = "t"; m.ws_url = "wss://e"
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
