"""Unit tests for P1-2 cache cleanup (4h activity-based LRU)."""

from __future__ import annotations

import time

from hermes_octo_plugin.adapter import (
    CACHE_MAX_AGE_S,
    OctoAdapter,
)
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    return make_bare_adapter()


def test_touch_cache_records_activity():
    a = _make_adapter()
    a._touch_cache("ch1")
    assert "ch1" in a._cache_activity
    assert a._cache_activity["ch1"] > 0


def test_touch_cache_ignores_empty_id():
    a = _make_adapter()
    a._touch_cache("")
    assert a._cache_activity == {}


def test_cleanup_evicts_stale_channel_entries():
    a = _make_adapter()
    # Seed two channels: one stale, one fresh.
    now = time.monotonic()
    a._cache_activity["stale_ch"] = now - (CACHE_MAX_AGE_S + 60)
    a._cache_activity["fresh_ch"] = now - 10

    # Populate per-channel state for both.
    for ch in ("stale_ch", "fresh_ch"):
        a._group_histories[ch] = [{"sender": "x", "body": "y"}]
        a._group_md_cache[ch] = {"content": "x", "version": 1}
        a._group_md_checked.add(ch)
        a._chat_kind[ch] = 2  # ChannelType.Group
        a._group_cache_timestamps[ch] = 12345

    removed = a._cleanup_caches()

    assert removed == 1
    # Stale evicted from every per-channel dict
    assert "stale_ch" not in a._group_histories
    assert "stale_ch" not in a._group_md_cache
    assert "stale_ch" not in a._group_md_checked
    assert "stale_ch" not in a._chat_kind
    assert "stale_ch" not in a._group_cache_timestamps
    assert "stale_ch" not in a._cache_activity
    # Fresh kept
    assert "fresh_ch" in a._group_histories
    assert "fresh_ch" in a._group_md_cache
    assert "fresh_ch" in a._group_md_checked
    assert "fresh_ch" in a._chat_kind
    assert "fresh_ch" in a._group_cache_timestamps
    assert "fresh_ch" in a._cache_activity


def test_cleanup_idempotent_when_nothing_stale():
    a = _make_adapter()
    a._touch_cache("ch1")
    a._touch_cache("ch2")
    assert a._cleanup_caches() == 0
    assert set(a._cache_activity) == {"ch1", "ch2"}


def test_cleanup_handles_channel_with_no_state():
    """An activity record without any populated dicts should still be cleaned
    up cleanly (no KeyError)."""
    a = _make_adapter()
    a._cache_activity["ghost"] = time.monotonic() - (CACHE_MAX_AGE_S + 1)
    assert a._cleanup_caches() == 1
    assert "ghost" not in a._cache_activity


def test_touch_refreshes_activity_timestamp():
    a = _make_adapter()
    a._cache_activity["ch"] = time.monotonic() - (CACHE_MAX_AGE_S + 60)
    a._touch_cache("ch")
    assert a._cleanup_caches() == 0, "freshly touched channel must not be evicted"
