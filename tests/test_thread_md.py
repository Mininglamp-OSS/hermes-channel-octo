"""Unit tests for P2-1 THREAD.md support."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_octo_plugin.adapter import OctoAdapter
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    a = make_bare_adapter()
    a._http_session = object()  # truthy
    a._api_url = "https://example.test"
    a._bot_token = "tok"
    return a


# ─── _split_thread_channel_id ────────────────────────────────────────────────


class TestSplitThreadChannelId:
    def test_group_channel(self):
        a = _make_adapter()
        assert a._split_thread_channel_id("g1") == ("g1", None)

    def test_thread_channel(self):
        a = _make_adapter()
        assert a._split_thread_channel_id("g1____thread_abc") == ("g1", "thread_abc")

    def test_thread_with_empty_short_id_treated_as_no_thread(self):
        """A malformed `gid____` (separator but no short id) should fall back
        to group semantics so we don't ever hit the thread endpoint with an
        empty short_id."""
        a = _make_adapter()
        parent, short = a._split_thread_channel_id("g1____")
        assert parent == "g1"
        assert short is None


# ─── _ensure_group_md / _ensure_thread_md ────────────────────────────────────


@pytest.mark.asyncio
class TestEnsureMd:
    async def test_ensure_group_md_caches(self):
        a = _make_adapter()

        async def fake_get(_s, _u, _t, gid):
            assert gid == "g1"  # must be parent group_no, NOT thread channel_id
            return {"content": "Group MD body", "version": 5}

        with patch("hermes_octo_plugin.adapter.api.get_group_md", new=fake_get):
            await a._ensure_group_md("g1")
        assert a._group_md_cache["g1"] == {"content": "Group MD body", "version": 5}
        assert "g1" in a._group_md_checked

    async def test_ensure_group_md_idempotent(self):
        a = _make_adapter()
        calls = 0

        async def fake_get(_s, _u, _t, _gid):
            nonlocal calls
            calls += 1
            return {"content": "x", "version": 1}

        with patch("hermes_octo_plugin.adapter.api.get_group_md", new=fake_get):
            await a._ensure_group_md("g1")
            await a._ensure_group_md("g1")
        assert calls == 1

    async def test_ensure_thread_md_caches_under_composite_key(self):
        a = _make_adapter()

        async def fake_thread_md(_s, _u, _t, *, group_no, short_id):
            assert group_no == "g1"
            assert short_id == "thr_abc"
            return {"content": "Thread MD body", "version": 2}

        with patch("hermes_octo_plugin.adapter.api.get_thread_md", new=fake_thread_md):
            await a._ensure_thread_md("g1", "thr_abc")
        assert a._group_md_cache["g1____thr_abc"] == {
            "content": "Thread MD body", "version": 2,
        }
        assert "g1____thr_abc" in a._group_md_checked

    async def test_ensure_group_md_swallows_errors(self):
        a = _make_adapter()

        async def fake_raise(*_a, **_kw):
            raise RuntimeError("boom")

        with patch("hermes_octo_plugin.adapter.api.get_group_md", new=fake_raise):
            await a._ensure_group_md("g1")
        assert "g1" not in a._group_md_cache
        # _checked is still set so we don't retry on every message
        assert "g1" in a._group_md_checked


# ─── _handle_group_md_event ──────────────────────────────────────────────────


class TestHandleGroupMdEvent:
    def test_group_md_deleted_clears_cache_only_for_target(self):
        a = _make_adapter()
        a._group_md_cache["g1"] = {"content": "x", "version": 1}
        a._group_md_cache["g1____t1"] = {"content": "y", "version": 1}
        a._group_md_checked.update({"g1", "g1____t1"})

        a._handle_group_md_event("g1____t1", "group_md_deleted")

        # Thread's MD evicted, parent group's MD intact
        assert "g1" in a._group_md_cache
        assert "g1____t1" not in a._group_md_cache
        assert "g1____t1" not in a._group_md_checked

    def test_group_md_updated_marks_only_target_stale(self):
        a = _make_adapter()
        a._group_md_checked.update({"g1", "g1____t1"})

        a._handle_group_md_event("g1", "group_md_updated")

        # Parent invalidated, thread record kept
        assert "g1" not in a._group_md_checked
        assert "g1____t1" in a._group_md_checked


# ─── _refresh_group_md routes to the right API ───────────────────────────────


@pytest.mark.asyncio
class TestRefreshGroupMd:
    async def test_refresh_routes_thread_to_thread_md_api(self):
        a = _make_adapter()
        called_thread = False

        async def fake_group_md(*_a, **_kw):
            raise AssertionError("group_md API must not be called for a thread channel_id")

        async def fake_thread_md(_s, _u, _t, *, group_no, short_id):
            nonlocal called_thread
            called_thread = True
            assert group_no == "g1"
            assert short_id == "t9"
            return {"content": "fresh thread md", "version": 7}

        with patch("hermes_octo_plugin.adapter.api.get_group_md", new=fake_group_md), \
             patch("hermes_octo_plugin.adapter.api.get_thread_md", new=fake_thread_md):
            await a._refresh_group_md("g1____t9")

        assert called_thread
        assert a._group_md_cache["g1____t9"] == {"content": "fresh thread md", "version": 7}

    async def test_refresh_routes_group_to_group_md_api(self):
        a = _make_adapter()
        called_group = False

        async def fake_group_md(_s, _u, _t, gid):
            nonlocal called_group
            called_group = True
            assert gid == "g1"
            return {"content": "fresh group md", "version": 4}

        async def fake_thread_md(*_a, **_kw):
            raise AssertionError("thread_md API must not be called for a bare group_no")

        with patch("hermes_octo_plugin.adapter.api.get_group_md", new=fake_group_md), \
             patch("hermes_octo_plugin.adapter.api.get_thread_md", new=fake_thread_md):
            await a._refresh_group_md("g1")

        assert called_group
        assert a._group_md_cache["g1"] == {"content": "fresh group md", "version": 4}
