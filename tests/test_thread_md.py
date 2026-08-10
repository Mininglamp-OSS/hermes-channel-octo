"""Unit tests for P2-1 THREAD.md support."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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

    async def test_ensure_thread_md_transport_failure_remains_retryable(self):
        a = _make_adapter()
        get_thread_md = AsyncMock(
            side_effect=[
                RuntimeError("temporary outage"),
                {"content": "recovered thread", "version": 3},
            ]
        )

        with patch(
            "hermes_octo_plugin.adapter.api.get_thread_md", new=get_thread_md
        ):
            await a._ensure_thread_md("g1", "t1")
            assert "g1____t1" not in a._group_md_checked
            assert "g1____t1" not in a._group_md_cache

            await a._ensure_thread_md("g1", "t1")

        assert get_thread_md.await_count == 2
        assert "g1____t1" in a._group_md_checked
        assert a._group_md_cache["g1____t1"] == {
            "content": "recovered thread",
            "version": 3,
        }

    async def test_ensure_group_md_transport_failure_remains_retryable(self):
        a = _make_adapter()
        get_group_md = AsyncMock(
            side_effect=[
                RuntimeError("temporary outage"),
                {"content": "recovered", "version": 2},
            ]
        )

        with patch(
            "hermes_octo_plugin.adapter.api.get_group_md", new=get_group_md
        ):
            await a._ensure_group_md("g1")
            assert "g1" not in a._group_md_checked
            assert "g1" not in a._group_md_cache

            await a._ensure_group_md("g1")

        assert get_group_md.await_count == 2
        assert "g1" in a._group_md_checked
        assert a._group_md_cache["g1"] == {
            "content": "recovered",
            "version": 2,
        }

    @pytest.mark.parametrize("scope", ["group", "thread"])
    async def test_inflight_md_fetch_cannot_restore_evicted_group_scope(self, scope):
        a = _make_adapter()
        a._write_md_to_disk = MagicMock()
        a._delete_md_from_disk = MagicMock()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_group_md(_session, _url, _token, _group_no):
            entered.set()
            await release.wait()
            return {"content": "stale group", "version": 7}

        async def blocked_thread_md(
            _session, _url, _token, *, group_no, short_id
        ):
            assert (group_no, short_id) == ("g1", "t1")
            entered.set()
            await release.wait()
            return {"content": "stale thread", "version": 8}

        target = "g1" if scope == "group" else "g1____t1"
        fetch = (
            a._ensure_group_md("g1")
            if scope == "group"
            else a._ensure_thread_md("g1", "t1")
        )
        with (
            patch(
                "hermes_octo_plugin.adapter.api.get_group_md",
                new=blocked_group_md,
            ),
            patch(
                "hermes_octo_plugin.adapter.api.get_thread_md",
                new=blocked_thread_md,
            ),
        ):
            task = asyncio.create_task(fetch)
            await entered.wait()
            await a._evict_group_scope("g1")
            release.set()
            await task

        assert target not in a._group_md_cache
        assert target not in a._group_md_checked
        a._write_md_to_disk.assert_not_called()

    @pytest.mark.parametrize("scope", ["group", "thread"])
    async def test_md_fetch_scheduled_before_eviction_but_started_afterward_is_ignored(
        self, scope
    ):
        a = _make_adapter()
        a._known_group_ids.add("g1")
        a._write_md_to_disk = MagicMock()
        a._delete_md_from_disk = MagicMock()
        target = "g1" if scope == "group" else "g1____t1"
        get_group_md = AsyncMock(
            return_value={"content": "late group", "version": 14}
        )
        get_thread_md = AsyncMock(
            return_value={"content": "late thread", "version": 15}
        )

        fetch = (
            a._ensure_group_md("g1")
            if scope == "group"
            else a._ensure_thread_md("g1", "t1")
        )
        await a._evict_group_scope("g1")
        with (
            patch(
                "hermes_octo_plugin.adapter.api.get_group_md",
                get_group_md,
            ),
            patch(
                "hermes_octo_plugin.adapter.api.get_thread_md",
                get_thread_md,
            ),
        ):
            await fetch

        assert target not in a._group_md_cache
        assert target not in a._group_md_checked
        get_group_md.assert_not_awaited()
        get_thread_md.assert_not_awaited()

    @pytest.mark.parametrize("scope", ["group", "thread"])
    async def test_md_delete_event_invalidates_an_older_inflight_fetch(self, scope):
        a = _make_adapter()
        a._write_md_to_disk = MagicMock()
        a._delete_md_from_disk = MagicMock()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_group_md(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return {"content": "deleted group", "version": 12}

        async def blocked_thread_md(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return {"content": "deleted thread", "version": 13}

        target = "g1" if scope == "group" else "g1____t1"
        fetch = (
            a._ensure_group_md("g1")
            if scope == "group"
            else a._ensure_thread_md("g1", "t1")
        )
        with (
            patch(
                "hermes_octo_plugin.adapter.api.get_group_md",
                new=blocked_group_md,
            ),
            patch(
                "hermes_octo_plugin.adapter.api.get_thread_md",
                new=blocked_thread_md,
            ),
        ):
            task = asyncio.create_task(fetch)
            await entered.wait()
            a._handle_group_md_event(target, "group_md_deleted")
            release.set()
            await task

        assert target not in a._group_md_cache
        assert target not in a._group_md_checked
        a._write_md_to_disk.assert_not_called()


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

    @pytest.mark.parametrize("scope", ["group", "thread"])
    async def test_inflight_refresh_cannot_restore_evicted_group_scope(self, scope):
        a = _make_adapter()
        a._write_md_to_disk = MagicMock()
        a._delete_md_from_disk = MagicMock()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_group_md(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return {"content": "stale group", "version": 10}

        async def blocked_thread_md(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return {"content": "stale thread", "version": 11}

        target = "g1" if scope == "group" else "g1____t1"
        with (
            patch(
                "hermes_octo_plugin.adapter.api.get_group_md",
                new=blocked_group_md,
            ),
            patch(
                "hermes_octo_plugin.adapter.api.get_thread_md",
                new=blocked_thread_md,
            ),
        ):
            task = asyncio.create_task(a._refresh_group_md(target))
            await entered.wait()
            await a._evict_group_scope("g1")
            release.set()
            await task

        assert target not in a._group_md_cache
        a._write_md_to_disk.assert_not_called()

    @pytest.mark.parametrize("scope", ["group", "thread"])
    async def test_refresh_scheduled_by_update_cannot_run_after_delete(self, scope):
        a = _make_adapter()
        a._known_group_ids.add("g1")
        a._write_md_to_disk = MagicMock()
        a._delete_md_from_disk = MagicMock()
        target = "g1" if scope == "group" else "g1____t1"
        get_group_md = AsyncMock(
            return_value={"content": "deleted group", "version": 16}
        )
        get_thread_md = AsyncMock(
            return_value={"content": "deleted thread", "version": 17}
        )

        a._handle_group_md_event(target, "group_md_updated")
        refresh = a._refresh_group_md(target)
        a._handle_group_md_event(target, "group_md_deleted")
        with (
            patch(
                "hermes_octo_plugin.adapter.api.get_group_md",
                get_group_md,
            ),
            patch(
                "hermes_octo_plugin.adapter.api.get_thread_md",
                get_thread_md,
            ),
        ):
            await refresh

        assert target not in a._group_md_cache
        get_group_md.assert_not_awaited()
        get_thread_md.assert_not_awaited()
