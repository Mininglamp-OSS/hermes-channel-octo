"""Unit tests for P1-3 startup prefetch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin.adapter import OctoAdapter
from hermes_octo_plugin.types import ChannelType, GroupInfo, GroupMember
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    a = make_bare_adapter()
    a._http_session = MagicMock()  # truthy is enough; api calls are patched
    a._api_url = "https://example.test"
    a._bot_token = "tok"
    a._robot_id = "bot_self"
    return a


@pytest.mark.asyncio
async def test_prefetch_seeds_known_groups_and_md_and_members():
    a = _make_adapter()
    groups = [
        GroupInfo(group_no="g1", name="Group 1"),
        GroupInfo(group_no="g2", name="Group 2"),
    ]
    md_by_group = {
        "g1": {"content": "G1 README", "version": 3},
        "g2": {"content": "", "version": 0},  # empty MD should NOT be cached
    }
    members_by_group = {
        "g1": [GroupMember(uid="u1", name="Alice"), GroupMember(uid="u2", name="Bob")],
        "g2": [GroupMember(uid="u3", name="Carol")],
    }

    async def fake_groups(*_a, **_kw):
        return groups

    async def fake_md(_s, _u, _t, gid):
        return md_by_group.get(gid)

    async def fake_members(_s, _u, _t, gid):
        return members_by_group.get(gid, [])

    with patch("hermes_octo_plugin.adapter.api.fetch_bot_groups", new=fake_groups), \
         patch("hermes_octo_plugin.adapter.api.get_group_md", new=fake_md), \
         patch("hermes_octo_plugin.adapter.api.get_group_members", new=fake_members):
        await a._prefetch_groups_and_members()

    # known_group_ids populated for both groups
    assert a._known_group_ids == {"g1", "g2"}
    # chat_kind seeded as Group for both
    assert a._chat_kind == {"g1": ChannelType.Group, "g2": ChannelType.Group}
    # GROUP.md cached only when content non-empty
    assert "g1" in a._group_md_cache
    assert a._group_md_cache["g1"] == {"content": "G1 README", "version": 3}
    assert "g1" in a._group_md_checked
    assert "g2" not in a._group_md_cache
    # member maps populated for all members
    assert a._member_map == {"Alice": "u1", "Bob": "u2", "Carol": "u3"}
    assert a._uid_to_name == {"u1": "Alice", "u2": "Bob", "u3": "Carol"}
    # group_cache_timestamps set so the next refresh_group_member_cache call
    # won't immediately re-fetch
    assert "g1" in a._group_cache_timestamps
    assert "g2" in a._group_cache_timestamps
    # Activity recorded so the cleanup loop doesn't evict on first sweep
    assert "g1" in a._cache_activity
    assert "g2" in a._cache_activity


@pytest.mark.asyncio
async def test_prefetch_swallows_per_group_errors():
    """One bad endpoint must not poison the whole warmup."""
    a = _make_adapter()
    groups = [GroupInfo(group_no="g_good", name="Good"), GroupInfo(group_no="g_bad", name="Bad")]

    async def fake_groups(*_a, **_kw):
        return groups

    async def fake_md(_s, _u, _t, gid):
        if gid == "g_bad":
            raise RuntimeError("404")
        return {"content": "OK", "version": 1}

    async def fake_members(_s, _u, _t, gid):
        if gid == "g_bad":
            raise RuntimeError("500")
        return [GroupMember(uid="u_good", name="GoodPerson")]

    with patch("hermes_octo_plugin.adapter.api.fetch_bot_groups", new=fake_groups), \
         patch("hermes_octo_plugin.adapter.api.get_group_md", new=fake_md), \
         patch("hermes_octo_plugin.adapter.api.get_group_members", new=fake_members):
        await a._prefetch_groups_and_members()

    # known_group_ids still records both even when their data failed
    assert a._known_group_ids == {"g_good", "g_bad"}
    # Good group's data made it through
    assert a._group_md_cache.get("g_good", {}).get("content") == "OK"
    assert a._member_map.get("GoodPerson") == "u_good"
    # Bad group's data is absent but caused no exception
    assert "g_bad" not in a._group_md_cache


@pytest.mark.asyncio
async def test_prefetch_noop_when_fetch_bot_groups_fails():
    """A failed group-list call returns early without touching state."""
    a = _make_adapter()

    async def fake_groups(*_a, **_kw):
        raise RuntimeError("network down")

    with patch("hermes_octo_plugin.adapter.api.fetch_bot_groups", new=fake_groups):
        await a._prefetch_groups_and_members()

    assert a._known_group_ids == set()
    assert a._chat_kind == {}
    assert a._group_md_cache == {}
    assert a._member_map == {}


@pytest.mark.asyncio
async def test_prefetch_noop_when_no_http_session():
    """Without a session, prefetch is a no-op (defensive guard)."""
    a = _make_adapter()
    a._http_session = None
    # If this called fetch_bot_groups it would explode because session is None,
    # so reaching the assert at all means the early return fired.
    await a._prefetch_groups_and_members()
    assert a._known_group_ids == set()
