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
async def test_prefetch_seeds_group_ids_without_loading_group_content_or_members():
    a = _make_adapter()
    groups = [
        GroupInfo(group_no="g1", name="Group 1"),
        GroupInfo(group_no="g2", name="Group 2"),
    ]
    get_md = AsyncMock()
    get_members = AsyncMock()

    with (
        patch(
            "hermes_octo_plugin.adapter.api.fetch_bot_groups",
            new=AsyncMock(return_value=groups),
        ),
        patch("hermes_octo_plugin.adapter.api.get_group_md", new=get_md),
        patch("hermes_octo_plugin.adapter.api.get_group_members", new=get_members),
    ):
        await a._prefetch_groups_and_members()

    assert a._known_group_ids == {"g1", "g2"}
    assert a._chat_kind == {"g1": ChannelType.Group, "g2": ChannelType.Group}
    assert a._cache_activity.keys() == {"g1", "g2"}
    assert a._group_md_cache == {}
    assert a._group_member_rosters == {}
    assert a._member_map == {}
    assert a._uid_to_name == {}
    get_md.assert_not_awaited()
    get_members.assert_not_awaited()



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
