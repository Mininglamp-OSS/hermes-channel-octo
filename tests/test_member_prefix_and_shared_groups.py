"""Tests for Batch 2: member list prefix + shared-groups reverse index."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from hermes_octo_plugin.adapter import OctoAdapter
from hermes_octo_plugin.types import GroupMember
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    a = make_bare_adapter()
    a._robot_id = "bot_self"
    return a


# ─── Member list prefix ─────────────────────────────────────────────────────


class TestMemberListPrefix:
    def test_empty_returns_empty_string(self):
        a = _make_adapter()
        assert a._build_member_list_prefix() == ""

    def test_small_group_lists_all_members(self):
        a = _make_adapter()
        a._uid_to_name = OrderedDict([("u1", "Alice"), ("u2", "Bob")])
        out = a._build_member_list_prefix()
        assert "[Group Members]" in out
        assert "Alice (u1)" in out
        assert "Bob (u2)" in out
        assert "@[uid:displayName]" in out
        # Example uses the first member to teach the format
        assert "@[u1:Alice]" in out

    def test_large_group_shows_count_only(self):
        a = _make_adapter()
        a._uid_to_name = {f"u{i}": f"User{i}" for i in range(15)}
        out = a._build_member_list_prefix()
        assert "15 members" in out
        assert "octo_management" in out
        # Should NOT list every member
        assert "User7" not in out

    def test_boundary_exactly_ten_lists(self):
        a = _make_adapter()
        a._uid_to_name = OrderedDict((f"u{i}", f"User{i}") for i in range(10))
        out = a._build_member_list_prefix()
        assert "[Group Members]" in out
        assert "User9 (u9)" in out


# ─── Reverse index ──────────────────────────────────────────────────────────


class TestUserGroupIndex:
    def test_update_adds_members(self):
        a = _make_adapter()
        members = [GroupMember(uid="u1", name="A"), GroupMember(uid="u2", name="B")]
        a._update_user_group_index("g1", members)
        assert a._user_group_index["u1"] == {"g1"}
        assert a._user_group_index["u2"] == {"g1"}

    def test_update_drops_bot_self_uid(self):
        """Bot is technically a member of its own groups — but it's never a
        useful answer to "groups I share with the bot"."""
        a = _make_adapter()
        members = [GroupMember(uid="bot_self", name="Bot"), GroupMember(uid="u1", name="A")]
        a._update_user_group_index("g1", members)
        assert "bot_self" not in a._user_group_index
        assert a._user_group_index["u1"] == {"g1"}

    def test_update_idempotent_for_same_roster(self):
        a = _make_adapter()
        members = [GroupMember(uid="u1", name="A")]
        a._update_user_group_index("g1", members)
        a._update_user_group_index("g1", members)
        assert a._user_group_index["u1"] == {"g1"}

    def test_update_removes_stale_members(self):
        """Member kicked from a group should disappear from shared-groups
        on next refresh."""
        a = _make_adapter()
        a._update_user_group_index("g1", [GroupMember(uid="u1", name="A"), GroupMember(uid="u2", name="B")])
        # u2 leaves g1
        a._update_user_group_index("g1", [GroupMember(uid="u1", name="A")])
        assert a._user_group_index["u1"] == {"g1"}
        assert "u2" not in a._user_group_index  # Removed entirely (had no other groups)

    def test_user_in_multiple_groups(self):
        a = _make_adapter()
        a._update_user_group_index("g1", [GroupMember(uid="u1", name="A")])
        a._update_user_group_index("g2", [GroupMember(uid="u1", name="A")])
        assert a._user_group_index["u1"] == {"g1", "g2"}

    def test_user_leaving_one_group_keeps_other(self):
        a = _make_adapter()
        a._update_user_group_index("g1", [GroupMember(uid="u1", name="A")])
        a._update_user_group_index("g2", [GroupMember(uid="u1", name="A")])
        # u1 leaves g1 but stays in g2
        a._update_user_group_index("g1", [])
        assert a._user_group_index["u1"] == {"g2"}


# ─── find_shared_groups ─────────────────────────────────────────────────────


class TestFindSharedGroups:
    def test_unknown_uid_returns_empty(self):
        a = _make_adapter()
        assert a.find_shared_groups("ghost") == []

    def test_returns_groups_with_names(self):
        a = _make_adapter()
        a._user_group_index["u1"] = {"g1", "g2"}
        a._group_names["g1"] = "Group One"
        a._group_names["g2"] = "Group Two"
        out = a.find_shared_groups("u1")
        # Sorted by group_no
        assert out == [
            {"group_no": "g1", "name": "Group One"},
            {"group_no": "g2", "name": "Group Two"},
        ]

    def test_missing_name_falls_back_to_group_no(self):
        a = _make_adapter()
        a._user_group_index["u1"] = {"g_unknown"}
        out = a.find_shared_groups("u1")
        assert out == [{"group_no": "g_unknown", "name": "g_unknown"}]

    def test_empty_uid_returns_empty(self):
        a = _make_adapter()
        assert a.find_shared_groups("") == []
