"""Unit tests for permission.py — parse_target + check_permission."""

from __future__ import annotations

import pytest

from hermes_octo_plugin.permission import check_permission, parse_target
from hermes_octo_plugin.types import ChannelType, GroupMember


OWNER = "owner_uid_abc"
ALICE = "uid_alice"
BOB = "uid_bob"
GROUP_ALPHA = "group_alpha"
GROUP_BETA = "group_beta"


async def _members_alice_only(_group_no: str):
    return [GroupMember(uid=ALICE, name="Alice")]


async def _members_alice_bob(_group_no: str):
    return [GroupMember(uid=ALICE, name="Alice"), GroupMember(uid=BOB, name="Bob")]


async def _members_raise(_group_no: str):
    raise RuntimeError("API down")


# ─── parse_target ─────────────────────────────────────────────────────────────


class TestParseTarget:
    def test_explicit_group_prefix(self):
        assert parse_target("group:abc") == ("abc", ChannelType.Group)

    def test_explicit_user_prefix(self):
        assert parse_target("user:xyz") == ("xyz", ChannelType.DM)

    def test_explicit_channel_prefix(self):
        assert parse_target("channel:abc") == ("abc", ChannelType.Group)

    def test_thread_via_group_prefix(self):
        cid, ct = parse_target("group:g1____t9")
        assert cid == "g1____t9"
        assert ct == ChannelType.CommunityTopic

    def test_thread_via_bare(self):
        cid, ct = parse_target("g1____t9")
        assert cid == "g1____t9"
        assert ct == ChannelType.CommunityTopic

    def test_octo_prefix_stripped(self):
        assert parse_target("octo:user1") == ("user1", ChannelType.DM)

    def test_bare_known_group(self):
        cid, ct = parse_target("abc", known_group_ids={"abc"})
        assert cid == "abc"
        assert ct == ChannelType.Group

    def test_bare_unknown_group_is_dm(self):
        cid, ct = parse_target("xyz", known_group_ids={"abc"})
        assert ct == ChannelType.DM

    def test_bare_no_known_set_is_dm(self):
        assert parse_target("xyz") == ("xyz", ChannelType.DM)


# ─── check_permission ────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestCheckPermission:

    async def test_unknown_requester_denied(self):
        r = await check_permission(
            requester_uid=None,
            channel_id=ALICE,
            channel_type=ChannelType.DM,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_only,
        )
        assert not r.allowed
        assert "无法识别" in r.reason

    async def test_owner_full_dm_access(self):
        r = await check_permission(
            requester_uid=OWNER,
            channel_id=ALICE,           # Someone else's DM
            channel_type=ChannelType.DM,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_only,
        )
        assert r.allowed

    async def test_owner_full_group_access_skips_member_check(self):
        r = await check_permission(
            requester_uid=OWNER,
            channel_id=GROUP_BETA,       # bot's owner is not in group_beta
            channel_type=ChannelType.Group,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_only,  # would deny Alice/Bob
        )
        assert r.allowed

    async def test_dm_own_allowed(self):
        r = await check_permission(
            requester_uid=ALICE,
            channel_id=ALICE,
            channel_type=ChannelType.DM,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_only,
        )
        assert r.allowed

    async def test_dm_other_denied(self):
        r = await check_permission(
            requester_uid=ALICE,
            channel_id=BOB,
            channel_type=ChannelType.DM,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_only,
        )
        assert not r.allowed
        assert "他人" in r.reason

    async def test_group_member_allowed(self):
        r = await check_permission(
            requester_uid=BOB,
            channel_id=GROUP_ALPHA,
            channel_type=ChannelType.Group,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_bob,
        )
        assert r.allowed

    async def test_group_non_member_denied(self):
        r = await check_permission(
            requester_uid=BOB,
            channel_id=GROUP_ALPHA,
            channel_type=ChannelType.Group,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_only,
        )
        assert not r.allowed
        assert "不在该群" in r.reason

    async def test_group_fetch_failure_denied(self):
        r = await check_permission(
            requester_uid=BOB,
            channel_id=GROUP_ALPHA,
            channel_type=ChannelType.Group,
            owner_uid=OWNER,
            fetch_group_members=_members_raise,
        )
        assert not r.allowed
        assert "查询失败" in r.reason

    async def test_thread_parent_member_allowed(self):
        r = await check_permission(
            requester_uid=BOB,
            channel_id=f"{GROUP_ALPHA}____thread1",
            channel_type=ChannelType.CommunityTopic,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_bob,
        )
        assert r.allowed

    async def test_thread_parent_non_member_denied(self):
        r = await check_permission(
            requester_uid=BOB,
            channel_id=f"{GROUP_ALPHA}____thread1",
            channel_type=ChannelType.CommunityTopic,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_only,
        )
        assert not r.allowed
        assert "子区" in r.reason

    async def test_thread_invalid_channel_id_denied(self):
        r = await check_permission(
            requester_uid=BOB,
            channel_id="",
            channel_type=ChannelType.CommunityTopic,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_bob,
        )
        assert not r.allowed

    async def test_unsupported_channel_type(self):
        r = await check_permission(
            requester_uid=ALICE,
            channel_id="x",
            channel_type=99,
            owner_uid=OWNER,
            fetch_group_members=_members_alice_only,
        )
        assert not r.allowed
        assert "不支持" in r.reason
