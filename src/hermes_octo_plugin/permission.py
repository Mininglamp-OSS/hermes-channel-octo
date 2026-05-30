"""
Cross-channel access control for Octo message queries.

Rules (mirrors openclaw-channel-octo/src/permission.ts):
- Owner of the bot → full access (all DMs and groups).
- DM channels    → requester may only query their own DM with the bot.
- Group channels → requester must currently be a member of the group.
- Thread (CommunityTopic, channel_type=5) → requester must be a member of
  the parent group (parent group_no is the prefix before "____").
- Unknown requester → denied.

The module is intentionally framework-free: callers pass in the requester
uid, channel info, owner_uid, and a member-fetch coroutine. This keeps the
adapter wiring narrow and the rules easy to unit-test.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .types import ChannelType, GroupMember

# Coroutine signature: group_no -> list[GroupMember]
GroupMembersFetcher = Callable[[str], Awaitable[list[GroupMember]]]


@dataclass
class PermissionResult:
    allowed: bool
    reason: str | None = None


async def check_permission(
    *,
    requester_uid: str | None,
    channel_id: str,
    channel_type: int,
    owner_uid: str | None,
    fetch_group_members: GroupMembersFetcher,
) -> PermissionResult:
    if not requester_uid:
        return PermissionResult(False, "无法识别调用者身份")

    if owner_uid and requester_uid == owner_uid:
        return PermissionResult(True)

    if channel_type == ChannelType.DM:
        if channel_id != requester_uid:
            return PermissionResult(False, "无权查询他人与Bot的私信")
        return PermissionResult(True)

    if channel_type == ChannelType.Group:
        return await _check_group_membership(
            requester_uid, channel_id, fetch_group_members
        )

    if channel_type == ChannelType.CommunityTopic:
        # Thread channel_id format: groupNo____shortId
        group_no = channel_id.split("____", 1)[0] if channel_id else ""
        if not group_no:
            return PermissionResult(False, "无效的子区频道ID")
        return await _check_group_membership(
            requester_uid, group_no, fetch_group_members,
            denied_reason="你不在该群中，无权访问子区",
        )

    return PermissionResult(False, f"不支持的频道类型: {channel_type}")


async def _check_group_membership(
    requester_uid: str,
    group_no: str,
    fetch_group_members: GroupMembersFetcher,
    denied_reason: str = "你不在该群中，无权查询",
) -> PermissionResult:
    try:
        members = await fetch_group_members(group_no)
    except Exception as e:  # network failure should not silently grant access
        return PermissionResult(False, f"群成员查询失败: {e}")

    member_uids = {m.uid for m in members if m.uid}
    if requester_uid not in member_uids:
        return PermissionResult(False, denied_reason)
    return PermissionResult(True)


# ─── Target parsing ──────────────────────────────────────────────────────────


def parse_target(
    target: str,
    *,
    known_group_ids: set[str] | None = None,
) -> tuple[str, ChannelType]:
    """
    Parse a target string into (channel_id, channel_type).

    Mirrors openclaw-channel-octo/src/actions.ts :: parseTarget. Explicit
    prefixes (``group:`` / ``channel:`` / ``user:``) always win. Bare IDs
    containing ``____`` are CommunityTopic; otherwise resolved by membership
    in *known_group_ids* (defaults to DM when the set is empty or missing).
    """
    THREAD_SEP = "____"

    if target.startswith("group:"):
        channel_id = target[len("group:"):]
        if THREAD_SEP in channel_id:
            return channel_id, ChannelType.CommunityTopic
        return channel_id, ChannelType.Group

    if target.startswith("channel:"):
        channel_id = target[len("channel:"):]
        if THREAD_SEP in channel_id:
            return channel_id, ChannelType.CommunityTopic
        return channel_id, ChannelType.Group

    if target.startswith("user:"):
        return target[len("user:"):], ChannelType.DM

    bare = target[len("octo:"):] if target.startswith("octo:") else target

    if THREAD_SEP in bare:
        return bare, ChannelType.CommunityTopic

    is_group = bool(known_group_ids and bare in known_group_ids)
    return bare, ChannelType.Group if is_group else ChannelType.DM
