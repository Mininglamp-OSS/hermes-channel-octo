"""
``octo_management`` agent tool.

LLM-callable tool that gives the model direct access to Octo group / thread /
GROUP.md / voice-context / cross-channel-read APIs without going through the
inbound message pipeline.

Exposes a single tool with an ``action`` enum that fans out to the
corresponding API call.

Event loop hygiene:
  Hermes ``_run_async`` bridges sync tool dispatch to async handlers by
  spinning up a *fresh* event loop on a worker thread when the caller is
  already inside one. ``adapter._http_session`` was created in the gateway's
  main loop, so reusing it from the worker loop raises
  ``RuntimeError: Timeout context manager should be used inside a task``
  the first time aiohttp tries to arm a per-request timer.
  We sidestep this by opening a fresh ``aiohttp.ClientSession`` inside the
  handler — it lives only for the call, lives in the current loop, and is
  cleaned up on exit. The adapter is only consulted for live identity
  (api_url / bot_token / owner_uid / known_group_ids), never for I/O.

Permissions:
  - All operations that hit a specific channel honour
    :meth:`OctoAdapter.check_read_permission` semantics (replicated locally
    so we can use the per-call session).
  - Requester identity comes only from Hermes' trusted task-local Octo session;
    the model-facing schema never asks the model to repeat an authenticated UID.
  - Mutating operations (create-group / add-members / group-md-update /
    voice-context-update) require that trusted requester to equal the bot owner.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC
from typing import Any

import aiohttp

from . import api
from .permission import check_permission, parse_target
from .types import ChannelType

logger = logging.getLogger(__name__)

_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _new_guarded_http_session(api_url: str) -> aiohttp.ClientSession:
    """Create a per-call session without bypassing the adapter's SSRF policy."""
    from .adapter import _new_guarded_http_session as _factory

    return _factory(api_url)


def _valid_resource_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_RESOURCE_ID_RE.fullmatch(value))


def _valid_target_channel_id(channel_id: str, channel_type: ChannelType) -> bool:
    if channel_type == ChannelType.DM:
        return True
    if channel_type == ChannelType.Group:
        return _valid_resource_id(channel_id)
    if channel_type == ChannelType.CommunityTopic:
        group_id, separator, short_id = channel_id.partition("____")
        return bool(
            separator
            and _valid_resource_id(group_id)
            and _valid_resource_id(short_id)
        )
    return False


# ---------------------------------------------------------------------------
# Adapter resolution
# ---------------------------------------------------------------------------


def _resolve_adapter():
    """Return the live Octo adapter from the running gateway, or None."""
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref
    except Exception:
        return None
    runner = _gateway_runner_ref()
    if runner is None:
        return None
    try:
        return runner.adapters.get(Platform("octo"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


ACTIONS = [
    # Read / search
    "list-groups", "group-info", "group-members", "search-members",
    "read-messages", "search-shared-groups",
    # Send (active, agent-initiated delivery to any DM/group/thread)
    "send-message",
    # GROUP.md
    "group-md-read", "group-md-update",
    # Group admin
    "create-group", "update-group", "add-members", "remove-members",
    # Threads
    "create-thread", "list-threads", "get-thread", "delete-thread",
    "list-thread-members", "join-thread", "leave-thread",
    # THREAD.md
    "thread-md-read", "thread-md-update",
    # Voice correction context
    "voice-context-read", "voice-context-update", "voice-context-delete",
]


TOOL_SCHEMA = {
    "name": "octo_management",
    "description": (
        "Manage Octo groups, threads, GROUP.md, voice-correction "
        "context, and read messages across channels. Single tool with "
        "an action parameter — pick the action and supply the required "
        "fields. Cross-channel read enforces permissions: only the bot "
        "owner may read another user's DM; group/thread reads require "
        "the trusted current-session requester to be a member."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ACTIONS,
                "description": "Operation to perform.",
            },
            "group_id": {
                "type": "string",
                "description": (
                    "group_no. Required for group-info, group-members, "
                    "group-md-*, update-group, add-members, "
                    "remove-members, *-thread*, thread-md-*."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "Target channel. Accepts `user:<uid>` (DM), "
                    "`group:<group_no>` (group), or "
                    "`group:<group_no>____<short_id>` (thread). Bare ids are "
                    "resolved against the bot's known groups. Required for "
                    "read-messages and send-message."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max messages for read-messages (1-100, default 20).",
            },
            "content": {
                "type": "string",
                "description": (
                    "Message body for send-message; new content for "
                    "group-md-update / thread-md-update / voice-context-update."
                ),
            },
            "reply_to_message_id": {
                "type": "string",
                "description": (
                    "Optional message_id to reply to. Only meaningful for "
                    "send-message."
                ),
            },
            "mention_uids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional uids to @mention. Only meaningful for "
                    "send-message in group / thread channels."
                ),
            },
            "mention_all": {
                "type": "boolean",
                "description": (
                    "If true, @all in the target channel. Only meaningful for "
                    "send-message in group / thread channels."
                ),
            },
            "keyword": {
                "type": "string",
                "description": "Fuzzy keyword for search-members.",
            },
            "members": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of member uids. Required for create-group, "
                    "add-members, remove-members."
                ),
            },
            "name": {
                "type": "string",
                "description": "Group name for create-group / update-group.",
            },
            "notice": {
                "type": "string",
                "description": "Group notice / announcement for update-group.",
            },
            "creator": {
                "type": "string",
                "description": (
                    "uid of the user who becomes the group owner. Required "
                    "for create-group."
                ),
            },
            "thread_name": {
                "type": "string",
                "description": "Thread name for create-thread.",
            },
            "short_id": {
                "type": "string",
                "description": (
                    "Thread short id. Required for get-thread, "
                    "delete-thread, list-thread-members, join-thread, "
                    "leave-thread, thread-md-read, thread-md-update."
                ),
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audit(
    action: str,
    requester: str | None,
    target: str,
    channel_type: int | None = None,
    result: str = "allowed",
    reason: str | None = None,
    count: int | None = None,
) -> None:
    """Emit a structured audit-log line for cross-channel queries.

    JSON-encoded so log shippers can parse without bespoke regex. We use
    the module logger at INFO so operators can grep ``[AUDIT] octo-query``
    to find every cross-channel read/search the agent performed.
    """
    try:
        from datetime import datetime
        ts = datetime.now(UTC).isoformat(timespec="seconds")
    except Exception:
        ts = ""
    entry: dict = {
        "ts": ts, "action": action, "requester": requester, "target": target,
        "result": result,
    }
    if channel_type is not None:
        entry["channelType"] = channel_type
    if reason:
        entry["reason"] = reason
    if count is not None:
        entry["count"] = count
    try:
        logger.info("[AUDIT] octo-query %s",
                    json.dumps(entry, ensure_ascii=False, default=str))
    except Exception:
        # Never let audit logging interrupt a tool call.
        pass


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def _ok(data: Any) -> str:
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False, default=str)


def _require(args: dict, *keys: str) -> str | None:
    missing = [k for k in keys if not args.get(k)]
    if missing:
        return _err(f"missing required argument(s): {', '.join(missing)}")
    return None


def _require_owner(adapter, requester_uid: str | None, action: str) -> str | None:
    """Mutating admin actions require the call to be from the bot's owner."""
    owner = adapter._owner_uid or ""
    if requester_uid and owner and requester_uid == owner:
        return None
    if not requester_uid:
        return _err(f"action '{action}' requires requester_uid")
    return _err(
        f"action '{action}' requires bot-owner privileges; "
        f"requester {requester_uid!r} is not the owner"
    )


def _trusted_requester_uid() -> str | None:
    """Read caller identity from Hermes' task-local gateway session context.

    Tool arguments are model-controlled and therefore cannot authenticate the
    requester. Hermes 0.20 propagates these ContextVars into async tool worker
    threads. Older/incompatible runtimes that cannot provide the context are
    denied rather than falling back to a claimed uid.
    """
    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
        requester_uid = get_session_env("HERMES_SESSION_USER_ID", "").strip()
    except Exception:
        return None
    if platform != "octo" or not requester_uid:
        return None
    return requester_uid


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


OWNER_ONLY_ACTIONS = frozenset({
    # Space-wide enumeration and voice context are not scoped to one group.
    "list-groups", "search-members", "voice-context-read",
    "create-group", "update-group", "add-members", "remove-members",
    "group-md-update", "create-thread", "delete-thread", "join-thread",
    "leave-thread", "thread-md-update",
    "voice-context-update", "voice-context-delete",
})


# Metadata reads are scoped to the requesting user's membership in the parent
# group.  A topic (thread) is represented as ``{group_no}____{short_id}``, so
# its authorization must use the parent group roster rather than treating the
# thread identifier as a standalone group.
GROUP_READ_ACTIONS = frozenset({
    "group-info", "group-members", "group-md-read", "list-threads",
})
THREAD_READ_ACTIONS = frozenset({
    "get-thread", "list-thread-members", "thread-md-read",
})


async def _authorize_group_or_thread_read(
    *,
    action: str,
    args: dict[str, Any],
    requester_uid: str,
    adapter: Any,
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
) -> str | None:
    """Return an error when a scoped metadata read lacks parent membership."""
    if action not in GROUP_READ_ACTIONS | THREAD_READ_ACTIONS:
        return None

    group_id = args.get("group_id")
    if not group_id:
        return None

    if action in THREAD_READ_ACTIONS:
        short_id = args.get("short_id")
        if not short_id:
            return None
        channel_id = f"{group_id}____{short_id}"
        channel_type = ChannelType.CommunityTopic
    else:
        channel_id = group_id
        channel_type = ChannelType.Group

    async def _fetch_members(parent_group_id: str):
        return await api.get_group_members(
            session, api_url, bot_token, parent_group_id
        )

    result = await check_permission(
        requester_uid=requester_uid,
        channel_id=channel_id,
        channel_type=channel_type,
        owner_uid=adapter._owner_uid,
        fetch_group_members=_fetch_members,
    )
    if result.allowed:
        return None

    _audit(
        action=action,
        requester=requester_uid,
        target=channel_id,
        channel_type=int(channel_type),
        result="denied",
        reason=result.reason,
    )
    return _err(result.reason or "permission denied")


async def octo_management_handler(args: dict, **_kwargs) -> str:  # noqa: PLR0911,PLR0912
    """Dispatch a octo_management call. Returns a JSON string."""
    action = args.get("action")
    if action not in ACTIONS:
        return _err(f"unknown action: {action!r} (valid: {sorted(ACTIONS)})")

    adapter = _resolve_adapter()
    if adapter is None:
        return _err("Octo adapter is not running in this process")

    api_url = adapter._api_url
    bot_token = adapter._bot_token
    if not api_url or not bot_token:
        return _err("Octo adapter is not configured (missing api_url / bot_token)")

    requester_uid = _trusted_requester_uid()
    if not requester_uid:
        return _err(
            f"action '{action}' requires a trusted Octo session requester"
        )

    # These values become URL path segments in the API layer. Reject malformed
    # input before opening a session or performing a membership lookup; API
    # helpers validate again as defense in depth.
    for field in ("group_id", "short_id"):
        value = args.get(field)
        if value is not None and not _valid_resource_id(value):
            return _err(f"invalid {field}")

    if action in OWNER_ONLY_ACTIONS:
        err = _require_owner(adapter, requester_uid, action)
        if err:
            return err

    group_id = args.get("group_id")
    short_id = args.get("short_id")
    content = args.get("content")

    # Always open a per-call aiohttp session in the current loop. The adapter's
    # session lives in the gateway loop and would explode under _run_async's
    # worker-thread loop with "Timeout context manager should be used inside a
    # task". The session closes on context exit so we don't leak FDs.
    async with _new_guarded_http_session(api_url) as session:
        try:
            if (err := await _authorize_group_or_thread_read(
                action=action,
                args=args,
                requester_uid=requester_uid,
                adapter=adapter,
                session=session,
                api_url=api_url,
                bot_token=bot_token,
            )):
                return err

            if action == "list-groups":
                groups = await api.fetch_bot_groups(session, api_url, bot_token)
                return _ok({
                    "groups": [g.__dict__ if hasattr(g, "__dict__") else g for g in groups]
                })

            if action == "group-info":
                if (e := _require(args, "group_id")):
                    return e
                info = await api.get_group_info(session, api_url, bot_token, group_no=group_id)
                return _ok({
                    "group_no": info.group_no, "name": info.name, **(info.extra or {}),
                })

            if action == "group-members":
                if (e := _require(args, "group_id")):
                    return e
                members = await api.get_group_members(session, api_url, bot_token, group_id)
                return _ok({"members": [m.__dict__ for m in members]})

            if action == "search-members":
                keyword = args.get("keyword")
                limit = int(args.get("limit") or 50)
                results = await api.search_space_members(
                    session, api_url, bot_token, keyword=keyword, limit=limit,
                )
                _audit(
                    action="search-members",
                    requester=requester_uid,
                    target=keyword or "<all>",
                    count=len(results),
                )
                return _ok({"members": results})

            if action == "search-shared-groups":
                subject = (args.get("target") or "").strip() or (requester_uid or "")
                if not subject:
                    _audit(action="search-shared-groups", requester=requester_uid,
                           target="<none>", result="denied", reason="missing requester_uid/target")
                    return _err("search-shared-groups requires requester_uid or target")
                owner = adapter._owner_uid or ""
                if subject != requester_uid and (not owner or requester_uid != owner):
                    _audit(action="search-shared-groups", requester=requester_uid,
                           target=subject, result="denied", reason="not owner")
                    return _err("only the bot owner may query someone else's shared groups")
                groups = adapter.find_shared_groups(subject)
                _audit(action="search-shared-groups", requester=requester_uid,
                       target=subject, count=len(groups))
                return _ok({"uid": subject, "groups": groups, "count": len(groups)})

            if action == "read-messages":
                if (e := _require(args, "target")):
                    return e
                limit = int(args.get("limit") or 20)
                # Cross-channel read: parse target + permission check + read.
                channel_id, channel_type = parse_target(
                    args["target"], known_group_ids=adapter._known_group_ids,
                )
                if not _valid_target_channel_id(channel_id, channel_type):
                    return _err("invalid target channel id")

                async def _fetch_members(group_no: str):
                    return await api.get_group_members(session, api_url, bot_token, group_no)

                pres = await check_permission(
                    requester_uid=requester_uid,
                    channel_id=channel_id,
                    channel_type=channel_type,
                    owner_uid=adapter._owner_uid,
                    fetch_group_members=_fetch_members,
                )
                if not pres.allowed:
                    _audit(
                        action="read-messages", requester=requester_uid,
                        target=args["target"], channel_type=int(channel_type),
                        result="denied", reason=pres.reason,
                    )
                    return _err(pres.reason or "permission denied")
                messages = await api.get_channel_messages(
                    session, api_url, bot_token,
                    channel_id=channel_id,
                    channel_type=ChannelType(channel_type),
                    limit=max(1, min(limit, 100)),
                )
                _audit(
                    action="read-messages", requester=requester_uid,
                    target=args["target"], channel_type=int(channel_type),
                    count=len(messages),
                )
                return _ok({
                    "channel_id": channel_id,
                    "channel_type": int(channel_type),
                    "messages": messages,
                })

            if action == "send-message":
                if (e := _require(args, "target", "content")):
                    return e
                channel_id, channel_type = parse_target(
                    args["target"], known_group_ids=adapter._known_group_ids,
                )
                if not _valid_target_channel_id(channel_id, channel_type):
                    return _err("invalid target channel id")

                async def _fetch_members(group_no: str):
                    return await api.get_group_members(session, api_url, bot_token, group_no)

                pres = await check_permission(
                    requester_uid=requester_uid,
                    channel_id=channel_id,
                    channel_type=channel_type,
                    owner_uid=adapter._owner_uid,
                    fetch_group_members=_fetch_members,
                )
                if not pres.allowed:
                    _audit(
                        action="send-message", requester=requester_uid,
                        target=args["target"], channel_type=int(channel_type),
                        result="denied", reason=pres.reason,
                    )
                    return _err(pres.reason or "permission denied")

                mention_uids = args.get("mention_uids") or None
                mention_all = bool(args.get("mention_all") or False)
                # @mentions only make sense in group / thread channels.
                if channel_type == ChannelType.DM:
                    mention_uids = None
                    mention_all = False

                # Joining/leaving a thread mutates bot membership and remains
                # an explicit owner-only action. Sending must not smuggle that
                # mutation past the permission boundary.
                client_msg_no = str(uuid.uuid4())
                send_result = await api.send_message(
                    session, api_url, bot_token,
                    channel_id=channel_id,
                    channel_type=channel_type,
                    content=content,
                    mention_uids=mention_uids,
                    mention_all=mention_all,
                    reply_msg_id=args.get("reply_to_message_id") or None,
                    client_msg_no=client_msg_no,
                    on_behalf_of=adapter.on_behalf_of,
                )
                _audit(
                    action="send-message", requester=requester_uid,
                    target=args["target"], channel_type=int(channel_type),
                )
                response: dict[str, Any] = {
                    "sent": True,
                    "channel_id": channel_id,
                    "channel_type": int(channel_type),
                }
                for field in ("message_id", "message_seq", "client_msg_no"):
                    value = getattr(send_result, field, None)
                    if value is not None:
                        response[field] = value
                return _ok(response)

            if action == "group-md-read":
                if (e := _require(args, "group_id")):
                    return e
                md = await api.get_group_md(session, api_url, bot_token, group_id)
                return _ok(md or {"content": "", "version": 0})

            if action == "group-md-update":
                if (e := _require(args, "group_id", "content")):
                    return e
                result = await api.update_group_md(
                    session, api_url, bot_token, group_no=group_id, content=content,
                )
                # Update the live adapter's local cache so the next inbound
                # message picks up the change immediately.
                try:
                    version = (result or {}).get("version", 0)
                    adapter._group_md_cache[group_id] = {
                        "content": content,
                        "version": version,
                    }
                    adapter._group_md_checked.add(group_id)
                    # Persist to disk so the change survives a restart.
                    if hasattr(adapter, "_write_md_to_disk"):
                        adapter._write_md_to_disk(group_id, content, version)
                except Exception:
                    pass
                return _ok({"updated": True, "version": (result or {}).get("version", 0)})

            if action == "create-group":
                if (e := _require(args, "members", "creator")):
                    return e
                result = await api.create_group(
                    session, api_url, bot_token,
                    members=args["members"],
                    creator=args["creator"],
                    name=args.get("name"),
                )
                return _ok(result)

            if action == "update-group":
                if (e := _require(args, "group_id")):
                    return e
                if args.get("name") is None and args.get("notice") is None:
                    return _err("update-group requires at least one of: name, notice")
                await api.update_group(
                    session, api_url, bot_token,
                    group_no=group_id,
                    name=args.get("name"),
                    notice=args.get("notice"),
                )
                return _ok({"updated": True, "group_id": group_id})

            if action == "add-members":
                if (e := _require(args, "group_id", "members")):
                    return e
                result = await api.add_group_members(
                    session, api_url, bot_token,
                    group_no=group_id, members=args["members"],
                )
                return _ok(result)

            if action == "remove-members":
                if (e := _require(args, "group_id", "members")):
                    return e
                result = await api.remove_group_members(
                    session, api_url, bot_token,
                    group_no=group_id, members=args["members"],
                )
                return _ok(result)

            if action == "create-thread":
                if (e := _require(args, "group_id", "thread_name")):
                    return e
                result = await api.create_thread(
                    session, api_url, bot_token,
                    group_no=group_id, name=args["thread_name"],
                )
                # Steer the agent to send any follow-up M1 INTO the new thread,
                # not back to the parent group. The generic send_message tool
                # only targets the originating chat (parent group), so the agent
                # must explicitly call octo_management send-message with the
                # thread target. Failing to do so silently sends M1 to the
                # parent group and leaves the new thread empty.
                short_id = (result or {}).get("short_id")
                thread_target = (
                    f"group:{group_id}____{short_id}" if short_id else None
                )
                hint = {
                    "next_step": (
                        "To send a message INTO this newly-created thread "
                        "(not the parent group), you MUST call this same tool "
                        "again with action='send-message' and "
                        f"target='{thread_target}'. Do NOT use the generic "
                        "send_message tool — it will deliver to the parent "
                        "group instead of the thread."
                    ),
                    "thread_target": thread_target,
                }
                payload = dict(result or {})
                payload["_agent_hint"] = hint
                return _ok(payload)

            if action == "list-threads":
                if (e := _require(args, "group_id")):
                    return e
                threads = await api.list_threads(session, api_url, bot_token, group_no=group_id)
                return _ok({"threads": threads})

            if action == "get-thread":
                if (e := _require(args, "group_id", "short_id")):
                    return e
                t = await api.get_thread(
                    session, api_url, bot_token, group_no=group_id, short_id=short_id
                )
                return _ok(t)

            if action == "delete-thread":
                if (e := _require(args, "group_id", "short_id")):
                    return e
                await api.delete_thread(
                    session, api_url, bot_token, group_no=group_id, short_id=short_id
                )
                return _ok({"deleted": True, "group_id": group_id, "short_id": short_id})

            if action == "list-thread-members":
                if (e := _require(args, "group_id", "short_id")):
                    return e
                members = await api.list_thread_members(
                    session, api_url, bot_token, group_no=group_id, short_id=short_id,
                )
                return _ok({"members": members})

            if action == "join-thread":
                if (e := _require(args, "group_id", "short_id")):
                    return e
                await api.join_thread(
                    session, api_url, bot_token, group_no=group_id, short_id=short_id
                )
                return _ok({"joined": True})

            if action == "leave-thread":
                if (e := _require(args, "group_id", "short_id")):
                    return e
                await api.leave_thread(
                    session, api_url, bot_token, group_no=group_id, short_id=short_id
                )
                return _ok({"left": True})

            if action == "thread-md-read":
                if (e := _require(args, "group_id", "short_id")):
                    return e
                md = await api.get_thread_md(
                    session, api_url, bot_token, group_no=group_id, short_id=short_id
                )
                return _ok(md)

            if action == "thread-md-update":
                if (e := _require(args, "group_id", "short_id", "content")):
                    return e
                result = await api.update_thread_md(
                    session, api_url, bot_token,
                    group_no=group_id, short_id=short_id, content=content,
                )
                # Mirror the cache update we do for GROUP.md so the next
                # inbound thread message sees the new content immediately.
                try:
                    key = f"{group_id}____{short_id}"
                    version = (result or {}).get("version", 0)
                    adapter._group_md_cache[key] = {
                        "content": content,
                        "version": version,
                    }
                    adapter._group_md_checked.add(key)
                    # Persist to disk too.
                    if hasattr(adapter, "_write_md_to_disk"):
                        adapter._write_md_to_disk(key, content, version)
                except Exception:
                    pass
                return _ok({"updated": True, "version": (result or {}).get("version", 0)})

            if action == "voice-context-read":
                ctx = await api.get_voice_context(session, api_url, bot_token)
                return _ok(ctx)

            if action == "voice-context-update":
                if (e := _require(args, "content")):
                    return e
                await api.update_voice_context(session, api_url, bot_token, content=content)
                return _ok({"updated": True})

            if action == "voice-context-delete":
                await api.delete_voice_context(session, api_url, bot_token)
                return _ok({"deleted": True})

        except Exception as e:
            # API/backend exception text can contain response diagnostics or
            # credentials.  Keep the tool boundary deterministic and safe;
            # the exception type is logged without serializing its message.
            logger.error(
                "octo_management action %s failed (%s)", action, type(e).__name__
            )
            return _err(f"{action} failed: upstream request failed")

    return _err(f"unhandled action: {action}")
