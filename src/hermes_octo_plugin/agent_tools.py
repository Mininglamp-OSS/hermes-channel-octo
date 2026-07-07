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
  - Mutating operations (create-group / add-members / group-md-update /
    voice-context-update) require *requester_uid* to equal the bot owner.
"""

from __future__ import annotations

import errno
import json
import logging
import os
from datetime import UTC
from pathlib import Path
from typing import Any

import aiohttp

from . import api
from .permission import check_permission, parse_target
from .types import ChannelType

logger = logging.getLogger(__name__)


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
    # Owner secret resolution / local file write
    "write-secret",
]


TOOL_SCHEMA = {
    "name": "octo_management",
    "description": (
        "Manage Octo groups, threads, GROUP.md, voice-correction "
        "context, and read messages across channels. Single tool with "
        "an action parameter — pick the action and supply the required "
        "fields. Cross-channel read enforces permissions: only the bot "
        "owner may read another user's DM; group/thread reads require "
        "the requester to be a member. Pass `requester_uid` (the uid of "
        "the human currently chatting with the bot) for ALL read / "
        "search / mutation calls so permission checks have something to "
        "evaluate; omit it only for owner-only admin calls that you are "
        "running on the owner's behalf."
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
            "requester_uid": {
                "type": "string",
                "description": (
                    "uid of the human who triggered this call (the sender "
                    "of the current Octo message). Used for permission "
                    "checks on read / search / mutation actions. Omit only "
                    "for owner-initiated admin calls."
                ),
            },
            "alias": {
                "type": "string",
                "description": (
                    "For write-secret only. Alias or secret_id for one of the "
                    "bot owner's stored secrets. Pass the alias, never a raw "
                    "secret value; the tool resolves plaintext internally."
                ),
            },
            "file_path": {
                "type": "string",
                "description": (
                    "For write-secret only. Destination path relative to the "
                    "configured secrets jail root (e.g. '.env'). Paths escaping "
                    "the jail are rejected."
                ),
            },
            "template": {
                "type": "string",
                "description": (
                    "For write-secret only. Optional content template containing "
                    "the '{{secret}}' placeholder where the resolved value should "
                    "be inserted. If omitted or blank, writes the raw secret."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": (
                    "For write-secret only. 'overwrite' replaces the file; "
                    "'append' appends to it. Defaults to overwrite."
                ),
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SECRET_PLACEHOLDER = "{{secret}}"
SECRET_FILE_MODE = 0o600


def _default_secrets_root() -> Path:
    """Default write-secret jail under the active Hermes profile."""
    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(hermes_home).expanduser() / "workspace" / "octo" / "secrets"


def _secrets_root() -> Path:
    configured = os.environ.get("OCTO_SECRETS_FILE_ROOT", "").strip()
    return Path(configured).expanduser() if configured else _default_secrets_root()


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _canonicalize_existing_prefix(path: Path) -> Path:
    """Resolve symlinks through the nearest existing ancestor."""
    cur = path
    missing: list[str] = []
    while not cur.exists():
        if cur.parent == cur:
            break
        missing.append(cur.name)
        cur = cur.parent
    base = cur.resolve(strict=True) if cur.exists() else cur.resolve(strict=False)
    for part in reversed(missing):
        base = base / part
    return base


def _confine_secret_path(file_path: str, root_input: Path | None = None) -> tuple[Path, Path] | str:
    """Return (safe_abs, canonical_root) or an error string."""
    if not file_path or "\x00" in file_path:
        return "file_path is required for write-secret and must not contain NUL bytes"

    root_raw = root_input or _secrets_root()
    try:
        root = _canonicalize_existing_prefix(root_raw)
    except Exception as exc:  # pragma: no cover - rare platform/fs failure
        return f"write-secret is not configured: could not resolve secrets root ({type(exc).__name__})"

    if root.anchor == str(root):
        return "Refusing to use filesystem root as the write-secret jail root"

    requested = Path(file_path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    candidate = candidate.resolve(strict=False)

    if not _is_inside(root, candidate):
        return (
            f"Refusing to write the secret outside the allowed directory. "
            f"{file_path!r} resolves outside the permitted root."
        )

    rel = candidate.relative_to(root)
    current = root
    for part in rel.parts:
        current = current / part
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            break
        if os.path.islink(current):
            try:
                real = current.resolve(strict=True)
            except OSError:
                return (
                    f"Refusing to write the secret: {file_path!r} passes through "
                    "a symlink that cannot be verified to stay inside the allowed directory."
                )
            if not _is_inside(root, real):
                return (
                    f"Refusing to write the secret: {file_path!r} resolves through "
                    "a symlink that escapes the allowed directory."
                )
    return candidate, root


async def _handle_write_secret(
    session: aiohttp.ClientSession,
    api_url: str,
    bot_token: str,
    *,
    alias: str,
    file_path: str,
    template: str | None = None,
    mode: str = "overwrite",
) -> str:
    """Resolve a stored secret and write it locally without exposing plaintext."""
    if not alias:
        return _err("alias is required for write-secret")
    if not file_path:
        return _err("file_path is required for write-secret")
    if mode not in ("overwrite", "append"):
        return _err(f"invalid mode for write-secret: {mode!r}; use 'overwrite' or 'append'")
    if template is not None and template.strip() and SECRET_PLACEHOLDER not in template:
        return _err(
            f"template was provided but does not contain the {SECRET_PLACEHOLDER} placeholder"
        )

    confined = _confine_secret_path(file_path)
    if isinstance(confined, str):
        return _err(confined)
    abs_path, root = confined

    try:
        resolved = await api.resolve_secret(session, api_url, bot_token, alias)
    except Exception as exc:
        return _err(
            f"Could not resolve secret {alias!r}. The key service is unavailable "
            f"or the secret may need to be re-set. ({type(exc).__name__}: {exc})"
        )

    status = resolved.get("status")
    if status == "not_found":
        return _err(f"No stored secret matches {alias!r}. Ask the user to add this key first, then try again.")
    if status == "rate_limited":
        return _err(f"The key service is busy right now (rate limited). Wait a moment and retry write-secret for {alias!r}.")
    if status == "ambiguous":
        return _ok({
            "written": False,
            "ambiguous": True,
            "message": f"Multiple stored secrets match {alias!r}. Ask the user which one, then retry write-secret with the chosen display_name or secret_id.",
            "candidates": resolved.get("candidates", []),
        })
    if status != "resolved" or not isinstance(resolved.get("value"), str):
        return _err("resolveSecret returned an unusable response")

    value = resolved["value"]
    content = (
        template.replace(SECRET_PLACEHOLDER, value)
        if template and SECRET_PLACEHOLDER in template
        else value
    )

    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        real_parent = abs_path.parent.resolve(strict=True)
        if not _is_inside(root, real_parent):
            return _err("Refusing to write the secret: the destination directory escaped the allowed root after creation.")
        open_target = real_parent / abs_path.name
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode == "append" else os.O_TRUNC)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(open_target, flags, SECRET_FILE_MODE)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                return _err("Refusing to write the secret: the destination file is a symlink.")
            raise
        with os.fdopen(fd, "a" if mode == "append" else "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fchmod(f.fileno(), SECRET_FILE_MODE)
    except Exception as exc:
        rel = os.path.relpath(abs_path, root)
        return _err(f"Resolved the secret but failed to write it to {rel!r}: {type(exc).__name__}: {exc}")

    result: dict[str, Any] = {
        "written": True,
        "path": os.path.relpath(abs_path, root),
        "mode": mode,
    }
    if isinstance(resolved.get("display_name"), str):
        result["display_name"] = resolved["display_name"]
    return _ok(result)



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
        # Implicit owner attribution is OK only when we have an owner on file.
        # Otherwise refuse — better than mutating with no accountability.
        if owner:
            return None
        return _err(f"action '{action}' requires requester_uid; owner unknown")
    return _err(
        f"action '{action}' requires bot-owner privileges; "
        f"requester {requester_uid!r} is not the owner"
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


OWNER_ONLY_ACTIONS = frozenset({
    "create-group", "update-group", "add-members", "remove-members",
    "group-md-update", "create-thread", "delete-thread", "join-thread",
    "leave-thread", "thread-md-update",
    "voice-context-update", "voice-context-delete",
    "write-secret",
})


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

    requester_uid: str | None = args.get("requester_uid")

    if action == "write-secret" and not requester_uid:
        return _err("action 'write-secret' requires explicit requester_uid")

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
    async with aiohttp.ClientSession() as session:
        try:
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

                # Threads: the server accepts messages from non-members (HTTP 2xx)
                # but does not push them to subscribers. Best-effort join before
                # send so newly-created threads receive M1. Already-joined / race
                # errors are expected and swallowed.
                if channel_type == ChannelType.CommunityTopic and "____" in channel_id:
                    group_no, short_id = channel_id.split("____", 1)
                    try:
                        await api.join_thread(
                            session, api_url, bot_token,
                            group_no=group_no, short_id=short_id,
                        )
                    except Exception as exc:
                        logger.debug(
                            "join_thread best-effort before send-message failed "
                            "(likely already joined): %s", exc
                        )

                await api.send_message(
                    session, api_url, bot_token,
                    channel_id=channel_id,
                    channel_type=channel_type,
                    content=content,
                    mention_uids=mention_uids,
                    mention_all=mention_all,
                    reply_msg_id=args.get("reply_to_message_id") or None,
                )
                _audit(
                    action="send-message", requester=requester_uid,
                    target=args["target"], channel_type=int(channel_type),
                )
                return _ok({
                    "sent": True,
                    "channel_id": channel_id,
                    "channel_type": int(channel_type),
                })

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

            if action == "write-secret":
                alias = (args.get("alias") or "").strip()
                file_path = (args.get("file_path") or args.get("filePath") or "").strip()
                template = args.get("template")
                mode = args.get("mode") or "overwrite"
                return await _handle_write_secret(
                    session, api_url, bot_token,
                    alias=alias, file_path=file_path,
                    template=template if isinstance(template, str) else None,
                    mode=mode,
                )

        except Exception as e:
            logger.exception("octo_management action %s failed", action)
            return _err(f"{action} failed: {type(e).__name__}: {e}")

    return _err(f"unhandled action: {action}")
