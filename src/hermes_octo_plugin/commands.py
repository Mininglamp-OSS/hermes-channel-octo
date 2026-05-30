"""
Slash commands for the Octo plugin.

Registered via ``ctx.register_command()`` so they appear in:
  - the hermes CLI (``hermes`` REPL)
  - every gateway session (Octo DM/group, Telegram, Discord, Slack...)
  - Discord's native slash picker (auto-mirrored)
  - Telegram's command menu (auto-mirrored)

Commands are intentionally **read-only** and scoped to "ops / debugging":
they expose internal adapter state so operators can inspect health, group
membership, and cached GROUP.md without leaving the chat or tailing
``gateway.log``. Mutations (cache refresh, voice context, etc.) are also
provided but **gated to the bot owner**.

Cross-platform behaviour
------------------------
All slash commands in hermes are global — a Telegram user can also type
``/octo_doctor``. We don't refuse those calls (some answers like
``/octo_groups`` are useful regardless of which platform asked), but
each handler logs the calling platform so audit trails are accurate.

Adapter resolution
------------------
Like ``octo_management``, we resolve the live adapter via
``gateway.run._gateway_runner_ref()``. When no gateway is running (CLI-only
session, no platform connected), commands return a clean
"adapter not running" message instead of crashing.

Event-loop hygiene
------------------
We avoid touching ``adapter._http_session`` directly — the same
worker-thread-loop trap that bit ``octo_management`` applies. Read-only
commands stay sync; the one async command (``/octo_refresh``) opens a
fresh ``aiohttp.ClientSession`` in the current loop.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


# ── Resolution helpers (shared with agent_tools) ────────────────────────────


def _resolve_adapter():
    """Return the live OctoAdapter, or None if no gateway is running."""
    try:
        from .agent_tools import _resolve_adapter as _r
        return _r()
    except Exception:
        return None


def _session_user_id() -> str:
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_USER_ID", "") or ""
    except Exception:
        return ""


def _session_platform() -> str:
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_PLATFORM", "") or ""
    except Exception:
        return ""


def _gate_owner(adapter) -> str | None:
    """Return None if the calling session belongs to the bot owner, else an
    error string. Falls back to denying when no session context is available
    — better to refuse than to silently allow from CLI / cron contexts."""
    owner = getattr(adapter, "_owner_uid", "") or ""
    requester = _session_user_id()
    if not owner:
        return (
            "⚠ bot owner not yet known (registration incomplete) — "
            "this command is unavailable"
        )
    if not requester:
        return (
            "⚠ this command requires owner privileges and no session "
            "identity is available (CLI / cron context)"
        )
    if requester != owner:
        return (
            f"⚠ owner-only command — calling user `{requester[:8]}…` is "
            f"not the bot owner `{owner[:8]}…`"
        )
    return None


def _no_adapter() -> str:
    return (
        "❌ Octo adapter is not running in this gateway process. "
        "Start the gateway with `OCTO_API_URL` + `OCTO_BOT_TOKEN` "
        "configured."
    )


# ── /octo_doctor ──────────────────────────────────────────────────────────


def doctor(_raw_args: str) -> str:
    """Connection + cache health snapshot. Public (no owner gate)."""
    adapter = _resolve_adapter()
    if adapter is None:
        return _no_adapter()
    info = {
        "connected": getattr(adapter, "_connected", False),
        "need_reconnect": getattr(adapter, "_need_reconnect", False),
        "reconnect_attempts": getattr(adapter, "_reconnect_attempts", 0),
        "reconnect_in_progress": getattr(adapter, "_reconnect_in_progress", False),
        "robot_id": getattr(adapter, "_robot_id", ""),
        "owner_uid": getattr(adapter, "_owner_uid", ""),
        "api_url": getattr(adapter, "_api_url", ""),
        "known_groups": len(getattr(adapter, "_known_group_ids", set())),
        "cached_member_names": len(getattr(adapter, "_uid_to_name", {})),
        "cached_group_md": len(getattr(adapter, "_group_md_cache", {})),
        "cached_chat_kinds": len(getattr(adapter, "_chat_kind", {})),
        "cache_activity_entries": len(getattr(adapter, "_cache_activity", {})),
        "calling_platform": _session_platform() or "<none>",
    }
    return "Octo doctor:\n" + json.dumps(info, ensure_ascii=False, indent=2)


# ── /octo_info ────────────────────────────────────────────────────────────


def info(_raw_args: str) -> str:
    """Plugin / bot version info. Public."""
    adapter = _resolve_adapter()
    try:
        from . import __version__ as plugin_version
    except Exception:
        plugin_version = "unknown"
    payload: dict = {
        "plugin_version": plugin_version,
    }
    if adapter is not None:
        reg = getattr(adapter, "_registration", None)
        payload["robot_id"] = getattr(adapter, "_robot_id", "")
        payload["owner_uid"] = getattr(adapter, "_owner_uid", "")
        payload["api_url"] = getattr(adapter, "_api_url", "")
        if reg is not None:
            payload["ws_url"] = getattr(reg, "ws_url", "")
            payload["owner_channel_id"] = getattr(reg, "owner_channel_id", "")
    else:
        payload["adapter"] = "not running"
    return "Octo info:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# ── /octo_groups ──────────────────────────────────────────────────────────


def groups(_raw_args: str) -> str:
    """List the groups the bot is currently known to be a member of.

    Source is the in-memory cache (populated by startup prefetch + inbound
    self-heal). If no groups are known yet, suggests running /octo_refresh.
    """
    adapter = _resolve_adapter()
    if adapter is None:
        return _no_adapter()
    names = getattr(adapter, "_group_names", {}) or {}
    known = getattr(adapter, "_known_group_ids", set()) or set()
    # Merge: groups with names go first, then any ids known but not yet named.
    all_ids = sorted(known | set(names))
    if not all_ids:
        return (
            "(no groups known yet — startup prefetch may still be running, "
            "or this bot isn't in any groups. Try `/octo_refresh`.)"
        )
    lines = [f"Octo groups ({len(all_ids)}):"]
    for gid in all_ids:
        gname = names.get(gid, "<no name>")
        lines.append(f"  {gname}  ({gid})")
    return "\n".join(lines)


# ── /octo_md <group_no_or_thread_key> ─────────────────────────────────────


def show_md(raw_args: str) -> str:
    """Display the cached GROUP.md or THREAD.md for the given key.

    Key format: ``<group_no>`` for group-level, ``<group_no>____<short_id>``
    for thread-level. Public — operators may want to inspect what context
    the LLM sees when replying in a given group.
    """
    key = raw_args.strip()
    if not key:
        return (
            "Usage: /octo_md <group_no>\n"
            "       /octo_md <group_no>____<short_id>   (for a thread)"
        )
    adapter = _resolve_adapter()
    if adapter is None:
        return _no_adapter()
    cache = getattr(adapter, "_group_md_cache", {}) or {}
    entry = cache.get(key)
    if not entry:
        return (
            f"(no cached MD for `{key}` — either no GROUP.md exists for "
            "this group, or the cache hasn't been populated yet. Try "
            "`/octo_refresh` then retry.)"
        )
    content = entry.get("content", "")
    version = entry.get("version", 0)
    header = f"GROUP.md for `{key}` (v{version}, {len(content)} chars):"
    return f"{header}\n```\n{content}\n```"


# ── /octo_refresh  (owner-only) ───────────────────────────────────────────


async def refresh_caches(_raw_args: str) -> str:
    """Force a full prefetch of group list + GROUP.md + member roster.

    Owner-only because it triggers N API calls and resets caches that may
    be in active use by other handlers. Returns immediately after kicking
    off the prefetch task; check ``/octo_doctor`` afterwards to see the
    new counts.
    """
    adapter = _resolve_adapter()
    if adapter is None:
        return _no_adapter()
    err = _gate_owner(adapter)
    if err:
        return err
    # Reset the check-once guards so prefetch actually re-fetches.
    try:
        adapter._group_md_checked.clear()
    except Exception:
        pass
    try:
        adapter._group_cache_timestamps.clear()
    except Exception:
        pass
    # Fire-and-forget; prefetch is bounded and logs its own progress.
    try:
        import asyncio
        asyncio.create_task(adapter._prefetch_groups_and_members())
    except Exception as e:
        return f"❌ refresh failed to schedule: {e}"
    return (
        "✅ Refresh scheduled. GROUP.md, member rosters, and known-group "
        "set will be re-fetched in the background. Run `/octo_doctor` "
        "in a few seconds to verify updated counts."
    )


# ── /octo_audit [lines]  (owner-only) ─────────────────────────────────────


def tail_audit(raw_args: str) -> str:
    """Show the most recent ``[AUDIT] octo-query`` lines from gateway.log.

    Owner-only because audit content can include channel ids / uids of
    third parties the requester shouldn't necessarily see. Defaults to
    20 lines; cap at 200 to keep responses readable.
    """
    adapter = _resolve_adapter()
    if adapter is None:
        return _no_adapter()
    err = _gate_owner(adapter)
    if err:
        return err
    n = 20
    arg = raw_args.strip()
    if arg:
        try:
            n = max(1, min(200, int(arg)))
        except ValueError:
            return f"⚠ invalid line count `{arg}`; usage: /octo_audit [N]"
    # Find the gateway log; fall back gracefully when unknown.
    log_path: str | None = None
    try:
        from hermes_constants import get_hermes_home
        candidate = os.path.join(get_hermes_home(), "logs", "gateway.log")
        if os.path.exists(candidate):
            log_path = candidate
    except Exception:
        pass
    if log_path is None:
        return "(gateway log path not resolvable in this environment)"
    try:
        # Read tail without loading whole file — typical log can be MBs.
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Heuristic: read last ~256 KB; should comfortably hold a few
            # hundred audit lines plus other gateway chatter.
            window = min(size, 256 * 1024)
            f.seek(size - window)
            tail = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"❌ failed to read log: {e}"
    audit_lines = [ln for ln in tail.splitlines() if "[AUDIT] octo-query" in ln]
    if not audit_lines:
        return "(no audit entries in the last 256 KB of gateway.log)"
    audit_lines = audit_lines[-n:]
    return (
        f"Octo audit log (last {len(audit_lines)} entries):\n"
        + "\n".join(audit_lines)
    )


# ── Registration entry point ────────────────────────────────────────────────


def register_all(ctx) -> None:
    """Register all Octo ops slash commands.

    Failures on a single command don't prevent the others from registering
    — slash commands are an ops nicety, not load-bearing.

    Name shape: we register with **hyphens** (``octo-doctor``). Hermes'
    gateway dispatch normalises any user-typed form via
    ``command.replace("_", "-")`` before looking up the handler — register
    with underscores and the lookup misses (registered name is stored
    verbatim while the lookup key is hyphenated), surfacing as
    "Unknown command /octo_doctor" in chat.
    """
    specs = [
        ("octo-doctor", doctor, "Octo bot connection + cache health snapshot.", ""),
        ("octo-info", info, "Octo plugin / bot version + registration info.", ""),
        ("octo-groups", groups, "List cached groups the bot is a member of.", ""),
        ("octo-md", show_md, "Show cached GROUP.md / THREAD.md for a group.", "<group_no>"),
        (
            "octo-refresh",
            refresh_caches,
            "(owner) Force re-fetch of group list / GROUP.md / members.",
            "",
        ),
        (
            "octo-audit",
            tail_audit,
            "(owner) Tail recent cross-channel query audit entries.",
            "[N]",
        ),
    ]
    for name, handler, description, args_hint in specs:
        try:
            ctx.register_command(
                name, handler=handler, description=description, args_hint=args_hint
            )
        except Exception as e:
            logger.warning("[Octo] register_command(%s) failed: %s", name, e)
