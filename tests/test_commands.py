"""Unit tests for slash commands in hermes_octo_plugin.commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hermes_octo_plugin import commands as cmds
from hermes_octo_plugin.identity import IdentityStateError


# ─── Helpers ────────────────────────────────────────────────────────────────


def _mock_runtime(**overrides) -> MagicMock:
    """Build a mock identity runtime with every field the commands read.

    Defaults to a fully populated, owner-known, healthy identity.
    """
    defaults = {
        "_connected": True,
        "_need_reconnect": True,
        "_reconnect_attempts": 0,
        "_reconnect_in_progress": False,
        "_robot_id": "bot_self",
        "_owner_uid": "owner_uid_abc",
        "_known_group_ids": {"g1", "g2"},
        "_uid_to_name": {"u1": "Alice", "u2": "Bob"},
        "_group_md_cache": {"g1": {"content": "G1 README", "version": 3}},
        "_chat_kind": {"g1": 2},
        "_cache_activity": {"g1": 0.0},
        "_group_names": {"g1": "Group One", "g2": "Group Two"},
        "_registration": None,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _mock_adapter(**overrides) -> MagicMock:
    """Build a mock adapter owning one identity runtime.

    Identity-scoped overrides are applied to the runtime, because that is where
    connection state, owner and authenticated caches live.
    """
    adapter_fields = {
        "is_connected": True,
        "_api_url": "https://im.deepminer.com.cn/api",
        "sole_identity_mode": True,
        "identity_phase": "single",
    }
    runtime_overrides = {
        key: value for key, value in overrides.items() if key not in adapter_fields
    }
    adapter_fields.update(
        {key: value for key, value in overrides.items() if key in adapter_fields}
    )
    runtime = _mock_runtime(**runtime_overrides)
    m = MagicMock()
    for k, v in adapter_fields.items():
        setattr(m, k, v)
    m.runtimes = (runtime,)
    m.resolve_trusted_runtime = MagicMock(return_value=runtime)
    m.routes.route_count.return_value = 0
    m.routes.session_count.return_value = 0
    m.routes.capacity.return_value = 4096
    m.routes.remaining_capacity.return_value = 4096
    m.routes.capacity_exhausted.return_value = False
    return m


def _set_session_user(uid: str):
    return patch("hermes_octo_plugin.commands._session_user_id", return_value=uid)


# ─── /octo_doctor ─────────────────────────────────────────────────────────


class TestDoctor:
    def test_no_adapter_returns_friendly_error(self):
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=None):
            out = cmds.doctor("")
        assert "not running" in out

    def test_reports_connection_and_cache_counts(self):
        a = _mock_adapter()
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.doctor("")
        # Strip the header line — body is JSON
        body = out.split("\n", 1)[1]
        data = json.loads(body)
        assert data["connected"] is True
        assert data["identities"] == 1
        assert data["sole_identity_mode"] is True
        identity = data["runtimes"][0]
        assert identity["robot_id"] == "bot_self"
        assert identity["owner_uid"] == "owner_uid_abc"
        assert identity["known_groups"] == 2
        assert identity["cached_member_names"] == 2
        assert identity["cached_group_md"] == 1

    def test_snapshot_never_exposes_credentials(self):
        a = _mock_adapter()
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.doctor("")
        assert "token" not in out.lower()

    def test_multi_identity_snapshot_exposes_only_the_calling_runtime(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        caller = adapter.runtimes[0]
        foreign = _mock_runtime(
            _robot_id="bot_foreign",
            _owner_uid="foreign-owner",
        )
        adapter.runtimes = (caller, foreign)
        adapter.resolve_trusted_runtime.return_value = caller
        adapter.routes.route_count.return_value = 0
        adapter.routes.session_count.return_value = 0

        with patch(
            "hermes_octo_plugin.commands._resolve_adapter",
            return_value=adapter,
        ):
            out = cmds.doctor("")

        data = json.loads(out.split("\n", 1)[1])
        assert data["identities"] == 2
        assert [item["robot_id"] for item in data["runtimes"]] == ["bot_self"]
        assert "bot_foreign" not in out
        assert "foreign-owner" not in out

    def test_multi_identity_snapshot_reports_route_capacity(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        adapter.routes.route_count.return_value = 4095
        adapter.routes.session_count.return_value = 4095
        adapter.routes.capacity.return_value = 4096
        adapter.routes.remaining_capacity.return_value = 1
        adapter.routes.capacity_exhausted.return_value = False

        with patch(
            "hermes_octo_plugin.commands._resolve_adapter",
            return_value=adapter,
        ):
            out = cmds.doctor("")

        data = json.loads(out.split("\n", 1)[1])
        assert data["route_capacity"] == 4096
        assert data["route_capacity_remaining"] == 1
        assert data["route_capacity_exhausted"] is False


# ─── /octo_info ───────────────────────────────────────────────────────────


class TestInfo:
    def test_includes_plugin_version_and_robot_id(self):
        a = _mock_adapter()
        a.runtimes[0]._registration = MagicMock(
            ws_url="wss://im.test/ws", owner_channel_id="oc_x"
        )
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.info("")
        body = out.split("\n", 1)[1]
        data = json.loads(body)
        assert "plugin_version" in data
        identity = data["identities"][0]
        assert identity["robot_id"] == "bot_self"
        assert identity["ws_url"] == "wss://im.test/ws"
        assert identity["owner_channel_id"] == "oc_x"

    def test_multi_identity_info_exposes_only_the_calling_runtime(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        caller = adapter.runtimes[0]
        foreign = _mock_runtime(
            _robot_id="bot_foreign",
            _owner_uid="foreign-owner",
            _registration=MagicMock(
                ws_url="wss://internal.invalid/ws",
                owner_channel_id="foreign-channel",
            ),
        )
        adapter.runtimes = (caller, foreign)
        adapter.resolve_trusted_runtime.return_value = caller

        with patch(
            "hermes_octo_plugin.commands._resolve_adapter",
            return_value=adapter,
        ):
            out = cmds.info("")

        data = json.loads(out.split("\n", 1)[1])
        assert [item["robot_id"] for item in data["identities"]] == ["bot_self"]
        assert "bot_foreign" not in out
        assert "foreign-owner" not in out
        assert "internal.invalid" not in out

    def test_no_adapter_still_returns_plugin_version(self):
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=None):
            out = cmds.info("")
        body = out.split("\n", 1)[1]
        data = json.loads(body)
        assert "plugin_version" in data
        assert data.get("adapter") == "not running"


# ─── /octo_groups ─────────────────────────────────────────────────────────


class TestGroups:
    def test_lists_named_groups(self):
        a = _mock_adapter()
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.groups("")
        assert "Group One" in out
        assert "Group Two" in out
        assert "(g1)" in out
        assert "(g2)" in out
        assert "(2):" in out  # header count

    def test_includes_unnamed_groups(self):
        a = _mock_adapter(
            _known_group_ids={"g1", "g_unnamed"},
            _group_names={"g1": "Group One"},
        )
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.groups("")
        assert "Group One" in out
        assert "g_unnamed" in out
        assert "<no name>" in out

    def test_empty_suggests_refresh(self):
        a = _mock_adapter(_known_group_ids=set(), _group_names={})
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.groups("")
        assert "no groups known" in out
        assert "/octo_refresh" in out


# ─── /octo_md ─────────────────────────────────────────────────────────────


class TestShowMd:
    def test_empty_args_shows_usage(self):
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=_mock_adapter()):
            out = cmds.show_md("")
        assert out.lower().startswith("usage")

    def test_existing_group_md_shown(self):
        a = _mock_adapter()
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.show_md("g1")
        assert "v3" in out
        assert "G1 README" in out
        assert "9 chars" in out  # len("G1 README") = 9

    def test_missing_md_suggests_refresh(self):
        a = _mock_adapter(_group_md_cache={})
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.show_md("g1")
        assert "no cached MD" in out
        assert "/octo_refresh" in out

    def test_thread_key_lookup(self):
        a = _mock_adapter(
            _group_md_cache={"g1____thr_x": {"content": "thread content", "version": 7}},
        )
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a):
            out = cmds.show_md("g1____thr_x")
        assert "v7" in out
        assert "thread content" in out


# ─── Owner gating (used by refresh + audit) ─────────────────────────────────


class TestOwnerGate:
    """The gate runs against the identity that owns the calling conversation."""

    def test_owner_session_passes(self):
        a = _mock_adapter()
        with _set_session_user("owner_uid_abc"):
            err = cmds._gate_owner(a.runtimes[0])
        assert err is None

    def test_non_owner_denied(self):
        a = _mock_adapter()
        with _set_session_user("random_user"):
            err = cmds._gate_owner(a.runtimes[0])
        assert err is not None
        assert "owner-only" in err

    def test_no_session_identity_denied(self):
        a = _mock_adapter()
        with _set_session_user(""):
            err = cmds._gate_owner(a.runtimes[0])
        assert err is not None
        assert "no session identity" in err

    def test_no_owner_uid_denied(self):
        a = _mock_adapter(_owner_uid="")
        with _set_session_user("anyone"):
            err = cmds._gate_owner(a.runtimes[0])
        assert err is not None
        assert "owner not yet known" in err


# ─── /octo_refresh ────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRefresh:
    async def test_non_owner_denied(self):
        a = _mock_adapter()
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a), \
             _set_session_user("intruder"):
            out = await cmds.refresh_caches("")
        assert "owner-only" in out

    async def test_owner_clears_check_guards_and_schedules(self):
        a = _mock_adapter()
        runtime = a.runtimes[0]
        runtime._group_md_checked = {"g1"}
        runtime._group_cache_timestamps = {"g1": 123}
        # Make _prefetch awaitable so create_task accepts it
        async def fake_prefetch():
            return None
        runtime._prefetch_groups_and_members = fake_prefetch

        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a), \
             _set_session_user("owner_uid_abc"):
            out = await cmds.refresh_caches("")

        assert "Refresh scheduled" in out
        assert runtime._group_md_checked == set()
        assert runtime._group_cache_timestamps == {}



# ─── /octo-route-forget ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestForgetRoute:
    async def test_non_owner_is_denied(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        with (
            patch(
                "hermes_octo_plugin.commands._resolve_adapter",
                return_value=adapter,
            ),
            _set_session_user("intruder"),
        ):
            out = await cmds.forget_route("group retired")

        assert "owner-only" in out
        adapter.routes.forget_target.assert_not_called()

    async def test_group_requires_transcript_inheritance_confirmation(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        with (
            patch(
                "hermes_octo_plugin.commands._resolve_adapter",
                return_value=adapter,
            ),
            _set_session_user("owner_uid_abc"),
        ):
            out = await cmds.forget_route("group retired")

        assert "inherit" in out
        assert "--confirm-transcript-inheritance" in out
        adapter.routes.forget_target.assert_not_called()

    async def test_owner_can_forget_confirmed_group_route(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        adapter.routes.forget_target.return_value = True
        with (
            patch(
                "hermes_octo_plugin.commands._resolve_adapter",
                return_value=adapter,
            ),
            _set_session_user("owner_uid_abc"),
        ):
            out = await cmds.forget_route(
                "group retired --confirm-transcript-inheritance"
            )

        assert "forgot route" in out
        assert "existing Hermes transcript" in out
        adapter.routes.forget_target.assert_called_once_with(
            robot_id="bot_self",
            chat_type="group",
            chat_id="retired",
        )

    async def test_dm_route_does_not_require_group_confirmation(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        adapter.routes.forget_target.return_value = True
        with (
            patch(
                "hermes_octo_plugin.commands._resolve_adapter",
                return_value=adapter,
            ),
            _set_session_user("owner_uid_abc"),
        ):
            out = await cmds.forget_route("dm retired-peer")

        assert "forgot route" in out
        assert "transcript" not in out

    async def test_persistence_failure_is_reported_as_not_forgotten(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        adapter.routes.forget_target.side_effect = IdentityStateError(
            "Octo route deletion could not be persisted"
        )
        with (
            patch(
                "hermes_octo_plugin.commands._resolve_adapter",
                return_value=adapter,
            ),
            _set_session_user("owner_uid_abc"),
        ):
            out = await cmds.forget_route("dm retired-peer")

        assert "was not forgotten" in out
        assert "could not be persisted" in out

    async def test_usage_requires_type_and_chat_id(self):
        adapter = _mock_adapter(sole_identity_mode=False)
        with patch(
            "hermes_octo_plugin.commands._resolve_adapter",
            return_value=adapter,
        ):
            out = await cmds.forget_route("retired")
        assert "Usage:" in out

# ─── /octo_audit ──────────────────────────────────────────────────────────


class TestAudit:
    def test_non_owner_denied(self):
        a = _mock_adapter()
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a), \
             _set_session_user("intruder"):
            out = cmds.tail_audit("")
        assert "owner-only" in out

    def test_invalid_line_count(self):
        a = _mock_adapter()
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a), \
             _set_session_user("owner_uid_abc"):
            out = cmds.tail_audit("abc")
        assert "invalid line count" in out

    def test_reads_log_and_filters_audit_entries(self, tmp_path):
        log_file = tmp_path / "gateway.log"
        log_file.write_text(
            "irrelevant line 1\n"
            "2026-05-19 12:00:00 INFO foo: [AUDIT] octo-query {\"action\":\"read-messages\"}\n"
            "2026-05-19 12:00:01 INFO foo: ordinary log line\n"
            "2026-05-19 12:00:02 INFO foo: [AUDIT] octo-query {\"action\":\"search-members\"}\n",
            encoding="utf-8",
        )
        a = _mock_adapter()

        # Patch hermes_constants.get_hermes_home so the audit reader finds
        # our tmp log at <home>/logs/gateway.log.
        logs_dir = tmp_path
        # The function expects logs/ subdir; create that layout.
        (tmp_path / "logs").mkdir(exist_ok=True)
        target = tmp_path / "logs" / "gateway.log"
        target.write_text(log_file.read_text(encoding="utf-8"), encoding="utf-8")

        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a), \
             _set_session_user("owner_uid_abc"), \
             patch("hermes_constants.get_hermes_home", return_value=str(tmp_path)):
            out = cmds.tail_audit("5")
        assert "read-messages" in out
        assert "search-members" in out
        assert "ordinary log line" not in out

    def test_no_audit_entries_friendly_message(self, tmp_path):
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "gateway.log").write_text(
            "boring line\nanother line\n", encoding="utf-8",
        )
        a = _mock_adapter()
        with patch("hermes_octo_plugin.commands._resolve_adapter", return_value=a), \
             _set_session_user("owner_uid_abc"), \
             patch("hermes_constants.get_hermes_home", return_value=str(tmp_path)):
            out = cmds.tail_audit("")
        assert "no audit entries" in out


# ─── register_all ───────────────────────────────────────────────────────────


class TestRegistration:
    def test_register_all_invokes_register_command_for_each(self):
        ctx = MagicMock()
        cmds.register_all(ctx)
        names = {call.args[0] for call in ctx.register_command.call_args_list}
        assert names == {
            "octo-doctor",
            "octo-info",
            "octo-groups",
            "octo-md",
            "octo-refresh",
            "octo-route-forget",
            "octo-audit",
        }

    def test_register_all_swallows_per_command_errors(self):
        ctx = MagicMock()
        # Make register_command raise for one specific command but succeed for others
        def fake_register(name, **_kw):
            if name == "octo-md":
                raise RuntimeError("simulated conflict")

        ctx.register_command.side_effect = fake_register
        # Should not raise
        cmds.register_all(ctx)
        # Still attempted all 7
        assert ctx.register_command.call_count == 7
