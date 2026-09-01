"""Multi-token Octo identity routing contracts.

One ``octo`` platform, one ``OctoAdapter``, several bot tokens.  These tests
pin the behaviour that keeps a conversation on its own identity: durable
routes, fail-closed unknown routes, one-shot legacy migration, token rotation,
per-identity card bindings, and the absence of any credential in persisted
state, logs or tool surfaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin import identity
from hermes_octo_plugin.adapter import IdentityRuntime, OctoAdapter
from hermes_octo_plugin.card_sessions import CardSession, CardSessionRegistry
from hermes_octo_plugin.identity import (
    BIND_BOUND,
    BIND_CAPACITY,
    BIND_CONFLICT,
    BIND_IDEMPOTENT,
    PHASE_MIGRATED,
    PHASE_PENDING,
    PHASE_SINGLE,
    CardBindingStore,
    IdentityRouteRegistry,
    IdentityConflictError,
    IdentityStateError,
    IdentityMetadata,
    IdentityStateStore,
    PlannedRoute,
    TokenConfigError,
    parse_bot_tokens,
)
from hermes_octo_plugin.protocol import PacketType
from tests.conftest import make_bare_adapter, make_bare_octo_adapter
from hermes_octo_plugin.types import BotRegisterResp, ChannelType, SendMessageResult

TOKEN_A = "secret-token-alpha"
TOKEN_B = "secret-token-bravo"
TOKEN_C = "secret-token-charlie"


# ── Configuration contract ────────────────────────────────────────────────


class TestTokenParsing:
    def test_single_token_is_unchanged(self):
        assert parse_bot_tokens(TOKEN_A) == (TOKEN_A,)

    def test_semicolons_separate_tokens_and_whitespace_is_trimmed(self):
        assert parse_bot_tokens(f" {TOKEN_A} ; {TOKEN_B};{TOKEN_C} ") == (
            TOKEN_A,
            TOKEN_B,
            TOKEN_C,
        )

    def test_absent_configuration_yields_no_tokens(self):
        assert parse_bot_tokens("") == ()
        assert parse_bot_tokens("   ") == ()
        assert parse_bot_tokens(None) == ()

    @pytest.mark.parametrize(
        "raw",
        [
            f"{TOKEN_A};",
            f";{TOKEN_A}",
            f"{TOKEN_A};;{TOKEN_B}",
            ";",
        ],
    )
    def test_empty_entries_are_configuration_errors(self, raw):
        with pytest.raises(TokenConfigError) as excinfo:
            parse_bot_tokens(raw)
        assert "empty" in str(excinfo.value)

    def test_repeated_token_is_a_configuration_error(self):
        with pytest.raises(TokenConfigError) as excinfo:
            parse_bot_tokens(f"{TOKEN_A};{TOKEN_B};{TOKEN_A}")
        assert "#3 repeats entry #1" in str(excinfo.value)

    @pytest.mark.parametrize(
        "raw",
        [f"{TOKEN_A};", f"{TOKEN_A};{TOKEN_A}", f"{TOKEN_A};;{TOKEN_B}"],
    )
    def test_errors_never_echo_a_token(self, raw):
        with pytest.raises(TokenConfigError) as excinfo:
            parse_bot_tokens(raw)
        message = str(excinfo.value)
        for token in (TOKEN_A, TOKEN_B):
            assert token not in message

    def test_non_string_configuration_is_rejected(self):
        with pytest.raises(TokenConfigError):
            parse_bot_tokens(["a", "b"])


# ── Adapter construction ──────────────────────────────────────────────────


def _config(tokens: str, **extra) -> SimpleNamespace:
    return SimpleNamespace(
        extra={
            "api_url": "https://api.octo.invalid",
            "bot_token": tokens,
            **extra,
        }
    )
def _adapter(
    tmp_path: Path,
    tokens: str,
    *,
    seed_legacy_identity: bool = True,
    **extra,
) -> OctoAdapter:
    adapter = OctoAdapter(_config(tokens, **extra), state_base_dir=tmp_path)
    parsed = parse_bot_tokens(tokens)
    if (
        seed_legacy_identity
        and len(parsed) > 1
        and adapter.identity_state.load_metadata() is None
    ):
        first_robot_id = {
            TOKEN_A: "bot_alpha",
            TOKEN_B: "bot_bravo",
            TOKEN_C: "bot_charlie",
        }.get(parsed[0], "bot_alpha")
        adapter.identity_state.save_metadata(
            IdentityMetadata(
                phase=PHASE_SINGLE,
                legacy_robot_id=first_robot_id,
            )
        )
    return adapter


def _bare_multi_adapter():
    first = make_bare_adapter()
    second = make_bare_adapter()
    adapter = make_bare_octo_adapter(runtimes=(first, second))
    first._adapter = adapter
    second._adapter = adapter
    return adapter, first, second


class TestAdapterInventory:
    def test_single_token_creates_one_identity_runtime(self, tmp_path):
        adapter = _adapter(tmp_path, TOKEN_A)
        assert len(adapter.runtimes) == 1
        assert adapter.sole_identity_mode is True
        assert adapter.runtimes[0].bot_token == TOKEN_A

    def test_three_tokens_share_one_adapter_and_platform(self, tmp_path):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B};{TOKEN_C}")
        assert isinstance(adapter, OctoAdapter)
        assert len(adapter.runtimes) == 3
        assert [runtime.bot_token for runtime in adapter.runtimes] == [
            TOKEN_A,
            TOKEN_B,
            TOKEN_C,
        ]
        # One platform, one adapter: every runtime points back at the same one.
        assert {id(runtime.adapter) for runtime in adapter.runtimes} == {id(adapter)}
        assert adapter.platform is adapter.runtimes[0].adapter.platform

    def test_shared_configuration_reaches_every_runtime(self, tmp_path):
        adapter = _adapter(
            tmp_path,
            f"{TOKEN_A};{TOKEN_B}",
            cdn_url="https://cdn.octo.invalid",
            require_mention=False,
            history_limit=7,
        )
        for runtime in adapter.runtimes:
            assert runtime._api_url == "https://api.octo.invalid"
            assert runtime._cdn_url == "https://cdn.octo.invalid"
            assert runtime._require_mention is False
            assert runtime._history_limit == 7

    def test_unconfigured_platform_still_constructs(self, tmp_path):
        adapter = _adapter(tmp_path, "")
        assert adapter.runtimes == ()

    @pytest.mark.asyncio
    async def test_unconfigured_platform_refuses_to_connect(self, tmp_path, caplog):
        adapter = _adapter(tmp_path, "")
        caplog.set_level(logging.ERROR, logger="hermes_octo_plugin.adapter")
        assert await adapter.connect() is False
        assert "OCTO_API_URL and OCTO_BOT_TOKEN must be set" in caplog.text


# ── Connect / lifecycle with fake registrations ───────────────────────────


def _install_fake_connect(monkeypatch, mapping: dict[str, str], *, failing=()):
    """Replace runtime.connect with a register-only stand-in.

    ``mapping`` maps bot token to the ``robot_id`` the server would return.
    Tokens in ``failing`` raise, mimicking an identity that cannot register.
    """

    async def fake_connect(self, *, is_reconnect: bool = False) -> bool:
        del is_reconnect
        if self.bot_token in failing:
            raise RuntimeError("registration refused")
        robot_id = mapping[self.bot_token]
        self._adapter._claim_identity(self, robot_id)
        self._robot_id = robot_id
        self._owner_uid = f"owner-of-{robot_id}"
        await self._adapter._on_runtime_registered(self)
        self._connected = True
        self._mark_connected()
        return True

    async def fake_disconnect(self) -> None:
        self._connected = False

    monkeypatch.setattr(IdentityRuntime, "connect", fake_connect)
    monkeypatch.setattr(IdentityRuntime, "disconnect", fake_disconnect)


class TestConnectAggregation:
    @pytest.mark.asyncio
    async def test_single_token_records_only_the_single_phase_sentinel(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})

        assert await adapter.connect() is True
        assert adapter.sole_identity_mode is True
        metadata = adapter.identity_state.load_metadata()
        assert metadata.phase == PHASE_SINGLE
        assert metadata.legacy_robot_id == "bot_alpha"
        # No route snapshot is written on the single-identity fast path.
        assert not adapter.routes.path.exists()

    @pytest.mark.asyncio
    async def test_single_identity_does_not_listen_if_robot_id_cannot_persist(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        monkeypatch.setattr(
            adapter.identity_state,
            "record_single_identity",
            MagicMock(side_effect=OSError("read-only filesystem")),
        )

        assert await adapter.connect() is False
        assert adapter.runtimes[0].is_runtime_connected is False


    @pytest.mark.asyncio
    async def test_single_token_identity_change_keeps_the_original_sentinel(
        self, tmp_path, monkeypatch
    ):
        original = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        assert await original.connect() is True
        await original.disconnect()

        legacy_key = "agent:main:octo:dm:legacy-peer"
        replacement = _adapter(tmp_path, TOKEN_C)
        replacement.set_session_store(
            _Store([_entry(legacy_key, "legacy-peer")])
        )
        _install_fake_connect(monkeypatch, {TOKEN_C: "bot_charlie"})
        assert await replacement.connect() is True
        metadata = replacement.identity_state.load_metadata()
        assert metadata.phase == PHASE_MIGRATED
        assert metadata.legacy_robot_id == "bot_alpha"
        assert replacement.routes.robot_id_for_session(legacy_key) == "bot_alpha"
        assert replacement.runtimes[0].robot_id == "bot_charlie"
        assert replacement.runtimes[0].is_runtime_connected is True

        runtime, _resolved, error = replacement.resolve_outbound("legacy-peer")
        assert runtime is None
        assert "route is not established" in error
        assert _bind(
            replacement, "bot_charlie", "new-peer", ChannelType.DM
        ) is True

    @pytest.mark.asyncio
    async def test_one_failing_identity_does_not_stop_the_others(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
            failing=(TOKEN_A,),
        )

        assert await adapter.connect() is True
        assert adapter.runtimes[0].is_runtime_connected is False
        assert adapter.runtimes[1].is_runtime_connected is True
        assert adapter.is_connected is True

    @pytest.mark.asyncio
    async def test_platform_is_offline_only_when_every_identity_fails(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
            failing=(TOKEN_A, TOKEN_B),
        )

        assert await adapter.connect() is False
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_duplicate_identity_across_tokens_fails_that_runtime(
        self, tmp_path, monkeypatch, caplog
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_alpha"}
        )
        caplog.set_level(logging.ERROR, logger="hermes_octo_plugin.adapter")

        assert await adapter.connect() is True
        connected = [r for r in adapter.runtimes if r.is_runtime_connected]
        assert len(connected) == 1
        assert adapter.runtime_for_robot_id("bot_alpha") is connected[0]
        assert "registered as the same identity" in caplog.text

    @pytest.mark.asyncio
    async def test_runtime_connect_propagates_identity_conflicts(
        self, tmp_path, monkeypatch
    ):
        runtime = _adapter(tmp_path, TOKEN_A).runtimes[0]
        monkeypatch.setattr(runtime, "_new_http_session", MagicMock(return_value=object()))
        monkeypatch.setattr(
            runtime,
            "_do_connect",
            AsyncMock(
                side_effect=IdentityConflictError(
                    "two configured Octo tokens registered as the same identity"
                )
            ),
        )
        finalize = AsyncMock(return_value=None)
        monkeypatch.setattr(
            runtime, "_finalize_disconnect_resources_shielded", finalize
        )

        with pytest.raises(IdentityConflictError):
            await runtime._connect_locked()
        finalize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initial_connection_failure_schedules_a_runtime_retry(
        self, tmp_path, monkeypatch
    ):
        runtime = _adapter(tmp_path, TOKEN_A).runtimes[0]
        monkeypatch.setattr(
            runtime, "_new_http_session", MagicMock(return_value=object())
        )
        monkeypatch.setattr(
            runtime,
            "_do_connect",
            AsyncMock(side_effect=RuntimeError("registration unavailable")),
        )
        monkeypatch.setattr(
            runtime,
            "_finalize_disconnect_resources_shielded",
            AsyncMock(return_value=None),
        )
        spawn = MagicMock()
        monkeypatch.setattr(runtime, "_spawn_reconnect_task", spawn)

        assert await runtime._connect_locked() is False
        spawn.assert_called_once_with()


    @pytest.mark.asyncio
    async def test_receive_loop_exit_refreshes_aggregate_connection_state(self):
        adapter, first, second = _bare_multi_adapter()
        for runtime in (first, second):
            runtime._connected = True
            runtime._need_reconnect = False
            runtime._ws = SimpleNamespace(
                recv=AsyncMock(side_effect=RuntimeError("socket closed"))
            )
        adapter._refresh_connection_state()

        first_task = asyncio.create_task(first._receive_loop())
        first._recv_task = first_task
        await first_task
        assert adapter.is_connected is True

        second_task = asyncio.create_task(second._receive_loop())
        second._recv_task = second_task
        await second_task
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_server_disconnect_refreshes_aggregate_connection_state(self):
        adapter, first, second = _bare_multi_adapter()
        for runtime in (first, second):
            runtime._connected = True
            runtime._ws = MagicMock(close=AsyncMock())
        adapter._refresh_connection_state()

        with patch(
            "hermes_octo_plugin.adapter.decode_packet",
            return_value=(PacketType.DISCONNECT, {}),
        ):
            await first._handle_frame(b"disconnect")
            assert adapter.is_connected is True
            await second._handle_frame(b"disconnect")

        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_ping_timeout_refreshes_aggregate_connection_state(self):
        adapter, first, second = _bare_multi_adapter()
        first._connected = False
        second._connected = True
        second._ping_max_retry = 0
        second._ws = MagicMock(close=AsyncMock())
        adapter._refresh_connection_state()

        with patch("hermes_octo_plugin.adapter.asyncio.sleep", AsyncMock()):
            await second._heartbeat_loop()

        assert adapter.is_connected is False
    @pytest.mark.asyncio
    async def test_disconnect_drains_every_identity_and_flushes_routes(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_bravo", "s9_peer", ChannelType.DM)

        await adapter.disconnect()

        assert all(not r.is_runtime_connected for r in adapter.runtimes)
        assert adapter.is_connected is False
        assert adapter.routes.path.exists()


# ── Route binding ─────────────────────────────────────────────────────────


def _source(chat_id: str, *, chat_type: str = "dm", user_id: str = "u1"):
    return SimpleNamespace(
        platform=SimpleNamespace(value="octo"),
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_id_alt=None,
        thread_id=None,
        prospective_thread_id=None,
    )


def _bind(adapter, robot_id, chat_id, channel_type, *, user_id="u1"):
    chat_type = "dm" if channel_type == ChannelType.DM else "group"
    return adapter.bind_inbound_route(
        robot_id=robot_id,
        source=_source(chat_id, chat_type=chat_type, user_id=user_id),
        channel_type=channel_type,
    )


class TestRouteBinding:
    def test_repeated_binding_of_the_same_identity_is_idempotent(self):
        routes = IdentityRouteRegistry(path=Path("/nonexistent/routes.json"))
        first = routes.bind(
            robot_id="bot_alpha",
            session_key="agent:main:octo:dm:s1_x",
            chat_type="dm",
            chat_id="s1_x",
            channel_type=int(ChannelType.DM),
        )
        second = routes.bind(
            robot_id="bot_alpha",
            session_key="agent:main:octo:dm:s1_x",
            chat_type="dm",
            chat_id="s1_x",
            channel_type=int(ChannelType.DM),
        )
        assert first == BIND_BOUND
        assert second == BIND_IDEMPOTENT
        assert routes.robot_id_for_session("agent:main:octo:dm:s1_x") == "bot_alpha"

    def test_second_identity_claiming_a_route_is_a_permanent_conflict(self):
        routes = IdentityRouteRegistry(path=Path("/nonexistent/routes.json"))
        routes.bind(
            robot_id="bot_alpha",
            session_key="agent:main:octo:dm:s1_x",
            chat_type="dm",
            chat_id="s1_x",
            channel_type=int(ChannelType.DM),
        )
        assert (
            routes.bind(
                robot_id="bot_bravo",
                session_key="agent:main:octo:dm:s1_x",
                chat_type="dm",
                chat_id="s1_x",
                channel_type=int(ChannelType.DM),
            )
            == BIND_CONFLICT
        )
        # Fail closed for both identities until state is repaired.
        assert routes.session_conflict("agent:main:octo:dm:s1_x") is True
        assert routes.robot_id_for_session("agent:main:octo:dm:s1_x") is None
        assert routes.target_conflict("dm", "s1_x") is True
        assert routes.lookup_chat_id("s1_x") is None
        assert (
            routes.bind(
                robot_id="bot_alpha",
                session_key="agent:main:octo:dm:s1_x",
                chat_type="dm",
                chat_id="s1_x",
                channel_type=int(ChannelType.DM),
            )
            == BIND_CONFLICT
        )

    def test_target_conflict_blocks_every_existing_session(self):
        routes = IdentityRouteRegistry(path=Path("/nonexistent/routes.json"))
        first_session = "agent:main:octo:group:g1:user-a"
        routes.bind(
            robot_id="bot_alpha",
            session_key=first_session,
            chat_type="group",
            chat_id="g1",
            channel_type=int(ChannelType.Group),
        )

        assert (
            routes.bind(
                robot_id="bot_bravo",
                session_key="agent:main:octo:group:g1:user-b",
                chat_type="group",
                chat_id="g1",
                channel_type=int(ChannelType.Group),
            )
            == BIND_CONFLICT
        )
        assert routes.target_conflict("group", "g1") is True
        assert routes.session_conflict(first_session) is True
        assert routes.robot_id_for_session(first_session) is None

    def test_full_route_store_rejects_new_target_without_evicting_incumbent(
        self, monkeypatch
    ):
        monkeypatch.setattr(identity, "MAX_PERSISTED_ROUTES", 1)
        routes = IdentityRouteRegistry(path=Path("/nonexistent/routes.json"))
        incumbent_session = "agent:main:octo:group:shared"

        assert (
            routes.bind(
                robot_id="bot_alpha",
                session_key=incumbent_session,
                chat_type="group",
                chat_id="shared",
                channel_type=int(ChannelType.Group),
            )
            == BIND_BOUND
        )
        assert (
            routes.bind(
                robot_id="bot_bravo",
                session_key="agent:main:octo:group:new",
                chat_type="group",
                chat_id="new",
                channel_type=int(ChannelType.Group),
            )
            == BIND_CAPACITY
        )

        assert routes.route_count() == 1
        assert routes.robot_id_for_target("group", "shared") == "bot_alpha"
        assert routes.robot_id_for_session(incumbent_session) == "bot_alpha"
        assert routes.robot_id_for_target("group", "new") is None

    def test_owner_can_durably_forget_exact_unconflicted_target(self, tmp_path):
        path = tmp_path / "routes.json"
        routes = IdentityRouteRegistry(path=path)
        session_key = "agent:main:octo:group:retired"
        assert (
            routes.bind(
                robot_id="bot_alpha",
                session_key=session_key,
                chat_type="group",
                chat_id="retired",
                channel_type=int(ChannelType.Group),
            )
            == BIND_BOUND
        )
        assert routes.flush(force=True) is True

        assert (
            routes.forget_target(
                robot_id="bot_bravo",
                chat_type="group",
                chat_id="retired",
            )
            is False
        )
        assert (
            routes.forget_target(
                robot_id="bot_alpha",
                chat_type="group",
                chat_id="retired",
            )
            is True
        )
        assert routes.robot_id_for_target("group", "retired") is None
        assert routes.robot_id_for_session(session_key) is None
        assert routes.dirty is False

        restored = IdentityRouteRegistry(path=path)
        restored.load()
        assert restored.robot_id_for_target("group", "retired") is None
        assert restored.robot_id_for_session(session_key) is None

        conflicted = IdentityRouteRegistry(path=tmp_path / "conflicted.json")
        conflicted.bind(
            robot_id="bot_alpha",
            session_key="conflicted-alpha",
            chat_type="group",
            chat_id="conflicted",
            channel_type=int(ChannelType.Group),
        )
        conflicted.bind(
            robot_id="bot_bravo",
            session_key="conflicted-bravo",
            chat_type="group",
            chat_id="conflicted",
            channel_type=int(ChannelType.Group),
        )
        assert (
            conflicted.forget_target(
                robot_id="bot_alpha",
                chat_type="group",
                chat_id="conflicted",
            )
            is False
        )

    def test_forget_target_rolls_back_when_snapshot_write_fails(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "routes.json"
        routes = IdentityRouteRegistry(path=path)
        session_key = "agent:main:octo:dm:retired"
        assert (
            routes.bind(
                robot_id="bot_alpha",
                session_key=session_key,
                chat_type="dm",
                chat_id="retired",
                channel_type=int(ChannelType.DM),
                wire_chat_id="retired",
            )
            == BIND_BOUND
        )
        assert routes.flush(force=True) is True

        def fail_write(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(identity, "write_state_document", fail_write)
        with pytest.raises(IdentityStateError, match="could not be persisted"):
            routes.forget_target(
                robot_id="bot_alpha",
                chat_type="dm",
                chat_id="retired",
            )

        assert routes.robot_id_for_target("dm", "retired") == "bot_alpha"
        assert routes.robot_id_for_session(session_key) == "bot_alpha"
        assert routes.dirty is False

        restored = IdentityRouteRegistry(path=path)
        restored.load()
        assert restored.robot_id_for_target("dm", "retired") == "bot_alpha"
        assert restored.robot_id_for_session(session_key) == "bot_alpha"

    @pytest.mark.asyncio
    async def test_capacity_refusal_logs_distinct_reason(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setattr(identity, "MAX_PERSISTED_ROUTES", 1)
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
        )
        assert await adapter.connect() is True
        assert _bind(adapter, "bot_alpha", "first", ChannelType.Group) is True
        caplog.set_level(logging.ERROR, logger="hermes_octo_plugin.adapter")

        assert _bind(adapter, "bot_alpha", "second", ChannelType.Group) is False
        assert "route capacity exhausted" in caplog.text

    def test_migrated_dm_wire_identity_is_unknown_until_trusted_inbound(
        self, tmp_path
    ):
        session_key = "agent:main:octo:dm:s5_u1"
        routes = IdentityRouteRegistry(path=tmp_path / "routes.json")
        plan = (
            PlannedRoute(
                session_key=session_key,
                chat_type="dm",
                chat_id="s5_u1",
                channel_type=int(ChannelType.DM),
            ),
        )

        assert routes.bind_plan(robot_id="bot_alpha", plan=plan) is True
        assert routes.robot_id_for_session(session_key) == "bot_alpha"
        assert routes.lookup_chat_id("s5_u1") is None
        assert routes.flush(force=True) is True

        restored = IdentityRouteRegistry(path=tmp_path / "routes.json")
        restored.load()
        assert restored.robot_id_for_session(session_key) == "bot_alpha"
        assert restored.lookup_chat_id("s5_u1") is None
        assert (
            restored.bind(
                robot_id="bot_alpha",
                session_key=session_key,
                chat_type="dm",
                chat_id="s5_u1",
                channel_type=int(ChannelType.DM),
                wire_chat_id="u1",
            )
            == BIND_BOUND
        )
        assert not restored.target_conflict("dm", "s5_u1")
        lookup = restored.lookup_chat_id("s5_u1")
        assert lookup is not None

        assert lookup.wire_chat_id == "u1"
    def test_legacy_placeholder_wire_snapshot_waits_for_trusted_inbound(
        self, tmp_path
    ):
        path = tmp_path / "legacy-routes.json"
        identity.write_state_document(
            path,
            {
                "sessions": [
                    {
                        "session_key": "agent:main:octo:dm:s5_u1",
                        "robot_id": "bot_alpha",
                        "chat_type": "dm",
                        "chat_id": "s5_u1",
                    }
                ],
                "targets": [
                    {
                        "chat_type": "dm",
                        "chat_id": "s5_u1",
                        "robot_id": "bot_alpha",
                        "channel_type": int(ChannelType.DM),
                        "wire_chat_id": "s5_u1",
                    }
                ],
                "session_conflicts": [],
                "target_conflicts": [],
            },
        )
        routes = IdentityRouteRegistry(path=path)
        routes.load()

        assert routes.lookup_chat_id("s5_u1") is None
        assert (
            routes.bind(
                robot_id="bot_alpha",
                session_key="agent:main:octo:dm:s5_u1",
                chat_type="dm",
                chat_id="s5_u1",
                channel_type=int(ChannelType.DM),
                wire_chat_id="u1",
            )
            == BIND_BOUND
        )
        lookup = routes.lookup_chat_id("s5_u1")
        assert lookup is not None
        assert lookup.wire_chat_id == "u1"

    def test_concurrent_binding_of_one_target_keeps_a_single_owner(self):
        routes = IdentityRouteRegistry(path=Path("/nonexistent/routes.json"))
        results: list[str] = []

        def claim(robot_id: str, index: int) -> None:
            results.append(
                routes.bind(
                    robot_id=robot_id,
                    session_key=f"agent:main:octo:group:g1:{index}",
                    chat_type="group",
                    chat_id="g1",
                    channel_type=int(ChannelType.Group),
                )
            )

        threads = [
            threading.Thread(
                target=claim, args=("bot_alpha" if i % 2 else "bot_bravo", i)
            )
            for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert BIND_CONFLICT in results
        assert routes.target_conflict("group", "g1") is True

    @pytest.mark.asyncio
    async def test_sole_identity_mode_binds_nothing(self, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        await adapter.connect()

        assert _bind(adapter, "bot_alpha", "s1_peer", ChannelType.DM) is True
        assert adapter.routes.route_count() == 0

    @pytest.mark.asyncio
    async def test_every_session_of_one_target_maps_to_one_identity(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()

        assert _bind(adapter, "bot_alpha", "g1", ChannelType.Group, user_id="u1")
        assert _bind(adapter, "bot_alpha", "g1", ChannelType.Group, user_id="u2")
        lookup = adapter.routes.lookup_chat_id("g1")
        assert lookup.robot_id == "bot_alpha"
        assert adapter.routes.session_count() == 2

    @pytest.mark.asyncio
    async def test_bind_during_route_write_schedules_another_snapshot(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        original_write = identity.write_state_document
        first_write = threading.Event()
        release_first = threading.Event()
        second_write = threading.Event()
        guard = threading.Lock()
        calls = 0

        def slow_write(path, body):
            nonlocal calls
            with guard:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_write.set()
                assert release_first.wait(2)
            else:
                second_write.set()
            original_write(path, body)

        monkeypatch.setattr(identity, "write_state_document", slow_write)
        assert _bind(adapter, "bot_alpha", "s-write-1", ChannelType.DM)
        assert await asyncio.to_thread(first_write.wait, 1)
        assert _bind(adapter, "bot_alpha", "s-write-2", ChannelType.DM)
        release_first.set()

        wrote_again = await asyncio.to_thread(second_write.wait, 1)
        for _ in range(100):
            if not adapter.routes.dirty and not adapter._route_flush_tasks:
                break
            await asyncio.sleep(0.01)

        assert wrote_again
        assert calls == 2
        assert adapter.routes.dirty is False

    @pytest.mark.asyncio
    async def test_different_spaces_never_cross_routes(self, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()

        assert _bind(adapter, "bot_alpha", "s1_peer", ChannelType.DM)
        assert _bind(adapter, "bot_bravo", "s2_peer", ChannelType.DM)

        assert adapter.routes.lookup_chat_id("s1_peer").robot_id == "bot_alpha"
        assert adapter.routes.lookup_chat_id("s2_peer").robot_id == "bot_bravo"

    @pytest.mark.asyncio
    async def test_conflicting_inbound_message_is_refused(self, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()

        assert _bind(adapter, "bot_alpha", "g1", ChannelType.Group) is True
        assert _bind(adapter, "bot_bravo", "g1", ChannelType.Group) is False


# ── Inbound binding happens before Hermes dispatch ────────────────────────


def _recv(channel_id: str, from_uid: str, channel_type: int):
    return SimpleNamespace(
        message_id="m-1",
        message_seq=1,
        from_uid=from_uid,
        channel_id=channel_id,
        channel_type=channel_type,
        timestamp=0,
        encrypted_payload=b"",
    )


async def _dispatch_text(runtime, *, text: str, channel_id: str, from_uid: str):
    payload = {"type": 1, "content": text}
    with patch(
        "hermes_octo_plugin.adapter.aes_decrypt",
        return_value=json.dumps(payload).encode("utf-8"),
    ):
        await runtime._handle_recv(
            _recv(channel_id, from_uid, int(ChannelType.DM))
        )


class TestInboundBinding:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["hello", "/new", "/reset"])
    async def test_route_is_bound_before_message_dispatch(
        self, tmp_path, monkeypatch, text
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        runtime = adapter.runtimes[0]

        bound_at_dispatch: list[str | None] = []

        async def capture(event):
            bound_at_dispatch.append(
                adapter.routes.robot_id_for_session(
                    adapter.session_key_for_source(event.source)
                )
            )

        runtime.handle_message = capture
        await _dispatch_text(
            runtime, text=text, channel_id="s3_peer", from_uid="s3_u1"
        )

        assert bound_at_dispatch == ["bot_alpha"]

    @pytest.mark.asyncio
    async def test_trusted_turn_identity_is_visible_to_tools(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        runtime = adapter.runtimes[1]
        runtime._robot_id = "bot_bravo"

        observed: list[object] = []

        async def capture(_event):
            observed.append(adapter.resolve_trusted_runtime())

        runtime.handle_message = capture
        await _dispatch_text(
            runtime, text="hi", channel_id="s4_peer", from_uid="s4_u1"
        )

        assert observed == [runtime]
        # The context variable is scoped to the turn.
        assert identity.current_robot_id.get() == ""

    @pytest.mark.asyncio
    async def test_same_bare_dm_uid_is_scoped_per_runtime(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        first, second = adapter.runtimes
        events: list[object] = []

        async def capture(event):
            events.append(event)

        first.handle_message = capture
        second.handle_message = capture
        peer_uid = "6e91530e4b39442894ec4d887753c178"
        await _dispatch_text(
            first, text="space one", channel_id=peer_uid, from_uid=peer_uid
        )
        await _dispatch_text(
            second, text="space two", channel_id=peer_uid, from_uid=peer_uid
        )

        assert len(events) == 2
        first_chat_id = events[0].source.chat_id
        second_chat_id = events[1].source.chat_id
        assert first_chat_id == peer_uid
        assert second_chat_id != peer_uid
        assert second.robot_id not in second_chat_id
        first_route = adapter.routes.lookup_chat_id(first_chat_id)
        second_route = adapter.routes.lookup_chat_id(second_chat_id)
        assert first_route.robot_id == first.robot_id
        assert second_route.robot_id == second.robot_id
        assert first_route.wire_chat_id == peer_uid
        assert second_route.wire_chat_id == peer_uid

        await adapter.disconnect()
        restarted = _adapter(tmp_path, f"{TOKEN_B};{TOKEN_A}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await restarted.connect()
        runtime, resolved, error = restarted.resolve_outbound(second_chat_id)
        assert error == ""
        assert runtime.robot_id == "bot_bravo"
        assert resolved["channel_type"] == int(ChannelType.DM)
        assert (
            runtime._outbound_channel_id(second_chat_id, ChannelType.DM)
            == peer_uid
        )

    @pytest.mark.asyncio
    async def test_conflicting_inbound_never_reaches_hermes(
        self, tmp_path, monkeypatch, caplog
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        first, second = adapter.runtimes
        caplog.set_level(logging.ERROR, logger="hermes_octo_plugin.adapter")

        first.handle_message = AsyncMock()
        second.handle_message = AsyncMock()
        second._send_typing_safe = AsyncMock()
        await _dispatch_text(
            first, text="one", channel_id="s5_peer", from_uid="s5_u1"
        )
        await _dispatch_text(
            second, text="two", channel_id="s5_peer", from_uid="s5_u1"
        )
        await asyncio.sleep(0)

        first.handle_message.assert_awaited_once()
        second.handle_message.assert_not_awaited()
        second._send_typing_safe.assert_not_awaited()
        assert "identity route conflict" in caplog.text


# ── Outbound resolution ───────────────────────────────────────────────────


class TestOutboundResolution:
    @pytest.mark.asyncio
    async def test_chat_id_only_send_uses_the_persisted_reverse_index(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_bravo", "s6_peer", ChannelType.DM)

        runtime, resolved, error = adapter.resolve_outbound("s6_peer")

        assert error == ""
        assert runtime is adapter.runtimes[1]
        # A Space DM must not be re-derived from a group-shaped chat id.
        assert resolved["channel_type"] == int(ChannelType.DM)

    @pytest.mark.asyncio
    async def test_restarted_space_dm_does_not_fall_back_to_a_heuristic(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "s7_peer", ChannelType.DM)
        await adapter.disconnect()

        restarted = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await restarted.connect()

        runtime, resolved, error = restarted.resolve_outbound("s7_peer")
        assert error == ""
        assert runtime is restarted.runtimes[0]
        assert resolved["channel_type"] == int(ChannelType.DM)
        # The runtime's own heuristic would have guessed a group here.
        assert runtime._resolve_channel_type("s7_peer") == ChannelType.Group
        assert (
            runtime._outbound_channel_id("s7_peer", ChannelType.DM)
            == "peer"
        )

    @pytest.mark.asyncio
    async def test_unknown_route_fails_closed(self, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()

        runtime, _resolved, error = adapter.resolve_outbound("never-seen")
        assert runtime is None
        assert "route is not established" in error

        result = await adapter.send("never-seen", "hello")
        assert result.success is False
        assert "route is not established" in result.error

    @pytest.mark.asyncio
    async def test_home_channel_uses_the_stable_legacy_primary_only(
        self, tmp_path, monkeypatch
    ):
        IdentityStateStore(base_dir=tmp_path / "identity").save_metadata(
            IdentityMetadata(
                phase=PHASE_SINGLE,
                legacy_robot_id="bot_alpha",
            )
        )
        first_config = _config(f"{TOKEN_A};{TOKEN_B}")
        first_config.home_channel = SimpleNamespace(chat_id="home-group")
        first = OctoAdapter(first_config, state_base_dir=tmp_path)
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await first.connect()
        await first.disconnect()

        reordered_config = _config(f"{TOKEN_B};{TOKEN_A}")
        reordered_config.home_channel = SimpleNamespace(chat_id="home-group")
        reordered = OctoAdapter(reordered_config, state_base_dir=tmp_path)
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await reordered.connect()

        runtime, _resolved, error = reordered.resolve_outbound("home-group")
        assert error == ""
        assert runtime is reordered.runtimes[1]
        assert runtime.robot_id == "bot_alpha"
        runtime._connected = False
        runtime, _resolved, error = reordered.resolve_outbound("home-group")
        assert runtime is None
        assert "bot_alpha" in error
        assert "not routed through another identity" in error

        runtime, _resolved, error = reordered.resolve_outbound("another-group")
        assert runtime is None
        assert "route is not established" in error

    @pytest.mark.asyncio
    async def test_ambiguous_chat_id_fails_closed(self, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        # Same bare chat id reached as a DM by one identity and as a group by
        # the other: the reverse index must refuse to choose.
        _bind(adapter, "bot_alpha", "ambiguous", ChannelType.DM)
        _bind(adapter, "bot_bravo", "ambiguous", ChannelType.Group)

        assert adapter.routes.lookup_chat_id("ambiguous") is None
        runtime, _resolved, error = adapter.resolve_outbound("ambiguous")
        assert runtime is None
        assert "route is not established" in error

    @pytest.mark.asyncio
    async def test_conflicted_home_channel_does_not_fall_back_to_primary(
        self, tmp_path, monkeypatch
    ):
        config = _config(f"{TOKEN_A};{TOKEN_B}")
        config.home_channel = SimpleNamespace(chat_id="home-group")
        adapter = OctoAdapter(config, state_base_dir=tmp_path)
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "home-group", ChannelType.Group)
        assert _bind(
            adapter, "bot_bravo", "home-group", ChannelType.Group
        ) is False

        runtime, _resolved, error = adapter.resolve_outbound("home-group")
        assert runtime is None
        assert "route is not established" in error

    @pytest.mark.asyncio
    async def test_offline_identity_is_reported_without_switching_tokens(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "s8_peer", ChannelType.DM)
        # The owning identity disappears from the live registry.
        del adapter._runtimes_by_robot_id["bot_alpha"]

        runtime, _resolved, error = adapter.resolve_outbound("s8_peer")
        assert runtime is None
        assert "bot_alpha" in error
        assert "not routed through another identity" in error

    @pytest.mark.asyncio
    async def test_disconnected_identity_is_reported_without_sending(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "s8_offline", ChannelType.DM)
        owner = adapter.runtimes[0]
        owner._connected = False
        owner.send = AsyncMock()

        runtime, _resolved, error = adapter.resolve_outbound("s8_offline")
        result = await adapter.send("s8_offline", "must not leave")

        assert runtime is None
        assert "bot_alpha" in error
        assert "not routed through another identity" in error
        assert result.success is False
        owner.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejoining_the_same_identity_restores_its_routes(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "s8_peer", ChannelType.DM)
        del adapter._runtimes_by_robot_id["bot_alpha"]
        adapter._claim_identity(adapter.runtimes[0], "bot_alpha")

        runtime, _resolved, error = adapter.resolve_outbound("s8_peer")
        assert error == ""
        assert runtime is adapter.runtimes[0]

    @pytest.mark.asyncio
    async def test_trusted_turn_identity_wins_over_the_target_index(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        token = identity.current_robot_id.set("bot_bravo")
        try:
            runtime, _resolved, error = adapter.resolve_outbound("never-bound")
        finally:
            identity.current_robot_id.reset(token)
        assert error == ""
        assert runtime is adapter.runtimes[1]


    @pytest.mark.asyncio
    async def test_trusted_turn_rejects_a_foreign_target_route(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "owned-by-alpha", ChannelType.Group)

        token = identity.current_robot_id.set("bot_bravo")
        try:
            runtime, _resolved, error = adapter.resolve_outbound(
                "owned-by-alpha"
            )
        finally:
            identity.current_robot_id.reset(token)

        assert runtime is None
        assert "route is not established" in error

    @pytest.mark.asyncio
    async def test_trusted_turn_rejects_an_explicitly_conflicted_target(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "conflicted", ChannelType.Group)
        assert _bind(
            adapter, "bot_bravo", "conflicted", ChannelType.Group
        ) is False

        token = identity.current_robot_id.set("bot_alpha")
        try:
            runtime, _resolved, error = adapter.resolve_outbound("conflicted")
        finally:
            identity.current_robot_id.reset(token)

        assert runtime is None
        assert "route is not established" in error
    @pytest.mark.asyncio
    async def test_media_sends_route_to_the_owning_identity(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_bravo", "s10_peer", ChannelType.DM)
        target = adapter.runtimes[1]

        for name, args in (
            ("send", ("s10_peer", "text")),
            ("send_image", ("s10_peer", "https://cdn.invalid/a.png")),
            ("send_document", ("s10_peer", "/tmp/a.pdf")),
            ("send_voice", ("s10_peer", "/tmp/a.ogg")),
            ("send_video", ("s10_peer", "/tmp/a.mp4")),
            ("get_chat_info", ("s10_peer",)),
        ):
            spy = AsyncMock(return_value="ok")
            monkeypatch.setattr(target, name, spy)
            other = AsyncMock()
            monkeypatch.setattr(adapter.runtimes[0], name, other)
            await getattr(adapter, name)(*args)
            spy.assert_awaited_once()
            other.assert_not_awaited()


# ── Legacy migration state machine ────────────────────────────────────────


    @pytest.mark.asyncio
    async def test_platform_read_methods_delegate_to_the_trusted_runtime(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        first, second = adapter.runtimes
        first.check_read_permission = AsyncMock()
        second.check_read_permission = AsyncMock(return_value=("allowed", "g1", 2))
        first.read_channel_messages = AsyncMock()
        second.read_channel_messages = AsyncMock(return_value={"ok": True})

        token = identity.current_robot_id.set("bot_bravo")
        try:
            permission = await adapter.check_read_permission("u1", "g1")
            messages = await adapter.read_channel_messages(
                requester_uid="u1",
                target="g1",
                limit=7,
            )
        finally:
            identity.current_robot_id.reset(token)

        assert permission == ("allowed", "g1", 2)
        assert messages == {"ok": True}
        second.check_read_permission.assert_awaited_once_with("u1", "g1")
        second.read_channel_messages.assert_awaited_once_with(
            requester_uid="u1",
            target="g1",
            limit=7,
        )
        first.check_read_permission.assert_not_awaited()
        first.read_channel_messages.assert_not_awaited()

class _Store:
    """Minimal stand-in for the Hermes session store's public enumeration."""

    def __init__(self, entries):
        self._entries = entries

    def list_sessions(self):
        return self._entries


def _entry(session_key: str, chat_id: str, chat_type: str = "dm", origin=True):
    origin_obj = (
        SimpleNamespace(
            platform=SimpleNamespace(value="octo"),
            chat_id=chat_id,
            chat_type=chat_type,
        )
        if origin
        else None
    )
    return SimpleNamespace(session_key=session_key, origin=origin_obj)


class TestLegacyMigration:
    @pytest.mark.asyncio
    async def test_empty_legacy_plan_still_records_the_primary_and_finishes(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )

        assert await adapter.connect() is True
        metadata = adapter.identity_state.load_metadata()
        assert adapter.identity_phase == PHASE_MIGRATED
        assert metadata.legacy_robot_id == "bot_alpha"
        assert not adapter.identity_state.plan_path.exists()

    @pytest.mark.asyncio
    async def test_verified_primary_is_durable_before_applying_plan(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        observed: list[str] = []
        original_apply = adapter.identity_state.apply_legacy_plan

        def apply_after_primary_is_durable(*, robot_id, plan):
            metadata = adapter.identity_state.load_metadata()
            observed.append(metadata.legacy_robot_id)
            return original_apply(robot_id=robot_id, plan=plan)

        monkeypatch.setattr(
            adapter.identity_state,
            "apply_legacy_plan",
            apply_after_primary_is_durable,
        )
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )

        assert await adapter.connect() is True
        assert observed == ["bot_alpha"]

    @pytest.mark.asyncio
    async def test_first_multi_token_start_requires_a_verified_single_identity(
        self, tmp_path, monkeypatch
    ):
        legacy_key = "agent:main:octo:dm:legacy-peer"
        adapter = _adapter(
            tmp_path,
            f"{TOKEN_A};{TOKEN_B}",
            seed_legacy_identity=False,
        )
        adapter.set_session_store(
            _Store([_entry(legacy_key, "legacy-peer")])
        )
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
        )

        assert await adapter.connect() is False
        assert adapter.identity_state.load_metadata() is None
        assert not adapter.identity_state.plan_path.exists()
        assert adapter.routes.robot_id_for_session(legacy_key) is None

    def test_oversized_legacy_plan_is_refused_without_state_transition(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(identity, "MAX_PERSISTED_ROUTES", 1)
        store = IdentityStateStore(base_dir=tmp_path)
        store.save_metadata(
            IdentityMetadata(
                phase=PHASE_SINGLE,
                legacy_robot_id="bot_alpha",
            )
        )

        with pytest.raises(IdentityStateError, match="exceeds route capacity"):
            store.begin(
                token_count=2,
                enumerate_legacy_sessions=lambda: [
                    ("session-one", "dm", "peer-one"),
                    ("session-two", "dm", "peer-two"),
                ],
            )

        metadata = store.load_metadata()
        assert metadata.phase == PHASE_SINGLE
        assert not store.plan_path.exists()

    def test_pending_metadata_plan_size_mismatch_is_refused(self, tmp_path):
        store = IdentityStateStore(base_dir=tmp_path)
        store.save_metadata(
            IdentityMetadata(
                phase=PHASE_PENDING,
                legacy_robot_id="bot_alpha",
                plan_size=2,
            )
        )
        store.save_plan(
            (
                PlannedRoute(
                    session_key="session-one",
                    chat_type="dm",
                    chat_id="peer-one",
                    channel_type=int(ChannelType.DM),
                ),
            )
        )

        with pytest.raises(IdentityStateError, match="plan size"):
            store.begin(
                token_count=2,
                enumerate_legacy_sessions=lambda: (),
            )

    @pytest.mark.asyncio
    async def test_empty_single_sentinel_cannot_authorize_multi_token_start(
        self, tmp_path, monkeypatch
    ):
        single = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha"},
            failing=(TOKEN_A,),
        )
        assert await single.connect() is False
        metadata = single.identity_state.load_metadata()
        assert metadata.phase == PHASE_SINGLE
        assert metadata.legacy_robot_id == ""

        multi = _adapter(
            tmp_path,
            f"{TOKEN_A};{TOKEN_B}",
            seed_legacy_identity=False,
        )
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
        )
        assert await multi.connect() is False
        assert multi.identity_state.load_metadata().phase == PHASE_SINGLE
        assert not multi.identity_state.plan_path.exists()

    @pytest.mark.asyncio
    async def test_pending_plan_with_known_primary_allows_safe_revert(
        self, tmp_path, monkeypatch
    ):
        legacy_key = "agent:main:octo:dm:legacy-peer"
        single = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        assert await single.connect() is True
        await single.disconnect()

        pending = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        pending.set_session_store(
            _Store([_entry(legacy_key, "legacy-peer")])
        )
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
            failing=(TOKEN_A,),
        )
        assert await pending.connect() is True
        assert pending.identity_state.load_metadata().legacy_robot_id == "bot_alpha"
        await pending.disconnect()

        reverted = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        assert await reverted.connect() is True
        assert reverted.identity_phase == PHASE_MIGRATED
        assert reverted.routes.robot_id_for_session(legacy_key) == "bot_alpha"

    @pytest.mark.asyncio
    async def test_pending_migration_connects_primary_before_secondary(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        secondary_phases: list[str] = []

        async def fake_connect(runtime, *, is_reconnect=False):
            del is_reconnect
            if runtime.bot_token == TOKEN_B:
                secondary_phases.append(adapter.identity_phase)
            robot_id = (
                "bot_alpha" if runtime.bot_token == TOKEN_A else "bot_bravo"
            )
            adapter._claim_identity(runtime, robot_id)
            runtime._robot_id = robot_id
            await adapter._on_runtime_registered(runtime)
            runtime._connected = True
            return True

        monkeypatch.setattr(IdentityRuntime, "connect", fake_connect)

        assert await adapter.connect() is True
        assert secondary_phases == [PHASE_MIGRATED]

    @pytest.mark.asyncio
    async def test_first_multi_token_start_freezes_then_migrates(
        self, tmp_path, monkeypatch
    ):
        # A profile that has only ever run one identity.
        single = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        await single.connect()
        peer_uid = "6e91530e4b39442894ec4d887753c178"
        legacy_session_key = f"agent:main:octo:dm:{peer_uid}"

        multi = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        multi.set_session_store(
            _Store([_entry(legacy_session_key, peer_uid)])
        )
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        assert await multi.connect() is True

        assert multi.identity_phase == PHASE_MIGRATED
        assert (
            multi.routes.robot_id_for_session(legacy_session_key)
            == "bot_alpha"
        )
        assert not multi.identity_state.plan_path.exists()
        observed_session_keys: list[str] = []

        async def capture(event):
            observed_session_keys.append(multi.session_key_for_source(event.source))

        multi.runtimes[0].handle_message = capture
        await _dispatch_text(
            multi.runtimes[0],
            text="legacy primary",
            channel_id=peer_uid,
            from_uid=peer_uid,
        )
        assert observed_session_keys == [legacy_session_key]

    @pytest.mark.asyncio
    async def test_migrated_space_qualified_dm_survives_first_inbound(
        self, tmp_path, monkeypatch
    ):
        legacy_chat_id = "s5_u1"
        legacy_key = f"agent:main:octo:dm:{legacy_chat_id}"
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        adapter.set_session_store(
            _Store([_entry(legacy_key, legacy_chat_id)])
        )
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
        )
        assert await adapter.connect() is True
        assert adapter.routes.robot_id_for_session(legacy_key) == "bot_alpha"

        runtime, _metadata, error = adapter.resolve_outbound(
            legacy_chat_id,
            session_key=legacy_key,
        )
        assert runtime is None
        assert "route is not established" in error

        primary = adapter.runtimes[0]
        primary.handle_message = AsyncMock()
        primary._send_typing_safe = AsyncMock()
        await _dispatch_text(
            primary,
            text="hello again",
            channel_id=legacy_chat_id,
            from_uid="s5_u1",
        )

        assert not adapter.routes.target_conflict("dm", legacy_chat_id)
        assert adapter.routes.robot_id_for_session(legacy_key) == "bot_alpha"
        lookup = adapter.routes.lookup_chat_id(legacy_chat_id)
        assert lookup is not None
        assert lookup.wire_chat_id == "u1"
        primary.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_primary_registration_failure_keeps_the_plan_pending(
        self, tmp_path, monkeypatch
    ):
        single = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        await single.connect()

        multi = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        multi.set_session_store(
            _Store([_entry("agent:main:octo:dm:s1_peer", "s1_peer")])
        )
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
            failing=(TOKEN_A,),
        )
        assert await multi.connect() is True

        # Other identities serve new traffic while the frozen set waits.
        assert multi.identity_state.load_metadata().phase == PHASE_PENDING
        assert (
            multi.routes.robot_id_for_session("agent:main:octo:dm:s1_peer") is None
        )
        assert _bind(multi, "bot_bravo", "s2_peer", ChannelType.DM) is True
        assert _bind(multi, "bot_bravo", "s1_peer", ChannelType.DM) is False
        assert not multi.routes.target_conflict("dm", "s1_peer")

        result = await multi.send("s1_peer", "hi")
        assert result.success is False
        assert "route is not established" in result.error

    @pytest.mark.asyncio
    async def test_recovered_primary_migrates_only_the_frozen_set(
        self, tmp_path, monkeypatch
    ):
        single = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        await single.connect()

        pending = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        pending.set_session_store(
            _Store([_entry("agent:main:octo:dm:s1_peer", "s1_peer")])
        )
        _install_fake_connect(
            monkeypatch,
            {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"},
            failing=(TOKEN_A,),
        )
        await pending.connect()
        _bind(pending, "bot_bravo", "s2_peer", ChannelType.DM)
        await pending.disconnect()

        # Primary comes back. A session created meanwhile must not be absorbed.
        recovered = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        recovered.set_session_store(
            _Store(
                [
                    _entry("agent:main:octo:dm:s1_peer", "s1_peer"),
                    _entry("agent:main:octo:dm:s2_peer", "s2_peer"),
                ]
            )
        )
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await recovered.connect()

        assert recovered.identity_phase == PHASE_MIGRATED
        assert (
            recovered.routes.robot_id_for_session("agent:main:octo:dm:s1_peer")
            == "bot_alpha"
        )
        assert (
            recovered.routes.robot_id_for_session("agent:main:octo:dm:s2_peer")
            == "bot_bravo"
        )

    @pytest.mark.asyncio
    async def test_first_token_identity_mismatch_refuses_to_migrate(
        self, tmp_path, monkeypatch, caplog
    ):
        single = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        await single.connect()

        # Operator put a different bot's token first.
        multi = _adapter(tmp_path, f"{TOKEN_C};{TOKEN_A}")
        multi.set_session_store(
            _Store([_entry("agent:main:octo:dm:s1_peer", "s1_peer")])
        )
        _install_fake_connect(
            monkeypatch, {TOKEN_C: "bot_charlie", TOKEN_A: "bot_alpha"}
        )
        caplog.set_level(logging.ERROR, logger="hermes_octo_plugin.adapter")
        await multi.connect()

        assert multi.identity_state.load_metadata().phase == PHASE_PENDING
        assert (
            multi.routes.robot_id_for_session("agent:main:octo:dm:s1_peer") is None
        )
        assert "previous identity was bot_alpha" in caplog.text


    def test_migration_conflict_rolls_back_and_keeps_the_plan(
        self, tmp_path
    ):
        store = IdentityStateStore(base_dir=tmp_path)
        store.save_metadata(
            IdentityMetadata(
                phase=PHASE_SINGLE,
                legacy_robot_id="bot_alpha",
            )
        )
        startup = store.begin(
            token_count=2,
            enumerate_legacy_sessions=lambda: [
                ("legacy-session", "group", "shared-group")
            ],
        )
        store.routes.bind(
            robot_id="bot_bravo",
            session_key="new-session",
            chat_type="group",
            chat_id="shared-group",
            channel_type=int(ChannelType.Group),
        )
        assert store.routes.flush(force=True) is True

        with pytest.raises(IdentityStateError):
            store.apply_legacy_plan(
                robot_id="bot_alpha",
                plan=startup.plan,
            )

        assert store.load_metadata().phase == PHASE_PENDING
        assert store.plan_path.exists()
        assert (
            store.routes.robot_id_for_target("group", "shared-group")
            == "bot_bravo"
        )
        assert not store.routes.target_conflict("group", "shared-group")

    def test_migration_flush_failure_keeps_phase_pending(
        self, tmp_path, monkeypatch
    ):
        store = IdentityStateStore(base_dir=tmp_path)
        store.save_metadata(
            IdentityMetadata(
                phase=PHASE_SINGLE,
                legacy_robot_id="bot_alpha",
            )
        )
        startup = store.begin(
            token_count=2,
            enumerate_legacy_sessions=lambda: [
                ("legacy-session", "dm", "legacy-peer")
            ],
        )
        monkeypatch.setattr(
            store.routes, "flush", MagicMock(return_value=False)
        )

        with pytest.raises(OSError):
            store.apply_legacy_plan(
                robot_id="bot_alpha",
                plan=startup.plan,
            )

        assert store.load_metadata().phase == PHASE_PENDING
        assert store.plan_path.exists()
    @pytest.mark.asyncio
    async def test_migration_never_runs_twice(self, tmp_path, monkeypatch):
        single = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        await single.connect()

        first = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        first.set_session_store(
            _Store([_entry("agent:main:octo:dm:s1_peer", "s1_peer")])
        )
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await first.connect()
        await first.disconnect()

        # Reordered tokens, and a session that now belongs to the other bot.
        second = _adapter(tmp_path, f"{TOKEN_B};{TOKEN_A}")
        second.set_session_store(
            _Store([_entry("agent:main:octo:dm:s2_peer", "s2_peer")])
        )
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await second.connect()

        assert second.identity_phase == PHASE_MIGRATED
        assert (
            second.routes.robot_id_for_session("agent:main:octo:dm:s1_peer")
            == "bot_alpha"
        )
        assert (
            second.routes.robot_id_for_session("agent:main:octo:dm:s2_peer") is None
        )

    @pytest.mark.asyncio
    async def test_sessions_without_structured_origin_are_not_planned(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        adapter.set_session_store(
            _Store(
                [
                    _entry("agent:main:octo:dm:with_origin", "s1_peer"),
                    _entry("agent:main:octo:dm:no_origin", "", origin=False),
                ]
            )
        )
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()

        assert (
            adapter.routes.robot_id_for_session("agent:main:octo:dm:with_origin")
            == "bot_alpha"
        )
        assert (
            adapter.routes.robot_id_for_session("agent:main:octo:dm:no_origin")
            is None
        )

    @pytest.mark.asyncio
    async def test_token_reordering_does_not_change_existing_routes(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_bravo", "s11_peer", ChannelType.DM)
        await adapter.disconnect()

        reordered = _adapter(tmp_path, f"{TOKEN_B};{TOKEN_A}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await reordered.connect()

        runtime, _resolved, error = reordered.resolve_outbound("s11_peer")
        assert error == ""
        assert runtime.bot_token == TOKEN_B

    @pytest.mark.asyncio
    async def test_rotated_token_with_the_same_identity_keeps_every_route(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "s12_peer", ChannelType.DM)
        await adapter.disconnect()

        rotated_token = "secret-token-alpha-rotated"
        rotated = _adapter(tmp_path, f"{rotated_token};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {rotated_token: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await rotated.connect()

        runtime, _resolved, error = rotated.resolve_outbound("s12_peer")
        assert error == ""
        assert runtime.bot_token == rotated_token

    @pytest.mark.asyncio
    async def test_new_identity_for_a_token_does_not_inherit_old_routes(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_alpha", "s13_peer", ChannelType.DM)
        await adapter.disconnect()

        replaced = _adapter(tmp_path, f"{TOKEN_C};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_C: "bot_delta", TOKEN_B: "bot_bravo"}
        )
        await replaced.connect()

        runtime, _resolved, error = replaced.resolve_outbound("s13_peer")
        assert runtime is None
        assert "bot_alpha" in error

    def test_route_snapshot_writes_are_serialized(self, tmp_path, monkeypatch):
        routes = IdentityRouteRegistry(path=tmp_path / "routes.json")
        routes.bind(
            robot_id="bot_alpha",
            session_key="s-1",
            chat_type="dm",
            chat_id="c1",
            channel_type=int(ChannelType.DM),
        )
        original_write = identity.write_state_document
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        guard = threading.Lock()
        calls = 0
        active = 0
        max_active = 0

        def slow_write(path, body):
            nonlocal calls, active, max_active
            with guard:
                calls += 1
                call_number = calls
                active += 1
                max_active = max(max_active, active)
            try:
                if call_number == 1:
                    first_entered.set()
                    assert release_first.wait(2)
                else:
                    second_entered.set()
                original_write(path, body)
            finally:
                with guard:
                    active -= 1

        monkeypatch.setattr(identity, "write_state_document", slow_write)
        first = threading.Thread(target=routes.flush, kwargs={"force": True})
        second = threading.Thread(target=routes.flush, kwargs={"force": True})
        first.start()
        assert first_entered.wait(1)
        second.start()
        overlapped = second_entered.wait(0.1)
        release_first.set()
        first.join(2)
        second.join(2)

        assert not overlapped
        assert max_active == 1
        assert not first.is_alive()
        assert not second.is_alive()

    @pytest.mark.asyncio
    async def test_dropping_back_to_one_token_keeps_honouring_routes(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_bravo", "s14_peer", ChannelType.DM)
        await adapter.disconnect()

        shrunk = _adapter(tmp_path, TOKEN_A)
        _install_fake_connect(monkeypatch, {TOKEN_A: "bot_alpha"})
        await shrunk.connect()

        assert shrunk.sole_identity_mode is False
        runtime, _resolved, error = shrunk.resolve_outbound("s14_peer")
        assert runtime is None
        assert "bot_bravo" in error


# ── Durable state recovery ────────────────────────────────────────────────


class TestDurableStateRecovery:
    def test_corrupt_snapshot_restores_the_previous_generation(self, tmp_path):
        store = IdentityStateStore(base_dir=tmp_path)
        store.routes.bind(
            robot_id="bot_alpha",
            session_key="s-1",
            chat_type="dm",
            chat_id="c1",
            channel_type=int(ChannelType.DM),
        )
        store.routes.flush(force=True)
        store.routes.bind(
            robot_id="bot_bravo",
            session_key="s-2",
            chat_type="dm",
            chat_id="c2",
            channel_type=int(ChannelType.DM),
        )
        store.routes.flush(force=True)
        store.routes.path.write_text("{ not json", encoding="utf-8")

        reloaded = IdentityStateStore(base_dir=tmp_path)
        reloaded.routes.load()
        assert reloaded.routes.restored_from_backup is True
        assert reloaded.routes.robot_id_for_session("s-1") == "bot_alpha"

    def test_two_corrupt_generations_fail_closed(self, tmp_path):
        store = IdentityStateStore(base_dir=tmp_path)
        store.routes.bind(
            robot_id="bot_alpha",
            session_key="s-1",
            chat_type="dm",
            chat_id="c1",
            channel_type=int(ChannelType.DM),
        )
        store.routes.flush(force=True)
        store.routes.flush(force=True)
        store.routes.path.write_text("{ not json", encoding="utf-8")
        store.routes.backup_path.write_text("also not json", encoding="utf-8")

        reloaded = IdentityStateStore(base_dir=tmp_path)
        reloaded.routes.load()
        assert reloaded.routes.snapshot_corrupt is True
        assert reloaded.routes.robot_id_for_session("s-1") is None

    def test_missing_sentinel_with_surviving_route_state_is_ambiguous(self, tmp_path):
        store = IdentityStateStore(base_dir=tmp_path)
        store.save_metadata(
            identity.IdentityMetadata(phase=PHASE_MIGRATED, legacy_robot_id="bot_a")
        )
        store.routes.bind(
            robot_id="bot_a",
            session_key="s-1",
            chat_type="dm",
            chat_id="c1",
            channel_type=int(ChannelType.DM),
        )
        store.routes.flush(force=True)
        store.metadata_path.unlink()

        fresh = IdentityStateStore(base_dir=tmp_path)
        with pytest.raises(IdentityStateError):
            fresh.begin(token_count=2, enumerate_legacy_sessions=lambda: [])


    def test_failed_flush_after_backup_restore_keeps_the_good_backup(
        self, tmp_path, monkeypatch
    ):
        store = IdentityStateStore(base_dir=tmp_path)
        store.routes.bind(
            robot_id="bot_alpha",
            session_key="s-1",
            chat_type="dm",
            chat_id="c1",
            channel_type=int(ChannelType.DM),
        )
        assert store.routes.flush(force=True) is True
        store.routes.bind(
            robot_id="bot_alpha",
            session_key="s-2",
            chat_type="dm",
            chat_id="c2",
            channel_type=int(ChannelType.DM),
        )
        assert store.routes.flush(force=True) is True
        store.routes.path.write_text("{ corrupt", encoding="utf-8")

        restored = IdentityRouteRegistry(path=store.routes.path)
        restored.load()
        assert restored.restored_from_backup is True
        assert restored.robot_id_for_session("s-1") == "bot_alpha"
        restored.bind(
            robot_id="bot_alpha",
            session_key="s-3",
            chat_type="dm",
            chat_id="c3",
            channel_type=int(ChannelType.DM),
        )
        monkeypatch.setattr(
            identity,
            "write_state_document",
            MagicMock(side_effect=OSError("disk unavailable")),
        )
        assert restored.flush(force=True) is False

        recovered = IdentityRouteRegistry(path=store.routes.path)
        recovered.load()
        assert recovered.snapshot_corrupt is False
        assert recovered.robot_id_for_session("s-1") == "bot_alpha"
    def test_missing_sentinel_with_surviving_card_binding_is_ambiguous(
        self, tmp_path
    ):
        registry = CardSessionRegistry(
            persistence=CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        )
        registry.register(_card_session())

        fresh = IdentityStateStore(base_dir=tmp_path / "identity")
        with pytest.raises(IdentityStateError):
            fresh.begin(token_count=2, enumerate_legacy_sessions=lambda: [])

    def test_corrupt_sentinel_never_re_migrates(self, tmp_path):
        store = IdentityStateStore(base_dir=tmp_path)
        store.save_metadata(
            identity.IdentityMetadata(phase=PHASE_MIGRATED, legacy_robot_id="bot_a")
        )
        store.metadata_path.write_text("{ garbage", encoding="utf-8")

        fresh = IdentityStateStore(base_dir=tmp_path)
        with pytest.raises(IdentityStateError):
            fresh.begin(token_count=2, enumerate_legacy_sessions=lambda: [])

    @pytest.mark.parametrize("tokens", [TOKEN_A, f"{TOKEN_A};{TOKEN_B}"])
    @pytest.mark.asyncio
    async def test_unwritable_identity_state_stops_every_identity(
        self, tmp_path, monkeypatch, tokens
    ):
        adapter = _adapter(tmp_path, tokens)
        started: list[str] = []

        async def refuse(self, *, is_reconnect: bool = False):
            started.append(self.bot_token)
            return True

        monkeypatch.setattr(IdentityRuntime, "connect", refuse)
        monkeypatch.setattr(
            identity.IdentityStateStore,
            "begin",
            MagicMock(side_effect=OSError("read-only filesystem")),
        )

        assert await adapter.connect() is False
        assert started == []

    def test_persisted_state_never_contains_a_credential(self, tmp_path):
        store = IdentityStateStore(base_dir=tmp_path)
        startup = store.begin(token_count=1, enumerate_legacy_sessions=lambda: [])
        assert startup.phase == PHASE_SINGLE
        store.record_single_identity("bot_alpha")
        store.routes.bind(
            robot_id="bot_alpha",
            session_key="s-1",
            chat_type="dm",
            chat_id="c1",
            channel_type=int(ChannelType.DM),
        )
        store.routes.flush(force=True)

        digests = {
            hashlib.sha256(token.encode()).hexdigest()
            for token in (TOKEN_A, TOKEN_B, TOKEN_C)
        }
        for path in tmp_path.rglob("*"):
            if not path.is_file():
                continue
            blob = path.read_text(encoding="utf-8")
            for token in (TOKEN_A, TOKEN_B, TOKEN_C):
                assert token not in blob
            for digest in digests:
                assert digest not in blob


# ── Per-identity card bindings ────────────────────────────────────────────


def _card_session(message_id: str = "card-1") -> CardSession:
    return CardSession(
        message_id=message_id,
        binding_id="binding-1",
        session_key="agent:main:octo:dm:s1_peer",
        chat_id="s1_peer",
        channel_id="peer",
        channel_type=ChannelType.DM,
        requester_uid="u1",
        card={"type": "AdaptiveCard", "body": []},
        plain="a card",
        action_labels={"ok": "OK"},
        input_ids=("field",),
        action_channel_ids=("peer",),
    )

def test_robot_identity_change_resets_runtime_scoped_state(tmp_path):
    adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
    runtime = adapter.runtimes[0]
    runtime._registration = BotRegisterResp(
        robot_id="bot_alpha",
        im_token="im-alpha",
        ws_url="wss://octo.invalid/ws",
        api_url="https://octo.invalid",
        owner_uid="owner-alpha",
        owner_channel_id="owner-alpha",
    )
    runtime._robot_id = "bot_alpha"
    adapter._runtimes_by_robot_id["bot_alpha"] = runtime
    runtime._attach_card_binding_store()
    runtime._card_sessions.register(_card_session("old-card"))
    runtime._known_group_ids.add("old-group")
    runtime._uid_to_name["old-user"] = "Old User"
    runtime._group_md_cache["old-group"] = {"content": "old"}
    runtime._chat_kind["old-chat"] = ChannelType.Group
    runtime._space_dm_targets["old-dm"] = "old-wire"

    runtime._registration = BotRegisterResp(
        robot_id="bot_bravo",
        im_token="im-bravo",
        ws_url="wss://octo.invalid/ws",
        api_url="https://octo.invalid",
        owner_uid="owner-bravo",
        owner_channel_id="owner-bravo",
    )
    adapter._claim_identity(runtime, "bot_bravo")
    runtime._robot_id = "bot_bravo"
    runtime._attach_card_binding_store()

    assert runtime._known_group_ids == set()
    assert runtime._uid_to_name == {}
    assert runtime._group_md_cache == {}
    assert runtime._chat_kind == {}
    assert runtime._space_dm_targets == {}
    assert runtime._card_sessions.peek("old-card") is None
    assert len(
        CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path).load()
    ) == 1
    assert (
        CardBindingStore(robot_id="bot_bravo", base_dir=tmp_path).load()
        == []
    )
    poller = MagicMock()
    runtime._event_poller = poller
    runtime._event_task = MagicMock()

    runtime._registration = BotRegisterResp(
        robot_id="bot_charlie",
        im_token="im-charlie",
        ws_url="wss://octo.invalid/ws",
        api_url="https://octo.invalid",
        owner_uid="owner-charlie",
        owner_channel_id="owner-charlie",
    )
    adapter._claim_identity(runtime, "bot_charlie")

    poller.stop.assert_called_once_with()
    assert runtime._event_poller is None
    assert runtime._event_task is None


@pytest.mark.asyncio
async def test_robot_identity_change_cancels_previous_identity_tasks(tmp_path):
    adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
    runtime = adapter.runtimes[0]
    started = asyncio.Event()

    async def wait_for_cancellation():
        started.set()
        await asyncio.Event().wait()

    prefetch = asyncio.create_task(wait_for_cancellation())
    progress = asyncio.create_task(wait_for_cancellation())
    runtime._prefetch_task = prefetch
    runtime._progress_tasks = {progress}
    await started.wait()

    with patch(
        "hermes_octo_plugin.card_progress.cancel_adapter_progress"
    ) as cancel_progress:
        runtime._reset_identity_scoped_state()
        await asyncio.sleep(0)

    assert prefetch.cancelled()
    assert progress.cancelled()
    assert runtime._prefetch_task is None
    assert runtime._progress_tasks == set()
    cancel_progress.assert_called_once_with(runtime)



class TestCardActionIdentity:
    @pytest.mark.asyncio
    async def test_card_action_binds_and_scopes_the_receiving_identity(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin.card_events import CardAction

        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        adapter._message_handler = AsyncMock()
        runtime = adapter.runtimes[0]
        session = _card_session()
        runtime._register_card_session(session)
        observed: list[tuple[object, str | None, int | None]] = []

        async def capture(_event):
            _, metadata, _ = adapter.resolve_outbound(session.chat_id)
            observed.append(
                (
                    adapter.resolve_trusted_runtime(),
                    adapter.routes.robot_id_for_session(session.session_key),
                    (metadata or {}).get("channel_type"),
                )
            )

        runtime.handle_message = capture
        status = await runtime._handle_card_action_event(
            CardAction(
                event_id=1,
                message_id=session.message_id,
                channel_id=session.channel_id,
                channel_type=session.channel_type,
                action_id="ok",
                inputs={},
                operator_uid=session.requester_uid,
                data={"_octo_binding": session.binding_id},
            )
        )

        assert status == "completed"
        assert observed == [(runtime, "bot_alpha", int(ChannelType.DM))]
        assert identity.current_robot_id.get() == ""


class _ToggleCardPersistence:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.fail = False

    def load(self):
        return json.loads(json.dumps(self.records))

    def save(self, records):
        if self.fail:
            raise OSError("disk unavailable")
        self.records = json.loads(json.dumps(records))


class TestCardBindingPersistence:
    def test_bindings_round_trip_for_one_identity(self, tmp_path):
        store = CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        registry = CardSessionRegistry(persistence=store)
        registry.register(_card_session())

        restored = CardSessionRegistry(
            persistence=CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        )
        assert restored.restore() == 1
        assert restored.peek("card-1").binding_id == "binding-1"

    def test_claim_write_failure_rolls_back_to_pending(self):
        persistence = _ToggleCardPersistence()
        registry = CardSessionRegistry(persistence=persistence)
        registry.register(_card_session())
        persistence.fail = True

        with pytest.raises(OSError, match="disk unavailable"):
            registry.claim("card-1", 7)

        persistence.fail = False
        claim = registry.claim("card-1", 7)
        assert claim.status == "claimed"
        assert claim.attempts == 1

    def test_completed_claim_is_not_duplicate_until_completion_is_durable(self):
        persistence = _ToggleCardPersistence()
        registry = CardSessionRegistry(persistence=persistence)
        registry.register(_card_session())
        assert registry.claim("card-1", 7).status == "claimed"
        persistence.fail = True

        with pytest.raises(OSError, match="disk unavailable"):
            registry.complete("card-1", 7)
        with pytest.raises(OSError, match="disk unavailable"):
            registry.claim("card-1", 7)

        persistence.fail = False
        assert registry.claim("card-1", 7).status == "duplicate"
        assert persistence.records[0]["state"] == "completed"

    def test_concurrent_persists_cannot_regress_completed_state(self):
        class BlockingPersistence(_ToggleCardPersistence):
            def __init__(self) -> None:
                super().__init__()
                self.block_next = False
                self.entered = threading.Event()
                self.release = threading.Event()

            def save(self, records):
                snapshot = json.loads(json.dumps(records))
                if self.block_next:
                    self.block_next = False
                    self.entered.set()
                    assert self.release.wait(2)
                self.records = snapshot

        persistence = BlockingPersistence()
        registry = CardSessionRegistry(persistence=persistence)
        registry.register(_card_session("interactive"))
        registry.register(_card_session("unrelated"))
        assert registry.claim("interactive", 7).status == "claimed"
        persistence.block_next = True

        discard = threading.Thread(
            target=registry.discard,
            args=("unrelated",),
        )
        complete = threading.Thread(
            target=registry.complete,
            args=("interactive", 7),
        )
        discard.start()
        assert persistence.entered.wait(1)
        complete.start()
        persistence.release.set()
        discard.join(2)
        complete.join(2)

        assert not discard.is_alive()
        assert not complete.is_alive()
        record = next(
            item for item in persistence.records
            if item["message_id"] == "interactive"
        )
        assert record["state"] == "completed"

    def test_card_binding_store_propagates_write_failure(self, tmp_path, monkeypatch):
        store = CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        monkeypatch.setattr(
            identity,
            "write_state_document",
            MagicMock(side_effect=OSError("disk unavailable")),
        )

        with pytest.raises(OSError, match="disk unavailable"):
            store.save([])

    def test_a_second_identity_sees_no_foreign_binding(self, tmp_path):
        store = CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        registry = CardSessionRegistry(persistence=store)
        registry.register(_card_session())

        other = CardSessionRegistry(
            persistence=CardBindingStore(robot_id="bot_bravo", base_dir=tmp_path)
        )
        assert other.restore() == 0
        assert other.peek("card-1") is None

    def test_claim_state_is_durable_before_the_cursor_can_advance(self, tmp_path):
        store = CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        registry = CardSessionRegistry(persistence=store)
        registry.register(_card_session())
        claim = registry.claim("card-1", 42)
        assert claim.status == "claimed"

        # Disk already reflects the attempt, so a restart cannot replay it
        # without bound, nor silently skip it.
        records = store.load()
        assert len(records) == 1
        assert records[0]["attempt_event_id"] == 42
        assert records[0]["dispatch_attempts"] == 1

    def test_completed_binding_survives_a_restart_to_block_replay(self, tmp_path):
        store = CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        registry = CardSessionRegistry(persistence=store)
        registry.register(_card_session())
        registry.claim("card-1", 7)
        registry.complete("card-1", 7)

        restored = CardSessionRegistry(
            persistence=CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        )
        assert restored.restore() == 1
        assert restored.claim("card-1", 7).status == "duplicate"

    def test_expired_bindings_are_dropped_on_restore(self, tmp_path):
        store = CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        registry = CardSessionRegistry(persistence=store, ttl_seconds=1.0)
        registry.register(_card_session())
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        payload["body"]["records"][0]["expires_at"] = 1.0
        # Re-checksum so the document stays valid; only the TTL is stale.
        store.save(payload["body"]["records"])

        restored = CardSessionRegistry(
            persistence=CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        )
        assert restored.restore() == 0

    def test_corrupt_store_fails_closed_for_that_identity(self, tmp_path, caplog):
        store = CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        registry = CardSessionRegistry(persistence=store)
        registry.register(_card_session())
        store.path.write_text("{ not json", encoding="utf-8")

        caplog.set_level(logging.ERROR, logger="hermes_octo_plugin.identity")
        restored = CardSessionRegistry(
            persistence=CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        )
        assert restored.restore() == 0
        assert restored.peek("card-1") is None

    def test_persisted_bindings_carry_no_credential(self, tmp_path):
        store = CardBindingStore(robot_id="bot_alpha", base_dir=tmp_path)
        registry = CardSessionRegistry(persistence=store)
        registry.register(_card_session())
        blob = store.path.read_text(encoding="utf-8")
        for token in (TOKEN_A, TOKEN_B, TOKEN_C):
            assert token not in blob
        assert "Authorization" not in blob
        assert "im_token" not in blob


# ── Model-facing surface stays identity-free ───────────────────────────────


class TestModelSurface:
    def test_no_tool_schema_exposes_an_identity_field(self):
        from hermes_octo_plugin.agent_tools import TOOL_SCHEMA
        from hermes_octo_plugin.card_tools import (
            DISPLAY_CARD_TOOL_SCHEMA,
            INTERACTIVE_CARD_TOOL_SCHEMA,
        )
        from hermes_octo_plugin.message_tools import MESSAGE_TOOL_SCHEMAS

        forbidden = {"bot_token", "token", "robot_id", "space_id", "identity"}
        schemas = [
            TOOL_SCHEMA,
            DISPLAY_CARD_TOOL_SCHEMA,
            INTERACTIVE_CARD_TOOL_SCHEMA,
            *MESSAGE_TOOL_SCHEMAS,
        ]
        for schema in schemas:
            blob = json.dumps(schema)
            for name in forbidden:
                assert f'"{name}"' not in blob, (schema["name"], name)

    def test_session_keys_never_carry_an_identity(self, tmp_path):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        key = adapter.session_key_for_source(_source("s1_peer"))
        assert key == "agent:main:octo:dm:s1_peer"
        for value in (TOKEN_A, TOKEN_B, "bot_alpha", "robot_id"):
            assert value not in key

    @pytest.mark.asyncio
    async def test_tool_resolution_refuses_an_unrouted_conversation(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin import agent_tools

        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        monkeypatch.setattr(agent_tools, "_resolve_adapter", lambda: adapter)

        assert agent_tools._resolve_runtime() is None

        token = identity.current_robot_id.set("bot_bravo")
        try:
            assert agent_tools._resolve_runtime() is adapter.runtimes[1]
        finally:
            identity.current_robot_id.reset(token)

    @pytest.mark.asyncio
    async def test_tool_resolution_refuses_the_offline_owning_identity(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin import agent_tools

        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        adapter.runtimes[1]._connected = False
        monkeypatch.setattr(agent_tools, "_resolve_adapter", lambda: adapter)

        token = identity.current_robot_id.set("bot_bravo")
        try:
            assert agent_tools._resolve_runtime() is None
        finally:
            identity.current_robot_id.reset(token)


# ── Profile sharing ───────────────────────────────────────────────────────


class TestProfileSharing:
    @pytest.mark.asyncio
    async def test_identities_share_the_profile_and_isolate_credentials(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        first, second = adapter.runtimes

        # Shared: one Hermes platform, config, session store and message bridge.
        assert first.config is second.config is adapter.config
        assert first.adapter is second.adapter is adapter

        # Isolated: credential, identity, owner and every authenticated cache.
        assert first.bot_token != second.bot_token
        assert first.robot_id != second.robot_id
        assert first.owner_uid != second.owner_uid
        for field in (
            "_card_sessions",
            "_card_profile_cache",
            "_known_group_ids",
            "_group_md_cache",
            "_chat_kind",
            "_uid_to_name",
            "_command_menu_force_event",
            "_lifecycle_lock",
        ):
            assert getattr(first, field) is not getattr(second, field), field

    @pytest.mark.asyncio
    async def test_one_identity_disconnecting_leaves_the_other_serving(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()
        _bind(adapter, "bot_bravo", "s15_peer", ChannelType.DM)

        adapter.runtimes[0]._connected = False
        adapter._refresh_connection_state()

        assert adapter.is_connected is True
        runtime, _resolved, error = adapter.resolve_outbound("s15_peer")
        assert error == ""
        assert runtime is adapter.runtimes[1]


# ── Logging hygiene ───────────────────────────────────────────────────────


class TestLoggingHygiene:
    @pytest.mark.asyncio
    async def test_connect_logs_identities_without_credentials(
        self, tmp_path, monkeypatch, caplog
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        caplog.set_level(logging.DEBUG, logger="hermes_octo_plugin.adapter")

        await adapter.connect()

        assert "bot_alpha" in caplog.text
        for token in (TOKEN_A, TOKEN_B):
            assert token not in caplog.text

    @pytest.mark.asyncio
    async def test_user_visible_route_errors_hide_credentials(
        self, tmp_path, monkeypatch
    ):
        adapter = _adapter(tmp_path, f"{TOKEN_A};{TOKEN_B}")
        _install_fake_connect(
            monkeypatch, {TOKEN_A: "bot_alpha", TOKEN_B: "bot_bravo"}
        )
        await adapter.connect()

        result = await adapter.send("unrouted", "hello")
        assert result.success is False
        for token in (TOKEN_A, TOKEN_B):
            assert token not in result.error


# ── Standalone cron delivery ──────────────────────────────────────────────


class TestStandaloneDelivery:
    @pytest.mark.asyncio
    async def test_single_token_is_used_directly(self, tmp_path, monkeypatch):
        from hermes_octo_plugin import adapter as adapter_module

        monkeypatch.setattr(
            adapter_module,
            "IdentityStateStore",
            lambda: IdentityStateStore(base_dir=tmp_path / "identity"),
        )
        assert await adapter_module._standalone_route(
            "https://api.invalid",
            TOKEN_A,
            chat_id="any-group",
            home_chat_id="",
        ) == (TOKEN_A, None, "any-group", "")

    @pytest.mark.asyncio
    async def test_multi_token_resolves_the_legacy_primary_identity(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin import adapter as adapter_module

        store = IdentityStateStore(base_dir=tmp_path / "identity")
        store.save_metadata(
            identity.IdentityMetadata(
                phase=PHASE_MIGRATED, legacy_robot_id="bot_bravo"
            )
        )
        monkeypatch.setattr(
            adapter_module,
            "IdentityStateStore",
            lambda: IdentityStateStore(base_dir=tmp_path / "identity"),
        )

        registrations = {
            TOKEN_A: SimpleNamespace(robot_id="bot_alpha"),
            TOKEN_B: SimpleNamespace(robot_id="bot_bravo"),
        }

        async def fake_register(_session, _api_url, token, **_kwargs):
            return registrations[token]

        monkeypatch.setattr(
            adapter_module.api, "register_bot", AsyncMock(side_effect=fake_register)
        )
        monkeypatch.setattr(
            adapter_module,
            "_new_guarded_http_session",
            lambda *_args, **_kwargs: _NullSession(),
        )

        resolved = await adapter_module._standalone_route(
            "https://api.invalid",
            f"{TOKEN_A};{TOKEN_B}",
            chat_id="home-group",
            home_chat_id="home-group",
        )
        assert resolved == (TOKEN_B, None, "home-group", "")


    @pytest.mark.asyncio
    async def test_standalone_home_without_verified_primary_never_uses_first_token(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin import adapter as adapter_module

        state_dir = tmp_path / "identity"
        monkeypatch.setattr(
            adapter_module,
            "IdentityStateStore",
            lambda: IdentityStateStore(base_dir=state_dir),
        )
        register = AsyncMock()
        monkeypatch.setattr(adapter_module.api, "register_bot", register)

        token, _channel_type, _wire_chat_id, error = (
            await adapter_module._standalone_route(
                "https://api.invalid",
                f"{TOKEN_A};{TOKEN_B}",
                chat_id="home-group",
                home_chat_id="home-group",
            )
        )

        assert token == ""
        assert "route is not established" in error
        register.assert_not_awaited()
    @pytest.mark.asyncio
    async def test_standalone_conflicted_home_never_uses_primary(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin import adapter as adapter_module

        state_dir = tmp_path / "identity"
        store = IdentityStateStore(base_dir=state_dir)
        store.save_metadata(
            identity.IdentityMetadata(
                phase=PHASE_MIGRATED,
                legacy_robot_id="bot_alpha",
            )
        )
        store.routes.bind(
            robot_id="bot_alpha",
            session_key="home-alpha",
            chat_type="group",
            chat_id="home-group",
            channel_type=int(ChannelType.Group),
        )
        store.routes.bind(
            robot_id="bot_bravo",
            session_key="home-bravo",
            chat_type="group",
            chat_id="home-group",
            channel_type=int(ChannelType.Group),
        )
        assert store.routes.flush(force=True) is True
        monkeypatch.setattr(
            adapter_module,
            "IdentityStateStore",
            lambda: IdentityStateStore(base_dir=state_dir),
        )

        token, _channel_type, _wire_chat_id, error = (
            await adapter_module._standalone_route(
                "https://api.invalid",
                f"{TOKEN_A};{TOKEN_B}",
                chat_id="home-group",
                home_chat_id="home-group",
            )
        )
        assert token == ""
        assert "route is not established" in error

    @pytest.mark.asyncio
    async def test_standalone_send_uses_target_route_token_and_channel_type(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin import adapter as adapter_module

        state_dir = tmp_path / "identity"
        store = IdentityStateStore(base_dir=state_dir)
        store.save_metadata(
            identity.IdentityMetadata(
                phase=PHASE_MIGRATED, legacy_robot_id="bot_alpha"
            )
        )
        scoped_chat_id = identity.scoped_dm_chat_id("bot_bravo", "wire-peer")
        store.routes.bind(
            robot_id="bot_bravo",
            session_key=f"agent:main:octo:dm:{scoped_chat_id}",
            chat_type="dm",
            chat_id=scoped_chat_id,
            channel_type=int(ChannelType.DM),
            wire_chat_id="wire-peer",
        )
        store.routes.flush(force=True)
        monkeypatch.setattr(
            adapter_module,
            "IdentityStateStore",
            lambda: IdentityStateStore(base_dir=state_dir),
        )
        registrations = {
            TOKEN_A: SimpleNamespace(robot_id="bot_alpha"),
            TOKEN_B: SimpleNamespace(robot_id="bot_bravo"),
        }

        async def fake_register(_session, _api_url, token, **_kwargs):
            return registrations[token]

        send = AsyncMock(
            return_value=SendMessageResult(
                message_id="m1", message_seq=1, client_msg_no="c1"
            )
        )
        monkeypatch.setattr(
            adapter_module.api, "register_bot", AsyncMock(side_effect=fake_register)
        )
        monkeypatch.setattr(adapter_module.api, "send_message", send)
        monkeypatch.setattr(
            adapter_module,
            "_new_guarded_http_session",
            lambda *_args, **_kwargs: _NullSession(),
        )

        result = await adapter_module._standalone_send(
            _config(f"{TOKEN_A};{TOKEN_B}"),
            scoped_chat_id,
            "hello",
        )

        assert result["success"] is True
        assert send.await_args.args[2] == TOKEN_B
        assert send.await_args.kwargs["channel_type"] == ChannelType.DM
        assert send.await_args.kwargs["channel_id"] == "wire-peer"

    @pytest.mark.asyncio
    async def test_no_matching_identity_refuses_to_deliver(self, tmp_path, monkeypatch):
        from hermes_octo_plugin import adapter as adapter_module

        store = IdentityStateStore(base_dir=tmp_path / "identity")
        store.save_metadata(
            identity.IdentityMetadata(
                phase=PHASE_MIGRATED, legacy_robot_id="bot_missing"
            )
        )
        monkeypatch.setattr(
            adapter_module,
            "IdentityStateStore",
            lambda: IdentityStateStore(base_dir=tmp_path / "identity"),
        )
        monkeypatch.setattr(
            adapter_module.api,
            "register_bot",
            AsyncMock(return_value=SimpleNamespace(robot_id="bot_alpha")),
        )
        monkeypatch.setattr(
            adapter_module,
            "_new_guarded_http_session",
            lambda *_args, **_kwargs: _NullSession(),
        )

        token, _channel_type, _wire_chat_id, error = (
            await adapter_module._standalone_route(
                "https://api.invalid",
                f"{TOKEN_A};{TOKEN_B}",
                chat_id="home-group",
                home_chat_id="home-group",
            )
        )
        assert token == ""
        assert "bot_missing" in error

    @pytest.mark.asyncio
    async def test_corrupt_identity_metadata_never_selects_the_first_token(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin import adapter as adapter_module

        state_dir = tmp_path / "identity"
        store = IdentityStateStore(base_dir=state_dir)
        store.save_metadata(
            identity.IdentityMetadata(
                phase=PHASE_MIGRATED, legacy_robot_id="bot_alpha"
            )
        )
        store.metadata_path.write_text("{ corrupt", encoding="utf-8")
        monkeypatch.setattr(
            adapter_module,
            "IdentityStateStore",
            lambda: IdentityStateStore(base_dir=state_dir),
        )

        token, _channel_type, _wire_chat_id, error = (
            await adapter_module._standalone_route(
                "https://api.invalid",
                f"{TOKEN_A};{TOKEN_B}",
                chat_id="home-group",
                home_chat_id="home-group",
            )
        )
        assert token == ""
        assert "unusable" in error

    @pytest.mark.asyncio
    async def test_missing_metadata_with_routes_never_selects_the_first_token(
        self, tmp_path, monkeypatch
    ):
        from hermes_octo_plugin import adapter as adapter_module

        state_dir = tmp_path / "identity"
        store = IdentityStateStore(base_dir=state_dir)
        store.routes.bind(
            robot_id="bot_alpha",
            session_key="agent:main:octo:dm:s1_peer",
            chat_type="dm",
            chat_id="s1_peer",
            channel_type=int(ChannelType.DM),
        )
        store.routes.flush(force=True)
        monkeypatch.setattr(
            adapter_module,
            "IdentityStateStore",
            lambda: IdentityStateStore(base_dir=state_dir),
        )

        token, _channel_type, _wire_chat_id, error = (
            await adapter_module._standalone_route(
                "https://api.invalid",
                f"{TOKEN_A};{TOKEN_B}",
                chat_id="s1_peer",
                home_chat_id="",
            )
        )
        assert token == ""
        assert "ambiguous" in error

    @pytest.mark.asyncio
    async def test_send_refuses_when_no_identity_matches(self, tmp_path, monkeypatch):
        from hermes_octo_plugin import adapter as adapter_module

        monkeypatch.setattr(
            adapter_module,
            "_standalone_route",
            AsyncMock(return_value=("", None, None, "route unavailable")),
        )
        result = await adapter_module._standalone_send(
            SimpleNamespace(
                extra={
                    "api_url": "https://api.octo.invalid",
                    "bot_token": f"{TOKEN_A};{TOKEN_B}",
                }
            ),
            "g1",
            "hello",
        )
        assert result == {"error": "route unavailable"}


class _NullSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


# ── Event cursors stay per identity ───────────────────────────────────────


class TestPerIdentityCursors:
    @pytest.mark.asyncio
    async def test_each_identity_gets_its_own_cursor_file(self, tmp_path, monkeypatch):
        from hermes_octo_plugin.card_events import FileEventCursorStore

        first = FileEventCursorStore(owner_id="bot_alpha", base_dir=tmp_path)
        second = FileEventCursorStore(owner_id="bot_bravo", base_dir=tmp_path)
        assert first.path != second.path

        await first.save(11)
        await second.save(22)
        assert await first.load() == 11
        assert await second.load() == 22
