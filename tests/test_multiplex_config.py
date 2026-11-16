"""Regression tests: ``OCTO_*`` configuration is profile-scoped on a multiplexed gateway.

Hermes with ``gateway.multiplex_profiles`` serves several profiles from one gateway process and
keeps each profile's ``.env`` in a context-local secret scope instead of ``os.environ``
(``agent.secret_scope``). A secondary profile's ``OCTO_*`` values are therefore invisible to
``os.getenv``, and a scoped miss must stay a miss rather than borrowing the default (launch)
profile's process env value.

These tests install real profile scopes through Hermes' own API -- ``load_env_file`` (the ``.env``
tokenizer ``build_profile_secret_scope`` layers into a scope), ``set_secret_scope`` /
``reset_secret_scope`` and ``set_multiplex_active`` -- and then drive the plugin's public hooks:

* plugin registration (the gate that decides whether the LLM-callable surface is registered),
* adapter enablement (``_env_enablement`` / ``_is_connected``),
* adapter construction (``OctoAdapter.__init__`` field resolution),
* safety settings (``OCTO_ALLOW_PRIVATE_HOSTS`` in both the adapter validator and the transport
  policy, and the legacy ``OCTO_CARD_MESSAGE_ENABLED`` fallback),
* the out-of-process cron sender (``_standalone_send``).

They fail while any of those reads still use ``os.getenv`` and decode a process env that belongs
to another profile. The scope is built from ``load_env_file`` instead of
``build_profile_secret_scope`` so the test does not inherit this machine's managed/external
secret layers; the result is the same mapping Hermes installs for the profile's own ``.env``.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# These regression tests exercise Hermes' multiplexing secret scope and the scope-aware reader the
# plugin delegates to. Older Hermes installations have neither; skip the module there instead of
# failing collection (the plugin itself falls back to the legacy process-env reads).
_secret_scope = pytest.importorskip(
    "agent.secret_scope",
    reason="Hermes gateway multiplexing (agent.secret_scope) is required",
)
pytest.importorskip(
    "gateway.platforms._shared",
    reason="Hermes' scope-aware secret reader (gateway.platforms._shared) is required",
)
if "profile_home" not in inspect.signature(_secret_scope.set_secret_scope).parameters:
    pytest.skip(
        "Hermes secret scope does not record profile provenance",
        allow_module_level=True,
    )

UnscopedSecretError = _secret_scope.UnscopedSecretError
get_secret = _secret_scope.get_secret
is_multiplex_active = _secret_scope.is_multiplex_active
load_env_file = _secret_scope.load_env_file
reset_secret_scope = _secret_scope.reset_secret_scope
set_multiplex_active = _secret_scope.set_multiplex_active
set_secret_scope = _secret_scope.set_secret_scope

from hermes_octo_plugin import adapter as adapter_module
from hermes_octo_plugin import api
from hermes_octo_plugin import register as register_plugin
from hermes_octo_plugin.adapter import (
    DEFAULT_COMMAND_MENU_MAX_CHARS,
    OctoAdapter,
    _env_enablement,
    _is_connected,
    _standalone_send,
    _validate_octo_url,
    _validate_octo_ws_url,
)
from hermes_octo_plugin.card_tools import _profile_enabled
from hermes_octo_plugin.message_tools import _edit_profile_enabled
from hermes_octo_plugin.transport import private_hosts_enabled
from hermes_octo_plugin.types import (
    CARD_PROFILE_V1,
    CardProfileManifest,
    SendMessageResult,
)


# Every env name the plugin reads (plus the ones it only declares to core). Cleared before every
# test so a developer shell cannot make a scoped-miss assertion pass accidentally.
_OCTO_ENV_NAMES = (
    "OCTO_API_URL",
    "OCTO_BOT_TOKEN",
    "OCTO_WS_URL",
    "OCTO_CDN_URL",
    "OCTO_HOME_CHANNEL",
    "OCTO_ALLOW_PRIVATE_HOSTS",
    "OCTO_ON_BEHALF_OF",
    "OCTO_EVENT_POLL_INTERVAL_S",
    "OCTO_EVENT_POLL_WAIT_S",
    "OCTO_EVENT_POLL_LIMIT",
    "OCTO_PROGRESS_CARD_RENDERER",
    "OCTO_COMMAND_MENU_MAX_CHARS",
    "OCTO_ALLOWED_USERS",
    "OCTO_ALLOW_ALL_USERS",
    "OCTO_CARD_MESSAGE_ENABLED",
)

# The launch/default profile: its .env values are installed into os.environ by Hermes' single-profile
# startup path, which is exactly the value a secondary profile must never borrow.
_DEFAULT_PROFILE_ENV = {
    "OCTO_API_URL": "https://default.example.test",
    "OCTO_BOT_TOKEN": "default-token",
    "OCTO_WS_URL": "wss://default-ws.example.test/ws",
    "OCTO_CDN_URL": "https://default-cdn.example.test",
    "OCTO_HOME_CHANNEL": "default-home",
    "OCTO_ON_BEHALF_OF": "default-grantor",
    "OCTO_COMMAND_MENU_MAX_CHARS": "4096",
    "OCTO_EVENT_POLL_INTERVAL_S": "0.01",
    "OCTO_EVENT_POLL_WAIT_S": "1",
    "OCTO_EVENT_POLL_LIMIT": "5",
    "OCTO_PROGRESS_CARD_RENDERER": "registry",
    "OCTO_ALLOW_PRIVATE_HOSTS": "true",
    "OCTO_CARD_MESSAGE_ENABLED": "1",
}

# A secondary profile with its own credentials and tuning, but deliberately missing the optional
# WS URL, on-behalf grantor, private-host trust and card flag.
_SECONDARY_PROFILE_ENV = {
    "OCTO_API_URL": "https://secondary.example.test",
    "OCTO_BOT_TOKEN": "secondary-token-a;secondary-token-b",
    "OCTO_CDN_URL": "https://secondary-cdn.example.test",
    "OCTO_HOME_CHANNEL": "secondary-home",
    "OCTO_COMMAND_MENU_MAX_CHARS": "2048",
    "OCTO_EVENT_POLL_INTERVAL_S": "7.5",
    "OCTO_EVENT_POLL_WAIT_S": "30",
    "OCTO_EVENT_POLL_LIMIT": "75",
    "OCTO_PROGRESS_CARD_RENDERER": "local",
    "OCTO_CARD_MESSAGE_ENABLED": "0",
}

_PRIVATE_TRUSTED_ENV = {"OCTO_ALLOW_PRIVATE_HOSTS": "true"}
_CARDS_ENABLED_ENV = {"OCTO_CARD_MESSAGE_ENABLED": "1"}


def _write_profile_home(home: Path, values: dict[str, str]) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{name}={value}\n" for name, value in values.items())
    (home / ".env").write_text(body, encoding="utf-8")
    return home


@contextmanager
def _profile_config(home: Path):
    """Install *home*'s ``.env`` as the active profile scope, the way Hermes does per profile."""

    token = set_secret_scope(
        load_env_file(home / ".env"),
        profile_home=str(home),
    )
    try:
        yield
    finally:
        reset_secret_scope(token)


@pytest.fixture(autouse=True)
def _clear_octo_process_env(monkeypatch):
    for name in _OCTO_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_installed_profile_scope():
    """A leaked scope from another test must not decide what "unscoped" means here."""

    token = set_secret_scope(None)
    try:
        yield
    finally:
        reset_secret_scope(token)


@pytest.fixture
def multiplex():
    """Mark this process as a profile multiplexer, restoring the real mode afterwards."""

    was_active = is_multiplex_active()
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(was_active)


@pytest.fixture
def default_process_env(monkeypatch, _clear_octo_process_env):
    """The launch profile's values in ``os.environ`` (Hermes' single-profile startup bridge)."""

    for name, value in _DEFAULT_PROFILE_ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def profile_homes(tmp_path):
    return SimpleNamespace(
        default=_write_profile_home(tmp_path / "default", _DEFAULT_PROFILE_ENV),
        secondary=_write_profile_home(tmp_path / "secondary", _SECONDARY_PROFILE_ENV),
        bare=_write_profile_home(tmp_path / "bare", {}),
        private_trusted=_write_profile_home(tmp_path / "private-trusted", _PRIVATE_TRUSTED_ENV),
        cards_enabled=_write_profile_home(tmp_path / "cards-enabled", _CARDS_ENABLED_ENV),
    )


class _Session:
    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *_args):
        return None


class _RecordingCtx:
    """Minimal plugin-context stand-in recording every registration the plugin makes."""

    def __init__(self) -> None:
        self.platforms: list[dict] = []
        self.tools: list[dict] = []
        self.skills: list[dict] = []
        self.commands: list[dict] = []
        self.hooks: list[tuple[str, object]] = []

    def register_platform(self, **kwargs) -> None:
        self.platforms.append(kwargs)

    def register_tool(self, **kwargs) -> None:
        self.tools.append(kwargs)

    def register_skill(self, **kwargs) -> None:
        self.skills.append(kwargs)

    def register_command(self, name, **kwargs) -> None:
        self.commands.append({"name": name, **kwargs})

    def register_hook(self, name, callback) -> None:
        self.hooks.append((name, callback))


def test_host_scope_is_fail_closed_when_no_profile_scope_is_installed(
    profile_homes, default_process_env, multiplex
):
    """Prerequisite: the host reader the plugin must use fails closed for a scoped miss."""

    with _profile_config(profile_homes.bare):
        assert get_secret("OCTO_BOT_TOKEN") is None
        assert get_secret("OCTO_BOT_TOKEN", "missing") == "missing"

    # No scope at all under multiplexing is the host's "programming error" case; the plugin must
    # therefore use the shared helper that falls back to the launch profile's process env for it.
    with pytest.raises(UnscopedSecretError):
        get_secret("OCTO_BOT_TOKEN")


def test_plugin_registration_reads_the_secondary_profile_scope(
    profile_homes, default_process_env, multiplex
):
    ctx = _RecordingCtx()
    with _profile_config(profile_homes.secondary):
        register_plugin(ctx)

    assert [platform["name"] for platform in ctx.platforms] == ["octo"]
    assert "octo_management" in {tool["name"] for tool in ctx.tools}
    assert ctx.skills and ctx.commands and ctx.hooks


def test_plugin_registration_never_borrows_the_default_profile(
    profile_homes, default_process_env, multiplex
):
    ctx = _RecordingCtx()
    with _profile_config(profile_homes.bare):
        register_plugin(ctx)

    assert [platform["name"] for platform in ctx.platforms] == ["octo"]
    assert ctx.tools == []
    assert ctx.skills == []
    assert ctx.commands == []
    assert ctx.hooks == []

    # Unscoped is the launch/default profile: its process env is its own configuration.
    ctx = _RecordingCtx()
    register_plugin(ctx)
    assert "octo_management" in {tool["name"] for tool in ctx.tools}
    assert ctx.skills and ctx.commands and ctx.hooks


def test_enablement_hooks_read_the_secondary_profile_scope(
    profile_homes, default_process_env, multiplex
):
    with _profile_config(profile_homes.secondary):
        seed = _env_enablement()

        assert seed is not None
        assert seed["bot_token"] == "secondary-token-a;secondary-token-b"
        assert seed["api_url"] == "https://secondary.example.test"
        assert seed["cdn_url"] == "https://secondary-cdn.example.test"
        assert seed["home_channel"]["chat_id"] == "secondary-home"
        assert _is_connected(SimpleNamespace(extra={})) is True


def test_enablement_hooks_never_borrow_the_default_profile(
    profile_homes, default_process_env, multiplex
):
    with _profile_config(profile_homes.bare):
        seed = _env_enablement()
        assert seed is None or seed == {}
        assert _is_connected(SimpleNamespace(extra={})) is False

    # Unscoped is the launch/default profile: its process env is its own configuration.
    assert _env_enablement()["bot_token"] == "default-token"
    assert _is_connected(SimpleNamespace(extra={})) is True


def test_adapter_construction_reads_the_secondary_profile_scope(
    profile_homes, default_process_env, multiplex, tmp_path
):
    with _profile_config(profile_homes.secondary):
        adapter = OctoAdapter(
            SimpleNamespace(extra={}),
            state_base_dir=tmp_path / "secondary-state",
        )

    assert adapter._shared.api_url == "https://secondary.example.test"
    assert adapter._bot_tokens == ("secondary-token-a", "secondary-token-b")
    # These names exist only in the default profile's process env; the secondary must see defaults,
    # never the other profile's values.
    assert adapter._shared.ws_url == ""
    assert adapter._shared.cdn_url == "https://secondary-cdn.example.test"
    assert adapter.on_behalf_of is None
    assert adapter._shared.command_menu_max_chars == "2048"
    assert adapter._shared.event_poll_interval_s == 7.5
    assert adapter._shared.event_poll_wait_s == 30
    assert adapter._shared.event_poll_limit == 75
    assert adapter.progress_card_renderer == "local"


def test_default_profile_construction_still_reads_its_process_env(
    default_process_env, multiplex, tmp_path
):
    """The launch profile constructs unscoped under multiplexing; it must not raise."""

    adapter = OctoAdapter(
        SimpleNamespace(extra={}),
        state_base_dir=tmp_path / "default-state",
    )

    assert adapter._shared.api_url == "https://default.example.test"
    assert adapter._bot_tokens == ("default-token",)
    assert adapter._shared.ws_url == "wss://default-ws.example.test/ws"
    assert adapter._shared.cdn_url == "https://default-cdn.example.test"
    assert adapter.on_behalf_of == "default-grantor"
    assert adapter._shared.command_menu_max_chars == "4096"
    assert adapter._shared.event_poll_interval_s == 0.01
    assert adapter._shared.event_poll_wait_s == 1
    assert adapter._shared.event_poll_limit == 5
    assert adapter.progress_card_renderer == "registry"


def test_profile_without_octo_values_constructs_unconfigured(
    profile_homes, default_process_env, multiplex, tmp_path
):
    with _profile_config(profile_homes.bare):
        adapter = OctoAdapter(
            SimpleNamespace(extra={}),
            state_base_dir=tmp_path / "bare-state",
        )

    assert adapter._shared.api_url == ""
    assert adapter._bot_tokens == ()
    assert adapter._shared.ws_url == ""
    assert adapter._shared.cdn_url == ""
    assert adapter.on_behalf_of is None
    assert adapter._shared.command_menu_max_chars == str(DEFAULT_COMMAND_MENU_MAX_CHARS)
    assert adapter._shared.event_poll_interval_s == 2.0
    assert adapter._shared.event_poll_wait_s == 25
    assert adapter._shared.event_poll_limit == 50
    assert adapter.progress_card_renderer == "local"


def test_explicit_platform_extra_still_wins_over_the_profile_scope(
    profile_homes, default_process_env, multiplex, tmp_path
):
    with _profile_config(profile_homes.secondary):
        adapter = OctoAdapter(
            SimpleNamespace(
                extra={
                    "api_url": "https://explicit.example.test",
                    "bot_token": "explicit-token",
                }
            ),
            state_base_dir=tmp_path / "explicit-state",
        )

    assert adapter._shared.api_url == "https://explicit.example.test"
    assert adapter._bot_tokens == ("explicit-token",)


def test_private_host_trust_is_profile_local(
    profile_homes, default_process_env, multiplex, tmp_path
):
    private_url = "http://10.24.0.7:8080"

    # The default profile trusts private origins (process env says true); the secondary does not
    # opt in, so it must neither trust nor validate a private origin.
    with _profile_config(profile_homes.secondary):
        assert private_hosts_enabled() is False

        with pytest.raises(ValueError, match="OCTO_ALLOW_PRIVATE_HOSTS"):
            _validate_octo_url(private_url, "OCTO_API_URL")
        with pytest.raises(ValueError, match="OCTO_ALLOW_PRIVATE_HOSTS"):
            _validate_octo_ws_url("ws://10.24.0.7:8080/ws")
        with pytest.raises(ValueError, match="OCTO_ALLOW_PRIVATE_HOSTS"):
            OctoAdapter(
                SimpleNamespace(
                    extra={"api_url": private_url, "bot_token": "secondary-token-a"}
                ),
                state_base_dir=tmp_path / "private-secondary",
            )

    # A profile that does opt in gets its own trust policy in both readers.
    with _profile_config(profile_homes.private_trusted):
        assert private_hosts_enabled() is True
        assert _validate_octo_url(private_url, "OCTO_API_URL") == private_url

        adapter = OctoAdapter(
            SimpleNamespace(
                extra={"api_url": private_url, "bot_token": "trusted-token"}
            ),
            state_base_dir=tmp_path / "private-trusted",
        )
        assert adapter._shared.api_url == private_url

    # Unscoped launch/default profile keeps the legacy process-env behaviour.
    assert private_hosts_enabled() is True


def test_legacy_card_message_flag_is_profile_local(
    profile_homes, default_process_env, multiplex
):
    # available=False makes the local configured flag the only gate (cards.card_delivery_enabled).
    manifest = CardProfileManifest(available=False, enabled=False)

    with _profile_config(profile_homes.secondary):
        assert _profile_enabled(manifest, CARD_PROFILE_V1) is False
        assert _edit_profile_enabled(manifest) is False

    with _profile_config(profile_homes.bare):
        assert _profile_enabled(manifest, CARD_PROFILE_V1) is False
        assert _edit_profile_enabled(manifest) is False

    with _profile_config(profile_homes.cards_enabled):
        assert _profile_enabled(manifest, CARD_PROFILE_V1) is True
        assert _edit_profile_enabled(manifest) is True

    # Unscoped default profile still reads its own process env flag.
    assert _profile_enabled(manifest, CARD_PROFILE_V1) is True
    assert _edit_profile_enabled(manifest) is True


@pytest.mark.asyncio
async def test_standalone_sender_reads_the_profile_scope(
    profile_homes, default_process_env, multiplex
):
    with _profile_config(profile_homes.secondary):
        config = SimpleNamespace(extra={}, token="")
        with (
            patch.object(
                adapter_module,
                "_standalone_route",
                AsyncMock(return_value=("secondary-token-a", None, "group-secondary", "")),
            ),
            patch.object(
                adapter_module,
                "_new_guarded_http_session",
                return_value=_Session(),
            ) as guarded_session,
            patch.object(
                api,
                "send_message",
                AsyncMock(return_value=SendMessageResult(message_id="message-1")),
            ) as send_message,
        ):
            result = await _standalone_send(config, "group-secondary", "hello")

    assert result == {"success": True, "message_id": "message-1"}
    guarded_session.assert_called_once_with("https://secondary.example.test")
    assert send_message.await_args.args[1] == "https://secondary.example.test"
    assert send_message.await_args.args[2] == "secondary-token-a"
    assert send_message.await_args.kwargs["channel_id"] == "group-secondary"
    # The default profile's grantor must not leak into the secondary's outbound message.
    assert send_message.await_args.kwargs["on_behalf_of"] is None


@pytest.mark.asyncio
async def test_standalone_sender_fails_closed_without_the_profiles_credentials(
    profile_homes, default_process_env, multiplex
):
    with _profile_config(profile_homes.bare):
        config = SimpleNamespace(extra={}, token="")
        with (
            patch.object(
                adapter_module,
                "_standalone_route",
                AsyncMock(return_value=("borrowed-token", None, "group-borrowed", "")),
            ),
            patch.object(
                adapter_module,
                "_new_guarded_http_session",
                return_value=_Session(),
            ) as guarded_session,
            patch.object(api, "send_message", AsyncMock()) as send_message,
        ):
            result = await _standalone_send(config, "group-default", "hello")

    assert result == {
        "error": "OCTO_API_URL and OCTO_BOT_TOKEN must be configured"
    }
    guarded_session.assert_not_called()
    send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_profile_tasks_keep_their_own_octo_config(
    profile_homes, default_process_env, multiplex, tmp_path
):
    """Adapters for two profiles are built concurrently; scopes are context-local, not global."""

    ready = (asyncio.Event(), asyncio.Event())
    release = asyncio.Event()
    results: dict[str, dict] = {}

    async def probe(index: int, home: Path, label: str) -> None:
        token = set_secret_scope(
            load_env_file(home / ".env"), profile_home=str(home)
        )
        try:
            results[label] = {
                "seed": _env_enablement(),
                "connected": _is_connected(SimpleNamespace(extra={})),
            }
            ready[index].set()
            await release.wait()
            adapter = OctoAdapter(
                SimpleNamespace(extra={}),
                state_base_dir=tmp_path / f"state-{label}",
            )
            results[label]["api_url"] = adapter._shared.api_url
            results[label]["tokens"] = adapter._bot_tokens
            results[label]["ws_url"] = adapter._shared.ws_url
            results[label]["on_behalf_of"] = adapter.on_behalf_of
        finally:
            reset_secret_scope(token)

    default_task = asyncio.create_task(
        probe(0, profile_homes.default, "default")
    )
    secondary_task = asyncio.create_task(
        probe(1, profile_homes.secondary, "secondary")
    )
    await asyncio.gather(*(event.wait() for event in ready))
    release.set()
    await asyncio.gather(default_task, secondary_task)

    assert results["default"]["connected"] is True
    assert results["default"]["api_url"] == "https://default.example.test"
    assert results["default"]["tokens"] == ("default-token",)
    assert results["default"]["ws_url"] == "wss://default-ws.example.test/ws"
    assert results["default"]["on_behalf_of"] == "default-grantor"
    assert results["default"]["seed"]["bot_token"] == "default-token"

    assert results["secondary"]["connected"] is True
    assert results["secondary"]["api_url"] == "https://secondary.example.test"
    assert results["secondary"]["tokens"] == (
        "secondary-token-a",
        "secondary-token-b",
    )
    # Keys present only in the launch process env must stay absent in the secondary task.
    assert results["secondary"]["ws_url"] == ""
    assert results["secondary"]["on_behalf_of"] is None
    assert (
        results["secondary"]["seed"]["bot_token"]
        == "secondary-token-a;secondary-token-b"
    )
