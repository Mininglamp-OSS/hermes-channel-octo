"""
L1 loading smoke: validate that the package wires itself in as a hermes-agent
entry-point plugin, and that ``register(ctx)`` calls the host's
register_platform / register_tool / register_skill / register_command hooks
correctly under both unconfigured and configured env.

These tests do not connect to a real Octo IM server. They exercise only the
plugin-discovery contract that hermes-agent relies on at startup.
"""

from __future__ import annotations

import importlib.metadata as md

import pytest

import hermes_octo_plugin

ENTRY_POINT_GROUP = "hermes_agent.plugins"
EXPECTED_NAME = "octo"
# Target is the MODULE — hermes_cli does ep.load() then getattr(module, "register").
# Pointing at "hermes_octo_plugin:register" would resolve to the function and
# discovery would skip the plugin with "no register() function".
EXPECTED_TARGET = "hermes_octo_plugin"


class FakeCtx:
    """Minimal plugin-context stand-in. Records every host-side registration
    call the plugin makes."""

    def __init__(self) -> None:
        self.platforms: list[dict] = []
        self.tools: list[dict] = []
        self.skills: list[dict] = []
        self.commands: list[dict] = []

    def register_platform(self, **kwargs) -> None:
        self.platforms.append(kwargs)

    def register_tool(self, **kwargs) -> None:
        self.tools.append(kwargs)

    def register_skill(self, **kwargs) -> None:
        self.skills.append(kwargs)

    def register_command(self, name, **kwargs) -> None:
        self.commands.append({"name": name, **kwargs})


# ---------- entry-point metadata ----------


def test_entry_point_declared():
    """`pip install`-time metadata exposes the octo plugin under the
    hermes-agent entry-point group."""
    eps = [e for e in md.entry_points(group=ENTRY_POINT_GROUP) if e.name == EXPECTED_NAME]
    assert len(eps) == 1, (
        f"expected exactly one '{EXPECTED_NAME}' entry-point in group "
        f"'{ENTRY_POINT_GROUP}', got {len(eps)}"
    )
    assert eps[0].value == EXPECTED_TARGET


def test_entry_point_loads_to_register_callable():
    """The advertised entry-point target resolves to the ``hermes_octo_plugin``
    module, which exposes ``register`` — matching hermes_cli's
    ``getattr(module, "register")`` lookup."""
    (ep,) = (e for e in md.entry_points(group=ENTRY_POINT_GROUP) if e.name == EXPECTED_NAME)
    loaded = ep.load()
    assert loaded is hermes_octo_plugin
    assert callable(getattr(loaded, "register", None))


# ---------- register() behaviour ----------


def test_register_without_env_only_registers_platform(monkeypatch):
    """When OCTO_API_URL / OCTO_BOT_TOKEN are not both set, the plugin must
    still register the platform (so it shows up in `hermes setup gateway`)
    but MUST NOT pollute the global tool / skill / command registries."""
    monkeypatch.delenv("OCTO_API_URL", raising=False)
    monkeypatch.delenv("OCTO_BOT_TOKEN", raising=False)

    ctx = FakeCtx()
    hermes_octo_plugin.register(ctx)

    assert len(ctx.platforms) == 1
    assert ctx.platforms[0]["name"] == "octo"
    assert ctx.tools == []
    assert ctx.skills == []
    assert ctx.commands == []


def test_register_with_env_wires_full_surface(monkeypatch):
    """With both env vars set, register() must wire the LLM-callable surface:
    octo_management tool, octo-bot-api skill, and the /octo-* slash commands,
    in addition to the platform entry."""
    monkeypatch.setenv("OCTO_API_URL", "http://localhost:1")
    monkeypatch.setenv("OCTO_BOT_TOKEN", "test-token")

    ctx = FakeCtx()
    hermes_octo_plugin.register(ctx)

    # Platform always
    assert [p["name"] for p in ctx.platforms] == ["octo"]

    # Exactly one tool, named octo_management, with an async handler
    assert len(ctx.tools) == 1
    tool = ctx.tools[0]
    assert tool["name"] == "octo_management"
    assert tool["toolset"] == "octo"
    assert tool["is_async"] is True
    assert callable(tool["handler"])

    # Bundled skill registered with a real path that exists on disk
    assert len(ctx.skills) == 1
    skill = ctx.skills[0]
    assert skill["name"] == "octo-bot-api"
    assert skill["path"].exists(), f"bundled SKILL.md missing at {skill['path']}"

    # Slash commands — at least the ops-baseline set
    cmd_names = {c["name"] for c in ctx.commands}
    expected = {
        "octo-doctor",
        "octo-info",
        "octo-groups",
        "octo-md",
        "octo-refresh",
        "octo-audit",
    }
    missing = expected - cmd_names
    assert not missing, f"missing slash command registrations: {missing}"


def test_register_platform_install_hint_points_to_github():
    """The install_hint must point at the standalone GitHub repo, not the
    not-yet-published PyPI package."""
    ctx = FakeCtx()
    hermes_octo_plugin.register(ctx)
    (entry,) = ctx.platforms
    assert "Mininglamp-OSS/hermes-channel-octo" in entry["install_hint"]
    assert "pip install hermes-channel-octo" not in entry["install_hint"]


# ---------- Platform enum resolution ----------


def test_platform_enum_resolves_octo():
    """After discover_plugins() runs (conftest.pytest_configure), the host's
    Platform enum must know about 'octo'. Skipped if hermes-agent is not
    installed in this test env (e.g. running just unit tests)."""
    try:
        from hermes_agent.types import Platform  # type: ignore
    except ImportError:
        pytest.skip("hermes-agent not installed in this test environment")

    p = Platform("octo")
    assert p.value == "octo"
