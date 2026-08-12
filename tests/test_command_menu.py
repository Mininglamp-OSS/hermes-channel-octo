from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from hermes_octo_plugin import command_menu
from hermes_octo_plugin.command_menu import CommandMenuManifest, build_command_menu


@dataclass(frozen=True)
class _CoreCommand:
    name: str
    description: str
    aliases: tuple[str, ...] = ()
    args_hint: str = ""
    available: bool = True
    gateway_config_gate: str | None = None
    cli_only: bool = False


def _build(
    *,
    core: list[_CoreCommand] | None = None,
    quick: dict[str, object] | None = None,
    plugins: dict[str, object] | None = None,
    bundles: dict[str, object] | None = None,
    skills: dict[str, object] | None = None,
    disabled: set[str] | None = None,
    skill_usage: dict[str, int] | None = None,
    max_chars: int = 0,
) -> CommandMenuManifest:
    commands = core or []
    return build_command_menu(
        core_commands=commands,
        config_overrides=set(),
        is_gateway_available=lambda command, _overrides: command.available,
        quick_commands=quick or {},
        plugin_commands=plugins or {},
        bundle_commands=bundles or {},
        skill_commands=skills or {},
        disabled_skill_names=disabled or set(),
        skill_usage_counts=skill_usage or {},
        max_chars=max_chars,
    )


def test_core_menu_uses_gateway_canonical_names_and_reserves_aliases():
    manifest = _build(
        core=[
            _CoreCommand(
                "new",
                "Start a new session",
                aliases=("reset",),
                args_hint="[name]",
            ),
            _CoreCommand("cli-only", "Not on gateways", available=False),
        ],
        quick={"reset": {"type": "exec", "command": "secret shell body"}},
        plugins={"new": {"description": "Plugin collision"}},
    )

    assert manifest.commands == (
        {"command": "/new", "description": "Start a new session [name]"},
    )
    assert manifest.source_counts == {
        "core": 1,
        "quick": 0,
        "plugin": 0,
        "bundle": 0,
        "skill": 0,
    }

@pytest.mark.parametrize(
    ("max_chars", "description"),
    (
        (0, "Gateway command"),
        (1000, ""),
    ),
)
def test_gateway_menu_only_publishes_curated_commands(
    max_chars: int,
    description: str,
):
    manifest = _build(
        core=[
            _CoreCommand("new", "Gateway command"),
            _CoreCommand("retry", "Gateway command"),
            _CoreCommand("stop", "Gateway command"),
            _CoreCommand("commands", "Gateway command"),
            _CoreCommand("status", "Gateway command"),
        ],
        max_chars=max_chars,
    )

    assert manifest.commands == (
        {"command": "/new", "description": description},
        {"command": "/stop", "description": description},
        {"command": "/commands", "description": description},
    )


def test_hidden_gateway_commands_still_reserve_dispatch_tokens():
    manifest = _build(
        core=[
            _CoreCommand("new", "Start a new session"),
            _CoreCommand("retry", "Retry", aliases=("again",)),
        ],
        quick={
            "retry": {"type": "exec"},
            "again": {"type": "exec"},
        },
    )

    assert manifest.commands == (
        {"command": "/new", "description": "Start a new session"},
    )


def test_gate_off_core_names_still_reserve_gateway_dispatch_tokens():
    manifest = _build(
        core=[
            _CoreCommand(
                "verbose",
                "Show tool details",
                aliases=("details",),
                available=False,
                gateway_config_gate="display.verbose",
                cli_only=True,
            ),
            _CoreCommand(
                "browser",
                "CLI-only browser command",
                available=False,
                cli_only=True,
            ),
        ],
        quick={
            "verbose": {"type": "exec"},
            "details": {"type": "exec"},
            "browser": {"type": "exec"},
        },
    )

    assert manifest.commands == (
        {"command": "/browser", "description": "Configured quick command"},
    )


def test_menu_follows_gateway_precedence_and_filters_octo_disabled_skills():
    manifest = _build(
        quick={
            "deploy": {"type": "exec", "command": "printf super-secret"},
            "shortcut": {"type": "alias", "target": "/new --private"},
        },
        plugins={
            "deploy": {"description": "Plugin collision"},
            "octo-doctor": {"description": "Inspect Octo health"},
        },
        skills={
            "/octo-doctor": {"name": "octo-doctor", "description": "Skill collision"},
            "/review": {"name": "review", "description": "Review a change"},
            "/private-skill": {
                "name": "private-skill",
                "description": "Must not appear on Octo",
            },
        },
        disabled={"private-skill"},
        bundles={
            "/review": {
                "name": "review",
                "description": "Bundle wins over skill",
                "skills": ["review"],
            }
        },
    )

    assert manifest.commands == (
        {"command": "/deploy", "description": "Configured quick command"},
        {"command": "/shortcut", "description": "Configured quick command alias"},
        {"command": "/octo-doctor", "description": "Inspect Octo health"},
        {"command": "/review", "description": "Bundle wins over skill"},
    )
    assert "super-secret" not in repr(manifest.commands)
    assert "--private" not in repr(manifest.commands)


def test_menu_rejects_overlong_names_but_clamps_descriptions():
    manifest = _build(
        plugins={
            "valid-name": {
                "description": " first\nsecond\u202e hidden " + ("界" * 120),
                "args_hint": "<value>",
            },
            "long-name-" + ("x" * 60): {"description": "Not executable"},
        }
    )

    assert manifest.commands == (
        {
            "command": "/valid-name",
            "description": manifest.commands[0]["description"],
        },
    )
    assert "\n" not in manifest.commands[0]["description"]
    assert "\u202e" not in manifest.commands[0]["description"]
    assert len(manifest.commands[0]["description"]) == 100


def test_source_specific_names_match_gateway_dispatch_equivalence():
    manifest = _build(
        quick={
            "foo_bar": {"type": "exec"},
            "Foo": {"type": "exec"},
            "/prefixed": {"type": "exec"},
        },
        plugins={
            "foo_bar": {"description": "Unreachable plugin key"},
            "real-name": {"description": "Executable plugin command"},
        },
    )

    assert manifest.commands == (
        {"command": "/foo_bar", "description": "Configured quick command"},
        {"command": "/real-name", "description": "Executable plugin command"},
    )


def test_manifest_digest_is_stable_and_changes_with_payload():
    first = _build(skills={"/review": {"name": "review", "description": "Review"}})
    same = _build(skills={"/review": {"name": "review", "description": "Review"}})
    changed = _build(skills={"/review": {"name": "review", "description": "Review safely"}})

    assert first.digest == same.digest
    assert first.digest != changed.digest

def test_bounded_menu_is_name_only_and_uses_product_priority():
    expected = (
        {"command": "/octo-doctor", "description": ""},
        {"command": "/new", "description": ""},
        {"command": "/review", "description": ""},
    )
    budget = len(json.dumps(expected, ensure_ascii=True, separators=(",", ":")))

    manifest = _build(
        core=[_CoreCommand("new", "Start a new session")],
        quick={"ship": {"type": "exec", "description": "Ship"}},
        plugins={
            "other-plugin": {"description": "Other"},
            "octo-doctor": {"description": "Doctor"},
        },
        bundles={
            "/tools": {
                "name": "tools",
                "description": "Tools",
                "skills": ["review"],
            }
        },
        skills={"/review": {"name": "review", "description": "Review"}},
        max_chars=budget,
    )

    assert manifest.commands == expected
    assert manifest.source_counts == {
        "core": 1,
        "quick": 0,
        "plugin": 1,
        "bundle": 0,
        "skill": 1,
    }
    assert manifest.collected_count == 6
    assert manifest.omitted_count == 3
    assert manifest.payload_chars == budget
    assert manifest.max_chars == budget

def test_bounded_menu_orders_skills_by_use_count_then_name():
    expected = (
        {"command": "/frequent", "description": ""},
        {"command": "/also-frequent", "description": ""},
    )
    budget = len(json.dumps(expected, ensure_ascii=True, separators=(",", ":")))

    manifest = _build(
        skills={
            "/unused": {"name": "unused", "description": "Unused"},
            "/also-frequent": {
                "name": "also-frequent",
                "description": "Also frequent",
            },
            "/frequent": {"name": "frequent", "description": "Frequent"},
        },
        skill_usage={"frequent": 20, "also-frequent": 10, "unused": 0},
        max_chars=budget,
    )

    assert manifest.commands == expected


def test_skill_usage_counts_are_optional_and_sanitized(monkeypatch):
    class _UsageModule:
        @staticmethod
        def load_usage():
            return {
                "valid": {"use_count": 7},
                "string": {"use_count": "4"},
                "negative": {"use_count": -1},
                "boolean": {"use_count": True},
                "broken": object(),
            }

    monkeypatch.setattr(
        command_menu.importlib,
        "import_module",
        lambda name: _UsageModule if name == "tools.skill_usage" else None,
    )

    assert command_menu._load_skill_usage_counts() == {
        "valid": 7,
        "string": 4,
    }


def test_bounded_menu_stops_instead_of_skipping_to_shorter_lower_priority_items():
    manifest = _build(
        plugins={
            "octo-" + ("x" * 40): {"description": "First"},
            "octo-a": {"description": "Would fit but must not jump the queue"},
        },
        max_chars=2,
    )

    assert manifest.commands == ()
    assert manifest.collected_count == 2
    assert manifest.omitted_count == 2
    assert manifest.payload_chars == 2


def test_unlimited_menu_preserves_dispatch_order_and_descriptions():
    manifest = _build(
        core=[_CoreCommand("new", "Start")],
        quick={"ship": {"type": "exec"}},
        plugins={"octo-doctor": {"description": "Doctor"}},
        skills={"/review": {"name": "review", "description": "Review"}},
        max_chars=0,
    )

    assert manifest.commands == (
        {"command": "/new", "description": "Start"},
        {"command": "/ship", "description": "Configured quick command"},
        {"command": "/octo-doctor", "description": "Doctor"},
        {"command": "/review", "description": "Review"},
    )
    assert manifest.omitted_count == 0
    assert manifest.max_chars == 0


def test_runtime_collector_uses_strict_octo_config_and_bundle_filtering(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.commands.COMMAND_REGISTRY",
        [_CoreCommand("new", "Start a new session")],
    )
    monkeypatch.setattr(
        "hermes_cli.commands._is_gateway_available",
        lambda command, _overrides: command.available,
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_commands", lambda: {})
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {
            "/review": {"name": "review", "description": "Skill collision"},
            "/private": {"name": "private", "description": "Private"},
        },
    )
    monkeypatch.setattr(
        "agent.skill_commands._load_skill_payload",
        lambda name, task_id=None: (
            ({"name": name}, None, name) if name in {"review", "private"} else None
        ),
    )
    monkeypatch.setattr(
        command_menu,
        "_read_strict_config",
        lambda: {
            "skills": {
                "disabled": ["globally-disabled"],
                "platform_disabled": {"octo": ["private"]},
            }
        },
    )
    monkeypatch.setattr(
        command_menu,
        "_load_skill_bundles",
        lambda: {
            "/review": {
                "name": "review",
                "description": "Review bundle",
                "skills": ["missing", "review"],
            },
            "/private-bundle": {
                "name": "private-bundle",
                "description": "No enabled members",
                "skills": ["private"],
            },
        },
    )

    manifest = command_menu.collect_runtime_command_menu({}, max_chars=0)

    assert manifest.commands == (
        {"command": "/new", "description": "Start a new session"},
        {"command": "/review", "description": "Review bundle"},
    )
    assert manifest.source_counts["bundle"] == 1


def test_strict_config_read_distinguishes_missing_path_from_broken_symlink(
    monkeypatch,
    tmp_path,
):
    missing = tmp_path / "missing-config.yaml"
    monkeypatch.setattr("hermes_constants.get_config_path", lambda: missing)
    assert command_menu._read_strict_config() == {}

    broken = tmp_path / "broken-config.yaml"
    broken.symlink_to(missing)
    monkeypatch.setattr("hermes_constants.get_config_path", lambda: broken)
    with pytest.raises(FileNotFoundError):
        command_menu._read_strict_config()


def test_strict_config_gate_snapshot_matches_nested_truthy_values():
    commands = [
        _CoreCommand(
            "gated",
            "Gated command",
            gateway_config_gate="features.gated",
        )
    ]

    assert command_menu._strict_config_gates(
        commands,
        {"features": {"gated": "true"}},
    ) == {"gated"}
    assert command_menu._strict_config_gates(commands, {}) == set()


def test_strict_disabled_snapshot_unions_global_and_octo_lists():
    assert command_menu._strict_disabled_skill_names(
        {
            "skills": {
                "disabled": ["global"],
                "platform_disabled": {"octo": ["octo-only"]},
            }
        }
    ) == {"global", "octo-only"}


def test_runtime_collector_propagates_strict_config_failure(monkeypatch):
    monkeypatch.setattr(
        command_menu,
        "_read_strict_config",
        lambda: (_ for _ in ()).throw(RuntimeError("invalid config")),
    )

    with pytest.raises(RuntimeError, match="invalid config"):
        command_menu.collect_runtime_command_menu({}, max_chars=0)
