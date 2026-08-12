"""Build the complete slash-command menu published to Octo."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

COMMAND_NAME_LIMIT = 50
COMMAND_DESCRIPTION_LIMIT = 100
_MENU_SOURCES = ("core", "quick", "plugin", "bundle", "skill")
_VISIBLE_GATEWAY_COMMANDS = frozenset({"new", "stop", "commands"})


@dataclass(frozen=True)
class CommandMenuManifest:
    """One deterministic Octo command-menu publication snapshot."""

    commands: tuple[dict[str, str], ...]
    digest: str
    source_counts: dict[str, int]
    collected_source_counts: dict[str, int] = field(default_factory=dict)
    collected_count: int = 0
    omitted_count: int = 0
    payload_chars: int = 0
    max_chars: int = 0


@dataclass(frozen=True)
class _CommandMenuEntry:
    source: str
    command: str
    description: str
    usage_count: int = 0


def _valid_token(bare: str) -> bool:
    return bool(bare) and len(bare) <= COMMAND_NAME_LIMIT and not any(
        char in {"/", "@"}
        or char.isspace()
        or unicodedata.category(char) in {"Cc", "Cf"}
        for char in bare
    )


def _exact_dispatch_name(value: object) -> str | None:
    """Validate a core/quick key, whose dispatch lookup is exact lowercase."""
    if not isinstance(value, str) or value != value.strip() or value != value.lower():
        return None
    return value if _valid_token(value) else None


def _normalized_dispatch_name(value: object) -> str | None:
    """Normalize names for Gateway sources that accept underscore aliases."""
    if not isinstance(value, str):
        return None
    bare = value.strip()
    if bare.startswith("/"):
        bare = bare[1:]
    bare = bare.lower().replace("_", "-")
    return bare if _valid_token(bare) else None


def _normalize_description(value: object, fallback: str, args_hint: object = "") -> str:
    text = value if isinstance(value, str) and value.strip() else fallback
    hint = args_hint.strip() if isinstance(args_hint, str) else ""
    if hint:
        text = f"{text} {hint}"
    visible = "".join(
        " " if char.isspace() else char
        for char in text
        if unicodedata.category(char) not in {"Cc", "Cf"}
    )
    return " ".join(visible.split())[:COMMAND_DESCRIPTION_LIMIT]


def _normalize_string_set(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise RuntimeError("disabled skill configuration must be a string or list")
    return {str(value).strip() for value in values if str(value).strip()}


def _read_strict_config() -> Mapping[str, object]:
    """Read the active profile config without swallowing I/O or YAML failures."""
    from agent.skill_utils import yaml_load
    from hermes_constants import get_config_path

    config_path = get_config_path()
    try:
        raw_config = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        try:
            config_path.lstat()
        except FileNotFoundError:
            return {}
        raise
    parsed = yaml_load(raw_config)
    if parsed is None:
        return {}
    if not isinstance(parsed, Mapping):
        raise RuntimeError("Hermes config root must be a mapping")
    return parsed


def _strict_config_gates(
    core_commands: Iterable[Any],
    config: Mapping[str, object],
) -> set[str]:
    from utils import is_truthy_value

    enabled: set[str] = set()
    for command in core_commands:
        gate = getattr(command, "gateway_config_gate", None)
        if not isinstance(gate, str) or not gate:
            continue
        value: object = config
        for key in gate.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if is_truthy_value(value, default=False):
            name = getattr(command, "name", None)
            if isinstance(name, str):
                enabled.add(name)
    return enabled


def _strict_disabled_skill_names(config: Mapping[str, object]) -> set[str]:
    """Union global and Octo-specific disabled skills on every Hermes line."""
    skills = config.get("skills")
    if skills is None:
        return set()
    if not isinstance(skills, Mapping):
        raise RuntimeError("Hermes skills config must be a mapping")

    disabled = _normalize_string_set(skills.get("disabled"))
    platform_disabled = skills.get("platform_disabled")
    if platform_disabled is None:
        return disabled
    if not isinstance(platform_disabled, Mapping):
        raise RuntimeError("skills.platform_disabled must be a mapping")
    return disabled | _normalize_string_set(platform_disabled.get("octo"))


def _load_skill_bundles() -> Mapping[str, object]:
    """Return the optional Hermes 0.20+ bundle registry."""
    try:
        module = importlib.import_module("agent.skill_bundles")
    except ModuleNotFoundError as exc:
        if exc.name == "agent.skill_bundles":
            return {}
        raise
    bundles = module.get_skill_bundles()
    if not isinstance(bundles, Mapping):
        raise RuntimeError("Hermes skill bundle registry is not a mapping")
    return bundles


def _executable_skill_bundles(
    bundles: Mapping[str, object],
    disabled_skill_names: set[str],
) -> dict[str, object]:
    """Keep bundles with at least one member that Octo dispatch can load."""
    from agent.skill_commands import _load_skill_payload

    executable: dict[str, object] = {}
    for slash_name, metadata in bundles.items():
        if not isinstance(metadata, Mapping):
            continue
        members = metadata.get("skills")
        if not isinstance(members, list):
            continue
        for member in members:
            if not isinstance(member, str) or not member.strip():
                continue
            identifier = member.strip()
            loaded = _load_skill_payload(identifier)
            if loaded is None:
                continue
            canonical_name = loaded[2]
            if (
                identifier not in disabled_skill_names
                and canonical_name not in disabled_skill_names
            ):
                executable[str(slash_name)] = metadata
                break
    return executable


def _require_mapping(value: object, source: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Hermes {source} registry is not a mapping")
    return value

def _server_storage_json(value: object) -> str:
    """Conservatively match Go's compact, HTML-escaped JSON storage length."""
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )

def _load_skill_usage_counts() -> dict[str, int]:
    """Read optional Hermes usage telemetry without making it load-bearing."""
    try:
        usage_module = importlib.import_module("tools.skill_usage")
        load_usage = getattr(usage_module, "load_usage")
        raw_usage = load_usage()
    except Exception:
        return {}
    if not isinstance(raw_usage, Mapping):
        return {}

    counts: dict[str, int] = {}
    for skill_name, record in raw_usage.items():
        if not isinstance(skill_name, str) or not isinstance(record, Mapping):
            continue
        raw_count = record.get("use_count")
        if isinstance(raw_count, bool):
            continue
        if isinstance(raw_count, int):
            count = raw_count
        elif isinstance(raw_count, str) and re.fullmatch(r"-?[0-9]+", raw_count):
            count = int(raw_count)
        else:
            continue
        if count >= 0:
            counts[skill_name] = count
    return counts


def build_command_menu(
    *,
    core_commands: Iterable[Any],
    config_overrides: set[str],
    is_gateway_available: Callable[[Any, set[str]], bool],
    quick_commands: Mapping[str, object],
    plugin_commands: Mapping[str, object],
    bundle_commands: Mapping[str, object],
    skill_commands: Mapping[str, object],
    disabled_skill_names: set[str],
    skill_usage_counts: Mapping[str, int],
    max_chars: int,
) -> CommandMenuManifest:
    """Collect by dispatch precedence, then apply the configured projection."""
    if isinstance(max_chars, bool) or max_chars < 0 or max_chars == 1:
        raise ValueError("command menu max_chars must be 0 or an integer of at least 2")

    entries: list[_CommandMenuEntry] = []
    collected_source_counts: dict[str, int] = dict.fromkeys(_MENU_SOURCES, 0)
    occupied: set[str] = set()

    def add(
        source: str,
        raw_name: object,
        description: object,
        fallback: str,
        hint: object = "",
        *,
        exact: bool = False,
        usage_count: int = 0,
    ) -> None:
        normalize = _exact_dispatch_name if exact else _normalized_dispatch_name
        name = normalize(raw_name)
        if name is None or name in occupied:
            return
        entries.append(
            _CommandMenuEntry(
                source=source,
                command=f"/{name}",
                description=_normalize_description(description, fallback, hint),
                usage_count=usage_count,
            )
        )
        occupied.add(name)
        collected_source_counts[source] += 1

    reserved_core_names: set[str] = set()
    for command in core_commands:
        gateway_recognizes = (
            not bool(getattr(command, "cli_only", False))
            or bool(getattr(command, "gateway_config_gate", None))
        )
        if gateway_recognizes:
            for raw_name in (
                getattr(command, "name", None),
                *getattr(command, "aliases", ()),
            ):
                reserved_name = _exact_dispatch_name(raw_name)
                if reserved_name:
                    reserved_core_names.add(reserved_name)
        if getattr(command, "name", None) not in _VISIBLE_GATEWAY_COMMANDS:
            continue
        if not is_gateway_available(command, config_overrides):
            continue
        add(
            "core",
            getattr(command, "name", None),
            getattr(command, "description", ""),
            "Hermes command",
            getattr(command, "args_hint", ""),
            exact=True,
        )
        for alias in getattr(command, "aliases", ()):
            alias_name = _exact_dispatch_name(alias)
            if alias_name:
                occupied.add(alias_name)
    occupied.update(reserved_core_names)

    for name in sorted(key for key in quick_commands if isinstance(key, str)):
        metadata = quick_commands[name]
        if not isinstance(metadata, Mapping):
            continue
        command_type = metadata.get("type")
        if command_type not in {"exec", "alias"}:
            continue
        fallback = (
            "Configured quick command alias"
            if command_type == "alias"
            else "Configured quick command"
        )
        add("quick", name, metadata.get("description"), fallback, exact=True)

    for name in sorted(key for key in plugin_commands if isinstance(key, str)):
        if _normalized_dispatch_name(name) != name:
            continue
        metadata = plugin_commands[name]
        if not isinstance(metadata, Mapping):
            continue
        add(
            "plugin",
            name,
            metadata.get("description"),
            "Plugin command",
            metadata.get("args_hint"),
            exact=True,
        )

    for slash_name in sorted(
        key for key in bundle_commands if isinstance(key, str)
    ):
        metadata = bundle_commands[slash_name]
        if not isinstance(metadata, Mapping):
            continue
        add(
            "bundle",
            slash_name,
            metadata.get("description"),
            f"Load the {metadata.get('name') or slash_name.lstrip('/')} skill bundle",
        )

    for slash_name in sorted(
        key for key in skill_commands if isinstance(key, str)
    ):
        metadata = skill_commands[slash_name]
        if not isinstance(metadata, Mapping):
            continue
        skill_name = metadata.get("name")
        if isinstance(skill_name, str) and skill_name in disabled_skill_names:
            continue
        usage_name = (
            skill_name
            if isinstance(skill_name, str)
            else str(slash_name).lstrip("/")
        )
        usage_count = max(0, skill_usage_counts.get(usage_name, 0))
        add(
            "skill",
            slash_name,
            metadata.get("description"),
            f"Invoke the {skill_name or slash_name.lstrip('/')} skill",
            usage_count=max(0, usage_count),
        )

    if max_chars:
        priority = {
            "core": 1,
            "skill": 2,
            "quick": 3,
            "plugin": 4,
            "bundle": 5,
        }
        ordered_entries = sorted(
            entries,
            key=lambda entry: (
                (
                    0
                    if entry.source == "plugin"
                    and entry.command.startswith("/octo-")
                    else priority[entry.source]
                ),
                -entry.usage_count if entry.source == "skill" else 0,
                entry.command if entry.source == "skill" else "",
            ),
        )
    else:
        ordered_entries = entries

    published: list[dict[str, str]] = []
    source_counts: dict[str, int] = dict.fromkeys(_MENU_SOURCES, 0)
    payload_chars = 2
    for entry in ordered_entries:
        item = {
            "command": entry.command,
            "description": "" if max_chars else entry.description,
        }
        item_chars = len(_server_storage_json(item))
        separator_chars = 1 if published else 0
        if max_chars and payload_chars + separator_chars + item_chars > max_chars:
            break
        published.append(item)
        source_counts[entry.source] += 1
        payload_chars += separator_chars + item_chars

    payload = json.dumps(published, ensure_ascii=False, separators=(",", ":"))
    return CommandMenuManifest(
        commands=tuple(published),
        digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        source_counts=source_counts,
        collected_source_counts=collected_source_counts,
        collected_count=len(entries),
        omitted_count=len(entries) - len(published),
        payload_chars=payload_chars,
        max_chars=max_chars,
    )


def collect_runtime_command_menu(
    quick_commands: Mapping[str, object],
    *,
    max_chars: int,
) -> CommandMenuManifest:
    """Read all live Hermes registries and build one complete Octo snapshot."""
    from agent.skill_commands import get_skill_commands
    from hermes_cli.commands import COMMAND_REGISTRY, _is_gateway_available
    from hermes_cli.plugins import get_plugin_commands

    config = _read_strict_config()
    disabled_skill_names = _strict_disabled_skill_names(config)
    plugin_commands = _require_mapping(get_plugin_commands(), "plugin command")
    skill_commands = _require_mapping(get_skill_commands(), "skill command")
    bundles = _executable_skill_bundles(
        _load_skill_bundles(),
        disabled_skill_names,
    )

    manifest = build_command_menu(
        core_commands=COMMAND_REGISTRY,
        config_overrides=_strict_config_gates(COMMAND_REGISTRY, config),
        is_gateway_available=_is_gateway_available,
        quick_commands=_require_mapping(quick_commands, "quick command"),
        plugin_commands=plugin_commands,
        bundle_commands=bundles,
        skill_commands=skill_commands,
        skill_usage_counts=_load_skill_usage_counts() if max_chars else {},
        disabled_skill_names=disabled_skill_names,
        max_chars=max_chars,
    )
    if manifest.collected_source_counts["core"] == 0:
        raise RuntimeError("Hermes exposed no gateway command registry entries")
    return manifest
