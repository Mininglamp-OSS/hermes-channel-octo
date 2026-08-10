"""Safe pure renderers for controlled Octo Type-17 cards."""

from __future__ import annotations

import json
import math
import re
import shlex
import threading
import time
from itertools import islice
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .types import (
    CARD_PROFILE_V1,
    CARD_PROFILE_V2,
    CARD_VERSION,
    CardProfileManifest,
    CardTemplatingCapability,
    MessageType,
)

ADAPTIVE_CARD_SCHEMA = "http://adaptivecards.io/schemas/adaptive-card.json"
DEFAULT_MAX_CARD_NODES = 200
DEFAULT_MAX_CARD_DEPTH = 16
DEFAULT_MAX_CARD_PAYLOAD_BYTES = 512 << 10
DEFAULT_MAX_VISIBLE_TEXT_BYTES = 64 << 10
DEFAULT_MAX_ACTION_DATA_BYTES = 16 << 10
DEFAULT_MAX_ACTION_DATA_VALUE_BYTES = 512
DEFAULT_MAX_DISPLAY_BLOCKS = 100


class CardLimitError(ValueError):
    """A rendered card exceeds a local or negotiated server limit."""


class CardProfileCache:
    """Short-lived, thread-safe manifest cache owned by one bot adapter."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = max(1.0, ttl_seconds)
        self._clock = clock
        self._manifest: CardProfileManifest | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> CardProfileManifest | None:
        with self._lock:
            if self._manifest is None or self._expires_at <= self._clock():
                self._manifest = None
                self._expires_at = 0.0
                return None
            return self._manifest

    def put(self, manifest: CardProfileManifest) -> None:
        with self._lock:
            self._manifest = manifest
            self._expires_at = self._clock() + self._ttl_seconds

    def clear(self) -> None:
        with self._lock:
            self._manifest = None
            self._expires_at = 0.0


@dataclass(frozen=True)
class CardRenderResult:
    """An Adaptive Card and its plain-text fallback."""

    card: dict[str, Any]
    plain: str


@dataclass(frozen=True)
class InteractiveCardRenderResult:
    """A controlled interactive card plus trusted session-binding metadata."""

    card: dict[str, Any]
    plain: str
    action_labels: dict[str, str]
    input_ids: tuple[str, ...]
    binding_id: str


@dataclass(frozen=True)
class CardCapabilities:
    """Normalized renderer capabilities advertised by the Octo server."""

    available: bool
    enabled: bool
    elements: frozenset[str] | None = None
    inputs: frozenset[str] | None = None
    actions: frozenset[str] | None = None
    profiles: frozenset[str] | None = None
    card_version: str | None = None
    authoritative: bool = False
    max_nodes: int | None = None
    max_depth: int | None = None
    max_payload_bytes: int | None = None
    max_input_text_bytes: int | None = None
    max_inputs_bytes: int | None = None


def card_delivery_enabled(
    manifest: CardProfileManifest,
    *,
    configured_enabled: bool,
) -> bool:
    """Use the explicit server gate when deployed, otherwise local config."""
    return manifest.enabled if manifest.available else configured_enabled


def _positive_limit(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return math.floor(value)


def derive_card_capabilities(manifest: CardProfileManifest) -> CardCapabilities:
    """Convert a manifest into authoritative renderer sets and safe limits."""
    authoritative = manifest.available
    profiles = frozenset(manifest.profiles or ()) if authoritative else None
    elements = (
        frozenset(manifest.elements or ())
        if authoritative
        else (frozenset(manifest.elements) if manifest.elements is not None else None)
    )
    inputs = (
        frozenset(manifest.inputs or ())
        if authoritative
        else (frozenset(manifest.inputs) if manifest.inputs is not None else None)
    )
    if authoritative or manifest.actions is not None or manifest.profiles is not None:
        action_values = set(manifest.actions or ())
        action_values.discard("Action.Submit")
        if CARD_PROFILE_V2 in (manifest.profiles or ()):
            action_values.add("Action.Submit")
        actions: frozenset[str] | None = frozenset(action_values)
    else:
        actions = None

    return CardCapabilities(
        available=manifest.available,
        enabled=manifest.enabled,
        elements=elements,
        inputs=inputs,
        actions=actions,
        profiles=profiles,
        card_version=manifest.card_version if authoritative else None,
        authoritative=authoritative,
        max_nodes=_positive_limit(manifest.limits.get("max_nodes")),
        max_depth=_positive_limit(manifest.limits.get("max_depth")),
        max_payload_bytes=_positive_limit(manifest.limits.get("max_payload_bytes")),
        max_input_text_bytes=_positive_limit(
            manifest.limits.get("max_input_text_bytes")
        ),
        max_inputs_bytes=_positive_limit(manifest.limits.get("max_inputs_bytes")),
    )


_SUMMARY_STRATEGY = {
    "apply_patch": "path",
    "bash": "shell",
    "browser_navigate": "url",
    "edit": "path",
    "exec": "shell",
    "exec_command": "shell",
    "fetch": "url",
    "find": "path",
    "glob": "query_scope",
    "grep": "query_scope",
    "ls": "path",
    "patch": "path",
    "process": "shell",
    "read": "path_range",
    "read_file": "path_range",
    "search": "query_scope",
    "search_files": "query_scope",
    "shell": "shell",
    "skill_view": "name",
    "terminal": "shell",
    "tool_call": "tool_call",
    "tool_describe": "name",
    "tool_search": "query",
    "web_extract": "url",
    "web_search": "query",
    "write": "path",
    "write_file": "path",
}
_TOOL_LABELS = {
    "__subagent_wait__": "等待子任务",
    "__thinking__": "分析问题",
    "apply_patch": "修改文件",
    "bash": "运行命令",
    "browser_back": "返回页面",
    "browser_click": "点击页面",
    "browser_console": "查看控制台",
    "browser_get_images": "查看图片",
    "browser_navigate": "打开网页",
    "browser_press": "发送按键",
    "browser_scroll": "滚动页面",
    "browser_snapshot": "读取网页",
    "browser_type": "填写表单",
    "browser_vision": "查看网页",
    "clarify": "确认需求",
    "delegate_task": "安排子任务",
    "edit": "修改文件",
    "exec": "运行命令",
    "exec_command": "运行命令",
    "fetch": "读取网页",
    "find": "搜索文件",
    "glob": "搜索文件",
    "grep": "搜索文件",
    "image_generate": "生成图片",
    "lcm_describe": "查看上下文",
    "lcm_expand": "展开上下文",
    "lcm_grep": "检索上下文",
    "lcm_inspect": "检查上下文",
    "ls": "列出文件",
    "memory": "更新记忆",
    "patch": "修改文件",
    "process": "运行命令",
    "read": "读取文件",
    "read_file": "读取文件",
    "search": "搜索文件",
    "search_files": "搜索文件",
    "session_search": "搜索会话",
    "shell": "运行命令",
    "skill_view": "读取技能",
    "skills_list": "列出技能",
    "terminal": "运行命令",
    "text_to_speech": "生成语音",
    "todo": "更新任务",
    "tool_call": "调用工具",
    "tool_describe": "读取工具说明",
    "tool_search": "查找工具",
    "vision_analyze": "查看图片",
    "web_extract": "读取网页",
    "web_search": "搜索网页",
    "write": "写入文件",
    "write_file": "写入文件",
}
_PROGRAM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./@:+-]+$")
_SAFE_TOOL_LABEL_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_SUMMARY_MAX_CHARS = 64
_ERROR_MAX_CHARS = 160
_CARD_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_MAX_INTERACTIVE_BUTTONS = 6
_MAX_INTERACTIVE_INPUTS = 5
_MAX_INTERACTIVE_TITLE_CHARS = 200
_MAX_INTERACTIVE_TEXT_CHARS = 2000
_MAX_INTERACTIVE_LABEL_CHARS = 64
_INPUT_TYPES = {
    "text": "Input.Text",
    "number": "Input.Number",
    "date": "Input.Date",
    "time": "Input.Time",
    "toggle": "Input.Toggle",
    "choice": "Input.ChoiceSet",
}


def _origin_domain(raw_url: str) -> str | None:
    try:
        parsed = urlsplit(raw_url)
        host = parsed.hostname
    except ValueError:
        return None
    if not parsed.scheme or not host:
        return None
    return f"{parsed.scheme.lower()}://{host}"


def sanitize_visible_text(text: str, *, generic: bool = True) -> str | None:
    """Normalize bounded visible card text without inspecting its meaning."""
    del generic
    if len(text.encode("utf-8")) > DEFAULT_MAX_VISIBLE_TEXT_BYTES:
        raise CardLimitError("card text bytes exceed local limit")
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized or None




def _first_string(params: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _shorten_path(path: str) -> str:
    segments = [segment for segment in re.split(r"[/\\]+", path) if segment]
    if len(segments) <= 2:
        return path
    return f"…/{segments[-2]}/{segments[-1]}"


def _summarize_shell(params: Mapping[str, object]) -> str:
    command = _first_string(params, ("command", "cmd")).strip()
    if not command:
        return ""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return ""
    index = 0
    while index < len(tokens) and re.match(
        r"^[A-Za-z_][A-Za-z0-9_]*=",
        tokens[index],
    ):
        index += 1
    program_token = tokens[index] if index < len(tokens) else ""
    if not _PROGRAM_TOKEN_RE.fullmatch(program_token):
        return ""
    program = program_token.rsplit("/", 1)[-1]
    remainder = tokens[index + 1 :]
    if program == "uv" and remainder:
        command_name = remainder[0]
        if command_name == "run":
            executable = next(
                (
                    token.rsplit("/", 1)[-1]
                    for token in remainder[1:]
                    if not token.startswith("-")
                    and _PROGRAM_TOKEN_RE.fullmatch(token)
                ),
                "",
            )
            return f"uv run {executable}" if executable else "uv run"
        if command_name in {"build", "lock", "sync"}:
            return f"uv {command_name}"
    if program in {"python", "python3"} and len(remainder) >= 2:
        if remainder[0] == "-m" and _PROGRAM_TOKEN_RE.fullmatch(remainder[1]):
            return f"{program} -m {remainder[1]}"
    allowed_subcommands = {
        "bun": {"run", "test"},
        "git": {
            "branch",
            "checkout",
            "diff",
            "fetch",
            "log",
            "merge",
            "pull",
            "push",
            "rebase",
            "restore",
            "rev-parse",
            "show",
            "status",
            "switch",
        },
        "hermes": {"gateway", "plugins"},
        "npm": {"run", "test"},
        "pnpm": {"run", "test"},
        "yarn": {"run", "test"},
    }
    first = remainder[0] if remainder else ""
    if first in allowed_subcommands.get(program, set()):
        return f"{program} {first}"
    return program


def _line_range_summary(params: Mapping[str, object]) -> str:
    offset = params.get("offset")
    limit = params.get("limit")
    if (
        isinstance(offset, int)
        and not isinstance(offset, bool)
        and offset > 0
    ):
        if (
            isinstance(limit, int)
            and not isinstance(limit, bool)
            and limit > 0
        ):
            return f"第 {offset}–{offset + limit - 1} 行"
        return f"从第 {offset} 行"
    start = params.get("start_line")
    end = params.get("end_line")
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and start > 0
    ):
        if (
            isinstance(end, int)
            and not isinstance(end, bool)
            and end >= start
        ):
            return f"第 {start}–{end} 行"
        return f"从第 {start} 行"
    return ""


def _tool_call_parts(
    params: Mapping[str, object],
) -> tuple[str, Mapping[str, object] | None]:
    name = _first_string(params, ("name", "tool_name", "tool")).strip()
    raw_arguments = params.get("arguments")
    if raw_arguments is None:
        raw_arguments = params.get("args")
    if raw_arguments is None:
        raw_arguments = params.get("params")
    arguments = raw_arguments if isinstance(raw_arguments, Mapping) else None
    return name, arguments


def summarize_tool_params(
    tool_name: str | None,
    params: object,
) -> str:
    """Return one allowlisted, bounded tool parameter summary."""
    if not tool_name or not isinstance(params, Mapping):
        return ""
    strategy = _SUMMARY_STRATEGY.get(tool_name)
    if strategy is None:
        return ""
    if strategy == "tool_call":
        inner_name, inner_args = _tool_call_parts(params)
        if (
            inner_name == tool_name
            or inner_name not in _SUMMARY_STRATEGY
            or inner_args is None
        ):
            return ""
        return summarize_tool_params(inner_name, inner_args)
    if strategy in {"path", "path_range"}:
        path = _shorten_path(_first_string(params, ("path", "file_path", "file")))
        parts = [path] if path else []
        if strategy == "path_range":
            line_range = _line_range_summary(params)
            if line_range:
                parts.append(line_range)
        summary = " · ".join(parts)
    elif strategy == "shell":
        summary = _summarize_shell(params)
    elif strategy == "url":
        summary = _first_string(params, ("url",))
    elif strategy == "name":
        summary = _first_string(
            params,
            ("name", "tool_name", "skill_name", "skill"),
        )
    elif strategy == "query_scope":
        raw_query = _first_string(params, ("query", "pattern"))
        safe_query = sanitize_visible_text(raw_query) if raw_query else None
        raw_path = _first_string(params, ("path", "root", "directory"))
        short_path = _shorten_path(raw_path) if raw_path else ""
        safe_path = (
            sanitize_visible_text(short_path, generic=False)
            if short_path
            else None
        )
        summary = " · ".join(
            part for part in (safe_query, safe_path) if part
        )
    else:
        summary = _first_string(params, ("query", "pattern"))
    summary = re.sub(r"\s+", " ", summary).strip()
    if not summary:
        return ""
    if len(summary) > _SUMMARY_MAX_CHARS:
        return f"{summary[:_SUMMARY_MAX_CHARS]}…"
    return summary


def safe_tool_label(tool_name: str | None) -> str:
    """Return a structurally bounded tool label."""
    if not tool_name:
        return "tool"
    if tool_name.startswith("mcp__"):
        return "MCP tool"
    if not _SAFE_TOOL_LABEL_RE.fullmatch(tool_name):
        return "tool"
    return tool_name


def localized_tool_label(
    tool_name: str | None,
    params: object = None,
) -> str:
    """Return a Chinese user-facing label for known Hermes tools."""
    if tool_name == "tool_call" and isinstance(params, Mapping):
        inner_name, inner_args = _tool_call_parts(params)
        if inner_name != tool_name and inner_name in _TOOL_LABELS:
            return localized_tool_label(inner_name, inner_args)
    if tool_name in _TOOL_LABELS:
        return _TOOL_LABELS[tool_name]
    safe = safe_tool_label(tool_name)
    if safe == "MCP tool":
        return "扩展工具"
    if safe == "tool":
        return "工具"
    return safe


def sanitize_error_text(error: object) -> str:
    """Return a short visible error summary without content inspection."""
    if not isinstance(error, str):
        return ""
    summary = re.sub(r"\s+", " ", error).strip()
    if len(summary) > _ERROR_MAX_CHARS:
        return f"{summary[:_ERROR_MAX_CHARS]}…"
    return summary


def sanitize_action_url(url: str) -> str:
    """Return a usable HTTP(S) action target or reject an invalid URL."""
    clean = url.strip()
    if not clean or any(character.isspace() for character in clean):
        raise ValueError("card action URL must be a safe http URL")
    try:
        parsed = urlsplit(clean)
        host = parsed.hostname
    except ValueError as exc:
        raise ValueError("card action URL must be a safe http URL") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("card action URL must be a safe http URL")
    return clean


def count_card_nodes(
    value: object,
    *,
    _root: bool = True,
    _seen: set[int] | None = None,
) -> int:
    """Count nested card objects while excluding arrays and the card root."""
    if not isinstance(value, (Mapping, list, tuple)):
        return 0
    seen = _seen if _seen is not None else set()
    marker = id(value)
    if marker in seen:
        return 0
    seen.add(marker)
    if isinstance(value, Mapping):
        return (0 if _root else 1) + sum(
            count_card_nodes(item, _root=False, _seen=seen) for item in value.values()
        )
    return sum(count_card_nodes(item, _root=False, _seen=seen) for item in value)


def card_max_depth(
    value: object,
    *,
    _depth: int = 0,
    _root: bool = True,
    _seen: set[int] | None = None,
) -> int:
    """Return object depth with the Adaptive Card root at depth zero."""
    if not isinstance(value, (Mapping, list, tuple)):
        return _depth
    seen = _seen if _seen is not None else set()
    marker = id(value)
    if marker in seen:
        return _depth
    seen.add(marker)
    if isinstance(value, Mapping):
        current_depth = _depth if _root else _depth + 1
        return max(
            (
                card_max_depth(
                    item,
                    _depth=current_depth,
                    _root=False,
                    _seen=seen,
                )
                for item in value.values()
            ),
            default=current_depth,
        )
    return max(
        (
            card_max_depth(
                item,
                _depth=_depth,
                _root=_root,
                _seen=seen,
            )
            for item in value
        ),
        default=_depth,
    )


def _go_json_bytes(value: object) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded = (
        encoded
        .replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )
    return encoded.encode("utf-8")


def card_payload_bytes(
    card: Mapping[str, object],
    plain: str | None,
    *,
    profile: str = CARD_PROFILE_V1,
    card_seq: int | None = None,
    transient: bool | None = None,
) -> int:
    """Measure the complete Type-17 payload with Go-compatible JSON escaping."""
    payload: dict[str, object] = {
        "type": MessageType.InteractiveCard,
        "profile": profile,
        "card_version": CARD_VERSION,
        "card": card,
    }
    if plain is not None:
        payload["plain"] = plain
    if card_seq is not None:
        payload["card_seq"] = card_seq
    if transient is not None:
        payload["transient"] = transient
    return len(_go_json_bytes(payload))


def validate_card_limits(
    card: Mapping[str, object],
    plain: str | None,
    capabilities: CardCapabilities | None,
    *,
    profile: str = CARD_PROFILE_V1,
    card_seq: int | None = None,
    transient: bool | None = None,
) -> None:
    """Reject output that exceeds server hard caps or tighter manifest caps."""
    max_nodes = (
        capabilities.max_nodes
        if capabilities is not None and capabilities.max_nodes is not None
        else DEFAULT_MAX_CARD_NODES
    )
    if count_card_nodes(card) > max_nodes:
        raise CardLimitError("card exceeds max_nodes")

    max_depth = (
        capabilities.max_depth
        if capabilities is not None and capabilities.max_depth is not None
        else DEFAULT_MAX_CARD_DEPTH
    )
    if card_max_depth(card) > max_depth:
        raise CardLimitError("card exceeds max_depth")

    max_payload_bytes = (
        capabilities.max_payload_bytes
        if capabilities is not None and capabilities.max_payload_bytes is not None
        else DEFAULT_MAX_CARD_PAYLOAD_BYTES
    )
    if (
        card_payload_bytes(
            card,
            plain,
            profile=profile,
            card_seq=card_seq,
            transient=transient,
        )
        > max_payload_bytes
    ):
        raise CardLimitError("card exceeds max_payload_bytes")


def _text_element(text: str, *, bold: bool = False) -> dict[str, Any]:
    element: dict[str, Any] = {
        "type": "TextBlock",
        "text": text,
        "wrap": True,
    }
    if bold:
        element["weight"] = "Bolder"
    return element


def _block_text(block: Mapping[str, object]) -> str | None:
    text = block.get("text")
    if not isinstance(text, str):
        raise ValueError("display card block text must be a string")
    return sanitize_visible_text(text)


def _profile_supported(
    capabilities: CardCapabilities | None,
    profile: str,
) -> bool:
    if capabilities is None or not capabilities.authoritative:
        return True
    return (
        capabilities.enabled
        and capabilities.card_version == CARD_VERSION
        and capabilities.profiles is not None
        and profile in capabilities.profiles
    )


def _require_element(
    capabilities: CardCapabilities | None,
    element: str,
) -> None:
    if capabilities is not None and not _supports(capabilities.elements, element):
        raise ValueError(f"{element} is not supported")


def _display_resource_url(value: object) -> str:
    if not isinstance(value, str) or any(character.isspace() for character in value):
        raise ValueError("display image URL must be a safe http URL")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError("display image URL must be a safe http URL") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("display image URL must be a safe http URL")
    return value


def build_display_card(
    *,
    blocks: Sequence[Mapping[str, object]],
    title: str | None = None,
    capabilities: CardCapabilities | None = None,
) -> CardRenderResult:
    """Render controlled display blocks with a same-source plain fallback."""
    if not _profile_supported(capabilities, CARD_PROFILE_V1):
        raise ValueError("octo/v1 is not supported")
    if isinstance(blocks, (str, bytes)) or len(blocks) > DEFAULT_MAX_DISPLAY_BLOCKS:
        raise CardLimitError("display card exceeds block limit")
    body: list[dict[str, Any]] = []
    plain_lines: list[str] = []

    def append_text(text: str, *, bold: bool = False) -> None:
        _require_element(capabilities, "TextBlock")
        body.append(_text_element(text, bold=bold))
        plain_lines.append(text)

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("display card title must be a string")
        clean_title = sanitize_visible_text(title)
        if clean_title is not None:
            append_text(clean_title, bold=True)

    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("display card blocks must be objects")
        block_type = block.get("type")
        if block_type in {"heading", "text"}:
            text = _block_text(block)
            if text is not None:
                append_text(text, bold=block_type == "heading")
            continue
        if block_type == "section":
            section_elements: list[dict[str, Any]] = []
            section_plain: list[str] = []
            for field, bold in (("title", True), ("text", False)):
                raw = block.get(field)
                if raw is None:
                    continue
                if not isinstance(raw, str):
                    raise ValueError(f"display section {field} must be a string")
                clean = sanitize_visible_text(raw)
                if clean is not None:
                    _require_element(capabilities, "TextBlock")
                    section_elements.append(_text_element(clean, bold=bold))
                    section_plain.append(clean)
            if not section_elements:
                continue
            if capabilities is None or _supports(capabilities.elements, "Container"):
                body.append({"type": "Container", "items": section_elements})
            else:
                body.extend(section_elements)
            plain_lines.extend(section_plain)
            continue
        if block_type == "facts":
            raw_items = block.get("items")
            if (
                not isinstance(raw_items, Sequence)
                or isinstance(raw_items, (str, bytes))
                or len(raw_items) > 50
            ):
                raise ValueError("display facts must contain at most 50 items")
            facts: list[dict[str, str]] = []
            lines: list[str] = []
            for raw_fact in raw_items:
                if not isinstance(raw_fact, Mapping):
                    raise ValueError("display facts must be objects")
                label = raw_fact.get("label")
                value = raw_fact.get("value")
                if not isinstance(label, str) or not isinstance(value, str):
                    raise ValueError("display fact fields must be strings")
                clean_label = sanitize_visible_text(label)
                clean_value = sanitize_visible_text(value)
                if clean_label is None or clean_value is None:
                    continue
                facts.append({"title": clean_label, "value": clean_value})
                lines.append(f"{clean_label}: {clean_value}")
            if not facts:
                continue
            if capabilities is None or _supports(capabilities.elements, "FactSet"):
                body.append({"type": "FactSet", "facts": facts})
            else:
                _require_element(capabilities, "TextBlock")
                body.extend(_text_element(line) for line in lines)
            plain_lines.extend(lines)
            continue
        if block_type == "image":
            resource_url = _display_resource_url(block.get("url"))
            alt = block.get("alt")
            if not isinstance(alt, str):
                raise ValueError("display image alt must be a string")
            clean_alt = sanitize_visible_text(alt)
            if clean_alt is None:
                continue
            origin = _origin_domain(resource_url)
            if origin is None:
                raise ValueError("display image URL must be a safe http URL")
            line = f"{clean_alt}: {origin}"
            if capabilities is None or _supports(capabilities.elements, "Image"):
                body.append({
                    "type": "Image",
                    "url": resource_url,
                    "altText": clean_alt,
                })
            else:
                _require_element(capabilities, "TextBlock")
                body.append(_text_element(line))
            plain_lines.append(line)
            continue
        if block_type == "actions":
            raw_items = block.get("items")
            if (
                not isinstance(raw_items, Sequence)
                or isinstance(raw_items, (str, bytes))
                or not raw_items
                or len(raw_items) > _MAX_INTERACTIVE_BUTTONS
            ):
                raise ValueError("display actions must contain 1-6 items")
            actions: list[dict[str, str]] = []
            lines: list[str] = []
            for raw_action in raw_items:
                if not isinstance(raw_action, Mapping):
                    raise ValueError("display actions must be objects")
                label = raw_action.get("label")
                url = raw_action.get("url")
                if not isinstance(label, str) or not isinstance(url, str):
                    raise ValueError("display action fields must be strings")
                clean_label = sanitize_visible_text(label)
                if clean_label is None:
                    continue
                safe_url = sanitize_action_url(url)
                actions.append({
                    "type": "Action.OpenUrl",
                    "title": clean_label,
                    "url": safe_url,
                })
                lines.append(f"{clean_label}: {safe_url}")
            if not actions:
                continue
            can_render_actions = (
                capabilities is None or _supports(capabilities.elements, "ActionSet")
            ) and (
                capabilities is None
                or _supports(capabilities.actions, "Action.OpenUrl")
            )
            if can_render_actions:
                body.append({"type": "ActionSet", "actions": actions})
            else:
                _require_element(capabilities, "TextBlock")
                body.extend(_text_element(line) for line in lines)
            plain_lines.extend(lines)
            continue
        raise ValueError("unsupported display card block type")

    plain = "\n".join(plain_lines).strip()
    if not body or not plain:
        raise ValueError("display card requires visible text")
    result = CardRenderResult(
        card={
            "$schema": ADAPTIVE_CARD_SCHEMA,
            "type": "AdaptiveCard",
            "version": CARD_VERSION,
            "body": body,
        },
        plain=plain,
    )
    validate_card_limits(result.card, result.plain, capabilities)
    return result


def _clean_interactive_text(
    value: object,
    *,
    field: str,
    max_chars: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    clean = sanitize_visible_text(value)
    if clean is None:
        raise ValueError(f"{field} must not be empty")
    if len(clean) > max_chars:
        raise ValueError(f"{field} exceeds length limit")
    return clean


def _clean_semantic_value(
    value: object,
    *,
    field: str,
    max_chars: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    candidate = value.strip()
    if not candidate or len(candidate) > max_chars:
        raise ValueError(f"{field} exceeds length limit")
    return candidate


def _sanitize_action_data(
    value: object,
    *,
    key: str = "",
    depth: int = 0,
) -> object:
    del key
    if isinstance(value, str):
        candidate = value.strip()
        if len(candidate.encode("utf-8")) > DEFAULT_MAX_ACTION_DATA_VALUE_BYTES:
            raise ValueError("action data value exceeds byte limit")
        return candidate
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("action data number must be finite")
        return value
    if depth >= 5:
        raise ValueError("action data exceeds depth limit")
    if isinstance(value, (list, tuple)):
        if len(value) > 50:
            raise ValueError("action data exceeds item limit")
        return [_sanitize_action_data(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for index, (child_key, child_value) in enumerate(islice(value.items(), 51)):
            if index == 50:
                raise ValueError("action data exceeds item limit")
            if not isinstance(child_key, str):
                raise ValueError("action data key must be a string")
            if child_key == "_octo_binding":
                continue
            if (
                child_key.startswith("_octo_")
                or not _CARD_ID_RE.fullmatch(child_key)
            ):
                raise ValueError("action data key must be a safe identifier")
            sanitized[child_key] = _sanitize_action_data(
                child_value,
                key=child_key,
                depth=depth + 1,
            )
        return sanitized
    raise ValueError("unsupported action data value")


def _require_card_id(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a safe identifier")
    candidate = value.strip()
    if not _CARD_ID_RE.fullmatch(candidate):
        raise ValueError(f"{field} must be a safe identifier")
    return candidate


def _require_trusted_binding_id(value: object) -> str:
    """Validate shape only; trusted bindings are intentionally high entropy."""
    if not isinstance(value, str):
        raise ValueError("binding_id must be a safe identifier")
    candidate = value.strip()
    if not _CARD_ID_RE.fullmatch(candidate):
        raise ValueError("binding_id must be a safe identifier")
    return candidate


def _supports(
    advertised: frozenset[str] | None,
    capability: str,
) -> bool:
    return advertised is None or capability in advertised


def build_interactive_card(
    *,
    title: str,
    buttons: Sequence[Mapping[str, object]],
    binding_id: str,
    text: str | None = None,
    inputs: Sequence[Mapping[str, object]] = (),
    capabilities: CardCapabilities | None = None,
) -> InteractiveCardRenderResult:
    """Build a bounded Type-17 form whose actions carry a trusted binding."""
    if not _profile_supported(capabilities, CARD_PROFILE_V2):
        raise ValueError("octo/v2 is not supported")
    trusted_binding = _require_trusted_binding_id(binding_id)
    clean_title = _clean_interactive_text(
        title,
        field="title",
        max_chars=_MAX_INTERACTIVE_TITLE_CHARS,
    )
    if capabilities is not None and not _supports(
        capabilities.elements,
        "TextBlock",
    ):
        raise ValueError("TextBlock is not supported")
    if capabilities is not None and not _supports(
        capabilities.actions,
        "Action.Submit",
    ):
        raise ValueError("Action.Submit is not supported")
    if not buttons or len(buttons) > _MAX_INTERACTIVE_BUTTONS:
        raise ValueError("interactive card requires 1-6 buttons")
    if len(inputs) > _MAX_INTERACTIVE_INPUTS:
        raise ValueError("interactive card supports at most 5 inputs")

    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": clean_title,
            "weight": "Bolder",
            "size": "Medium",
            "wrap": True,
        }
    ]
    plain_lines = [clean_title]
    if text is not None:
        clean_text = _clean_interactive_text(
            text,
            field="text",
            max_chars=_MAX_INTERACTIVE_TEXT_CHARS,
        )
        body.append({
            "type": "TextBlock",
            "text": clean_text,
            "wrap": True,
            "spacing": "Small",
        })
        plain_lines.append(clean_text)

    used_ids: set[str] = set()
    input_ids: list[str] = []
    text_input_count = sum(
        1
        for item in inputs
        if isinstance(item, Mapping) and item.get("kind", "text") == "text"
    )
    for raw_input in inputs:
        if not isinstance(raw_input, Mapping):
            raise ValueError("interactive card inputs must be objects")
        input_id = _require_card_id(raw_input.get("id"), "input id")
        if input_id in used_ids:
            raise ValueError(f"duplicate interactive card id: {input_id}")
        used_ids.add(input_id)
        kind = raw_input.get("kind", "text")
        if not isinstance(kind, str) or kind not in _INPUT_TYPES:
            raise ValueError("unsupported interactive card input kind")
        input_type = _INPUT_TYPES[kind]
        if capabilities is not None and not _supports(
            capabilities.inputs,
            input_type,
        ):
            raise ValueError(f"{input_type} is not supported")
        node: dict[str, Any] = {"type": input_type, "id": input_id}
        label = raw_input.get("label")
        if label is not None:
            node["label"] = _clean_interactive_text(
                label,
                field=f"input {input_id} label",
                max_chars=_MAX_INTERACTIVE_LABEL_CHARS,
            )
        choice_titles: list[str] = []
        if kind == "choice":
            raw_choices = raw_input.get("choices")
            if (
                not isinstance(raw_choices, Sequence)
                or isinstance(
                    raw_choices,
                    (str, bytes),
                )
                or not raw_choices
            ):
                raise ValueError(f"choice input {input_id} requires choices")
            choices: list[dict[str, str]] = []
            choice_values: set[str] = set()
            choice_titles = []
            for raw_choice in raw_choices[:128]:
                if not isinstance(raw_choice, Mapping):
                    raise ValueError("interactive card choices must be objects")
                choice_title = _clean_interactive_text(
                    raw_choice.get("title"),
                    field="choice title",
                    max_chars=_MAX_INTERACTIVE_LABEL_CHARS,
                )
                choice_value = _clean_semantic_value(
                    raw_choice.get("value"),
                    field="choice value",
                    max_chars=_MAX_INTERACTIVE_LABEL_CHARS,
                )
                if choice_value in choice_values:
                    raise ValueError("choice values must be unique")
                choice_values.add(choice_value)
                choice_titles.append(choice_title)
                choices.append({"title": choice_title, "value": choice_value})
            node["choices"] = choices
            multi_select = raw_input.get("multi_select", False)
            if not isinstance(multi_select, bool):
                raise ValueError("choice input multi_select must be a boolean")
            if multi_select:
                node["isMultiSelect"] = True
        else:
            placeholder = raw_input.get("placeholder")
            if placeholder is not None:
                node["placeholder"] = _clean_interactive_text(
                    placeholder,
                    field=f"input {input_id} placeholder",
                    max_chars=_MAX_INTERACTIVE_LABEL_CHARS,
                )
        if kind == "text" and capabilities is not None:
            byte_limits = [
                limit
                for limit in (
                    capabilities.max_input_text_bytes,
                    (
                        capabilities.max_inputs_bytes // max(1, text_input_count)
                        if capabilities.max_inputs_bytes is not None
                        else None
                    ),
                )
                if limit is not None
            ]
            if byte_limits:
                node["maxLength"] = max(1, min(byte_limits) // 4)
        body.append(node)
        input_ids.append(input_id)
        if kind == "choice":
            plain_lines.append(
                f"[{node.get('label', input_id)}: {' / '.join(choice_titles)}]"
            )
        else:
            plain_lines.append(f"[{node.get('label', input_id)}]")

    actions: list[dict[str, Any]] = []
    action_labels: dict[str, str] = {}
    for raw_button in buttons:
        if not isinstance(raw_button, Mapping):
            raise ValueError("interactive card buttons must be objects")
        action_id = _require_card_id(raw_button.get("id"), "button id")
        if action_id in used_ids:
            raise ValueError(f"duplicate interactive card id: {action_id}")
        used_ids.add(action_id)
        label = _clean_interactive_text(
            raw_button.get("label"),
            field=f"button {action_id} label",
            max_chars=_MAX_INTERACTIVE_LABEL_CHARS,
        )
        raw_data = raw_button.get("data")
        if raw_data is not None and not isinstance(raw_data, Mapping):
            raise ValueError(f"button {action_id} data must be an object")
        sanitized_data = _sanitize_action_data(raw_data or {})
        data = dict(sanitized_data) if isinstance(sanitized_data, Mapping) else {}
        if len(_go_json_bytes(data)) > DEFAULT_MAX_ACTION_DATA_BYTES:
            raise CardLimitError("action data exceeds byte limit")
        data["_octo_binding"] = trusted_binding
        action: dict[str, Any] = {
            "type": "Action.Submit",
            "id": action_id,
            "title": label,
            "data": data,
        }
        style = raw_button.get("style")
        if style in {"positive", "destructive"}:
            action["style"] = style
        elif style is not None:
            raise ValueError("unsupported interactive card button style")
        actions.append(action)
        action_labels[action_id] = label

    plain_lines.append(f"Actions: {' / '.join(action_labels.values())}")
    card = {
        "$schema": ADAPTIVE_CARD_SCHEMA,
        "type": "AdaptiveCard",
        "version": CARD_VERSION,
        "body": body,
        "actions": actions,
    }
    plain = "\n".join(plain_lines)
    validate_card_limits(
        card,
        plain,
        capabilities,
        profile=CARD_PROFILE_V2,
    )
    return InteractiveCardRenderResult(
        card=card,
        plain=plain,
        action_labels=action_labels,
        input_ids=tuple(input_ids),
        binding_id=trusted_binding,
    )


_REASONING_TEMPLATE_ID = "ai.reasoning-process"
_REASONING_TEMPLATE_WIRE = "template-ref/v1"
_REASONING_FALLBACK_THOUGHT = "正在分析…"
_REASONING_THOUGHT_MAX = 280
_REASONING_TOOL_NAME_MAX = 80
_REASONING_MAX_PHASES = 6
_REASONING_MAX_ACTIONS = 12
_REASONING_PHASE_NAMES = {
    "starting": "reasoning",
    "thinking": "reasoning",
    "running": "reasoning",
    "tool": "reasoning",
    "paused": "reasoning",
    "resuming": "reasoning",
    "answering": "answering",
    "completed": "completed",
    "done": "completed",
    "stopped": "stopped",
    "failed": "error",
    "error": "error",
    "expired": "error",
}
_REASONING_STATUS_MAP = {
    "running": "running",
    "complete": "done",
    "completed": "done",
    "done": "done",
    "ok": "done",
    "failed": "error",
    "error": "error",
    "cancelled": "error",
}
_REASONING_REQUIRED_VIEWS = {
    "active": (
        "octo/v2",
        frozenset({"reasoning", "answering"}),
        frozenset({"reasoning_stop"}),
    ),
    "error": (
        "octo/v2",
        frozenset({"error"}),
        frozenset({"reasoning_retry"}),
    ),
    "result": (
        "octo/v1",
        frozenset({"completed", "stopped"}),
        frozenset(),
    ),
}


def select_reasoning_process_template(
    templating: CardTemplatingCapability | None,
) -> dict[str, str] | None:
    """Select the sole Registry template compatible with the reasoning contract."""
    if (
        templating is None
        or not templating.supported
        or templating.wire != _REASONING_TEMPLATE_WIRE
    ):
        return None
    claimed = [
        template
        for template in templating.templates
        if template.id == _REASONING_TEMPLATE_ID
    ]
    compatible = []
    for template in claimed:
        if not template.version or template.version.strip() != template.version:
            continue
        valid = True
        for view_name, (
            wire_profile,
            required_states,
            _allowed_actions,
        ) in _REASONING_REQUIRED_VIEWS.items():
            views = [view for view in template.views if view.name == view_name]
            if len(views) != 1:
                valid = False
                break
            view = views[0]
            if (
                view.wire_profile != wire_profile
                or not required_states.issubset(view.states)
                or view.submit_actions
            ):
                valid = False
                break
        if valid:
            compatible.append(template)
    if len(compatible) != 1:
        return None
    selected = compatible[0]
    if sum(template.version == selected.version for template in claimed) != 1:
        return None
    return {"id": selected.id, "version": selected.version}


def format_progress_duration(duration_ms: object) -> str:
    """Format milliseconds exactly like OpenClaw's reasoning card."""
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, (int, float))
        or not math.isfinite(duration_ms)
        or duration_ms < 0
    ):
        return ""
    if duration_ms < 1_000:
        return f"{duration_ms:g}ms"
    total_seconds = round(duration_ms / 1_000)
    if total_seconds < 60:
        return f"{duration_ms / 1_000:.1f}s"
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{total_minutes}m {seconds}s"


def sanitize_reasoning_thought(text: object) -> str:
    """Return a bounded public reasoning summary, never raw protected content."""
    if not isinstance(text, str) or not text:
        return _REASONING_FALLBACK_THOUGHT
    normalized = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        " ",
        text,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if (
        not normalized
        or "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>" in normalized
        or "<<<END_OPENCLAW_INTERNAL_CONTEXT>>>" in normalized
    ):
        return _REASONING_FALLBACK_THOUGHT
    try:
        clean = sanitize_visible_text(normalized)
    except CardLimitError:
        return _REASONING_FALLBACK_THOUGHT
    if not clean:
        return _REASONING_FALLBACK_THOUGHT
    if len(clean) > _REASONING_THOUGHT_MAX:
        return f"{clean[:_REASONING_THOUGHT_MAX]}…"
    return clean


def _finite_count(value: object) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return math.floor(value)


def summarize_tool_result(tool_name: str | None, result: object) -> str:
    """Summarize only allowlisted structural result fields."""
    if result is None:
        return ""
    if isinstance(result, list):
        return f"{len(result)} 项结果"
    if not isinstance(result, Mapping):
        return "已完成"
    records: list[Mapping[str, object]] = [result]
    for key in ("details", "meta", "metadata", "summary"):
        value = result.get(key)
        if isinstance(value, Mapping):
            records.append(value)
    if tool_name in {
        "exec",
        "exec_command",
        "bash",
        "shell",
        "process",
        "terminal",
    }:
        for record in records:
            for key in ("exitCode", "exit_code", "code"):
                count = _finite_count(record.get(key))
                if count is not None:
                    return f"退出码 {count}"
    for record in records:
        for key in (
            "matchCount",
            "match_count",
            "resultCount",
            "result_count",
            "totalCount",
            "total_count",
        ):
            count = _finite_count(record.get(key))
            if count is not None:
                return f"{count} 项结果"
    for record in records:
        for key in ("fileCount", "file_count", "changedFiles"):
            count = _finite_count(record.get(key))
            if count is not None:
                return f"{count} 个文件"
        for key in ("bytes", "byteLength", "writtenBytes"):
            count = _finite_count(record.get(key))
            if count is not None:
                return f"{count} 字节"
    status_labels = {
        "accepted": "已接受",
        "queued": "排队中",
        "waiting": "等待中",
        "completed": "已完成",
        "complete": "已完成",
        "success": "已完成",
        "succeeded": "已完成",
        "ok": "已完成",
        "done": "已完成",
    }
    for record in records:
        status = record.get("status")
        if isinstance(status, str):
            label = status_labels.get(status.lower())
            if label is not None:
                return label
    return "已完成"


def localize_result_summary(summary: str) -> str | None:
    """Normalize common bounded result summaries into Chinese."""
    clean = sanitize_visible_text(summary)
    if not clean:
        return None
    lowered = clean.lower()
    if lowered in {"complete", "completed", "done", "success", "succeeded"}:
        return "已完成"
    match = re.fullmatch(r"(\d+)\s+(?:results?|items?)", lowered)
    if match is not None:
        return f"{match.group(1)} 项结果"
    return clean


def _reasoning_status(tool: Mapping[str, object]) -> str:
    status = tool.get("status")
    if not isinstance(status, str) or status not in _REASONING_STATUS_MAP:
        raise ValueError("unsupported progress tool status")
    return _REASONING_STATUS_MAP[status]


def _reasoning_tool_name(tool: Mapping[str, object]) -> str:
    raw_label = tool.get("label")
    if isinstance(raw_label, str):
        try:
            clean = sanitize_visible_text(raw_label)
        except CardLimitError:
            clean = None
        if clean:
            if len(clean) > _REASONING_TOOL_NAME_MAX:
                return f"{clean[:_REASONING_TOOL_NAME_MAX]}…"
            return clean
    raw_name = tool.get("tool_name")
    return localized_tool_label(raw_name if isinstance(raw_name, str) else None)


def _reasoning_action(tool: Mapping[str, object]) -> dict[str, str]:
    status = _reasoning_status(tool)
    tool_name = tool.get("tool_name")
    if tool_name == "__subagent_wait__":
        label = "正在等待子任务…" if status == "running" else "子任务已返回"
        duration = format_progress_duration(tool.get("duration_ms"))
        detail = f"{label} · {duration}" if duration else label
    else:
        summary = tool.get("summary")
        safe_summary = (
            sanitize_visible_text(summary, generic=False)
            if isinstance(summary, str)
            else None
        )
        result_summary = tool.get("result_summary")
        safe_result = (
            localize_result_summary(result_summary)
            if isinstance(result_summary, str)
            else None
        )
        error = sanitize_error_text(tool.get("error"))
        parts = [
            part
            for part in (
                safe_summary,
                error if status == "error" else safe_result,
            )
            if part
        ]
        if parts:
            detail = " · ".join(parts)
        elif status == "running":
            detail = "进行中"
        elif status == "error":
            detail = "调用失败"
        else:
            detail = "已完成"
    return {
        "tool": _reasoning_tool_name(tool),
        "detail": detail,
        "statusGlyph": (
            "◉" if status == "running" else "○" if status == "error" else "●"
        ),
        "statusTone": (
            "Accent"
            if status == "running"
            else "Attention"
            if status == "error"
            else "Good"
        ),
    }


def _reasoning_phases(
    tools: Sequence[Mapping[str, object]],
    *,
    synthesize_empty_actions: bool,
) -> list[dict[str, object]]:
    phases: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    thinking_steps: list[Mapping[str, object]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise ValueError("progress tool entries must be objects")
        if tool.get("tool_name") == "__thinking__":
            _reasoning_status(tool)
            thinking_steps.append(tool)
            current = {
                "thought": sanitize_reasoning_thought(tool.get("thought")),
                "actions": [],
            }
            phases.append(current)
            continue
        if current is None:
            current = {
                "thought": _REASONING_FALLBACK_THOUGHT,
                "actions": [],
            }
            phases.append(current)
        actions = current["actions"]
        assert isinstance(actions, list)
        actions.append(_reasoning_action(tool))
    if not phases:
        phases.append({"thought": _REASONING_FALLBACK_THOUGHT, "actions": []})
    if not synthesize_empty_actions:
        return phases
    for index, phase in enumerate(phases):
        actions = phase["actions"]
        assert isinstance(actions, list)
        if actions:
            continue
        thinking = thinking_steps[index] if index < len(thinking_steps) else None
        status = _reasoning_status(thinking) if thinking is not None else "done"
        detail = (
            "正在规划下一步…"
            if status == "running"
            else "该阶段已停止"
            if status == "error"
            else "该阶段已完成"
        )
        duration = (
            format_progress_duration(thinking.get("duration_ms"))
            if thinking is not None
            else ""
        )
        actions.append({
            "tool": "分析问题",
            "detail": f"{detail} · {duration}" if duration else detail,
            "statusGlyph": (
                "◉" if status == "running" else "○" if status == "error" else "●"
            ),
            "statusTone": (
                "Accent"
                if status == "running"
                else "Attention"
                if status == "error"
                else "Good"
            ),
        })
    return phases


def _trim_reasoning_phases(
    phases: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    remaining = _REASONING_MAX_ACTIONS
    visible: list[dict[str, object]] = []
    for phase in reversed(phases[-_REASONING_MAX_PHASES:]):
        if remaining <= 0:
            break
        raw_actions = phase.get("actions")
        actions = list(raw_actions) if isinstance(raw_actions, list) else []
        actions = actions[-remaining:]
        remaining -= len(actions)
        visible.append({
            "thought": phase.get("thought", _REASONING_FALLBACK_THOUGHT),
            "actions": actions,
        })
    visible.reverse()
    return visible


def _reasoning_process_data(
    *,
    phase: str,
    tools: Sequence[Mapping[str, object]],
    elapsed_ms: object,
    reasoning_id: str,
    phases: list[dict[str, object]],
) -> dict[str, object]:
    state = _REASONING_PHASE_NAMES.get(phase)
    if state is None:
        raise ValueError("unsupported progress card phase")
    elapsed = format_progress_duration(elapsed_ms) or "0ms"
    tool_count = sum(
        tool.get("tool_name") not in {"__thinking__", "__subagent_wait__"}
        for tool in tools
    )
    phase_count = len(phases)
    phase_label = f"{phase_count} 个阶段"
    tool_label = f"{tool_count} 次工具调用"
    active = state in {"reasoning", "answering"}
    error_message = (
        "等待后台任务超时。"
        if phase == "expired"
        else "处理被中断，已完成的步骤仍然保留。"
    )
    data: dict[str, object] = {
        "reasoningId": reasoning_id.strip() or "octo-progress",
        "state": state,
        "title": "处理进度",
        "statusLabel": (
            "进行中"
            if state == "reasoning"
            else "正在整理答案"
            if state == "answering"
            else "已完成"
            if state == "completed"
            else "已停止"
            if state == "stopped"
            else "处理失败"
        ),
        "statusTone": (
            "Accent"
            if state in {"reasoning", "answering"}
            else "Good"
            if state == "completed"
            else "Warning"
            if state == "stopped"
            else "Attention"
        ),
        "timerText": (
            "正在处理…"
            if state == "reasoning"
            else "正在整理答案…"
            if state == "answering"
            else f"{elapsed} · 已保留 {phase_label}"
            if state == "stopped"
            else "处理被中断"
            if state == "error"
            else f"{elapsed} · {phase_label} · {tool_label}"
        ),
        "traceExpanded": active or state == "error",
        "traceCollapsed": not active and state != "error",
        "collapsedSummary": (
            "分析已完成，正在整理答案"
            if state == "answering"
            else f"已保留停止前的 {phase_label}"
            if state == "stopped"
            else "处理被中断，可展开查看已完成的步骤"
            if state == "error"
            else f"{elapsed} · 执行详情已收起"
            if state == "completed"
            else "正在处理，可展开查看执行详情"
        ),
        "phases": phases,
    }
    if state == "reasoning":
        data["progressText"] = (
            "正在等待子任务…"
            if phase == "paused"
            else "子任务已返回，正在收尾…"
            if phase == "resuming"
            else "正在处理…"
        )
    elif state == "answering":
        data["progressText"] = "分析已完成，正在整理答案…"
    elif state == "error":
        data["errorTitle"] = "处理未完成"
        data["errorMessage"] = error_message
    return data


def build_reasoning_process_data(
    *,
    phase: str,
    tools: Sequence[Mapping[str, object]],
    elapsed_ms: object = None,
    reasoning_id: str = "",
) -> dict[str, object]:
    """Build the local OpenClaw-compatible reasoning view model."""
    phases = _reasoning_phases(tools, synthesize_empty_actions=True)
    return _reasoning_process_data(
        phase=phase,
        tools=tools,
        elapsed_ms=elapsed_ms,
        reasoning_id=reasoning_id,
        phases=phases,
    )


def build_reasoning_process_wire_data(
    *,
    phase: str,
    tools: Sequence[Mapping[str, object]],
    elapsed_ms: object = None,
    reasoning_id: str = "",
) -> dict[str, object] | None:
    """Build bounded Registry data without synthetic actions."""
    phases = [
        item
        for item in _reasoning_phases(
            tools,
            synthesize_empty_actions=False,
        )
        if item["actions"]
    ]
    if not phases:
        return None
    phases = _trim_reasoning_phases(phases)
    return _reasoning_process_data(
        phase=phase,
        tools=tools,
        elapsed_ms=elapsed_ms,
        reasoning_id=reasoning_id,
        phases=phases,
    )


def _reasoning_text_block(
    text: str,
    **extra: object,
) -> dict[str, object]:
    return {"type": "TextBlock", "text": text, "wrap": True, **extra}


def _reasoning_action_row(
    action: Mapping[str, object],
    *,
    first: bool,
    last: bool,
) -> dict[str, object]:
    glyph = str(action["statusGlyph"])
    rail = glyph if last else f"{glyph}\n│"
    return {
        "type": "ColumnSet",
        "spacing": "None" if first else "Small",
        "columns": [
            {
                "type": "Column",
                "width": "auto",
                "items": [
                    _reasoning_text_block(
                        rail,
                        color=str(action["statusTone"]),
                        size="Small",
                        spacing="None",
                    )
                ],
            },
            {
                "type": "Column",
                "width": "stretch",
                "items": [
                    _reasoning_text_block(
                        str(action["tool"]),
                        weight="Bolder",
                        size="Small",
                        spacing="None",
                    ),
                    _reasoning_text_block(
                        str(action["detail"]),
                        isSubtle=True,
                        size="Small",
                        spacing="None",
                        fontType="Monospace",
                    ),
                ],
            },
        ],
    }


def _reasoning_phase_block(
    phase: Mapping[str, object],
    *,
    first: bool,
) -> dict[str, object]:
    raw_actions = phase["actions"]
    assert isinstance(raw_actions, list)
    return {
        "type": "Container",
        "spacing": "None" if first else "Large",
        "separator": not first,
        "items": [
            _reasoning_text_block(
                str(phase["thought"]),
                size="Small",
                spacing="None",
            ),
            {
                "type": "Container",
                "spacing": "Small",
                "items": [
                    _reasoning_action_row(
                        action,
                        first=index == 0,
                        last=index == len(raw_actions) - 1,
                    )
                    for index, action in enumerate(raw_actions)
                    if isinstance(action, Mapping)
                ],
            },
        ],
    }


def build_reasoning_process_card(
    *,
    phase: str,
    tools: Sequence[Mapping[str, object]],
    elapsed_ms: object = None,
    reasoning_id: str = "",
    capabilities: CardCapabilities | None = None,
) -> CardRenderResult:
    """Render the local toggle-only OpenClaw reasoning card."""
    fallback_phase = (
        "completed"
        if _REASONING_PHASE_NAMES.get(phase) == "completed"
        else "failed"
        if _REASONING_PHASE_NAMES.get(phase) in {"error", "stopped"}
        else "running"
    )
    required = {"TextBlock", "Container", "ColumnSet"}
    if capabilities is not None and any(
        not _supports(capabilities.elements, element) for element in required
    ):
        return build_progress_card(
            phase=fallback_phase,
            tools=tools,
            capabilities=capabilities,
        )
    data = build_reasoning_process_data(
        phase=phase,
        tools=tools,
        elapsed_ms=elapsed_ms,
        reasoning_id=reasoning_id,
    )
    raw_phases = data["phases"]
    assert isinstance(raw_phases, list)
    phases = _trim_reasoning_phases(raw_phases)
    data["phases"] = phases
    can_toggle = (
        capabilities is not None
        and _supports(capabilities.elements, "ActionSet")
        and capabilities.actions is not None
        and "Action.ToggleVisibility" in capabilities.actions
    )
    trace_visible = bool(data["traceExpanded"]) if can_toggle else True
    body: list[dict[str, object]] = [
        {
            "type": "Container",
            "id": "octo-execution-trace-header",
            "style": "emphasis",
            "bleed": True,
            "spacing": "None",
            "items": [
                {
                    "type": "ColumnSet",
                    "spacing": "None",
                    "columns": [
                        {
                            "type": "Column",
                            "width": "stretch",
                            "items": [
                                _reasoning_text_block(
                                    str(data["title"]),
                                    weight="Bolder",
                                    spacing="None",
                                ),
                                _reasoning_text_block(
                                    str(data["timerText"]),
                                    size="Small",
                                    isSubtle=True,
                                    spacing="Small",
                                ),
                            ],
                        },
                        {
                            "type": "Column",
                            "width": "auto",
                            "items": [
                                _reasoning_text_block(
                                    str(data["statusLabel"]),
                                    color=data["statusTone"],
                                    weight="Bolder",
                                    size="Small",
                                    spacing="None",
                                )
                            ],
                        },
                    ],
                }
            ],
        },
        {
            "type": "Container",
            "id": "trace_panel",
            "isVisible": trace_visible,
            "spacing": "Large",
            "items": [
                *[
                    _reasoning_phase_block(item, first=index == 0)
                    for index, item in enumerate(phases)
                ],
                *(
                    [
                        _reasoning_text_block(
                            f"◉  {data['progressText']}",
                            color="Accent",
                            size="Small",
                            spacing="Large",
                        )
                    ]
                    if data.get("progressText")
                    else []
                ),
                *(
                    [
                        {
                            "type": "Container",
                            "style": "attention",
                            "spacing": "Large",
                            "items": [
                                _reasoning_text_block(
                                    str(data.get("errorTitle", "处理未完成")),
                                    weight="Bolder",
                                    color="Attention",
                                    spacing="None",
                                ),
                                _reasoning_text_block(
                                    str(data["errorMessage"]),
                                    size="Small",
                                    spacing="Small",
                                ),
                            ],
                        }
                    ]
                    if data.get("errorMessage")
                    else []
                ),
            ],
        },
        {
            "type": "Container",
            "id": "collapsed_panel",
            "isVisible": can_toggle and bool(data["traceCollapsed"]),
            "spacing": "Medium",
            "items": [
                _reasoning_text_block(
                    f"●  {data['collapsedSummary']}",
                    color=str(data["statusTone"]),
                    size="Small",
                    isSubtle=True,
                    spacing="None",
                )
            ],
        },
    ]
    if can_toggle:
        body.append({
            "type": "Container",
            "style": "emphasis",
            "bleed": True,
            "separator": True,
            "spacing": "Large",
            "items": [
                {
                    "type": "ActionSet",
                    "horizontalAlignment": "Right",
                    "actions": [
                        {
                            "type": "Action.ToggleVisibility",
                            "id": "reasoning_toggle",
                            "title": "展开/收起执行详情",
                            "targetElements": [
                                "trace_panel",
                                "collapsed_panel",
                            ],
                        }
                    ],
                }
            ],
        })
    plain_lines = [f"{data['title']} · {data['statusLabel']} · {data['timerText']}"]
    for item in phases:
        plain_lines.append(str(item["thought"]))
        actions = item["actions"]
        assert isinstance(actions, list)
        for action in actions:
            if isinstance(action, Mapping):
                plain_lines.append(f"{action['tool']} · {action['detail']}")
    if data.get("progressText"):
        plain_lines.append(str(data["progressText"]))
    if data.get("errorMessage"):
        plain_lines.append(str(data["errorMessage"]))
    result = CardRenderResult(
        card={
            "$schema": ADAPTIVE_CARD_SCHEMA,
            "type": "AdaptiveCard",
            "version": CARD_VERSION,
            "body": body,
            "metadata": {"octo_layout": "agent_progress_v1"},
        },
        plain="\n".join(plain_lines) or "[card]",
    )
    try:
        validate_card_limits(result.card, result.plain, capabilities)
    except CardLimitError:
        return build_progress_card(
            phase=fallback_phase,
            tools=tools,
            capabilities=capabilities,
        )
    return result


_PROGRESS_PHASES = frozenset({
    "thinking", "tool", "answering", "completed", "stopped", "failed", "expired",
    "starting", "running",
})
_PROGRESS_STATUSES = frozenset({"running", "complete", "failed"})
_PROGRESS_MAX_VISIBLE_STEPS = 12


def _progress_state(phase: str) -> tuple[str, str]:
    if phase in {"starting", "thinking", "running", "tool"}:
        return "进行中", "Accent"
    if phase == "answering":
        return "正在整理答案", "Accent"
    if phase == "completed":
        return "已完成", "Good"
    if phase == "stopped":
        return "已停止", "Warning"
    return "处理失败", "Attention"


def _progress_header(
    phase: str,
    tools: Sequence[Mapping[str, object]],
    elapsed_ms: object,
) -> str:
    status_label, _tone = _progress_state(phase)
    parts = ["处理进度", status_label]
    if tools:
        parts.append(f"{len(tools)} 个步骤")
    elapsed = format_progress_duration(elapsed_ms)
    if elapsed:
        parts.append(elapsed)
    return " · ".join(parts)


def _progress_tool_label(tool: Mapping[str, object]) -> str:
    raw_label = tool.get("label")
    if isinstance(raw_label, str):
        clean = sanitize_visible_text(raw_label)
        if clean:
            return clean
    raw_name = tool.get("tool_name")
    return localized_tool_label(raw_name if isinstance(raw_name, str) else None)


def _progress_step(tool: Mapping[str, object]) -> dict[str, str]:
    status = tool.get("status")
    if status not in _PROGRESS_STATUSES:
        raise ValueError("unsupported progress tool status")
    label = _progress_tool_label(tool)
    summary = tool.get("summary")
    raw_name = tool.get("tool_name")
    if isinstance(summary, str):
        safe_summary = sanitize_visible_text(summary, generic=False)
    else:
        safe_summary = summarize_tool_params(
            raw_name if isinstance(raw_name, str) else None,
            tool.get("args"),
        )
    result = tool.get("result_summary")
    safe_result = (
        localize_result_summary(result) if isinstance(result, str) else None
    )
    duration = format_progress_duration(tool.get("duration_ms"))
    if status == "running":
        detail_parts = [part for part in (safe_summary, "进行中") if part]
        glyph, tone = "◉", "Accent"
    elif status == "failed":
        error = sanitize_error_text(tool.get("error"))
        detail_parts = [
            part for part in (safe_summary, "失败", error, duration) if part
        ]
        glyph, tone = "○", "Attention"
    else:
        detail_parts = [
            part
            for part in (
                safe_summary,
                safe_result or "已完成",
                duration,
            )
            if part
        ]
        glyph, tone = "●", "Good"
    return {
        "tool": label,
        "detail": " · ".join(detail_parts),
        "statusGlyph": glyph,
        "statusTone": tone,
    }


def _progress_steps(
    tools: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    index = 0
    while index < len(tools):
        current = tools[index]
        if not isinstance(current, Mapping):
            raise ValueError("progress tool entries must be objects")
        name = current.get("tool_name")
        if current.get("status") == "complete":
            end = index + 1
            while (
                end < len(tools)
                and isinstance(tools[end], Mapping)
                and tools[end].get("tool_name") == name
                and tools[end].get("status") == "complete"
                and not tools[end].get("result_summary")
                and not current.get("result_summary")
            ):
                end += 1
            group = tools[index:end]
            if len(group) > 1:
                durations = [item.get("duration_ms") for item in group]
                valid = [
                    item
                    for item in durations
                    if isinstance(item, int)
                    and not isinstance(item, bool)
                    and item >= 0
                ]
                duration = (
                    format_progress_duration(sum(valid)) if valid else ""
                )
                latest = group[-1].get("summary")
                safe_latest = (
                    sanitize_visible_text(latest, generic=False)
                    if isinstance(latest, str)
                    else None
                )
                detail_parts = []
                if duration:
                    detail_parts.append(f"总计 {duration}")
                if safe_latest:
                    detail_parts.append(f"最近：{safe_latest}")
                steps.append({
                    "tool": f"{_progress_tool_label(current)} × {len(group)}",
                    "detail": " · ".join(detail_parts) or "已完成",
                    "statusGlyph": "●",
                    "statusTone": "Good",
                })
                index = end
                continue
        steps.append(_progress_step(current))
        index += 1
    return steps


def _progress_plain_line(step: Mapping[str, str]) -> str:
    detail = step.get("detail")
    suffix = f" · {detail}" if detail else ""
    return f"{step['statusGlyph']} {step['tool']}{suffix}"


def build_progress_card(
    *,
    phase: str,
    tools: Sequence[Mapping[str, object]] = (),
    elapsed_ms: object = None,
    capabilities: CardCapabilities | None = None,
) -> CardRenderResult:
    """Render a Chinese execution trace without exposing reasoning text."""
    if phase not in _PROGRESS_PHASES:
        raise ValueError("unsupported progress card phase")
    real_tools = [
        tool
        for tool in tools
        if isinstance(tool, Mapping) and tool.get("tool_name") != "__thinking__"
    ]
    hidden = max(0, len(real_tools) - _PROGRESS_MAX_VISIBLE_STEPS)
    visible = list(real_tools[-_PROGRESS_MAX_VISIBLE_STEPS:])
    steps = _progress_steps(visible)
    detail_lines = (
        [f"已隐藏前 {hidden} 个步骤"] if hidden else []
    ) + [_progress_plain_line(step) for step in steps]
    if not detail_lines:
        detail_lines = ["正在准备"]
    header = _progress_header(phase, real_tools, elapsed_ms)
    plain = "\n".join([header, *detail_lines])
    flat = build_display_card(
        title=header,
        blocks=[{"type": "text", "text": line} for line in detail_lines],
        capabilities=capabilities,
    )
    if (
        capabilities is None
        or not _supports(capabilities.elements, "ColumnSet")
        or not _supports(capabilities.elements, "Container")
    ):
        return CardRenderResult(card=flat.card, plain=plain)
    can_toggle = (
        _supports(capabilities.elements, "ActionSet")
        and capabilities.actions is not None
        and "Action.ToggleVisibility" in capabilities.actions
    )
    terminal = phase in {"completed", "stopped", "failed", "expired"}
    detail_visible = not (can_toggle and terminal)
    status_label, status_tone = _progress_state(phase)
    summary_parts = []
    if real_tools:
        complete_count = sum(
            tool.get("status") == "complete" for tool in real_tools
        )
        if terminal:
            summary_parts.append(f"{len(real_tools)} 个步骤")
        else:
            summary_parts.append(
                f"已完成 {complete_count}/{len(real_tools)} 个步骤"
            )
    elapsed = format_progress_duration(elapsed_ms)
    if elapsed:
        summary_parts.append(elapsed)
    summary_text = " · ".join(summary_parts) or "正在准备"
    status_items: list[dict[str, object]] = [
        _reasoning_text_block(
            status_label,
            color=status_tone,
            weight="Bolder",
            size="Small",
            spacing="None",
        )
    ]
    header_columns: list[dict[str, object]] = [
        {
            "type": "Column",
            "width": "stretch",
            "items": [
                _reasoning_text_block(
                    "处理进度",
                    weight="Bolder",
                    spacing="None",
                ),
                _reasoning_text_block(
                    summary_text,
                    size="Small",
                    isSubtle=True,
                    spacing="Small",
                ),
            ],
        },
        {
            "type": "Column",
            "width": "auto",
            "items": status_items,
        },
    ]
    if can_toggle:
        status_items.extend([
            {
                "type": "ActionSet",
                "id": "btn_collapse",
                "isVisible": detail_visible,
                "actions": [{
                    "type": "Action.ToggleVisibility",
                    "title": "收起执行详情",
                    "targetElements": [
                        {"elementId": "timeline_detail", "isVisible": False},
                        {"elementId": "btn_collapse", "isVisible": False},
                        {"elementId": "btn_expand", "isVisible": True},
                    ],
                }],
            },
            {
                "type": "ActionSet",
                "id": "btn_expand",
                "isVisible": not detail_visible,
                "actions": [{
                    "type": "Action.ToggleVisibility",
                    "title": "展开执行详情",
                    "targetElements": [
                        {"elementId": "timeline_detail", "isVisible": True},
                        {"elementId": "btn_collapse", "isVisible": True},
                        {"elementId": "btn_expand", "isVisible": False},
                    ],
                }],
            },
        ])
    trace_items: list[dict[str, object]] = []
    if hidden:
        trace_items.append(
            _reasoning_text_block(
                f"已隐藏前 {hidden} 个步骤",
                isSubtle=True,
                size="Small",
                spacing="None",
            )
        )
    trace_items.extend(
        _reasoning_action_row(
            step,
            first=index == 0 and not hidden,
            last=index == len(steps) - 1,
        )
        for index, step in enumerate(steps)
    )
    if not trace_items:
        trace_items.append(
            _reasoning_text_block(
                "正在准备",
                isSubtle=True,
                size="Small",
                spacing="None",
            )
        )
    card: dict[str, Any] = {
        "$schema": ADAPTIVE_CARD_SCHEMA,
        "type": "AdaptiveCard",
        "version": CARD_VERSION,
        "body": [
            {
                "type": "Container",
                "id": "octo-execution-trace-header",
                "style": "emphasis",
                "bleed": True,
                "spacing": "None",
                "items": [{
                    "type": "ColumnSet",
                    "spacing": "None",
                    "columns": header_columns,
                }],
            },
            {
                "type": "Container",
                "id": "timeline_detail",
                "isVisible": detail_visible,
                "spacing": "Medium",
                "items": trace_items,
            },
        ],
        "metadata": {"octo_layout": "agent_progress_v1"},
    }
    try:
        validate_card_limits(card, plain, capabilities)
    except CardLimitError:
        return CardRenderResult(card=flat.card, plain=plain)
    return CardRenderResult(card=card, plain=plain)


def build_agent_progress_card(
    *,
    phase: str,
    tools: Sequence[Mapping[str, object]],
    elapsed_ms: object = None,
    reasoning_id: str = "",
    reasoning_visible: bool,
    capabilities: CardCapabilities | None = None,
) -> CardRenderResult:
    """Select reasoning or fallback progress independently of delivery mode."""
    has_public_thought = any(
        tool.get("tool_name") == "__thinking__"
        and isinstance(tool.get("thought"), str)
        and bool(str(tool.get("thought")).strip())
        for tool in tools
    )
    if reasoning_visible and has_public_thought:
        return build_reasoning_process_card(
            phase=phase, tools=tools, elapsed_ms=elapsed_ms,
            reasoning_id=reasoning_id, capabilities=capabilities,
        )
    return build_progress_card(
        phase=phase, tools=tools, elapsed_ms=elapsed_ms, capabilities=capabilities,
    )
