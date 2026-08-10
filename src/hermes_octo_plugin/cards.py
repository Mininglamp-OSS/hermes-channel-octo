"""Safe pure renderers for controlled Octo Type-17 cards."""

from __future__ import annotations

import json
import math
import re
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
    profiles = (
        frozenset(manifest.profiles or ())
        if authoritative
        else None
    )
    elements = (
        frozenset(manifest.elements or ())
        if authoritative
        else (
            frozenset(manifest.elements)
            if manifest.elements is not None
            else None
        )
    )
    inputs = (
        frozenset(manifest.inputs or ())
        if authoritative
        else (
            frozenset(manifest.inputs)
            if manifest.inputs is not None
            else None
        )
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
        max_payload_bytes=_positive_limit(
            manifest.limits.get("max_payload_bytes")
        ),
        max_input_text_bytes=_positive_limit(
            manifest.limits.get("max_input_text_bytes")
        ),
        max_inputs_bytes=_positive_limit(
            manifest.limits.get("max_inputs_bytes")
        ),
    )


_MULTI_PART_TLDS = frozenset(
    {
        "ac.uk",
        "co.jp",
        "co.kr",
        "co.uk",
        "com.au",
        "com.br",
        "com.cn",
        "com.hk",
        "com.sg",
        "com.tw",
        "edu.cn",
        "gov.uk",
        "gov.cn",
        "net.cn",
        "org.cn",
        "org.uk",
    }
)
_URL_IN_TEXT_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s)\]}>\"']+"
)
_MARKDOWN_LINK_RE = re.compile(
    r"\[([^\]\r\n]{0,512})\]\(\s*([^\s)]+)(?:\s+[^)]*)?\)"
)
_PROTOCOL_RELATIVE_RE = re.compile(
    r"(^|[^A-Za-z0-9/:])"
    r"(//[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
    r"(?::\d+)?(?:/[^\s)\]}>\"']*)?)"
)
_SCHEMELESS_USERINFO_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+:[^\s/]+@"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
    r"(?::\d+)?(?:/[^\s)\]}>\"']*)?"
)
_SCHEMELESS_HOST_PATH_RE = re.compile(
    r"(^|[^A-Za-z0-9@._/:+-])"
    r"([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
    r"(?::\d+)?/[^\s)\]}>\"']+)"
)
_SECRET_KEYWORD_RE = re.compile(
    r"token|api[_-]?key|secret|password|passwd|pwd|authorization|bearer|"
    r"access[_-]?key|client[_-]?secret|credential",
    re.IGNORECASE,
)
_SECRET_PREFIX_RES = (
    re.compile(r"AKIA[0-9A-Z]{12,}"),
    re.compile(r"(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"[srp]k_(?:live|test)_[A-Za-z0-9]{10,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"xapp-[0-9]-[A-Za-z0-9-]{10,}"),
    re.compile(r"npm_[A-Za-z0-9]{30,}"),
    re.compile(r"shpat_[A-Fa-f0-9]{32,}"),
    re.compile(r"dop_v1_[A-Fa-f0-9]{32,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"),
)
_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_GENERIC_SECRET_RUN_RE = re.compile(r"[A-Za-z0-9_+/=-]{32,}")
_SUMMARY_STRATEGY = {
    "apply_patch": "path",
    "bash": "shell",
    "edit": "path",
    "exec": "shell",
    "fetch": "url",
    "find": "path",
    "glob": "path",
    "grep": "query",
    "ls": "path",
    "process": "shell",
    "read": "path",
    "search": "query",
    "shell": "shell",
    "web_search": "query",
    "write": "path",
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


def _registrable_domain(host: str) -> str:
    labels = host.rstrip(".").lower().split(".")
    if len(labels) <= 2 or ":" in host:
        return host.lower()
    keep = 3 if ".".join(labels[-2:]) in _MULTI_PART_TLDS else 2
    return ".".join(labels[-keep:])


def _origin_domain(raw_url: str) -> str | None:
    try:
        parsed = urlsplit(raw_url)
        host = parsed.hostname
    except ValueError:
        return None
    if not parsed.scheme or not host:
        return None
    safe_host = _registrable_domain(host)
    if ":" in safe_host:
        safe_host = f"[{safe_host}]"
    return f"{parsed.scheme.lower()}://{safe_host}"


def _markdown_link(match: re.Match[str]) -> str:
    label = match.group(1)
    target = match.group(2)
    candidate = f"https:{target}" if target.startswith("//") else target
    origin = _origin_domain(candidate)
    if origin is None or not origin.startswith(("http://", "https://")):
        return label
    return f"[{label}]({origin})"


def reduce_urls_in_text(text: str) -> str:
    """Reduce every URL-shaped visible sink to a disclosure-safe origin."""
    if len(text.encode("utf-8")) > DEFAULT_MAX_VISIBLE_TEXT_BYTES:
        raise CardLimitError("card text bytes exceed local limit")
    reduced = _MARKDOWN_LINK_RE.sub(_markdown_link, text)
    reduced = _URL_IN_TEXT_RE.sub(
        lambda match: _origin_domain(match.group(0)) or "",
        reduced,
    )
    reduced = _PROTOCOL_RELATIVE_RE.sub(
        lambda match: (
            match.group(1)
            + (_origin_domain(f"https:{match.group(2)}") or "")
        ),
        reduced,
    )
    reduced = _SCHEMELESS_USERINFO_RE.sub(
        lambda match: (
            _origin_domain(
                "https://" + match.group(0).rsplit("@", 1)[-1].split("/", 1)[0]
            )
            or ""
        ),
        reduced,
    )
    return _SCHEMELESS_HOST_PATH_RE.sub(
        lambda match: (
            match.group(1)
            + (_origin_domain(f"https://{match.group(2)}") or "")
        ),
        reduced,
    )


def is_sensitive(text: str, *, generic: bool = True) -> bool:
    """Detect credential names and common credential value shapes."""
    if _SECRET_KEYWORD_RE.search(text):
        return True
    if any(pattern.search(text) for pattern in _SECRET_PREFIX_RES):
        return True
    if not generic:
        return False
    if _LONG_HEX_RE.search(text):
        return True
    return any(
        (
            any(character.isdigit() for character in run)
            and any(character.isalpha() for character in run)
        )
        or any(character in "+/_=-" for character in run)
        for run in _GENERIC_SECRET_RUN_RE.findall(text)
    )


def sanitize_visible_text(text: str, *, generic: bool = True) -> str | None:
    """Reduce URL disclosure and drop a visible sink when it looks sensitive."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return None
    reduced = re.sub(r"\s+", " ", reduce_urls_in_text(normalized)).strip()
    if not reduced or is_sensitive(reduced, generic=generic):
        return None
    return reduced


def _first_string(params: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _shorten_path(path: str) -> str:
    segments = [segment for segment in re.split(r"[/\\]+", path) if segment]
    if len(segments) <= 3:
        return path
    return f"…/{segments[-2]}/{segments[-1]}"


def _summarize_shell(params: Mapping[str, object]) -> str:
    command = _first_string(params, ("command", "cmd")).strip()
    if not command:
        return ""
    tokens = command.split()
    index = 0
    while index < len(tokens) and re.match(
        r"^[A-Za-z_][A-Za-z0-9_]*=",
        tokens[index],
    ):
        index += 1
    program = tokens[index] if index < len(tokens) else ""
    if not _PROGRAM_TOKEN_RE.fullmatch(program):
        return ""
    return program.rsplit("/", 1)[-1]


def summarize_tool_params(
    tool_name: str | None,
    params: object,
) -> str:
    """Return one allowlisted, disclosure-safe tool parameter summary."""
    if not tool_name or not isinstance(params, Mapping):
        return ""
    strategy = _SUMMARY_STRATEGY.get(tool_name)
    if strategy is None:
        return ""
    if strategy == "path":
        summary = _shorten_path(
            _first_string(params, ("path", "file_path", "file"))
        )
    elif strategy == "shell":
        summary = _summarize_shell(params)
    elif strategy == "url":
        raw_url = _first_string(params, ("url",))
        summary = _origin_domain(raw_url) or ""
    else:
        summary = _first_string(params, ("query", "pattern"))
    summary = re.sub(r"\s+", " ", reduce_urls_in_text(summary)).strip()
    if not summary or is_sensitive(
        summary,
        generic=strategy in {"query", "url"},
    ):
        return ""
    if len(summary) > _SUMMARY_MAX_CHARS:
        return f"{summary[:_SUMMARY_MAX_CHARS]}…"
    return summary


def safe_tool_label(tool_name: str | None) -> str:
    """Return a bounded label without echoing arbitrary MCP or secret text."""
    if not tool_name:
        return "tool"
    if tool_name.startswith("mcp__"):
        return "MCP tool"
    if (
        not _SAFE_TOOL_LABEL_RE.fullmatch(tool_name)
        or is_sensitive(tool_name, generic=True)
    ):
        return "tool"
    return tool_name


def sanitize_error_text(error: object) -> str:
    """Return a short visible error summary or an empty fail-closed value."""
    if not isinstance(error, str):
        return ""
    summary = re.sub(r"\s+", " ", reduce_urls_in_text(error)).strip()
    if not summary or is_sensitive(summary, generic=True):
        return ""
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
            count_card_nodes(item, _root=False, _seen=seen)
            for item in value.values()
        )
    return sum(
        count_card_nodes(item, _root=False, _seen=seen)
        for item in value
    )


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
        encoded.replace("&", r"\u0026")
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
        if capabilities is not None
        and capabilities.max_payload_bytes is not None
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
                body.append(
                    {"type": "Image", "url": resource_url, "altText": clean_alt}
                )
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
                actions.append(
                    {"type": "Action.OpenUrl", "title": clean_label, "url": safe_url}
                )
                lines.append(f"{clean_label}: {safe_url}")
            if not actions:
                continue
            can_render_actions = (
                (capabilities is None or _supports(capabilities.elements, "ActionSet"))
                and (
                    capabilities is None
                    or _supports(capabilities.actions, "Action.OpenUrl")
                )
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
    clean = sanitize_visible_text(value.strip())
    if clean is None:
        raise ValueError(f"{field} must not contain sensitive data")
    if len(clean) > max_chars:
        return f"{clean[:max_chars]}…"
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
        raise ValueError(f"{field} exceeds safe limit")
    clean = sanitize_visible_text(candidate)
    if clean is None or clean != candidate:
        raise ValueError(f"{field} must not contain sensitive or URL data")
    return clean


def _sanitize_action_data(
    value: object,
    *,
    key: str = "",
    depth: int = 0,
) -> object:
    if _SECRET_KEYWORD_RE.fullmatch(key):
        return "[redacted]"
    if isinstance(value, str):
        candidate = value.strip()
        if len(candidate.encode("utf-8")) > DEFAULT_MAX_ACTION_DATA_VALUE_BYTES:
            raise ValueError("action data value exceeds byte limit")
        clean = sanitize_visible_text(candidate)
        if clean is None:
            return "[redacted]"
        if clean != candidate:
            raise ValueError("action data value must not contain URL data")
        return clean
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
        return [
            _sanitize_action_data(item, depth=depth + 1)
            for item in value
        ]
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for index, (child_key, child_value) in enumerate(
            islice(value.items(), 51)
        ):
            if index == 50:
                raise ValueError("action data exceeds item limit")
            if not isinstance(child_key, str):
                raise ValueError("action data key must be a string")
            if child_key == "_octo_binding":
                continue
            if (
                child_key.startswith("_octo_")
                or not _CARD_ID_RE.fullmatch(child_key)
                or (
                    is_sensitive(child_key, generic=True)
                    and not _SECRET_KEYWORD_RE.fullmatch(child_key)
                )
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
    if (
        not _CARD_ID_RE.fullmatch(candidate)
        or is_sensitive(candidate, generic=True)
    ):
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
        body.append(
            {
                "type": "TextBlock",
                "text": clean_text,
                "wrap": True,
                "spacing": "Small",
            }
        )
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
        if kind == "choice":
            raw_choices = raw_input.get("choices")
            if not isinstance(raw_choices, Sequence) or isinstance(
                raw_choices,
                (str, bytes),
            ) or not raw_choices:
                raise ValueError(f"choice input {input_id} requires choices")
            choices: list[dict[str, str]] = []
            choice_values: set[str] = set()
            choice_titles: list[str] = []
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



_PROGRESS_TITLES = {
    "starting": "Working",
    "running": "Working",
    "completed": "Completed",
    "failed": "Stopped",
}
_PROGRESS_STATUSES = frozenset({"running", "complete", "failed"})


def build_progress_card(
    *,
    phase: str,
    tools: Sequence[Mapping[str, object]] = (),
    capabilities: CardCapabilities | None = None,
) -> CardRenderResult:
    """Render lifecycle state without exposing prompts, reasoning, or raw output."""
    title = _PROGRESS_TITLES.get(phase)
    if title is None:
        raise ValueError("unsupported progress card phase")
    if len(tools) > 32:
        raise CardLimitError("progress card exceeds tool entry limit")
    blocks: list[dict[str, str]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise ValueError("progress tool entries must be objects")
        status = tool.get("status")
        if status not in _PROGRESS_STATUSES:
            raise ValueError("unsupported progress tool status")
        label = safe_tool_label(
            tool.get("tool_name") if isinstance(tool.get("tool_name"), str) else None
        )
        raw_summary = tool.get("summary")
        if isinstance(raw_summary, str):
            clean_summary = sanitize_visible_text(raw_summary)
            summary = clean_summary[:_SUMMARY_MAX_CHARS] if clean_summary else ""
        else:
            summary = summarize_tool_params(label, tool.get("args"))
        line = f"{label}{f' ({summary})' if summary else ''}: {status}"
        if status == "failed":
            safe_error = sanitize_error_text(tool.get("error"))
            if safe_error:
                line = f"{line} - {safe_error}"
        blocks.append({"type": "text", "text": line})
    if not blocks:
        blocks.append(
            {
                "type": "text",
                "text": "Preparing" if phase == "starting" else title,
            }
        )
    flat = build_display_card(
        title=title,
        blocks=blocks,
        capabilities=capabilities,
    )
    if (
        capabilities is None
        or not _supports(capabilities.elements, "ColumnSet")
        or not _supports(capabilities.elements, "Container")
    ):
        return flat
    detail = build_display_card(
        blocks=blocks,
        capabilities=capabilities,
    )
    card: dict[str, Any] = {
        "$schema": ADAPTIVE_CARD_SCHEMA,
        "type": "AdaptiveCard",
        "version": CARD_VERSION,
        "body": [
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [_text_element(title, bold=True)],
                    }
                ],
            },
            {
                "type": "Container",
                "id": "octo_progress_details",
                "items": detail.card["body"],
            },
        ],
        "metadata": {"octo_layout": "agent_progress_v1"},
    }
    try:
        validate_card_limits(card, flat.plain, capabilities)
    except CardLimitError:
        return flat
    return CardRenderResult(card=card, plain=flat.plain)