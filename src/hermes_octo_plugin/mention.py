"""
@mention parsing and conversion utilities.

Provides consistent mention detection across inbound and outbound code paths.

Supports two formats:
  - v1: @name (regex-based, positional pairing with uids)
  - v2: @[uid:name] (structured, precise mapping via entities)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .types import MentionEntity, MentionPayload

logger = logging.getLogger(__name__)

# Hard upper bound on how many @mentions we'll parse out of a single inbound
# message. Each mention triggers member-map lookups, regex passes, and
# back-to-front string rewrites — an attacker that spams hundreds of @
# tokens could otherwise wedge an event loop turn. 64 is well above any
# legitimate group-chat use.
MAX_MENTIONS_PER_MESSAGE = 64

# ─── Regex Patterns ──────────────────────────────────────────────────────────

# Matches @mentions in message content.
# Boundary: @ must be preceded by start-of-string or non-alphanumeric.
# Name chars: word chars, CJK, accented letters, dots, hyphens.
MENTION_PATTERN = re.compile(
    r"(?:^|(?<=\s|[^a-zA-Z0-9]))"
    r"@([\w\u00C0-\u024F\u4e00-\u9fff\u3040-\u30FF\uAC00-\uD7AF.\-]+)"
)

# Matches @[uid:displayName] format (adapter↔LLM internal use).
STRUCTURED_MENTION_PATTERN = re.compile(r"@\[([\w.\-]+):([^\]\n]+)\]")


# ─── Structured mention parse + convert (outbound: LLM reply → wire format) ──


@dataclass
class StructuredMention:
    """A single @[uid:name] occurrence in an outbound LLM reply.

    ``offset`` and ``length`` point into the *original* text, before
    conversion. They are recomputed against the converted text in
    :func:`convert_structured_mentions`.
    """
    uid: str
    name: str
    offset: int
    length: int


def parse_structured_mentions(text: str) -> list[StructuredMention]:
    """Find every ``@[uid:name]`` token in *text*.

    Used on outbound paths to detect LLM-emitted structured mentions before
    the bot sends the message — the wire format expects ``@name`` text plus
    a ``mention.entities`` sidecar, not the raw ``@[uid:name]`` token.
    """
    out: list[StructuredMention] = []
    for m in STRUCTURED_MENTION_PATTERN.finditer(text):
        out.append(StructuredMention(
            uid=m.group(1),
            name=m.group(2),
            offset=m.start(),
            length=len(m.group(0)),
        ))
    return out


def convert_structured_mentions(
    text: str,
    mentions: list[StructuredMention],
) -> tuple[str, list[MentionEntity], list[str]]:
    """Replace each ``@[uid:name]`` in *text* with ``@name`` and emit the
    matching wire-format sidecar.

    Algorithm (incremental, single pass):
      Sort mentions by ``offset`` and reconstruct the output string segment
      by segment. Tracking ``len(content)`` after each insertion gives the
      exact new offset of every ``@name`` — safer than reusing ``indexOf``
      which can bind duplicate names to the wrong position.

    Returns ``(content, entities, uids)`` where ``entities`` and ``uids``
    are in the same order as the original mentions (after offset sort).
    """
    sorted_mentions = sorted(mentions, key=lambda m: m.offset)
    entities: list[MentionEntity] = []
    uids: list[str] = []
    content_parts: list[str] = []
    cursor = 0
    running_len = 0
    for m in sorted_mentions:
        # Verbatim text between previous cursor and this mention.
        between = text[cursor:m.offset]
        content_parts.append(between)
        running_len += len(between)

        replacement = f"@{m.name}"
        entities.append(MentionEntity(
            uid=m.uid,
            offset=running_len,
            length=len(replacement),
        ))
        uids.append(m.uid)
        content_parts.append(replacement)
        running_len += len(replacement)

        cursor = m.offset + m.length
    # Tail after last mention.
    content_parts.append(text[cursor:])
    return "".join(content_parts), entities, uids


# ─── Extract UIDs from MentionPayload ────────────────────────────────────────


def _coerce_mention(mention: Any) -> MentionPayload | None:
    """Accept either a MentionPayload or a raw dict and return a MentionPayload."""
    if mention is None:
        return None
    if isinstance(mention, MentionPayload):
        return mention
    if isinstance(mention, dict):
        raw_entities = mention.get("entities")
        entities = None
        if isinstance(raw_entities, list):
            entities = [
                MentionEntity(uid=e["uid"], offset=e["offset"], length=e["length"])
                for e in raw_entities
                if isinstance(e, dict) and "uid" in e and "offset" in e and "length" in e
            ]
        uids = mention.get("uids") if isinstance(mention.get("uids"), list) else None
        all_flag = mention.get("all")
        return MentionPayload(uids=uids, entities=entities, all=all_flag)
    return None


def extract_mention_uids(mention: Any) -> list[str]:
    """Extract mention UIDs, preferring entities over uids.

    Accepts either a MentionPayload or a raw dict (e.g. from API messages).
    """
    mention = _coerce_mention(mention)
    if not mention:
        return []

    if mention.entities:
        valid_uids = [
            e.uid
            for e in mention.entities
            if isinstance(e, MentionEntity) and e.uid
        ]
        if valid_uids:
            return valid_uids

    if mention.uids:
        return [uid for uid in mention.uids if isinstance(uid, str)]

    return []


# ─── Convert @name → @[uid:name] for LLM Context ────────────────────────────


def convert_content_for_llm(
    content: str,
    mention: Any = None,
    member_map: dict[str, str] | None = None,
) -> str:
    """
    Convert @mentions in message content to @[uid:name] format for LLM context.

    Accepts either a MentionPayload or a raw dict for ``mention``.

    Path priority:
    1. entities valid → precise replacement (v2)
    2. entities invalid / not present → member_map lookup or uids positional pairing (v1)
    3. no mention → return original content

    Replacement proceeds from back to front to avoid offset drift.
    """
    mention = _coerce_mention(mention)
    if not mention:
        return content

    # Try entities (v2) — precise offset-based replacement
    if mention.entities:
        valid_entities = [
            e
            for e in mention.entities
            if (
                isinstance(e, MentionEntity)
                and e.uid
                and isinstance(e.offset, int)
                and isinstance(e.length, int)
                and e.offset >= 0
                and e.length > 0
                and e.offset + e.length <= len(content)
            )
        ]

        if valid_entities:
            sorted_entities = sorted(valid_entities, key=lambda e: e.offset, reverse=True)
            result = content
            for entity in sorted_entities:
                original = result[entity.offset : entity.offset + entity.length]
                if not original.startswith("@"):
                    continue
                name = original[1:]
                replacement = f"@[{entity.uid}:{name}]"
                result = (
                    result[: entity.offset]
                    + replacement
                    + result[entity.offset + entity.length :]
                )
            return result

    # Fallback (v1): member_map lookup or uids positional pairing
    has_member_map = member_map and len(member_map) > 0
    has_uids = mention.uids and len(mention.uids) > 0

    if has_member_map or has_uids:
        result = content
        uid_index = 0
        replacements: list[tuple[int, int, str]] = []  # (start, end, replacement)

        # Sort member names by length descending for longest-match-first
        sorted_names = sorted(member_map.keys(), key=len, reverse=True) if has_member_map else []

        for i, match in enumerate(MENTION_PATTERN.finditer(content)):
            if i >= MAX_MENTIONS_PER_MESSAGE:
                logger.warning(
                    "[octo] truncating @mentions in convert_content_for_llm at %d",
                    MAX_MENTIONS_PER_MESSAGE,
                )
                break
            name = match.group(1)
            uid: str | None = None
            matched_name = name

            if has_member_map and member_map:
                # Try longest prefix match (supports names with spaces)
                longer = _try_longest_member_match(content, match.start(), member_map, sorted_names)
                if longer:
                    uid = longer["uid"]
                    matched_name = longer["name"]
                else:
                    uid = member_map.get(name)
            elif has_uids and mention.uids and uid_index < len(mention.uids):
                candidate = mention.uids[uid_index]
                uid = candidate if isinstance(candidate, str) else None
                uid_index += 1

            if uid:
                replacements.append((
                    match.start(),
                    match.start() + 1 + len(matched_name),
                    f"@[{uid}:{matched_name}]",
                ))

        # Apply replacements from back to front
        for start, end, replacement in reversed(replacements):
            result = result[:start] + replacement + result[end:]

        return result

    return content


# ─── Build Entities from Plain @name ─────────────────────────────────────────


def build_entities_from_fallback(
    content: str,
    member_map: dict[str, str],
) -> tuple[list[MentionEntity], list[str]]:
    """
    Build mention entities from plain @name text using member_map (displayName → uid).

    This is the fallback path when structured @[uid:name] is not available.
    Uses longest-match-first to handle names with special characters.

    Returns:
        (entities, uids) — lists of MentionEntity and corresponding UIDs.
    """
    entities: list[MentionEntity] = []
    uids: list[str] = []

    sorted_names = sorted(member_map.keys(), key=len, reverse=True)

    for i, match in enumerate(MENTION_PATTERN.finditer(content)):
        if i >= MAX_MENTIONS_PER_MESSAGE:
            logger.warning(
                "[octo] truncating @mentions in build_entities_from_fallback at %d",
                MAX_MENTIONS_PER_MESSAGE,
            )
            break
        name = match.group(1)

        # Skip @all / @All
        if name.lower() == "all" or name == "所有人":
            continue

        uid: str | None = None
        matched_name = name

        # Try longest prefix match first
        longer = _try_longest_member_match(content, match.start(), member_map, sorted_names)
        if longer:
            uid = longer["uid"]
            matched_name = longer["name"]
        else:
            uid = member_map.get(name)

        if not uid:
            continue

        at_name = f"@{matched_name}"
        entities.append(MentionEntity(uid=uid, offset=match.start(), length=len(at_name)))
        uids.append(uid)

    return entities, uids


# ─── Internal Helpers ────────────────────────────────────────────────────────

# Name character class — mirrors MENTION_PATTERN's inner char set
_NAME_CHAR_RE = re.compile(r"[\w\u00C0-\u024F\u4e00-\u9fff\u3040-\u30FF\uAC00-\uD7AF.\-]")


def _try_longest_member_match(
    text: str,
    at_pos: int,
    member_map: dict[str, str],
    sorted_names: list[str],
) -> dict[str, str] | None:
    """
    From @at_pos, try to match the longest name in member_map.
    sorted_names must be sorted by length descending.

    Boundary check: character after matched name must be a terminator
    (non-name character), preventing partial matches.
    """
    after = text[at_pos + 1 :]  # text after @
    for candidate in sorted_names:
        if after.startswith(candidate):
            # Check boundary
            next_char_pos = at_pos + 1 + len(candidate)
            if next_char_pos >= len(text) or not _NAME_CHAR_RE.match(text[next_char_pos]):
                uid = member_map.get(candidate)
                if uid:
                    return {"name": candidate, "uid": uid}
    return None
