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

# Matches the UID component accepted by the structured mention envelope.
STRUCTURED_MENTION_UID_PATTERN = re.compile(r"[\w.\-]+")
STRUCTURED_MENTION_PATTERN = re.compile(
    rf"@\[({STRUCTURED_MENTION_UID_PATTERN.pattern}):([^\]\n]+)\]"
)


def strip_leading_self_mention_for_command(
    text: str,
    *,
    bot_uid: str,
    bot_name: str = "",
) -> str:
    """Expose a slash command that follows a leading mention of this bot.

    Octo groups require an explicit bot mention, while Hermes only recognizes
    commands whose first non-whitespace character is ``/``.  Inbound mention
    conversion turns ``@小爱 /new`` into ``@[bot_uid:小爱] /new``; strip that
    one routing mention only when the remaining text starts with ``/``.

    ``bot_name`` covers legacy clients that omit the structured mention
    sidecar and therefore leave the routing mention as plain ``@name`` text.
    """
    if not isinstance(text, str) or not bot_uid:
        return text

    match = re.match(
        rf"^\s*@\[{re.escape(bot_uid)}:[^\]\n]+\]\s*(/.*)$",
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1)

    if bot_name:
        match = re.match(
            rf"^\s*@{re.escape(bot_name)}\s+(/.*)$",
            text,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1)

    return text


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



def _utf16_length(text: str) -> int:
    """Return JS/NSString/Kotlin string length without allocating encoded bytes."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def neutralize_structured_mention_envelopes(text: str) -> str:
    """Keep user-authored ``@[...`` text visible but outside Hermes syntax."""
    return text.replace("@[", "＠[")



def _utf16_boundaries(text: str, targets: set[int]) -> dict[int, int]:
    """Map only requested UTF-16 boundaries, keeping memory O(entity count)."""
    boundaries = {0: 0} if 0 in targets else {}
    if not targets:
        return boundaries
    max_code_units = max(targets)
    code_units = 0
    for index, char in enumerate(text, 1):
        code_units += 2 if ord(char) > 0xFFFF else 1
        if code_units in targets:
            boundaries[code_units] = index
        if code_units >= max_code_units:
            break
    return boundaries


def _render_trusted_replacements(
    content: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    """Neutralize untrusted envelopes while inserting trusted replacements."""
    parts: list[str] = []
    cursor = 0
    for start, end, replacement in replacements:
        parts.append(neutralize_structured_mention_envelopes(content[cursor:start]))
        parts.append(replacement)
        cursor = end
    parts.append(neutralize_structured_mention_envelopes(content[cursor:]))
    return "".join(parts)




def convert_structured_mentions(
    text: str,
    mentions: list[StructuredMention],
    valid_uids: set[str] | None,
) -> tuple[str, list[MentionEntity], list[str]]:
    """Replace each ``@[uid:name]`` in *text* with ``@name`` and emit the
    matching wire-format sidecar.

    Algorithm (incremental, single pass):
      Sort mentions by ``offset`` and reconstruct the output string segment
      by segment. Track the rebuilt content in UTF-16 code units so entity
      offsets match JS/NSString/Kotlin indexing, including astral characters.

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
        running_len += _utf16_length(between)

        replacement = f"@{m.name}"
        if valid_uids is None or m.uid in valid_uids:
            entities.append(MentionEntity(
                uid=m.uid,
                offset=running_len,
                length=_utf16_length(replacement),
            ))
            uids.append(m.uid)
        content_parts.append(replacement)
        running_len += _utf16_length(replacement)

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
                for e in raw_entities[:MAX_MENTIONS_PER_MESSAGE]
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

    ``content`` is normally a ``str`` returned by the Octo message API. Some
    history payloads (rich text / multipart messages) surface ``content`` as a
    ``list`` or other non-string type; in that case we cannot meaningfully
    rewrite @mentions, so we return an empty string and log a warning rather
    than letting ``re.finditer`` raise ``TypeError`` and silently drop the
    surrounding frame upstream.
    """
    if not isinstance(content, str):
        logger.warning(
            "[octo] convert_content_for_llm: non-string content (type=%s); skipping mention rewrite",
            type(content).__name__,
        )
        return ""
    mention = _coerce_mention(mention)
    if not mention:
        return neutralize_structured_mention_envelopes(content)

    # Try entities (v2) — wire offsets and lengths are authoritative. Validate
    # the bounded sidecar before scanning any content, then inspect only the
    # prefix through the largest referenced UTF-16 endpoint.
    if mention.entities:
        valid_entities = [
            entity
            for entity in mention.entities
            if (
                isinstance(entity, MentionEntity)
                and entity.uid
                and isinstance(entity.offset, int)
                and not isinstance(entity.offset, bool)
                and isinstance(entity.length, int)
                and not isinstance(entity.length, bool)
                and entity.offset >= 0
                and entity.length > 0
            )
        ]
        boundary_targets = {
            boundary
            for entity in valid_entities
            for boundary in (entity.offset, entity.offset + entity.length)
            if boundary <= 2 * len(content)
        }
        max_end = max(boundary_targets, default=0)
        prefix = content[:max_end]
        boundaries = (
            None
            if all(ord(char) <= 0xFFFF for char in prefix)
            else _utf16_boundaries(content, boundary_targets)
        )
        entity_ranges: list[tuple[MentionEntity, int, int]] = []
        for entity in valid_entities:
            if boundaries is None:
                start = entity.offset if entity.offset <= len(content) else None
                end_offset = entity.offset + entity.length
                end = end_offset if end_offset <= len(content) else None
            else:
                start = boundaries.get(entity.offset)
                end = boundaries.get(entity.offset + entity.length)
            if start is None or end is None or start >= end or content[start] != "@":
                continue
            entity_ranges.append((entity, start, end))

        sorted_entities = sorted(entity_ranges, key=lambda item: item[1])
        overlaps = any(
            start < previous_end
            for (_, _, previous_end), (_, start, _) in zip(
                sorted_entities,
                sorted_entities[1:],
            )
        )
        if sorted_entities and not overlaps:
            replacements = [
                (
                    start,
                    end,
                    f"@[{entity.uid}:{content[start + 1:end]}]",
                )
                for entity, start, end in sorted_entities
            ]
            return _render_trusted_replacements(content, replacements)


    # Fallback (v1): member_map lookup or uids positional pairing
    has_member_map = member_map and len(member_map) > 0
    has_uids = mention.uids and len(mention.uids) > 0

    if has_member_map or has_uids:
        uid_index = 0
        replacements: list[tuple[int, int, str]] = []
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
                longer = _try_longest_member_match(content, match.start(), member_map, sorted_names)
                if longer:
                    matched_name = longer["name"]
                    uid = longer["uid"]
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

        return _render_trusted_replacements(
            content,
            sorted(replacements, key=lambda item: item[0]),
        )

    return neutralize_structured_mention_envelopes(content)



# ─── Build Entities from Plain @name ─────────────────────────────────────────


def build_entities_from_fallback(
    content: str,
    member_map: dict[str, str],
) -> tuple[list[MentionEntity], list[str]]:
    """
    Build mention entities from plain @name text using member_map (displayName → uid).

    This is the fallback path when structured @[uid:name] is not available.
    Uses longest-match-first to handle names with special characters.

    Like :func:`convert_content_for_llm`, this defensively returns empty results
    when ``content`` is not a ``str`` — some Octo payloads surface ``content``
    as a ``list`` (rich text / multipart) and ``re.finditer`` would otherwise
    raise ``TypeError``.

    Returns:
        (entities, uids) — lists of MentionEntity and corresponding UIDs.
    """
    if not isinstance(content, str):
        logger.warning(
            "[octo] build_entities_from_fallback: non-string content (type=%s); returning empty",
            type(content).__name__,
        )
        return [], []
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
        entities.append(MentionEntity(
            uid=uid,
            offset=_utf16_length(content[:match.start()]),
            length=_utf16_length(at_name),
        ))
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
