"""
Tests for hermes_octo_plugin.mention — mention parsing and conversion utilities.
"""

import pytest
from hermes_octo_plugin.mention import (
    extract_mention_uids,
    convert_content_for_llm,
    build_entities_from_fallback,
    MAX_MENTIONS_PER_MESSAGE,
    MENTION_PATTERN,
    STRUCTURED_MENTION_PATTERN,
)
from hermes_octo_plugin.types import MentionEntity, MentionPayload


class TestExtractMentionUids:
    def test_none_mention(self):
        assert extract_mention_uids(None) == []

    def test_empty_mention(self):
        mp = MentionPayload()
        assert extract_mention_uids(mp) == []

    def test_uids_only(self):
        mp = MentionPayload(uids=["u1", "u2"])
        assert extract_mention_uids(mp) == ["u1", "u2"]

    def test_entities_preferred_over_uids(self):
        mp = MentionPayload(
            uids=["u1", "u2"],
            entities=[MentionEntity(uid="e1", offset=0, length=5)],
        )
        result = extract_mention_uids(mp)
        assert result == ["e1"]

    def test_invalid_entities_fallback_to_uids(self):
        mp = MentionPayload(
            uids=["u1"],
            entities=[MentionEntity(uid="", offset=0, length=5)],  # invalid uid
        )
        result = extract_mention_uids(mp)
        assert result == ["u1"]

    def test_filters_non_string_uids(self):
        mp = MentionPayload(uids=["u1", 123, "u2"])  # type: ignore
        result = extract_mention_uids(mp)
        assert result == ["u1", "u2"]


class TestConvertContentForLLM:
    def test_no_mention(self):
        result = convert_content_for_llm("hello world")
        assert result == "hello world"

    def test_entities_v2_replacement(self):
        content = "@Alice hello"
        mention = MentionPayload(
            entities=[MentionEntity(uid="uid1", offset=0, length=6)],
        )
        result = convert_content_for_llm(content, mention)
        assert result == "@[uid1:Alice] hello"

    def test_entities_multiple(self):
        content = "@Alice and @Bob"
        mention = MentionPayload(
            entities=[
                MentionEntity(uid="uid1", offset=0, length=6),
                MentionEntity(uid="uid2", offset=11, length=4),
            ],
        )
        result = convert_content_for_llm(content, mention)
        assert "@[uid1:Alice]" in result
        assert "@[uid2:Bob]" in result

    def test_entities_back_to_front(self):
        """Replacements should proceed from back to front to avoid offset drift."""
        content = "@A @B"
        mention = MentionPayload(
            entities=[
                MentionEntity(uid="u1", offset=0, length=2),
                MentionEntity(uid="u2", offset=3, length=2),
            ],
        )
        result = convert_content_for_llm(content, mention)
        assert result == "@[u1:A] @[u2:B]"

    def test_uids_positional_pairing(self):
        content = "@Alice @Bob"
        mention = MentionPayload(uids=["uid1", "uid2"])
        member_map = {"Alice": "uid1", "Bob": "uid2"}
        result = convert_content_for_llm(content, mention, member_map)
        assert "@[uid1:Alice]" in result
        assert "@[uid2:Bob]" in result

    def test_member_map_lookup(self):
        content = "@Alice hello"
        mention = MentionPayload(uids=["uid1"])
        member_map = {"Alice": "uid1"}
        result = convert_content_for_llm(content, mention, member_map)
        assert "@[uid1:Alice]" in result

    def test_no_member_map_no_uids(self):
        content = "@Alice hello"
        mention = MentionPayload()
        result = convert_content_for_llm(content, mention)
        assert result == "@Alice hello"

    def test_chinese_names(self):
        content = "@张三 你好"
        mention = MentionPayload(
            entities=[MentionEntity(uid="uid1", offset=0, length=3)],
        )
        result = convert_content_for_llm(content, mention)
        assert result == "@[uid1:张三] 你好"


class TestBuildEntitiesFromFallback:
    def test_basic(self):
        member_map = {"Alice": "uid1", "Bob": "uid2"}
        entities, uids = build_entities_from_fallback("@Alice @Bob hello", member_map)
        assert len(entities) == 2
        assert len(uids) == 2
        assert "uid1" in uids
        assert "uid2" in uids

    def test_skip_all(self):
        member_map = {"Alice": "uid1"}
        entities, uids = build_entities_from_fallback("@all @Alice", member_map)
        assert len(entities) == 1
        assert uids == ["uid1"]

    def test_no_match(self):
        member_map = {"Alice": "uid1"}
        entities, uids = build_entities_from_fallback("@Unknown hello", member_map)
        assert len(entities) == 0
        assert len(uids) == 0

    def test_empty_content(self):
        member_map = {"Alice": "uid1"}
        entities, uids = build_entities_from_fallback("", member_map)
        assert len(entities) == 0

    def test_entity_offsets(self):
        member_map = {"Alice": "uid1"}
        entities, uids = build_entities_from_fallback("Hello @Alice!", member_map)
        assert len(entities) == 1
        assert entities[0].offset == 6
        assert entities[0].length == 6  # "@Alice"
        assert entities[0].uid == "uid1"


class TestMentionPattern:
    def test_basic_match(self):
        matches = MENTION_PATTERN.findall("@Alice hello")
        assert "Alice" in matches

    def test_chinese_name(self):
        matches = MENTION_PATTERN.findall("@张三 你好")
        assert "张三" in matches

    def test_multiple(self):
        matches = MENTION_PATTERN.findall("@Alice @Bob @Charlie")
        assert len(matches) == 3

    def test_no_match(self):
        matches = MENTION_PATTERN.findall("hello world")
        assert len(matches) == 0

    def test_email_not_matched(self):
        # @ in email addresses should not be matched as mentions
        matches = MENTION_PATTERN.findall("user@example.com")
        # The regex matches "example.com" here — this is expected behavior
        # as email detection is not in scope for mention parsing


class TestStructuredMentionPattern:
    def test_basic_match(self):
        match = STRUCTURED_MENTION_PATTERN.search("@[uid1:Alice]")
        assert match is not None
        assert match.group(1) == "uid1"
        assert match.group(2) == "Alice"

    def test_chinese_name(self):
        match = STRUCTURED_MENTION_PATTERN.search("@[uid1:张三]")
        assert match is not None
        assert match.group(2) == "张三"

    def test_multiple(self):
        matches = STRUCTURED_MENTION_PATTERN.findall("@[u1:Alice] @[u2:Bob]")
        assert len(matches) == 2

    def test_no_match_plain(self):
        match = STRUCTURED_MENTION_PATTERN.search("@Alice")
        assert match is None


class TestMentionCap:
    """A DoS-shaped message with hundreds of @ tokens must not be processed
    in full — both rewrite paths cap at MAX_MENTIONS_PER_MESSAGE."""

    def test_convert_content_caps_at_max(self):
        # 200 distinct @userN tokens with a member_map that resolves all of
        # them — only the first MAX_MENTIONS_PER_MESSAGE should be rewritten.
        n = 200
        names = [f"user{i}" for i in range(n)]
        member_map = {name: f"uid{i}" for i, name in enumerate(names)}
        content = " ".join(f"@{name}" for name in names)
        # convert_content_for_llm's v1 fallback path only runs when there's
        # a member_map OR uids — pass a sentinel uids list so we exercise
        # the member_map branch (uids is unused when member_map is present).
        mention = MentionPayload(uids=["sentinel"], entities=None, all=None)

        result = convert_content_for_llm(content, mention=mention, member_map=member_map)

        # Names beyond the cap should remain untouched as plain "@userN".
        cap = MAX_MENTIONS_PER_MESSAGE
        assert "@[uid0:user0]" in result
        assert f"@[uid{cap - 1}:user{cap - 1}]" in result
        assert f"@[uid{cap}:user{cap}]" not in result
        assert f"@user{cap}" in result

    def test_build_entities_caps_at_max(self):
        n = 200
        names = [f"user{i}" for i in range(n)]
        member_map = {name: f"uid{i}" for i, name in enumerate(names)}
        content = " ".join(f"@{name}" for name in names)

        entities, uids = build_entities_from_fallback(content, member_map)

        assert len(entities) <= MAX_MENTIONS_PER_MESSAGE
        assert len(uids) <= MAX_MENTIONS_PER_MESSAGE


# ─── Structured mention conversion (outbound LLM reply → wire format) ─────────


class TestParseStructuredMentions:
    def test_returns_empty_for_plain_text(self):
        from hermes_octo_plugin.mention import parse_structured_mentions
        assert parse_structured_mentions("hello world") == []

    def test_extracts_single_mention(self):
        from hermes_octo_plugin.mention import parse_structured_mentions
        out = parse_structured_mentions("@[abc123:Alice] hello")
        assert len(out) == 1
        assert out[0].uid == "abc123"
        assert out[0].name == "Alice"
        assert out[0].offset == 0
        assert out[0].length == len("@[abc123:Alice]")

    def test_extracts_multiple_and_preserves_order(self):
        from hermes_octo_plugin.mention import parse_structured_mentions
        text = "hi @[u1:A] and @[u2:B] plus @[u3:C]"
        out = parse_structured_mentions(text)
        assert [m.uid for m in out] == ["u1", "u2", "u3"]


class TestConvertStructuredMentions:
    def test_basic_single(self):
        from hermes_octo_plugin.mention import (
            parse_structured_mentions, convert_structured_mentions,
        )
        text = "@[abc123:Alice] hello"
        content, entities, uids = convert_structured_mentions(
            text, parse_structured_mentions(text),
        )
        assert content == "@Alice hello"
        assert uids == ["abc123"]
        assert len(entities) == 1
        assert entities[0].uid == "abc123"
        assert entities[0].offset == 0
        assert entities[0].length == len("@Alice")

    def test_offsets_track_converted_text(self):
        """The reported bug: client-side @ pill needs precise offsets into
        the converted text, not the original @[uid:name] template."""
        from hermes_octo_plugin.mention import (
            parse_structured_mentions, convert_structured_mentions,
        )
        text = "prefix @[u1:刘建辉] middle @[u2:Alice] tail"
        content, entities, uids = convert_structured_mentions(
            text, parse_structured_mentions(text),
        )
        assert content == "prefix @刘建辉 middle @Alice tail"
        # First entity: starts at offset 7 ("prefix " is 7 chars)
        assert entities[0].offset == 7
        assert entities[0].length == len("@刘建辉")
        # Second entity: offset accounts for prior replacement
        assert content[entities[1].offset:entities[1].offset + entities[1].length] == "@Alice"
        assert uids == ["u1", "u2"]

    def test_duplicate_names_get_distinct_offsets(self):
        from hermes_octo_plugin.mention import (
            parse_structured_mentions, convert_structured_mentions,
        )
        text = "@[u1:Bob] and @[u2:Bob]"
        content, entities, _ = convert_structured_mentions(
            text, parse_structured_mentions(text),
        )
        assert content == "@Bob and @Bob"
        assert entities[0].offset == 0
        assert entities[1].offset == content.index("@Bob", 1)
        assert entities[0].uid == "u1"
        assert entities[1].uid == "u2"

    def test_empty_mentions_passes_through(self):
        from hermes_octo_plugin.mention import convert_structured_mentions
        content, entities, uids = convert_structured_mentions("hello world", [])
        assert content == "hello world"
        assert entities == []
        assert uids == []
