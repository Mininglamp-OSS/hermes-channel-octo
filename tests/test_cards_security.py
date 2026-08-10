"""Security boundaries for outbound Octo card rendering."""

from __future__ import annotations
import json

import pytest

from hermes_octo_plugin import cards


def test_card_text_reduces_embedded_urls_before_reaching_visible_sinks() -> None:
    rendered = cards.build_display_card(
        blocks=[
            {
                "type": "text",
                "text": (
                    "Callback https://hooks.slack.com/services/T00/B00/secret"
                    "?token=also-secret"
                ),
            }
        ]
    )
    assert rendered.card["body"][0]["text"] == "Callback https://slack.com"
    assert rendered.plain == "Callback https://slack.com"



@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ("dsn user:p4ss@db.private.example.com/prod", ("p4ss", "private")),
        ("hook //hooks.private.example.com/services/K9x7", ("K9x7", "private")),
        ("hook hooks.private.example.com/services/K9x7", ("K9x7", "private")),
        ("click [here](javascript:alert(1))", ("javascript:",)),
    ],
)
def test_visible_text_reduces_schemeless_and_markdown_targets(
    text: str,
    forbidden: tuple[str, ...],
) -> None:
    rendered = cards.build_display_card(
        blocks=[{"type": "text", "text": text}]
    )

    serialized = str(rendered.card)
    for value in forbidden:
        assert value not in serialized
        assert value not in rendered.plain


@pytest.mark.parametrize(
    "token",
    [
        "xapp-1-A1234567890-B1234567890-C1234567890",
        "npm_123456789012345678901234567890123456",
        "shpat_" + "12345678901234567890123456789012",
        "dop_v1_1234567890123456789012345678901234567890",
    ],
)
def test_explicit_token_prefixes_are_hidden_even_in_non_generic_sinks(token: str) -> None:
    assert cards.sanitize_visible_text(token, generic=False) is None


def test_action_data_rejects_unsafe_keys_and_internal_namespace() -> None:
    for key in (
        "ghp_123456789012345678901234567890123456",
        "https://user:pass@example.com/private",
        "_octo_session",
    ):
        with pytest.raises(ValueError, match="action data key"):
            cards.build_interactive_card(
                title="Approval",
                buttons=[
                    {
                        "id": "approve",
                        "label": "Approve",
                        "data": {key: "value"},
                    }
                ],
                binding_id="binding-123",
            )


def test_action_url_preserves_the_safe_http_destination() -> None:
    assert (
        cards.sanitize_action_url("https://docs.example.com/private?id=section-2")
        == "https://docs.example.com/private?id=section-2"
    )


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://user:password@example.com/path",
        "//example.com/path",
        "not a url",
    ],
)
def test_action_url_rejects_dangerous_or_ambiguous_targets(url: str) -> None:
    with pytest.raises(ValueError, match="safe http"):
        cards.sanitize_action_url(url)


def test_recursive_limit_helpers_count_rendered_card_structure() -> None:
    card = {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "Container",
                "items": [{"type": "TextBlock", "text": "nested"}],
            }
        ],
    }

    assert cards.count_card_nodes(card) == 2
    assert cards.card_max_depth(card) == 2


def test_display_builder_enforces_negotiated_nodes_and_payload_bytes() -> None:
    with pytest.raises(cards.CardLimitError, match="max_nodes"):
        cards.build_display_card(
            blocks=[
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ],
            capabilities=cards.CardCapabilities(
                available=True,
                enabled=True,
                max_nodes=1,
            ),
        )

    rendered = cards.build_display_card(
        blocks=[{"type": "text", "text": "你好"}]
    )
    payload_bytes = cards.card_payload_bytes(rendered.card, rendered.plain)
    with pytest.raises(cards.CardLimitError, match="max_payload_bytes"):
        cards.build_display_card(
            blocks=[{"type": "text", "text": "你好"}],
            capabilities=cards.CardCapabilities(
                available=True,
                enabled=True,
                max_payload_bytes=payload_bytes - 1,
            ),
        )



def test_renderer_rejects_oversized_text_and_block_collections_before_building() -> None:
    with pytest.raises(cards.CardLimitError, match="text bytes"):
        cards.build_display_card(
            blocks=[{"type": "text", "text": "x" * (70 * 1024)}]
        )
    with pytest.raises(cards.CardLimitError, match="block limit"):
        cards.build_display_card(
            blocks=[{"type": "text", "text": "x"}] * 101
        )

def test_payload_bytes_match_go_json_escaping_and_edit_fields() -> None:
    card = {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [{"type": "TextBlock", "text": "<>&\u2028\u2029"}],
    }
    send_size = cards.card_payload_bytes(card, "<>&\u2028\u2029")
    edit_size = cards.card_payload_bytes(
        card,
        "<>&\u2028\u2029",
        card_seq=7,
        transient=True,
    )

    python_size = len(
        json.dumps(
            {
                "type": 17,
                "profile": "octo/v1",
                "card_version": "1.5",
                "card": card,
                "plain": "<>&\u2028\u2029",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )
    assert send_size == python_size + 42
    assert edit_size > send_size


def test_visible_card_text_drops_secret_shaped_blocks_from_card_and_plain() -> None:
    rendered = cards.build_display_card(
        blocks=[
            {"type": "text", "text": "token=AKIA1234567890ABCDEF"},
            {"type": "text", "text": "safe status"},
        ]
    )

    serialized = str(rendered.card)
    assert "AKIA1234567890ABCDEF" not in serialized
    assert "AKIA1234567890ABCDEF" not in rendered.plain
    assert rendered.plain == "safe status"


def test_tool_summaries_are_allowlisted_bounded_and_secret_safe() -> None:
    assert (
        cards.summarize_tool_params(
            "read",
            {"path": "/Users/example/workspace/project/config.py"},
        )
        == "…/project/config.py"
    )
    assert (
        cards.summarize_tool_params(
            "bash",
            {"command": "API_TOKEN=hidden python script.py --secret value"},
        )
        == "python"
    )
    assert cards.summarize_tool_params("mcp__unknown", {"query": "visible"}) == ""
    assert cards.summarize_tool_params("web_search", {"query": "token=hidden"}) == ""


def test_tool_labels_and_errors_do_not_echo_secrets() -> None:
    assert cards.safe_tool_label("read") == "read"
    assert cards.safe_tool_label("mcp__database_query") == "MCP tool"
    assert cards.safe_tool_label("token=AKIA1234567890ABCDEF") == "tool"
    assert (
        cards.sanitize_error_text(
            "request failed at https://private.example.com/path?id=secret"
        )
        == "request failed at https://example.com"
    )
    assert cards.sanitize_error_text("Authorization: Bearer hidden") == ""
