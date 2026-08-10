"""Security boundaries for outbound Octo card rendering."""

from __future__ import annotations
import json

import pytest

from hermes_octo_plugin import cards


def test_card_text_preserves_embedded_urls_and_content() -> None:
    text = (
        "Callback https://hooks.slack.com/services/T00/B00/secret"
        "?token=also-secret"
    )
    rendered = cards.build_display_card(
        blocks=[{"type": "text", "text": text}]
    )
    assert rendered.card["body"][0]["text"] == text
    assert rendered.plain == text



@pytest.mark.parametrize(
    "text",
    [
        "dsn user:p4ss@db.private.example.com/prod",
        "hook //hooks.private.example.com/services/K9x7",
        "hook hooks.private.example.com/services/K9x7",
        "click [here](javascript:alert(1))",
        "xapp-1-A1234567890-B1234567890-C1234567890",
        "npm_123456789012345678901234567890123456",
        "shpat_" + "12345678901234567890123456789012",
        "dop_v1_1234567890123456789012345678901234567890",
    ],
)
def test_visible_text_preserves_content_without_dlp_guessing(text: str) -> None:
    rendered = cards.build_display_card(
        blocks=[{"type": "text", "text": text}]
    )
    assert rendered.card["body"][0]["text"] == text
    assert rendered.plain == text


def test_action_data_preserves_content_but_rejects_internal_namespace() -> None:
    rendered = cards.build_interactive_card(
        title="Approval",
        buttons=[{
            "id": "approve",
            "label": "Approve",
            "data": {
                "token": "ghp_123456789012345678901234567890123456",
                "callback": "https://example.com/private?secret=value",
            },
        }],
        binding_id="binding-123",
    )
    submit = rendered.card["actions"][0]
    assert submit["data"]["token"].startswith("ghp_")
    assert submit["data"]["callback"].endswith("secret=value")

    with pytest.raises(ValueError, match="action data key"):
        cards.build_interactive_card(
            title="Approval",
            buttons=[{"id": "approve", "label": "Approve", "data": {"_octo_session": "x"}}],
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


def test_visible_card_text_preserves_secret_shaped_blocks() -> None:
    text = "token=AKIA1234567890ABCDEF"
    rendered = cards.build_display_card(
        blocks=[
            {"type": "text", "text": text},
            {"type": "text", "text": "safe status"},
        ]
    )
    assert rendered.card["body"][0]["text"] == text
    assert rendered.plain == f"{text}\nsafe status"


def test_tool_summaries_are_allowlisted_bounded_and_content_preserving() -> None:
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
    assert cards.summarize_tool_params("web_search", {"query": "token=hidden"}) == "token=hidden"


def test_tool_labels_and_errors_preserve_content_with_structural_bounds() -> None:
    assert cards.safe_tool_label("read") == "read"
    assert cards.safe_tool_label("mcp__database_query") == "MCP tool"
    assert cards.safe_tool_label("token") == "token"
    error = "request failed at https://private.example.com/path?id=secret"
    assert cards.sanitize_error_text(error) == error
    assert cards.sanitize_error_text("Authorization: Bearer hidden") == "Authorization: Bearer hidden"
