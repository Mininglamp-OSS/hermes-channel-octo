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


def test_visible_text_renders_markdown_links_as_literal_prose() -> None:
    text = "click [here](javascript:alert(1))"

    rendered = cards.build_display_card(blocks=[{"type": "text", "text": text}])

    assert rendered.card["body"][0]["text"] == r"click \[here](javascript:alert(1))"
    assert rendered.plain == text


def test_visible_text_renders_markdown_images_as_literal_prose() -> None:
    text = "preview ![report](https://cdn.example/report.png)"

    rendered = cards.build_display_card(blocks=[{"type": "text", "text": text}])

    assert rendered.card["body"][0]["text"] == (
        r"preview !\[report](https://cdn.example/report.png)"
    )
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



@pytest.mark.parametrize(
    "url",
    [
        "https://10.0.0.8/report.png",
        "https://metadata.google.internal/computeMetadata/v1/",
        "https://user:password@cdn.example/report.png",
        "file:///var/tmp/report.png",
    ],
)
def test_automatically_fetched_card_images_reject_unsafe_urls(url: str) -> None:
    with pytest.raises(ValueError, match="safe http"):
        cards.build_display_card(
            blocks=[{"type": "image", "url": url, "alt": "Report"}]
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://[fe80::1]/status",
    ],
)
def test_open_url_rejects_metadata_and_unconditionally_unsafe_addresses(
    url: str,
) -> None:
    with pytest.raises(ValueError, match="safe http"):
        cards.sanitize_action_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://10.24.8.12:8080/reports/today",
        "https://files.internal.example/download/42",
        "http://localhost:3000/status",
    ],
)
def test_open_url_preserves_legitimate_intranet_and_self_hosted_destinations(
    url: str,
) -> None:
    assert cards.sanitize_action_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/callback?token=secret",
        "https://example.com/object?X-Amz-Signature=secret",
        "https://example.com/object?X-Amz-Security-Token=secret",
        "https://example.com/callback?access_token=secret",
        "https://example.com/callback?api%5Fkey=secret",
    ],
)
def test_action_url_rejects_sensitive_query_credentials(url: str) -> None:
    with pytest.raises(ValueError, match="sensitive query"):
        cards.sanitize_action_url(url)


def test_action_url_preserves_non_sensitive_query_parameters() -> None:
    url = "https://example.com/search?q=octo&page=2"
    assert cards.sanitize_action_url(url) == url


def test_automatic_tool_url_summary_exposes_only_the_origin() -> None:
    summary = cards.summarize_tool_params(
        "fetch",
        {
            "url": (
                "https://cdn.example.com/private/report.pdf"
                "?X-Amz-Signature=signed-secret#download"
            )
        },
    )

    assert summary == "https://cdn.example.com"


def test_automatic_error_summary_redacts_only_explicit_credentials() -> None:
    summary = cards.sanitize_error_text(
        "401 from https://api.example/v1/items?token=ghp_url_secret&page=2 "
        "Authorization: Bearer sk-live-secret GitHub ghp_direct_secret"
    )

    assert summary.startswith("401 from https://api.example/v1/items?")
    assert "page=2" in summary
    assert summary.count("[redacted]") >= 2
    assert "ghp_url_secret" not in summary
    assert "sk-live-secret" not in summary
    assert "ghp_direct_secret" not in summary

@pytest.mark.parametrize(
    ("error", "credential"),
    [
        (
            "GET https://api.example/items?passwd=pw-value&page=2",
            "pw-value",
        ),
        (
            "GET https://api.example/items?x-goog-credential=cloud-value&page=2",
            "cloud-value",
        ),
        (
            "GET https://api.example/items?client_secret=client-value&page=2",
            "client-value",
        ),
        (
            "GET https://api.example/items?x-amz-credential=amz-value&page=2",
            "amz-value",
        ),
        (
            "GET https://api.example/items?key=key-value&page=2",
            "key-value",
        ),
        (
            "Authorization: Basic Zm9vOmJhcg== request failed",
            "Zm9vOmJhcg==",
        ),
        (
            'Authorization: Digest username="bot", response="digest-value"',
            "digest-value",
        ),
        (
            "{'Authorization': 'Basic cXVvdGVkLWJhc2lj'}",
            "cXVvdGVkLWJhc2lj",
        ),
        (
            '{"Authorization": "Digest username=\\"bot\\", response=\\"quoted-digest\\""}',
            "quoted-digest",
        ),
        (
            "psql: password=SuperSecret123 host=db",
            "SuperSecret123",
        ),
        (
            "request failed api_key=sk-proj-abcdefghijklmnopqrstuvwxyz",
            "sk-proj-abcdefghijklmnopqrstuvwxyz",
        ),
        (
            "X-Api-Key: sk-proj-abcdefghijklmnopqrstuvwxyz",
            "sk-proj-abcdefghijklmnopqrstuvwxyz",
        ),
        (
            "token: sk-live-9f8e7d6c5b4a39281706",
            "sk-live-9f8e7d6c5b4a39281706",
        ),
        (
            "Cookie: session=abcdef123456; csrftoken=zzz",
            "abcdef123456",
        ),
        (
            '{"password": "quoted-password", "host": "db"}',
            "quoted-password",
        ),
        (
            "{'api_key': 'quoted-api-key', 'mode': 'debug'}",
            "quoted-api-key",
        ),
        (
            '{"Cookie": "session=quoted-cookie; csrftoken=csrf"}',
            "quoted-cookie",
        ),
        (
            "Authorization: Token custom-authorization-secret",
            "custom-authorization-secret",
        ),
        (
            "Authorization: OAuth oauth_consumer_key=consumer, "
            "oauth_signature=oauth-secret",
            "oauth-secret",
        ),
        (
            "authorization=basic dXNlcjpwYXNz",
            "dXNlcjpwYXNz",
        ),
        (
            'authorization=digest username="alice", response="deadbeef"',
            "deadbeef",
        ),
    ],
)
def test_automatic_error_summary_redacts_all_recognized_credentials(
    error: str,
    credential: str,
) -> None:
    summary = cards.sanitize_error_text(error)

    assert credential not in summary
    assert "[redacted]" in summary


@pytest.mark.parametrize(
    ("error", "credential"),
    [
        ("SecretAccessKey: unquoted-access-key", "unquoted-access-key"),
        ("'SecretAccessKey': 'quoted-access-key'", "quoted-access-key"),
        ("SecretAccessKey=assigned-access-key", "assigned-access-key"),
        ("SessionToken: bare-session-token", "bare-session-token"),
        ('"SessionToken": "quoted-session-token"', "quoted-session-token"),
        ("SessionToken=assigned-session-token", "assigned-session-token"),
        ("Credentials: unquoted-credentials", "unquoted-credentials"),
        ("'Credentials': 'quoted-credentials'", "quoted-credentials"),
        ("Credentials=assigned-credentials", "assigned-credentials"),
    ],
)
def test_automatic_error_summary_redacts_aws_credential_field_variants(
    error: str,
    credential: str,
) -> None:
    summary = cards.sanitize_error_text(error)

    assert credential not in summary
    assert "[redacted]" in summary


def test_automatic_error_summary_preserves_credential_word_in_prose() -> None:
    error = "Credentials were unavailable after the remote service timed out"

    assert cards.sanitize_error_text(error) == error

def test_automatic_error_summary_preserves_noncredential_assignments() -> None:
    error = "worker failed: mode=debug retry_count=2"

    assert cards.sanitize_error_text(error) == error


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
    assert (
        cards.sanitize_error_text("Authorization: Bearer hidden")
        == "Authorization: Bearer [redacted]"
    )
