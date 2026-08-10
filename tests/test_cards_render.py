"""Pure renderer contracts for controlled outbound Octo cards."""

from __future__ import annotations
import pytest

from hermes_octo_plugin import cards, types
from hermes_octo_plugin.types import CardProfileManifest


def test_card_profile_cache_expires_and_can_be_cleared() -> None:
    now = [100.0]
    cache = cards.CardProfileCache(ttl_seconds=60, clock=lambda: now[0])
    manifest = CardProfileManifest(available=True, enabled=True)

    assert cache.get() is None
    cache.put(manifest)
    assert cache.get() is manifest
    now[0] = 161.0
    assert cache.get() is None

    cache.put(manifest)
    cache.clear()
    assert cache.get() is None


def test_display_card_uses_fixed_adaptive_card_envelope() -> None:
    rendered = cards.build_display_card(
        blocks=[{"type": "text", "text": "Status"}]
    )

    assert rendered.card == {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [{"type": "TextBlock", "text": "Status", "wrap": True}],
    }
    assert rendered.plain == "Status"


def test_display_card_renders_controlled_text_with_plain_fallback() -> None:
    rendered = cards.build_display_card(
        title="Approval request",
        blocks=[
            {"type": "heading", "text": "Details"},
            {"type": "text", "text": "Please review"},
        ],
    )

    assert rendered.card["body"] == [
        {
            "type": "TextBlock",
            "text": "Approval request",
            "wrap": True,
            "weight": "Bolder",
        },
        {
            "type": "TextBlock",
            "text": "Details",
            "wrap": True,
            "weight": "Bolder",
        },
        {"type": "TextBlock", "text": "Please review", "wrap": True},
    ]
    assert rendered.plain == "Approval request\nDetails\nPlease review"


def test_manifest_gate_distinguishes_missing_endpoint_from_disabled_server() -> None:
    unavailable = CardProfileManifest(available=False, enabled=False)
    disabled = CardProfileManifest(available=True, enabled=False)

    assert cards.card_delivery_enabled(unavailable, configured_enabled=True) is True
    assert cards.card_delivery_enabled(unavailable, configured_enabled=False) is False
    assert cards.card_delivery_enabled(disabled, configured_enabled=True) is False


def test_manifest_caps_fail_closed_when_deployed_lists_are_absent() -> None:
    absent = cards.derive_card_capabilities(
        CardProfileManifest(available=True, enabled=True)
    )
    legacy = cards.derive_card_capabilities(
        CardProfileManifest(available=False, enabled=False)
    )
    explicit = cards.derive_card_capabilities(
        CardProfileManifest(
            available=True,
            enabled=True,
            profiles=("octo/v1", "octo/v2"),
            elements=(),
            inputs=(),
            actions=("Action.OpenUrl",),
            limits={
                "max_nodes": 200.9,
                "max_depth": 0,
                "max_payload_bytes": float("inf"),
                "max_input_text_bytes": 4096.8,
                "max_inputs_bytes": 16384,
            },
        )
    )

    assert absent.elements == frozenset()
    assert absent.inputs == frozenset()
    assert absent.actions == frozenset()
    assert legacy.elements is None
    assert legacy.inputs is None
    assert legacy.actions is None
    assert explicit.elements == frozenset()
    assert explicit.inputs == frozenset()
    assert explicit.actions == frozenset({"Action.OpenUrl", "Action.Submit"})
    assert explicit.max_nodes == 200
    assert explicit.max_depth is None
    assert explicit.max_payload_bytes is None
    assert explicit.max_input_text_bytes == 4096
    assert explicit.max_inputs_bytes == 16384


def test_deployed_manifest_requires_exact_profile_and_card_version() -> None:
    missing = cards.derive_card_capabilities(
        CardProfileManifest(available=True, enabled=True)
    )
    wrong_version = cards.derive_card_capabilities(
        CardProfileManifest(
            available=True,
            enabled=True,
            profiles=("octo/v1",),
            card_version="1.4",
            elements=("TextBlock",),
        )
    )

    for capabilities in (missing, wrong_version):
        with pytest.raises(ValueError, match="octo/v1"):
            cards.build_display_card(
                blocks=[{"type": "text", "text": "Status"}],
                capabilities=capabilities,
            )


def test_display_card_fails_closed_without_textblock_capability() -> None:
    with pytest.raises(ValueError, match="TextBlock"):
        cards.build_display_card(
            blocks=[{"type": "text", "text": "Status"}],
            capabilities=cards.CardCapabilities(
                available=True,
                enabled=True,
                elements=frozenset({"Image"}),
            ),
        )


def test_display_card_rejects_empty_visible_fallback() -> None:
    with pytest.raises(ValueError, match="visible text"):
        cards.build_display_card(
            blocks=[{"type": "text", "text": "token=AKIA1234567890ABCDEF"}]
        )



def test_display_card_renders_controlled_rich_blocks_and_same_source_plain() -> None:
    rendered = cards.build_display_card(
        title="Release",
        blocks=[
            {"type": "section", "title": "Summary", "text": "Ready"},
            {
                "type": "facts",
                "items": [
                    {"label": "Owner", "value": "Platform"},
                    {"label": "State", "value": "Approved"},
                ],
            },
            {
                "type": "image",
                "url": "https://cdn.example.com/images/release.png",
                "alt": "Release diagram",
            },
            {
                "type": "actions",
                "items": [
                    {
                        "label": "Open runbook",
                        "url": "https://docs.example.com/private?token=redacted",
                    }
                ],
            },
        ],
        capabilities=cards.CardCapabilities(
            available=True,
            enabled=True,
            elements=frozenset(
                {"TextBlock", "Container", "FactSet", "Image", "ActionSet"}
            ),
            actions=frozenset({"Action.OpenUrl"}),
        ),
    )

    assert [element["type"] for element in rendered.card["body"]] == [
        "TextBlock",
        "Container",
        "FactSet",
        "Image",
        "ActionSet",
    ]
    assert rendered.card["body"][2]["facts"] == [
        {"title": "Owner", "value": "Platform"},
        {"title": "State", "value": "Approved"},
    ]
    assert rendered.card["body"][3]["url"] == (
        "https://cdn.example.com/images/release.png"
    )
    assert rendered.card["body"][4]["actions"][0]["url"] == (
        "https://docs.example.com/private?token=redacted"
    )
    assert rendered.plain == (
        "Release\nSummary\nReady\nOwner: Platform\nState: Approved\n"
        "Release diagram: https://example.com\n"
        "Open runbook: https://docs.example.com/private?token=redacted"
    )


def test_display_rich_blocks_degrade_only_to_advertised_textblock() -> None:
    rendered = cards.build_display_card(
        blocks=[
            {
                "type": "facts",
                "items": [{"label": "State", "value": "Approved"}],
            },
            {
                "type": "actions",
                "items": [{"label": "Docs", "url": "https://docs.example.com/x"}],
            },
        ],
        capabilities=cards.CardCapabilities(
            available=True,
            enabled=True,
            elements=frozenset({"TextBlock"}),
            actions=frozenset(),
        ),
    )

    assert {element["type"] for element in rendered.card["body"]} == {"TextBlock"}
    assert rendered.plain == "State: Approved\nDocs: https://docs.example.com/x"

def test_interactive_card_binds_controlled_inputs_and_submit_data() -> None:
    rendered = cards.build_interactive_card(
        title="Approval",
        text="Review this request",
        inputs=[
            {
                "id": "comment",
                "kind": "text",
                "label": "Comment",
                "placeholder": "Optional note",
            }
        ],
        buttons=[
            {
                "id": "approve",
                "label": "Approve",
                "data": {
                    "decision": "approve",
                    "token": "must-not-leak",
                    "_octo_binding": "forged",
                },
                "style": "positive",
            }
        ],
        binding_id="binding-123",
        capabilities=cards.CardCapabilities(
            available=True,
            enabled=True,
            elements=frozenset({"TextBlock"}),
            inputs=frozenset({"Input.Text"}),
            actions=frozenset({"Action.Submit"}),
            max_input_text_bytes=8,
            max_inputs_bytes=1024,
        ),
    )

    assert rendered.card["body"][-1] == {
        "type": "Input.Text",
        "id": "comment",
        "label": "Comment",
        "placeholder": "Optional note",
        "maxLength": 2,
    }
    assert rendered.card["actions"] == [
        {
            "type": "Action.Submit",
            "id": "approve",
            "title": "Approve",
            "data": {
                "decision": "approve",
                "token": "[redacted]",
                "_octo_binding": "binding-123",
            },
            "style": "positive",
        }
    ]
    assert rendered.action_labels == {"approve": "Approve"}
    assert rendered.input_ids == ("comment",)
    assert rendered.binding_id == "binding-123"
    assert types.resolve_card_profile(rendered.card) == "octo/v2"
    assert "must-not-leak" not in str(rendered.card)
    assert rendered.plain == (
        "Approval\nReview this request\n[Comment]\nActions: Approve"
    )


def test_interactive_card_fails_closed_on_unadvertised_capabilities() -> None:
    with pytest.raises(ValueError, match="Action.Submit"):
        cards.build_interactive_card(
            title="Approval",
            buttons=[{"id": "approve", "label": "Approve"}],
            binding_id="binding-123",
            capabilities=cards.CardCapabilities(
                available=True,
                enabled=True,
                elements=frozenset({"TextBlock"}),
                inputs=frozenset({"Input.Text"}),
                actions=frozenset(),
            ),
        )


def test_max_inputs_bytes_limits_submitted_values_not_card_definitions() -> None:
    rendered = cards.build_interactive_card(
        title="Approval",
        inputs=[{"id": "comment", "kind": "text"}],
        buttons=[{"id": "approve", "label": "Approve"}],
        binding_id="binding-123",
        capabilities=cards.CardCapabilities(
            available=True,
            enabled=True,
            elements=frozenset({"TextBlock"}),
            inputs=frozenset({"Input.Text"}),
            actions=frozenset({"Action.Submit"}),
            max_inputs_bytes=1,
        ),
    )

    assert rendered.input_ids == ("comment",)


def test_choice_plain_includes_options_and_semantic_values_do_not_truncate() -> None:
    rendered = cards.build_interactive_card(
        title="Choose",
        inputs=[
            {
                "id": "decision",
                "kind": "choice",
                "label": "Decision",
                "choices": [
                    {"title": "Approve", "value": "approve"},
                    {"title": "Reject", "value": "reject"},
                ],
            }
        ],
        buttons=[{"id": "submit", "label": "Submit"}],
        binding_id="binding-123",
    )

    assert "[Decision: Approve / Reject]" in rendered.plain
    with pytest.raises(ValueError, match="choice value"):
        cards.build_interactive_card(
            title="Choose",
            inputs=[
                {
                    "id": "decision",
                    "kind": "choice",
                    "choices": [{"title": "Long", "value": "x" * 65}],
                }
            ],
            buttons=[{"id": "submit", "label": "Submit"}],
            binding_id="binding-123",
        )


def test_progress_uses_specialized_layout_only_when_manifest_supports_it() -> None:
    enhanced = cards.build_progress_card(
        phase="running",
        tools=[{"tool_name": "read", "status": "complete"}],
        capabilities=cards.CardCapabilities(
            available=True,
            enabled=True,
            elements=frozenset({"TextBlock", "ColumnSet", "Column", "Container"}),
        ),
    )
    flat = cards.build_progress_card(
        phase="running",
        tools=[{"tool_name": "read", "status": "complete"}],
        capabilities=cards.CardCapabilities(
            available=True,
            enabled=True,
            elements=frozenset({"TextBlock"}),
        ),
    )

    assert enhanced.card["metadata"] == {"octo_layout": "agent_progress_v1"}
    assert [element["type"] for element in enhanced.card["body"]] == [
        "ColumnSet",
        "Container",
    ]
    assert "metadata" not in flat.card
    assert enhanced.plain == flat.plain
