"""Wire-level contracts for outbound Octo Type-17 cards and bot events."""

from __future__ import annotations
import json

from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from hermes_octo_plugin import api, cards, types
from hermes_octo_plugin.types import ChannelType


class _ProfileResponse:
    def __init__(self, *, status: int, payload=None, body: str = "") -> None:
        self.status = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self._body = body or ("{}" if payload is not None else "")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return self._body

    async def json(self, **_kwargs):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _ProfileSession:
    def __init__(self, response: _ProfileResponse) -> None:
        self.get = MagicMock(return_value=response)


def _card_with_submit() -> dict:
    return {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [{"type": "TextBlock", "text": "Choose"}],
        "actions": [{"type": "Action.Submit", "title": "Confirm"}],
    }


class TestCardTypes:
    def test_constants_and_interaction_detection_follow_the_wire_profiles(self):
        assert getattr(types, "CARD_PROFILE_V1", None) == "octo/v1"
        assert getattr(types, "CARD_PROFILE_V2", None) == "octo/v2"
        assert getattr(types, "CARD_VERSION", None) == "1.5"
        assert hasattr(types, "card_contains_interaction")
        assert types.card_contains_interaction(_card_with_submit()) is True
        assert (
            types.card_contains_interaction({"type": "AdaptiveCard", "body": []})
            is False
        )


class TestCardApi:
    @pytest.mark.asyncio
    async def test_send_card_uses_type17_envelope_auto_upgrades_and_requires_identity(
        self,
    ):
        assert hasattr(api, "send_card_message")
        with patch.object(
            api,
            "post_json",
            AsyncMock(return_value={"message_id": "card-42", "message_seq": 7}),
        ) as post_json:
            result = await api.send_card_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                channel_id="group-1",
                channel_type=ChannelType.Group,
                card=_card_with_submit(),
                plain="Choose",
                client_msg_no="client-card-42",
                on_behalf_of="grantor-1",
            )

        assert result.message_id == "card-42"
        post_json.assert_awaited_once_with(
            ANY,
            "https://api.example.invalid",
            "test-token",
            "/v1/bot/sendMessage",
            {
                "channel_id": "group-1",
                "channel_type": ChannelType.Group,
                "client_msg_no": "client-card-42",
                "on_behalf_of": "grantor-1",
                "payload": {
                    "type": 17,
                    "card": _card_with_submit(),
                    "plain": "Choose",
                    "profile": "octo/v2",
                    "card_version": "1.5",
                },
            },
        )

    @pytest.mark.asyncio
    async def test_send_card_rejects_a_success_response_without_message_id(self):
        assert hasattr(api, "send_card_message")
        with patch.object(api, "post_json", AsyncMock(return_value={})):
            with pytest.raises(api.OctoApiError, match="missing message_id"):
                await api.send_card_message(
                    MagicMock(),
                    "https://api.example.invalid",
                    "test-token",
                    channel_id="group-1",
                    channel_type=ChannelType.Group,
                    card={"type": "AdaptiveCard", "body": []},
                )

    @pytest.mark.asyncio
    async def test_send_card_generates_default_client_message_identity(self):
        with patch.object(
            api,
            "post_json",
            AsyncMock(return_value={"message_id": "card-43"}),
        ) as post_json:
            await api.send_card_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                channel_id="group-1",
                channel_type=ChannelType.Group,
                card={"type": "AdaptiveCard", "body": []},
            )

        client_msg_no = post_json.await_args.args[4]["client_msg_no"]
        assert str(UUID(client_msg_no)) == client_msg_no

    @pytest.mark.asyncio
    async def test_edit_card_serializes_complete_type17_frame_with_seq_and_transience(
        self,
    ):
        assert hasattr(api, "edit_card_message")
        with patch.object(api, "post_json", AsyncMock(return_value=None)) as post_json:
            await api.edit_card_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                channel_id="group-1",
                channel_type=ChannelType.Group,
                message_id="card-42",
                card={"type": "AdaptiveCard", "version": "1.5", "body": []},
                plain="working",
                card_seq=3,
                transient=True,
                on_behalf_of="grantor-1",
            )

        body = post_json.await_args.args[4]
        assert body["message_id"] == "card-42"
        assert body["channel_id"] == "group-1"
        assert body["channel_type"] == ChannelType.Group
        assert body["on_behalf_of"] == "grantor-1"
        assert body["content_edit"] == (
            '{"type": 17, "card": {"type": "AdaptiveCard", "version": "1.5", '
            '"body": []}, "profile": "octo/v1", "card_version": "1.5", '
            '"plain": "working", "card_seq": 3, "transient": true}'
        )

    @pytest.mark.asyncio
    async def test_edit_card_refuses_non_positive_sequence_without_network_io(self):
        assert hasattr(api, "edit_card_message")
        post_json = AsyncMock()
        with patch.object(api, "post_json", post_json):
            with pytest.raises(ValueError, match="positive"):
                await api.edit_card_message(
                    MagicMock(),
                    "https://api.example.invalid",
                    "test-token",
                    channel_id="group-1",
                    channel_type=ChannelType.Group,
                    message_id="card-42",
                    card={"type": "AdaptiveCard", "body": []},
                    card_seq=0,
                )
        post_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_template_card_send_and_edit_use_registry_wire_contract(self):
        template_ref = {"id": "ai.reasoning-process", "version": "0.3.0"}
        data = {
            "reasoningId": "session-1:turn-1",
            "state": "reasoning",
            "title": "Reasoning",
        }
        with patch.object(
            api,
            "post_json",
            AsyncMock(return_value={"message_id": "reasoning-1"}),
        ) as post_json:
            result = await api.send_template_card_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                channel_id="group-1",
                channel_type=ChannelType.Group,
                template_ref=template_ref,
                state="reasoning",
                data=data,
                client_msg_no="template-client-1",
            )
            await api.edit_template_card_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                channel_id="group-1",
                channel_type=ChannelType.Group,
                message_id="reasoning-1",
                template_ref=template_ref,
                state="completed",
                data={**data, "state": "completed"},
                card_seq=2,
                transient=False,
            )

        assert result.message_id == "reasoning-1"
        assert post_json.await_args_list[0].args[3:] == (
            "/v1/bot/sendMessage",
            {
                "channel_id": "group-1",
                "channel_type": ChannelType.Group,
                "client_msg_no": "template-client-1",
                "payload": {
                    "type": 17,
                    "template_ref": template_ref,
                    "state": "reasoning",
                    "data": data,
                },
            },
        )
        assert post_json.await_args_list[1].args[3:] == (
            "/v1/bot/message/edit",
            {
                "message_id": "reasoning-1",
                "channel_id": "group-1",
                "channel_type": ChannelType.Group,
                "template_ref": template_ref,
                "state": "completed",
                "data": {**data, "state": "completed"},
                "card_seq": 2,
                "transient": False,
            },
        )

    @pytest.mark.asyncio
    async def test_template_card_rejects_mismatched_state_before_network_io(self):
        post_json = AsyncMock()
        with patch.object(api, "post_json", post_json):
            with pytest.raises(ValueError, match="data.state"):
                await api.send_template_card_message(
                    MagicMock(),
                    "https://api.example.invalid",
                    "test-token",
                    channel_id="group-1",
                    channel_type=ChannelType.Group,
                    template_ref={
                        "id": "ai.reasoning-process",
                        "version": "0.3.0",
                    },
                    state="reasoning",
                    data={"state": "completed"},
                )
        post_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_and_edit_enforce_complete_type17_payload_bytes_before_io(self):
        card = {
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [{"type": "TextBlock", "text": "safe"}],
        }
        plain = "safe"
        send_size = cards.card_payload_bytes(card, plain)
        post_json = AsyncMock(return_value={"message_id": "card-1"})
        with (
            patch.object(cards, "DEFAULT_MAX_CARD_PAYLOAD_BYTES", send_size),
            patch.object(api, "post_json", post_json),
        ):
            await api.send_card_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                channel_id="group-1",
                channel_type=ChannelType.Group,
                card=card,
                plain=plain,
            )
            with pytest.raises(cards.CardLimitError, match="max_payload_bytes"):
                await api.edit_card_message(
                    MagicMock(),
                    "https://api.example.invalid",
                    "test-token",
                    channel_id="group-1",
                    channel_type=ChannelType.Group,
                    message_id="card-1",
                    card=card,
                    plain=plain,
                    card_seq=1,
                    transient=True,
                )

        assert post_json.await_count == 1

    @pytest.mark.asyncio
    async def test_card_profile_keeps_404_distinct_from_explicitly_disabled(self):
        assert hasattr(api, "get_card_profile")
        missing = await api.get_card_profile(
            _ProfileSession(_ProfileResponse(status=404)),
            "https://api.example.invalid",
            "test-token",
        )
        assert missing.available is False
        assert missing.enabled is False

        disabled = await api.get_card_profile(
            _ProfileSession(_ProfileResponse(status=200, payload={"enabled": 0})),
            "https://api.example.invalid",
            "test-token",
        )
        assert disabled.available is True
        assert disabled.enabled is False

    @pytest.mark.asyncio
    async def test_card_profile_uses_safe_error_for_transport_or_server_failure(self):
        assert hasattr(api, "get_card_profile")
        with pytest.raises(api.OctoApiError, match="HTTP 503") as exc_info:
            await api.get_card_profile(
                _ProfileSession(
                    _ProfileResponse(
                        status=503,
                        body="Bearer top-secret response body",
                    )
                ),
                "https://api.example.invalid",
                "test-token",
            )
        assert "top-secret" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_card_profile_preserves_validated_templating_capability(self):
        manifest = await api.get_card_profile(
            _ProfileSession(
                _ProfileResponse(
                    status=200,
                    payload={
                        "enabled": 1,
                        "templating": {
                            "supported": True,
                            "wire": "template-ref/v1",
                            "templates": [
                                {
                                    "id": "approval",
                                    "version": "2",
                                    "views": [
                                        {
                                            "name": "request",
                                            "wire_profile": "octo/v2",
                                            "states": ["pending"],
                                            "submit_actions": ["approve"],
                                        },
                                        {"name": 7, "wire_profile": "octo/v1"},
                                    ],
                                },
                                {"id": "missing-version"},
                            ],
                        },
                    },
                )
            ),
            "https://api.example.invalid",
            "test-token",
        )

        assert manifest.templating == types.CardTemplatingCapability(
            supported=True,
            wire="template-ref/v1",
            templates=(
                types.CardTemplateCapability(
                    id="approval",
                    version="2",
                    views=(
                        types.CardTemplateViewCapability(
                            name="request",
                            wire_profile="octo/v2",
                            states=("pending",),
                            submit_actions=("approve",),
                        ),
                    ),
                ),
            ),
        )

    @pytest.mark.asyncio
    async def test_events_use_the_server_post_contract_and_ack_exact_path(self):
        assert hasattr(api, "fetch_bot_events")
        assert hasattr(api, "ack_bot_event")
        with patch.object(
            api,
            "post_json",
            AsyncMock(return_value={"results": [{"event_id": 8}]}),
        ) as post_json:
            events = await api.fetch_bot_events(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                since_event_id=7,
                limit=40,
                wait_seconds=12,
            )
            await api.ack_bot_event(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                event_id=8,
            )

        assert events == [{"event_id": 8}]
        assert post_json.await_args_list[0].args[3:] == (
            "/v1/bot/events",
            {"event_id": 7, "limit": 40, "wait": 12},
        )
        assert post_json.await_args_list[1].args[3:] == (
            "/v1/bot/events/8/ack",
            {},
        )

    @pytest.mark.asyncio
    async def test_events_bound_batch_and_extend_timeout_past_long_poll(self):
        with patch.object(
            api,
            "post_json",
            AsyncMock(return_value={"results": []}),
        ) as post_json:
            await api.fetch_bot_events(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                since_event_id=7,
                limit=500,
                wait_seconds=60,
            )

        assert post_json.await_args.args[4] == {
            "event_id": 7,
            "limit": 100,
            "wait": 30,
        }
        assert post_json.await_args.kwargs["timeout"].total == 40
