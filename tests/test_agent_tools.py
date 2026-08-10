"""Behavioral tests for the ``octo_management`` tool boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin import agent_tools
from hermes_octo_plugin.types import GroupInfo, GroupMember, SendMessageResult


class _EmptyResponse:
    """A successful no-op HTTP response used to prove authorization is pre-I/O."""

    ok = True
    status = 200
    reason = "OK"
    request_info = None
    history = ()
    headers: dict[str, str] = {}
    content = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self, **_kwargs):
        return {}

    async def text(self):
        return ""


class _NoIoSession:
    """Session stub that makes accidental handler I/O deterministic in RED."""

    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def get(self, *_args, **_kwargs):
        return _EmptyResponse()

    def post(self, *_args, **_kwargs):
        return _EmptyResponse()

    def put(self, *_args, **_kwargs):
        return _EmptyResponse()

    def delete(self, *_args, **_kwargs):
        return _EmptyResponse()


def _configured_adapter() -> SimpleNamespace:
    return SimpleNamespace(
        _api_url="https://octo.invalid",
        _bot_token="test-token",
        on_behalf_of="grantor-1",
        _owner_uid="owner-uid",
        _known_group_ids={"group-1"},
        _group_md_cache={},
        _group_md_checked=set(),
        find_shared_groups=lambda _uid: [],
    )


_TRUST_CLAIM = object()


async def _call_handler(
    args: dict[str, object], trusted_uid: object = _TRUST_CLAIM
) -> str:
    """Run the tool with an explicit simulated Hermes session identity."""
    if trusted_uid is _TRUST_CLAIM:
        trusted_uid = args.get("requester_uid")
    with patch.object(
        agent_tools, "_trusted_requester_uid", return_value=trusted_uid
    ):
        return await agent_tools.octo_management_handler(args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("group_id", "../space/members?limit=500#"),
        ("group_id", "group/../../space"),
        ("group_id", "group%2F..%2Fspace"),
        ("group_id", "."),
        ("short_id", "../members"),
        ("short_id", "thread?admin=true"),
    ],
)
async def test_resource_ids_are_rejected_before_session_or_authorization_io(
    field: str,
    value: str,
):
    args = {
        **_args_for("get-thread"),
        "requester_uid": "member-1",
        field: value,
    }
    session_factory = MagicMock(side_effect=AssertionError("session must not open"))
    member_lookup = AsyncMock()
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", session_factory),
        patch.object(agent_tools.api, "get_group_members", member_lookup),
    ):
        result = json.loads(await _call_handler(args))

    assert result == {"error": f"invalid {field}"}
    session_factory.assert_not_called()
    member_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_target_path_injection_is_rejected_before_membership_io():
    args = {
        **_args_for("read-messages"),
        "requester_uid": "member-1",
        "target": "group:../space/members?limit=500#",
    }
    member_lookup = AsyncMock()
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(
            agent_tools, "_new_guarded_http_session", return_value=_NoIoSession()
        ),
        patch.object(agent_tools.api, "get_group_members", member_lookup),
    ):
        result = json.loads(await _call_handler(args))

    assert result == {"error": "invalid target channel id"}
    member_lookup.assert_not_awaited()


def _args_for(action: str) -> dict[str, object]:
    """Supply every action's ordinary required fields without a requester."""
    return {
        "action": action,
        "group_id": "group-1",
        "short_id": "thread-1",
        "target": "group:group-1",
        "content": "test content",
        "members": ["member-1"],
        "creator": "owner-uid",
        "thread_name": "test thread",
    }


def test_tool_schema_uses_trusted_session_identity_without_a_model_claim():
    assert set(agent_tools.TOOL_SCHEMA["parameters"]["required"]) == {"action"}
    assert "requester_uid" not in agent_tools.TOOL_SCHEMA["parameters"]["properties"]


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("octo", "member-1"), ("discord", None), ("", None)],
)
def test_trusted_requester_comes_only_from_an_octo_session_context(
    platform: str, expected: str | None
):
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform=platform, user_id="member-1")
    try:
        assert agent_tools._trusted_requester_uid() == expected
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", agent_tools.ACTIONS)
async def test_management_actions_fail_closed_without_trusted_session(action: str):
    """Every management action requires a trusted Octo session requester."""
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
    ):
        result = json.loads(await _call_handler(_args_for(action), trusted_uid=None))

    assert result == {
        "error": f"action '{action}' requires a trusted Octo session requester"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected_error"),
    [
        ("group-info", "You are not in this group; access denied"),
        ("group-members", "You are not in this group; access denied"),
        ("group-md-read", "You are not in this group; access denied"),
        ("list-threads", "You are not in this group; access denied"),
        ("get-thread", "You are not in this group; thread access denied"),
        ("list-thread-members", "You are not in this group; thread access denied"),
        ("thread-md-read", "You are not in this group; thread access denied"),
    ],
)
async def test_group_and_thread_reads_deny_requester_outside_parent_group(
    action: str,
    expected_error: str,
):
    """Group and thread metadata never bypass the parent-group membership gate."""
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(
            agent_tools.api,
            "get_group_members",
            AsyncMock(return_value=[GroupMember(uid="member-1", name="Member")]),
        ),
    ):
        result = json.loads(
            await _call_handler(
                {**_args_for(action), "requester_uid": "outsider"}
            )
        )

    assert result == {"error": expected_error}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["group-info", "get-thread"])
async def test_group_and_thread_reads_allow_current_parent_group_member(action: str):
    """A current parent-group member can complete the corresponding metadata read."""
    member_lookup = AsyncMock(
        return_value=[GroupMember(uid="member-1", name="Member")]
    )
    group_info_lookup = AsyncMock(
        return_value=GroupInfo(group_no="group-1", name="Test group")
    )
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools.api, "get_group_members", member_lookup),
        patch.object(agent_tools.api, "get_group_info", group_info_lookup),
    ):
        result = json.loads(
            await _call_handler(
                {**_args_for(action), "requester_uid": "member-1"}
            )
        )

    assert result["ok"] is True
    member_lookup.assert_awaited_once_with(
        ANY,
        "https://octo.invalid",
        "test-token",
        "group-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["list-groups", "search-members", "voice-context-read"])
@pytest.mark.parametrize(
    ("requester_uid", "expected_error"),
    [
        ("owner-uid", None),
        ("ordinary-member", "requires bot-owner privileges"),
    ],
)
async def test_space_wide_reads_require_the_explicit_owner(
    action: str,
    requester_uid: str,
    expected_error: str | None,
):
    """Space-wide data is visible only to a caller explicitly identified as owner."""
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
    ):
        result = json.loads(
            await _call_handler(
                {**_args_for(action), "requester_uid": requester_uid}
            )
        )

    if expected_error is None:
        assert result["ok"] is True
    else:
        assert expected_error in result["error"]


@pytest.mark.asyncio
async def test_group_member_lookup_failure_denies_metadata_read():
    """A failed membership lookup is never treated as an empty authorized result."""
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(
            agent_tools.api,
            "get_group_members",
            AsyncMock(side_effect=ConnectionError("Bearer secret-token-from-upstream")),
        ),
    ):
        result = json.loads(
            await _call_handler(
                {**_args_for("group-info"), "requester_uid": "member-1"}
            )
        )

    assert result == {"error": "Group member lookup failed"}
    assert "secret-token" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "cache_key", "api_call"),
    [
        ("group-md-update", "group-1", "update_group_md"),
        ("thread-md-update", "group-1____thread-1", "update_thread_md"),
    ],
)
async def test_failed_md_write_never_updates_local_state_or_leaks_secrets(
    action: str,
    cache_key: str,
    api_call: str,
):
    """A rejected MD write leaves cache/disk untouched and yields a safe error."""
    adapter = _configured_adapter()
    adapter._group_md_cache[cache_key] = {"content": "old", "version": 7}
    adapter._group_md_checked.add(cache_key)
    adapter._write_md_to_disk = MagicMock()
    failure = RuntimeError("Bearer secret-token-from-backend")

    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=adapter),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools.api, api_call, AsyncMock(side_effect=failure)),
    ):
        result = json.loads(
            await _call_handler(
                {**_args_for(action), "requester_uid": "owner-uid"}
            )
        )

    assert result == {"error": f"{action} failed: upstream request failed"}
    assert "secret-token" not in json.dumps(result)
    assert adapter._group_md_cache[cache_key] == {"content": "old", "version": 7}
    assert adapter._group_md_checked == {cache_key}
    adapter._write_md_to_disk.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_returns_the_server_message_identity():
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(
            agent_tools.api,
            "get_group_members",
            AsyncMock(return_value=[GroupMember(uid="owner-uid", name="Owner")]),
        ),
        patch.object(
            agent_tools.api,
            "send_message",
            AsyncMock(
                return_value=SendMessageResult(
                    message_id="9223372036854775807",
                    message_seq=9,
                    client_msg_no="client-1",
                )
            ),
        ),
    ):
        result = json.loads(
            await _call_handler(
                {**_args_for("send-message"), "requester_uid": "owner-uid"}
            )
        )

    assert result["ok"] is True
    assert result["data"]["message_id"] == "9223372036854775807"
    assert result["data"]["message_seq"] == 9
    assert result["data"]["client_msg_no"] == "client-1"


@pytest.mark.asyncio
async def test_thread_send_does_not_bypass_owner_only_join_permission():
    join_thread = AsyncMock()
    send_message = AsyncMock(
        return_value=SendMessageResult(message_id="server-thread-message")
    )
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(
            agent_tools.api,
            "get_group_members",
            AsyncMock(return_value=[GroupMember(uid="ordinary-member", name="Member")]),
        ),
        patch.object(agent_tools.api, "join_thread", join_thread),
        patch.object(agent_tools.api, "send_message", send_message),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("send-message"),
                    "requester_uid": "ordinary-member",
                    "target": "group-1____thread-1",
                }
            )
        )

    assert result["ok"] is True
    join_thread.assert_not_awaited()
    send_message.assert_awaited_once()
    assert send_message.await_args.kwargs["on_behalf_of"] == "grantor-1"


@pytest.mark.asyncio
async def test_update_group_without_any_mutation_field_does_not_report_success():
    update_group = AsyncMock()
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools.api, "update_group", update_group),
    ):
        result = json.loads(
            await _call_handler(
                {**_args_for("update-group"), "requester_uid": "owner-uid"}
            )
        )

    assert result == {"error": "update-group requires at least one of: name, notice"}
    update_group.assert_not_awaited()


@pytest.mark.asyncio
async def test_claimed_owner_uid_cannot_override_trusted_session_requester():
    update_group = AsyncMock()
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools.api, "update_group", update_group),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("update-group"),
                    "requester_uid": "owner-uid",
                    "name": "Unauthorized rename",
                },
                trusted_uid="ordinary-member",
            )
        )

    assert "requires bot-owner privileges" in result["error"]
    update_group.assert_not_awaited()


@pytest.mark.asyncio
async def test_owner_claim_without_trusted_octo_session_is_denied_before_io():
    update_group = AsyncMock()
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools.api, "update_group", update_group),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("update-group"),
                    "requester_uid": "owner-uid",
                    "name": "Unauthenticated rename",
                },
                trusted_uid=None,
            )
        )

    assert result == {"error": "action 'update-group' requires a trusted Octo session requester"}
    update_group.assert_not_awaited()
