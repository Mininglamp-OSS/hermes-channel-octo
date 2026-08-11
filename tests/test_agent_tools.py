"""Behavioral tests for the ``octo_management`` tool boundary."""

from __future__ import annotations

import json
import logging

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


@pytest.mark.asyncio
async def test_guarded_session_factory_accepts_api_and_cdn_origins() -> None:
    with patch.dict(
        "os.environ",
        {"OCTO_ALLOW_PRIVATE_HOSTS": "true"},
    ):
        session = agent_tools._new_guarded_http_session(
            "https://api.octo.invalid",
            "https://cdn.octo.invalid/assets",
        )
    try:
        assert session.transport_policy.trusted_download_origins() == frozenset({
            ("https", "api.octo.invalid", 443),
            ("https", "cdn.octo.invalid", 443),
        })
    finally:
        await session.close()


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
async def test_management_handler_uses_real_gateway_session_context() -> None:
    from gateway.session_context import clear_session_vars, set_session_vars

    fetch_groups = AsyncMock(return_value=[])
    tokens = set_session_vars(platform="octo", user_id="owner-uid")
    try:
        with (
            patch.object(
                agent_tools,
                "_resolve_adapter",
                return_value=_configured_adapter(),
            ),
            patch.object(
                agent_tools,
                "_new_guarded_http_session",
                return_value=_NoIoSession(),
            ),
            patch.object(
                agent_tools.api,
                "fetch_bot_groups",
                fetch_groups,
            ),
        ):
            result = json.loads(
                await agent_tools.octo_management_handler(
                    {"action": "list-groups"}
                )
            )
    finally:
        clear_session_vars(tokens)

    assert result == {"ok": True, "data": {"groups": []}}
    fetch_groups.assert_awaited_once()


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
        patch.object(
            agent_tools.api,
            "fetch_bot_groups",
            AsyncMock(return_value=[]),
        ),
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
async def test_management_send_filters_mentions_to_authoritative_target_roster():
    member_lookup = AsyncMock(
        return_value=[
            GroupMember(uid="owner-uid", name="Owner"),
            GroupMember(uid="member-1", name="Member"),
        ]
    )
    send_message = AsyncMock(
        return_value=SendMessageResult(message_id="server-message")
    )
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools.api, "get_group_members", member_lookup),
        patch.object(agent_tools.api, "send_message", send_message),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("send-message"),
                    "requester_uid": "owner-uid",
                    "mention_uids": ["member-1", "outsider"],
                }
            )
        )

    assert result["ok"] is True
    member_lookup.assert_awaited_once_with(
        ANY,
        "https://octo.invalid",
        "test-token",
        "group-1",
    )
    assert send_message.await_args.kwargs["mention_uids"] == ["member-1"]



@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    ["group:group-1", "group:group-1____thread-1"],
)
async def test_management_send_fails_closed_when_permission_skips_requested_roster(
    target: str,
) -> None:
    permission = AsyncMock(return_value=SimpleNamespace(allowed=True, reason=None))
    send_message = AsyncMock()
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools, "check_permission", permission),
        patch.object(agent_tools.api, "send_message", send_message),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("send-message"),
                    "target": target,
                    "requester_uid": "owner-uid",
                    "mention_uids": ["member-1"],
                }
            )
        )

    assert result == {
        "error": "Unable to verify target members for requested mentions"
    }
    send_message.assert_not_awaited()

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


def test_management_schema_declares_the_runtime_input_bounds():
    properties = agent_tools.TOOL_SCHEMA["parameters"]["properties"]

    assert properties["group_id"]["maxLength"] == 64
    assert properties["target"]["maxLength"] == 192
    assert properties["limit"]["minimum"] == 1
    assert properties["limit"]["maximum"] == 100
    assert properties["content"]["maxLength"] == 20_000
    assert properties["mention_uids"]["maxItems"] == 64
    assert properties["mention_uids"]["items"]["maxLength"] == 64
    assert properties["members"]["minItems"] == 1
    assert properties["members"]["maxItems"] == 100
    assert properties["members"]["items"]["maxLength"] == 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "override", "expected_error"),
    [
        ("group-md-update", {"content": {"unexpected": "object"}}, "invalid content"),
        ("read-messages", {"limit": True}, "invalid limit"),
        ("create-group", {"members": "member-1"}, "invalid members"),
        ("create-group", {"members": ["member-1"] * 101}, "invalid members"),
        ("create-group", {"members": ["m" * 65]}, "invalid members"),
        ("send-message", {"mention_uids": ["m" * 65]}, "invalid mention_uids"),
        ("update-group", {"name": "n" * 257}, "invalid name"),
    ],
)
async def test_management_rejects_runtime_malformed_values_before_io(
    action: str,
    override: dict[str, object],
    expected_error: str,
):
    session_factory = MagicMock(side_effect=AssertionError("session must not open"))
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", session_factory),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for(action),
                    **override,
                    "requester_uid": "owner-uid",
                },
                trusted_uid="owner-uid",
            )
        )

    assert result == {"error": expected_error}
    session_factory.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_group_md_update_cannot_poison_adapter_cache():
    adapter = _configured_adapter()
    adapter._group_md_cache["group-1"] = {"content": "safe", "version": 3}
    adapter._group_md_checked.add("group-1")
    adapter._write_md_to_disk = MagicMock()
    session_factory = MagicMock(side_effect=AssertionError("session must not open"))
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=adapter),
        patch.object(agent_tools, "_new_guarded_http_session", session_factory),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("group-md-update"),
                    "content": ["not", "text"],
                    "requester_uid": "owner-uid",
                },
                trusted_uid="owner-uid",
            )
        )

    assert result == {"error": "invalid content"}
    assert adapter._group_md_cache == {"group-1": {"content": "safe", "version": 3}}
    assert adapter._group_md_checked == {"group-1"}
    adapter._write_md_to_disk.assert_not_called()
    session_factory.assert_not_called()


@pytest.mark.asyncio
async def test_owner_mutation_does_not_require_group_membership_read():
    update_group = AsyncMock()
    membership_check = AsyncMock(
        side_effect=AssertionError("owner mutation must not read group membership")
    )
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools, "check_permission", membership_check),
        patch.object(agent_tools.api, "update_group", update_group),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("update-group"),
                    "name": "Owner-authorized rename",
                    "requester_uid": "owner-uid",
                },
                trusted_uid="owner-uid",
            )
        )

    assert result["ok"] is True
    membership_check.assert_not_awaited()
    update_group.assert_awaited_once()


@pytest.mark.asyncio
async def test_malformed_md_response_version_cannot_poison_adapter_cache():
    adapter = _configured_adapter()
    adapter._write_md_to_disk = MagicMock()
    update_group_md = AsyncMock(return_value={"version": ["malformed"]})
    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=adapter),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(agent_tools.api, "update_group_md", update_group_md),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("group-md-update"),
                    "content": "new content",
                    "requester_uid": "owner-uid",
                },
                trusted_uid="owner-uid",
            )
        )

    assert result == {"ok": True, "data": {"updated": True, "version": 0}}
    assert adapter._group_md_cache == {
        "group-1": {"content": "new content", "version": 0}
    }
    adapter._write_md_to_disk.assert_called_once_with("group-1", "new content", 0)



@pytest.mark.asyncio
async def test_search_audit_omits_requester_and_raw_keyword(caplog):
    requester = "requester-stable-id"
    keyword = "confidential-search-keyword"
    adapter = _configured_adapter()
    adapter._owner_uid = requester
    caplog.set_level(logging.INFO, logger="hermes_octo_plugin.agent_tools")

    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=adapter),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(
            agent_tools.api,
            "search_space_members",
            AsyncMock(return_value=[{"uid": "member-1"}]),
        ),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("search-members"),
                    "keyword": keyword,
                    "requester_uid": requester,
                },
                trusted_uid=requester,
            )
        )

    assert result["ok"] is True
    record = next(record for record in caplog.records if "[AUDIT]" in record.message)
    entry = json.loads(record.message.partition("octo-query ")[2])
    assert entry == {
        "action": "search-members",
        "result": "allowed",
        "count": 1,
    }
    assert requester not in caplog.text
    assert keyword not in caplog.text


@pytest.mark.asyncio
async def test_read_audit_omits_requester_and_raw_target(caplog):
    requester = "requester-stable-id"
    target = "group:confidential-target"
    caplog.set_level(logging.INFO, logger="hermes_octo_plugin.agent_tools")

    with (
        patch.object(agent_tools, "_resolve_adapter", return_value=_configured_adapter()),
        patch.object(agent_tools, "_new_guarded_http_session", _NoIoSession),
        patch.object(
            agent_tools.api,
            "get_group_members",
            AsyncMock(return_value=[GroupMember(uid=requester, name="Member")]),
        ),
        patch.object(agent_tools.api, "get_channel_messages", AsyncMock(return_value=[])),
    ):
        result = json.loads(
            await _call_handler(
                {
                    **_args_for("read-messages"),
                    "target": target,
                    "requester_uid": requester,
                },
                trusted_uid=requester,
            )
        )

    assert result["ok"] is True
    record = next(record for record in caplog.records if "[AUDIT]" in record.message)
    entry = json.loads(record.message.partition("octo-query ")[2])
    assert entry == {
        "action": "read-messages",
        "result": "allowed",
        "channelType": 2,
        "count": 0,
    }
    assert requester not in caplog.text
    assert target not in caplog.text