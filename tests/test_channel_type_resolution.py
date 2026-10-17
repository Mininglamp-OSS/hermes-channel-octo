"""Outbound channel-type resolution: OIDC UUID user ids must resolve to DM.

Regression guard for the misclassification where a 36-char hyphenated UUID
(octo users created through the OIDC/SSO login path) fell through the
32-char-hex heuristic and was sent as ``channel_type=2`` (group), making the
server answer ``400 group_status not_found`` for every typing/status call.
"""

from __future__ import annotations

import pytest

from hermes_octo_plugin.types import ChannelType
from tests.conftest import make_bare_adapter

UUID_UID = "baf54cde-cbf3-49c5-a43d-7733982b54e6"
HEX_UID = "59fd8053735d449ab9b18dc25c6399aa"


@pytest.mark.parametrize(
    "chat_id",
    [
        UUID_UID,  # lowercase
        UUID_UID.upper(),  # uppercase (octo does not normalise case)
        "01234567-89ab-cdef-0123-456789abcdef",
    ],
)
def test_oidc_uuid_user_id_resolves_to_dm(chat_id):
    adapter = make_bare_adapter()
    assert adapter._resolve_channel_type(chat_id) == ChannelType.DM


@pytest.mark.parametrize(
    "chat_id",
    [
        "group-1",  # plain non-hex id
        "baf54cde-cbf3-49c5-a43d-7733982b54e",  # 35 chars: truncated uuid
        "baf54cde-cbf3-49c5-a43d-7733982b54e67",  # 37 chars: over-long uuid
        "baf54cdecbf349c5a43d7733982b54e6" + "0" * 4,  # 36 chars, all hex
        "baf54cde-cbf3-49c5-a43d-7733982b54g6",  # non-hex trailing segment
    ],
)
def test_non_uuid_ids_are_not_treated_as_dm(chat_id):
    adapter = make_bare_adapter()
    assert adapter._resolve_channel_type(chat_id) == ChannelType.Group


def test_legacy_32_char_hex_uid_still_resolves_to_dm():
    adapter = make_bare_adapter()
    assert adapter._resolve_channel_type(HEX_UID) == ChannelType.DM


def test_thread_id_resolves_to_community_topic():
    adapter = make_bare_adapter()
    assert (
        adapter._resolve_channel_type("group-1____user-1")
        == ChannelType.CommunityTopic
    )


def test_explicit_metadata_wins_over_id_shape():
    adapter = make_bare_adapter()
    assert (
        adapter._resolve_channel_type(UUID_UID, {"channel_type": ChannelType.Group})
        == ChannelType.Group
    )


def test_remembered_inbound_kind_wins_over_id_shape():
    """A group whose id happens to look like a uuid must stay a group: the
    inbound path already resolved it and cached the answer."""
    adapter = make_bare_adapter()
    adapter._chat_kind[UUID_UID] = ChannelType.Group
    assert adapter._resolve_channel_type(UUID_UID) == ChannelType.Group
