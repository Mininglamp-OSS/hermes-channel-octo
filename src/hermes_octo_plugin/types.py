"""
Octo Bot API types.

Defines channel types, message types, and payload structures used
by the Octo Bot API and WuKongIM protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ChannelType(IntEnum):
    """Octo channel types."""
    DM = 1
    Group = 2
    CommunityTopic = 5  # Thread / sub-channel (子区)


class MessageType(IntEnum):
    """Octo message content types."""
    Text = 1
    Image = 2
    GIF = 3
    Voice = 4
    Video = 5
    Location = 6
    Card = 7
    File = 8
    MultipleForward = 11


@dataclass
class MentionEntity:
    """
    Precise position of a single @mention.

    offset/length units are UTF-16 code units (matching JS string.length).
    """
    uid: str
    offset: int
    length: int


@dataclass
class MentionPayload:
    """Mention metadata attached to a message."""
    uids: list[str] | None = None
    entities: list[MentionEntity] | None = None
    all: bool | None = None  # True or 1 = @all


@dataclass
class ReplyPayload:
    """Reply context for a message."""
    payload: dict[str, Any] | None = None
    from_uid: str | None = None
    from_name: str | None = None


@dataclass
class MessagePayload:
    """
    Octo message payload.

    The `type` field determines which other fields are populated.
    Additional unknown fields are captured in `extra`.
    """
    type: MessageType = MessageType.Text
    content: str | None = None
    url: str | None = None
    name: str | None = None
    mention: MentionPayload | None = None
    reply: ReplyPayload | None = None
    event: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessagePayload:
        """Parse a MessagePayload from a raw dict (e.g. from JSON)."""
        known_keys = {"type", "content", "url", "name", "mention", "reply", "event"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        mention = None
        if "mention" in data and data["mention"]:
            m = data["mention"]
            entities = None
            if m.get("entities"):
                entities = [
                    MentionEntity(uid=e["uid"], offset=e["offset"], length=e["length"])
                    for e in m["entities"]
                    if isinstance(e, dict) and "uid" in e
                ]
            mention = MentionPayload(
                uids=m.get("uids"),
                entities=entities,
                all=m.get("all"),
            )

        reply = None
        if "reply" in data and data["reply"]:
            r = data["reply"]
            reply = ReplyPayload(
                payload=r.get("payload"),
                from_uid=r.get("from_uid"),
                from_name=r.get("from_name"),
            )

        # Tolerate unknown message types from the server (e.g. system
        # notifications). Fall back to Text so the adapter doesn't crash.
        raw_type = data.get("type", 1)
        try:
            msg_type = MessageType(raw_type)
        except ValueError:
            msg_type = MessageType.Text

        return cls(
            type=msg_type,
            content=data.get("content"),
            url=data.get("url"),
            name=data.get("name"),
            mention=mention,
            reply=reply,
            event=data.get("event"),
            extra=extra,
        )


@dataclass
class BotMessage:
    """
    Incoming message received via WuKongIM WebSocket.

    Represents a fully decoded RECV packet with decrypted payload.
    """
    message_id: str
    message_seq: int
    from_uid: str
    channel_id: str
    channel_type: int
    timestamp: int
    payload: MessagePayload


@dataclass
class BotRegisterResp:
    """Response from /v1/bot/register API."""
    robot_id: str
    im_token: str
    ws_url: str
    api_url: str
    owner_uid: str
    owner_channel_id: str


@dataclass
class SendMessageResult:
    """Response from /v1/bot/sendMessage API."""
    message_id: int
    message_seq: int


@dataclass
class GroupMember:
    """A member of a Octo group."""
    uid: str
    name: str
    role: str | None = None  # admin/member
    robot: bool | None = None


@dataclass
class GroupInfo:
    """Basic group information."""
    group_no: str
    name: str
    extra: dict[str, Any] = field(default_factory=dict)
