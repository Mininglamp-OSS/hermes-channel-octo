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
    # 图文混排 (rich text). Contract defined by octo-lib
    # common/richtext.go — payload.content carries an ordered array of
    # {type:text|image} blocks. Field names must match octo-lib.
    RichText = 14


# RichText(=14) block type constants (aligned with octo-lib
# RichTextBlockText / RichTextBlockImage).
RICH_TEXT_BLOCK_TEXT = "text"
RICH_TEXT_BLOCK_IMAGE = "image"

# Placeholder injected when rendering a RichText image block as plain text
# (aligned with octo-lib RichTextImagePlaceholder).
RICH_TEXT_IMAGE_PLACEHOLDER = "[图片]"


@dataclass
class RichTextBlock:
    """One block inside a RichText(=14) `content` array.

    - type=text  → `text` (non-empty)
    - type=image → `url` (http/https), `width` and `height` (px, > 0),
                   `size` and `name` optional

    Server-side validation lives in octo-lib; this dataclass only carries
    the fields. Do NOT introduce `entities`/`offset`/`length` here — the
    RichText contract is deliberately positional, not offset-based.
    """
    type: str
    text: str | None = None
    url: str | None = None
    width: int | None = None
    height: int | None = None
    size: int | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the wire dict. Omits None-valued fields."""
        out: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            out["text"] = self.text
        if self.url is not None:
            out["url"] = self.url
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        if self.size is not None:
            out["size"] = self.size
        if self.name is not None:
            out["name"] = self.name
        return out


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
    # RichText(=14) only — ordered block array. Populated when the wire
    # `content` field is a list; text/other message types leave this None.
    blocks: list[dict[str, Any]] | None = None
    # RichText(=14) only — top-level `plain` string (server-authoritative
    # rendered text). None on other message types or when absent.
    plain: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessagePayload:
        """Parse a MessagePayload from a raw dict (e.g. from JSON)."""
        known_keys = {
            "type", "content", "url", "name",
            "mention", "reply", "event", "plain",
        }
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

        # RichText(=14): wire `content` is a list of blocks, and `plain`
        # is a top-level string. Legacy string-typed `content` on RichText
        # (old server or forward preview) is normalized into a single text
        # block downstream — here we just keep raw shapes intact.
        raw_content = data.get("content")
        blocks: list[dict[str, Any]] | None = None
        content_str: str | None = None
        if isinstance(raw_content, list):
            blocks = [b for b in raw_content if isinstance(b, dict)]
        elif isinstance(raw_content, str) or raw_content is None:
            content_str = raw_content

        plain_val = data.get("plain")
        plain_str = plain_val if isinstance(plain_val, str) else None

        return cls(
            type=msg_type,
            content=content_str,
            url=data.get("url"),
            name=data.get("name"),
            mention=mention,
            reply=reply,
            event=data.get("event"),
            blocks=blocks,
            plain=plain_str,
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
