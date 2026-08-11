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
    InteractiveCard = 17


# Type-17 (Adaptive Card) wire constants.  These are intentionally kept in
# the protocol types module rather than the renderer so API callers, the
# adapter lifecycle, and tools negotiate exactly the same values.
CARD_PROFILE_V1 = "octo/v1"
CARD_PROFILE_V2 = "octo/v2"
CARD_VERSION = "1.5"
CARD_PROFILES = frozenset({CARD_PROFILE_V1, CARD_PROFILE_V2})


def card_contains_interaction(value: Any, _seen: set[int] | None = None) -> bool:
    """Return whether a JSON-like card tree contains an interactive node.

    Adaptive Card interactions can appear in nested ``items`` / ``actions``
    arrays.  Keep this traversal cycle-safe even though production cards are
    JSON-shaped: callers may construct the controlled card object in Python
    before it is serialized.
    """
    if not isinstance(value, (dict, list, tuple)):
        return False
    seen = _seen if _seen is not None else set()
    marker = id(value)
    if marker in seen:
        return False
    seen.add(marker)
    if isinstance(value, dict):
        node_type = value.get("type")
        if isinstance(node_type, str) and (
            node_type.startswith("Input.") or node_type == "Action.Submit"
        ):
            return True
        return any(card_contains_interaction(item, seen) for item in value.values())
    return any(card_contains_interaction(item, seen) for item in value)


def resolve_card_profile(card: dict[str, Any], requested: str | None = None) -> str:
    """Select the required wire profile, upgrading interaction cards to v2."""
    if requested is not None and requested not in CARD_PROFILES:
        raise ValueError("unsupported Octo card profile")
    if card_contains_interaction(card):
        return CARD_PROFILE_V2
    return requested or CARD_PROFILE_V1


@dataclass(frozen=True)
class CardTemplateViewCapability:
    """One advertised view within a server-backed card template."""

    name: str
    wire_profile: str
    states: tuple[str, ...] = ()
    submit_actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CardTemplateCapability:
    """One validated template catalog entry."""

    id: str
    version: str
    views: tuple[CardTemplateViewCapability, ...] = ()


@dataclass(frozen=True)
class CardTemplatingCapability:
    """Optional template-ref capability advertised by the server."""

    supported: bool
    wire: str
    templates: tuple[CardTemplateCapability, ...] = ()


@dataclass(frozen=True)
class CardProfileManifest:
    """Safe normalized response from ``GET /v1/bot/card/profile``."""

    available: bool
    enabled: bool
    profiles: tuple[str, ...] | None = None
    card_version: str | None = None
    elements: tuple[str, ...] | None = None
    inputs: tuple[str, ...] | None = None
    actions: tuple[str, ...] | None = None
    limits: dict[str, Any] = field(default_factory=dict)
    templating: CardTemplatingCapability | None = None


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
    humans: bool | None = None
    ais: bool | None = None


def _coerce_wire_bool(value: Any) -> bool | None:
    """Normalize the protocol's boolean/0/1 flags without truthiness traps."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


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

    type: MessageType | int = MessageType.Text
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
            "type",
            "content",
            "url",
            "name",
            "mention",
            "reply",
            "event",
            "plain",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        mention = None
        raw_mention = data.get("mention")
        if isinstance(raw_mention, dict) and raw_mention:
            raw_entities = raw_mention.get("entities")
            entities: list[MentionEntity] | None = None
            if isinstance(raw_entities, list):
                from .mention import MAX_MENTIONS_PER_MESSAGE

                parsed_entities = []
                for entity in raw_entities[:MAX_MENTIONS_PER_MESSAGE]:
                    if not isinstance(entity, dict):
                        continue
                    uid = entity.get("uid")
                    offset = entity.get("offset")
                    length = entity.get("length")
                    if (
                        not isinstance(uid, str)
                        or not uid
                        or not isinstance(offset, int)
                        or isinstance(offset, bool)
                        or offset < 0
                        or not isinstance(length, int)
                        or isinstance(length, bool)
                        or length <= 0
                    ):
                        continue
                    parsed_entities.append(
                        MentionEntity(uid=uid, offset=offset, length=length)
                    )
                entities = parsed_entities or None
            raw_uids = raw_mention.get("uids")
            uids = (
                [uid for uid in raw_uids if isinstance(uid, str) and uid]
                if isinstance(raw_uids, list)
                else None
            )
            mention = MentionPayload(
                uids=uids or None,
                entities=entities,
                all=_coerce_wire_bool(raw_mention.get("all")),
                humans=_coerce_wire_bool(raw_mention.get("humans")),
                ais=_coerce_wire_bool(raw_mention.get("ais")),
            )

        reply = None
        raw_reply = data.get("reply")
        if isinstance(raw_reply, dict) and raw_reply:
            raw_reply_payload = raw_reply.get("payload")
            raw_from_uid = raw_reply.get("from_uid")
            raw_from_name = raw_reply.get("from_name")
            reply = ReplyPayload(
                payload=(
                    raw_reply_payload
                    if isinstance(raw_reply_payload, dict)
                    else None
                ),
                from_uid=raw_from_uid if isinstance(raw_from_uid, str) else None,
                from_name=(
                    raw_from_name
                    if isinstance(raw_from_name, str)
                    else None
                ),
            )

        # Preserve unknown numeric message types.  Coercing them to Text
        # hides protocol evolution and can misrepresent a non-text payload.
        raw_type = data.get("type", 1)
        try:
            msg_type = MessageType(raw_type)
        except (TypeError, ValueError):
            msg_type = (
                raw_type
                if isinstance(raw_type, int) and not isinstance(raw_type, bool)
                else -1
            )

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
            url=data.get("url") if isinstance(data.get("url"), str) else None,
            name=data.get("name") if isinstance(data.get("name"), str) else None,
            mention=mention,
            reply=reply,
            event=data.get("event") if isinstance(data.get("event"), dict) else None,
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

    message_id: str | None = None
    message_seq: int | None = None
    client_msg_no: str | None = None


@dataclass
class GroupMember:
    """A member of a Octo group."""

    uid: str
    name: str
    role: str | None = None  # admin/member
    robot: bool | int | None = None


@dataclass
class GroupInfo:
    """Basic group information."""

    group_no: str
    name: str
    extra: dict[str, Any] = field(default_factory=dict)
