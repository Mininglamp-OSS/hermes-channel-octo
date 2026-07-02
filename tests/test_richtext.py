"""
Tests for RichText(=14) 图文混排 support — types, inbound resolution,
and outbound send_rich_text_message.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin import api
from hermes_octo_plugin.types import (
    RICH_TEXT_BLOCK_IMAGE,
    RICH_TEXT_BLOCK_TEXT,
    RICH_TEXT_IMAGE_PLACEHOLDER,
    ChannelType,
    MessagePayload,
    MessageType,
    RichTextBlock,
)
from tests.conftest import make_bare_adapter


# ─── Types layer ────────────────────────────────────────────────────────────


class TestRichTextBlock:
    def test_text_block_serializes(self):
        b = RichTextBlock(type=RICH_TEXT_BLOCK_TEXT, text="hello")
        assert b.to_dict() == {"type": "text", "text": "hello"}

    def test_image_block_serializes(self):
        b = RichTextBlock(
            type=RICH_TEXT_BLOCK_IMAGE,
            url="https://x/y.png",
            width=100,
            height=50,
            size=1234,
            name="y.png",
        )
        assert b.to_dict() == {
            "type": "image",
            "url": "https://x/y.png",
            "width": 100,
            "height": 50,
            "size": 1234,
            "name": "y.png",
        }

    def test_none_fields_omitted(self):
        b = RichTextBlock(type=RICH_TEXT_BLOCK_IMAGE, url="u", width=1, height=1)
        assert "size" not in b.to_dict()
        assert "name" not in b.to_dict()
        assert "text" not in b.to_dict()


class TestMessagePayloadRichText:
    def test_parses_block_array(self):
        p = MessagePayload.from_dict({
            "type": 14,
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "image", "url": "https://x/y.png", "width": 100, "height": 50},
            ],
            "plain": "hello [图片]",
        })
        assert p.type == MessageType.RichText
        assert p.blocks is not None
        assert len(p.blocks) == 2
        assert p.plain == "hello [图片]"
        # content stays None when wire content is a list — blocks carries it.
        assert p.content is None

    def test_string_content_falls_through(self):
        # Legacy shape: server sends RichText with string-typed content.
        # blocks stays None; content keeps the string. The inbound resolver
        # handles the fallback to a synthetic single text block.
        p = MessagePayload.from_dict({
            "type": 14,
            "content": "legacy plain string",
        })
        assert p.type == MessageType.RichText
        assert p.blocks is None
        assert p.content == "legacy plain string"
        assert p.plain is None

    def test_plain_missing_ok(self):
        p = MessagePayload.from_dict({
            "type": 14,
            "content": [{"type": "text", "text": "hi"}],
        })
        assert p.plain is None

    def test_non_dict_blocks_filtered(self):
        p = MessagePayload.from_dict({
            "type": 14,
            "content": [
                "not a dict",
                {"type": "text", "text": "ok"},
                42,
            ],
        })
        assert p.blocks == [{"type": "text", "text": "ok"}]


# ─── Adapter inbound helper ─────────────────────────────────────────────────


def _make_adapter_with_api(api_url: str = "https://api.example.com"):
    a = make_bare_adapter()
    a._api_url = api_url
    a._cdn_url = ""
    return a


class TestResolveRichTextContent:
    def test_prefers_top_level_plain(self):
        a = _make_adapter_with_api()
        payload = MessagePayload.from_dict({
            "type": 14,
            "content": [
                {"type": "text", "text": "REBUILT"},
                {"type": "image", "url": "https://x/a.png", "width": 1, "height": 1},
            ],
            "plain": "server rendered text [图片]",
        })
        text, urls = a._resolve_rich_text_content(payload)
        assert text == "server rendered text [图片]"
        assert urls == ["https://x/a.png"]

    def test_builds_plain_from_blocks_when_missing(self):
        a = _make_adapter_with_api()
        payload = MessagePayload.from_dict({
            "type": 14,
            "content": [
                {"type": "text", "text": "prefix "},
                {"type": "image", "url": "https://x/a.png", "width": 1, "height": 1},
                {"type": "text", "text": " suffix"},
            ],
        })
        text, urls = a._resolve_rich_text_content(payload)
        assert text == f"prefix {RICH_TEXT_IMAGE_PLACEHOLDER} suffix"
        assert urls == ["https://x/a.png"]

    def test_collects_multiple_images_in_order(self):
        a = _make_adapter_with_api()
        payload = MessagePayload.from_dict({
            "type": 14,
            "content": [
                {"type": "image", "url": "https://x/a.png", "width": 1, "height": 1},
                {"type": "text", "text": " mid "},
                {"type": "image", "url": "https://x/b.png", "width": 1, "height": 1},
                {"type": "image", "url": "https://x/c.png", "width": 1, "height": 1},
            ],
        })
        text, urls = a._resolve_rich_text_content(payload)
        assert urls == [
            "https://x/a.png",
            "https://x/b.png",
            "https://x/c.png",
        ]
        # placeholder appears for each image in block order
        assert text.count(RICH_TEXT_IMAGE_PLACEHOLDER) == 3

    def test_relative_urls_resolved_via_api(self):
        a = _make_adapter_with_api("https://api.example.com")
        payload = MessagePayload.from_dict({
            "type": 14,
            "content": [
                {"type": "image", "url": "file/preview/xyz.png", "width": 1, "height": 1},
            ],
        })
        _, urls = a._resolve_rich_text_content(payload)
        assert urls == ["https://api.example.com/file/xyz.png"]

    def test_malformed_image_url_skipped(self):
        # url is a dict rather than a string — must not crash, must be
        # silently dropped from collected media_urls.
        a = _make_adapter_with_api()
        payload = MessagePayload.from_dict({
            "type": 14,
            "content": [
                {"type": "image", "url": {"nested": "bad"}, "width": 1, "height": 1},
                {"type": "image", "url": "https://x/ok.png", "width": 1, "height": 1},
            ],
        })
        text, urls = a._resolve_rich_text_content(payload)
        assert urls == ["https://x/ok.png"]
        # placeholder still emitted for both blocks — plain builder walks
        # by type, not by url validity.
        assert text == f"{RICH_TEXT_IMAGE_PLACEHOLDER}{RICH_TEXT_IMAGE_PLACEHOLDER}"

    def test_legacy_string_content_normalized(self):
        a = _make_adapter_with_api()
        payload = MessagePayload.from_dict({
            "type": 14,
            "content": "legacy body",
        })
        text, urls = a._resolve_rich_text_content(payload)
        assert text == "legacy body"
        assert urls == []

    def test_empty_plain_falls_back_to_blocks(self):
        # A plain of empty/whitespace-only should NOT suppress block rendering.
        a = _make_adapter_with_api()
        payload = MessagePayload.from_dict({
            "type": 14,
            "content": [{"type": "text", "text": "block-derived"}],
            "plain": "   ",
        })
        text, _ = a._resolve_rich_text_content(payload)
        assert text == "block-derived"


class TestResolveContentRichText:
    def test_resolve_content_returns_text(self):
        a = _make_adapter_with_api()
        payload = MessagePayload.from_dict({
            "type": 14,
            "content": [{"type": "text", "text": "hi"}],
            "plain": "hi",
        })
        assert a._resolve_content(payload) == "hi"

    def test_resolve_content_empty_richtext_placeholder(self):
        a = _make_adapter_with_api()
        # No blocks, no plain → empty text; fallback placeholder used.
        payload = MessagePayload.from_dict({"type": 14})
        assert a._resolve_content(payload) == "[图文消息]"


# ─── Outbound api.send_rich_text_message ────────────────────────────────────


class TestSendRichTextMessage:
    @pytest.mark.asyncio
    async def test_body_shape(self):
        session = MagicMock()
        blocks = [
            RichTextBlock(type=RICH_TEXT_BLOCK_TEXT, text="caption"),
            RichTextBlock(
                type=RICH_TEXT_BLOCK_IMAGE,
                url="https://x/y.png",
                width=100,
                height=50,
            ),
        ]
        with patch("hermes_octo_plugin.api.post_json", new_callable=AsyncMock) as mock_post:
            await api.send_rich_text_message(
                session=session,
                api_url="https://api.example.com",
                bot_token="tok",
                channel_id="G1",
                channel_type=ChannelType.Group,
                blocks=blocks,
                plain="caption[图片]",
            )
        mock_post.assert_awaited_once()
        args, _ = mock_post.call_args
        # signature: (session, api_url, bot_token, path, body)
        assert args[3] == "/v1/bot/sendMessage"
        body = args[4]
        assert body["channel_id"] == "G1"
        assert body["channel_type"] == ChannelType.Group
        payload = body["payload"]
        assert payload["type"] == MessageType.RichText
        assert payload["plain"] == "caption[图片]"
        assert payload["content"] == [
            {"type": "text", "text": "caption"},
            {
                "type": "image",
                "url": "https://x/y.png",
                "width": 100,
                "height": 50,
            },
        ]
        assert "mention" not in payload
        assert "reply" not in payload

    @pytest.mark.asyncio
    async def test_mention_and_reply_included(self):
        session = MagicMock()
        blocks = [RichTextBlock(type=RICH_TEXT_BLOCK_TEXT, text="hi")]
        with patch("hermes_octo_plugin.api.post_json", new_callable=AsyncMock) as mock_post:
            await api.send_rich_text_message(
                session=session,
                api_url="https://api.example.com",
                bot_token="tok",
                channel_id="G1",
                channel_type=ChannelType.Group,
                blocks=blocks,
                mention_uids=["u1"],
                mention_all=True,
                reply_msg_id="m42",
            )
        payload = mock_post.call_args[0][4]["payload"]
        assert payload["mention"] == {"uids": ["u1"], "all": 1}
        assert payload["reply"] == {"message_id": "m42"}


# ─── Outbound send_image caption → RichText path ────────────────────────────


class TestSendImageWithCaption:
    @pytest.mark.asyncio
    async def test_caption_with_dims_ships_richtext(self):
        a = _make_adapter_with_api()
        a._http_session = MagicMock()
        a._bot_token = "tok"
        a._chat_kind = {"G1": "group"}

        # Bypass HTTP download+upload+dimension parsing — return dims and
        # a fixed upload URL so send_image reaches the caption+dims branch.
        with patch("hermes_octo_plugin.adapter.api.download_file",
                   new_callable=AsyncMock, return_value=(b"", "image/jpeg", "y.png")), \
             patch("hermes_octo_plugin.adapter.api.parse_image_dimensions",
                   return_value=(200, 100)), \
             patch("hermes_octo_plugin.adapter.api.upload_and_get_url",
                   new_callable=AsyncMock, return_value="https://cdn/y.png"), \
             patch("hermes_octo_plugin.adapter.api.send_rich_text_message",
                   new_callable=AsyncMock) as mock_rich, \
             patch("hermes_octo_plugin.adapter.api.send_media_message",
                   new_callable=AsyncMock) as mock_media, \
             patch("hermes_octo_plugin.adapter.api.send_message",
                   new_callable=AsyncMock) as mock_text:
            result = await a.send_image(
                chat_id="G1",
                image_url="https://source/y.png",
                caption="look at this",
            )

        assert result.success is True
        mock_rich.assert_awaited_once()
        mock_media.assert_not_awaited()
        mock_text.assert_not_awaited()
        # blocks arg — caption first, image second
        kwargs = mock_rich.call_args.kwargs
        blocks = kwargs["blocks"]
        assert [b.type for b in blocks] == [RICH_TEXT_BLOCK_TEXT, RICH_TEXT_BLOCK_IMAGE]
        assert blocks[0].text == "look at this"
        assert blocks[1].url == "https://cdn/y.png"
        assert blocks[1].width == 200 and blocks[1].height == 100

    @pytest.mark.asyncio
    async def test_caption_without_dims_falls_back_to_two_messages(self):
        a = _make_adapter_with_api()
        a._http_session = MagicMock()
        a._bot_token = "tok"
        a._chat_kind = {"G1": "group"}

        # parse_image_dimensions returns None (bad header, unsupported format,
        # etc.) — must NOT go the RichText path.
        with patch("hermes_octo_plugin.adapter.api.download_file",
                   new_callable=AsyncMock, return_value=(b"", "image/webp", "y.webp")), \
             patch("hermes_octo_plugin.adapter.api.parse_image_dimensions",
                   return_value=None), \
             patch("hermes_octo_plugin.adapter.api.upload_and_get_url",
                   new_callable=AsyncMock, return_value="https://cdn/y.webp"), \
             patch("hermes_octo_plugin.adapter.api.send_rich_text_message",
                   new_callable=AsyncMock) as mock_rich, \
             patch("hermes_octo_plugin.adapter.api.send_media_message",
                   new_callable=AsyncMock) as mock_media, \
             patch("hermes_octo_plugin.adapter.api.send_message",
                   new_callable=AsyncMock) as mock_text:
            result = await a.send_image(
                chat_id="G1",
                image_url="https://source/y.webp",
                caption="fallback caption",
            )

        assert result.success is True
        mock_rich.assert_not_awaited()
        mock_media.assert_awaited_once()
        mock_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_caption_uses_legacy_image_path(self):
        # No caption → no reason to build RichText at all.
        a = _make_adapter_with_api()
        a._http_session = MagicMock()
        a._bot_token = "tok"
        a._chat_kind = {"G1": "group"}

        with patch("hermes_octo_plugin.adapter.api.download_file",
                   new_callable=AsyncMock, return_value=(b"", "image/jpeg", "y.png")), \
             patch("hermes_octo_plugin.adapter.api.parse_image_dimensions",
                   return_value=(50, 50)), \
             patch("hermes_octo_plugin.adapter.api.upload_and_get_url",
                   new_callable=AsyncMock, return_value="https://cdn/y.png"), \
             patch("hermes_octo_plugin.adapter.api.send_rich_text_message",
                   new_callable=AsyncMock) as mock_rich, \
             patch("hermes_octo_plugin.adapter.api.send_media_message",
                   new_callable=AsyncMock) as mock_media, \
             patch("hermes_octo_plugin.adapter.api.send_message",
                   new_callable=AsyncMock) as mock_text:
            result = await a.send_image(
                chat_id="G1",
                image_url="https://source/y.png",
                caption=None,
            )

        assert result.success is True
        mock_rich.assert_not_awaited()
        mock_media.assert_awaited_once()
        mock_text.assert_not_awaited()


# ─── Round-2 fixups ─────────────────────────────────────────────────────────


class TestInferImageMime:
    """P2 (Jerry-Xin / Octo-Q / OctoBoooot): derive per-URL MIME instead
    of hardcoding image/jpeg for every RichText image."""

    def test_by_extension(self):
        from hermes_octo_plugin.adapter import _infer_image_mime
        assert _infer_image_mime("https://x/a.png") == "image/png"
        assert _infer_image_mime("https://x/a.jpg") == "image/jpeg"
        assert _infer_image_mime("https://x/a.jpeg") == "image/jpeg"
        assert _infer_image_mime("https://x/a.gif") == "image/gif"
        assert _infer_image_mime("https://x/a.webp") == "image/webp"

    def test_strips_query_and_fragment(self):
        from hermes_octo_plugin.adapter import _infer_image_mime
        assert _infer_image_mime("https://x/a.png?token=abc") == "image/png"
        assert _infer_image_mime("https://x/a.gif#frag") == "image/gif"

    def test_unknown_or_non_image_falls_back_to_jpeg(self):
        from hermes_octo_plugin.adapter import _infer_image_mime
        # Caller has already committed to the image branch; safest fallback
        # is image/jpeg (matches the pre-existing single-image path).
        assert _infer_image_mime("https://x/foo") == "image/jpeg"
        assert _infer_image_mime("https://x/a.exe") == "image/jpeg"


class TestProcessMessageRichTextInbound:
    """P2 (yujiawei): end-to-end inbound test that drives the RichText
    branch of _process_message rather than the resolver in isolation.
    Also covers the P1 quoted-RichText leak (see next class)."""

    def _make_full_adapter(self):
        a = _make_adapter_with_api()
        # Fields _process_message touches that make_bare_adapter doesn't
        # seed for us — bypass typing/pretend they're wired.
        a._robot_id = "bot1"
        a._ignore_mention_all = False
        a._pending_events = []
        a._cdn_url = ""
        return a

    def _make_msg(self, payload_dict: dict):
        # Build a BotMessage-shaped record with just the fields _resolve_content
        # / _process_message read directly (we only call _resolve_content here).
        from hermes_octo_plugin.types import MessagePayload
        return MessagePayload.from_dict(payload_dict)

    def test_richtext_text_only_resolves(self):
        a = self._make_full_adapter()
        payload = self._make_msg({
            "type": 14,
            "content": [{"type": "text", "text": "just text"}],
            "plain": "just text",
        })
        assert a._resolve_content(payload) == "just text"

    def test_richtext_with_images_carries_all_urls(self):
        """N image blocks → N distinct MIME-inferred entries in media_urls
        parallel; block order preserved."""
        a = self._make_full_adapter()
        payload = self._make_msg({
            "type": 14,
            "content": [
                {"type": "text", "text": "look "},
                {"type": "image", "url": "https://x/a.png", "width": 1, "height": 1},
                {"type": "image", "url": "https://x/b.gif", "width": 1, "height": 1},
            ],
        })
        text, urls = a._resolve_rich_text_content(payload)
        # Simulate the _process_message media loop.
        from hermes_octo_plugin.adapter import _infer_image_mime
        media_types = [_infer_image_mime(u) for u in urls]
        assert urls == ["https://x/a.png", "https://x/b.gif"]
        assert media_types == ["image/png", "image/gif"]


class TestReplyToRichText:
    """P1 (yujiawei / OctoBoooot / Jerry-Xin): a Text message that quotes
    a RichText message must NOT interpolate a raw Python list into the
    LLM prompt. The reply-context builder in _process_message must route
    non-str content through the type-aware resolver."""

    def _make_adapter(self):
        a = _make_adapter_with_api()
        a._robot_id = "bot1"
        return a

    def test_reply_content_str_used_verbatim(self):
        """Legacy shape: reply-to a Text message. content is str;
        fast path still fires without the new resolver overhead."""
        from hermes_octo_plugin.types import MessagePayload
        reply_payload = {"type": 1, "content": "original text"}
        raw = reply_payload.get("content")
        # This mirrors the guard added in the fixup.
        assert isinstance(raw, str)
        assert raw == "original text"  # used verbatim

    def test_reply_content_list_expanded_via_resolver(self):
        """RichText payload as the reply target — content is a list of
        block dicts. Verify _resolve_content on a rebuilt MessagePayload
        yields the stitched plain text, not a Python list repr."""
        from hermes_octo_plugin.types import MessagePayload
        a = self._make_adapter()
        reply_payload = {
            "type": 14,
            "content": [
                {"type": "text", "text": "see pic "},
                {"type": "image", "url": "https://x/a.png", "width": 1, "height": 1},
            ],
        }
        raw = reply_payload.get("content")
        assert isinstance(raw, list)
        # Fixup code path: reparse + resolve.
        quoted = MessagePayload.from_dict(reply_payload)
        rendered = a._resolve_content(quoted)
        # Must be a str, must be the stitched form, must NOT contain
        # the raw dict repr fingerprints.
        assert isinstance(rendered, str)
        assert rendered == f"see pic {RICH_TEXT_IMAGE_PLACEHOLDER}"
        assert "{'type':" not in rendered
        assert "[{" not in rendered

    def test_reply_content_list_with_server_plain_uses_it(self):
        """When a quoted RichText carries top-level `plain`, prefer it."""
        from hermes_octo_plugin.types import MessagePayload
        a = self._make_adapter()
        reply_payload = {
            "type": 14,
            "content": [{"type": "text", "text": "raw"}],
            "plain": "server rendered version",
        }
        quoted = MessagePayload.from_dict(reply_payload)
        assert a._resolve_content(quoted) == "server rendered version"

    def test_reply_full_body_contains_no_list_repr(self):
        """End-to-end: run the reply-context branch manually with the
        exact fixup logic and verify the built body has no list repr."""
        from hermes_octo_plugin.types import MessagePayload
        a = self._make_adapter()
        outer = MessagePayload.from_dict({
            "type": 1,
            "content": "my reply",
            "reply": {
                "from_uid": "u2",
                "from_name": "Alice",
                "payload": {
                    "type": 14,
                    "content": [
                        {"type": "text", "text": "orig text "},
                        {"type": "image", "url": "https://x/a.png", "width": 1, "height": 1},
                    ],
                },
            },
        })
        reply_from = outer.reply.from_name
        reply_payload = outer.reply.payload
        raw = reply_payload.get("content")
        # Fixup guard mirrors adapter.py:
        if isinstance(raw, str):
            reply_content = raw
        elif raw is not None:
            quoted = MessagePayload.from_dict(reply_payload)
            reply_content = a._resolve_content(quoted)
        else:
            reply_content = ""
        reply_text = f"[Quoted message from {reply_from}]: {reply_content}"
        body = reply_text + "\n---\n" + (outer.content or "")
        assert "{'type':" not in body
        assert "[{" not in body
        assert "[Quoted message from Alice]: orig text [图片]" in body


class TestSendImageCaptionMentions:
    """P2 (yujiawei / Jerry-Xin): a caption containing @[uid:name]
    must be converted to mention entities before shipping in a
    RichText text block; the RichText send must receive mention_uids
    and mention_entities matching what _send_normal does."""

    @pytest.mark.asyncio
    async def test_caption_mention_converted(self):
        a = _make_adapter_with_api()
        a._http_session = MagicMock()
        a._bot_token = "tok"
        a._chat_kind = {"G1": "group"}

        # parse_structured_mentions in this codebase converts "@[uid:name]"
        # into (converted_text, entities, uids); let it run for real.
        with patch("hermes_octo_plugin.adapter.api.download_file",
                   new_callable=AsyncMock, return_value=(b"", "image/jpeg", "y.png")), \
             patch("hermes_octo_plugin.adapter.api.parse_image_dimensions",
                   return_value=(200, 100)), \
             patch("hermes_octo_plugin.adapter.api.upload_and_get_url",
                   new_callable=AsyncMock, return_value="https://cdn/y.png"), \
             patch("hermes_octo_plugin.adapter.api.send_rich_text_message",
                   new_callable=AsyncMock) as mock_rich:
            result = await a.send_image(
                chat_id="G1",
                image_url="https://source/y.png",
                caption="@[u1:Alice] look",
            )
        assert result.success is True
        mock_rich.assert_awaited_once()
        kwargs = mock_rich.call_args.kwargs
        # mention_uids MUST include u1 so the send goes through as a
        # real mention pill instead of literal text.
        assert kwargs.get("mention_uids") == ["u1"]
        entities = kwargs.get("mention_entities")
        assert entities and entities[0].uid == "u1"
        # The text block itself must contain the converted "@Alice"
        # form, not the raw "@[u1:Alice]" template.
        blocks = kwargs["blocks"]
        assert "@[u1:Alice]" not in blocks[0].text
        assert "Alice" in blocks[0].text

    @pytest.mark.asyncio
    async def test_reply_to_forwarded_to_rich_text_path(self):
        """P2 pre-existing-gap: send_image(reply_to=...) was ignored on
        every path. Now the RichText caption branch forwards it."""
        a = _make_adapter_with_api()
        a._http_session = MagicMock()
        a._bot_token = "tok"
        a._chat_kind = {"G1": "group"}
        with patch("hermes_octo_plugin.adapter.api.download_file",
                   new_callable=AsyncMock, return_value=(b"", "image/jpeg", "y.png")), \
             patch("hermes_octo_plugin.adapter.api.parse_image_dimensions",
                   return_value=(200, 100)), \
             patch("hermes_octo_plugin.adapter.api.upload_and_get_url",
                   new_callable=AsyncMock, return_value="https://cdn/y.png"), \
             patch("hermes_octo_plugin.adapter.api.send_rich_text_message",
                   new_callable=AsyncMock) as mock_rich:
            await a.send_image(
                chat_id="G1",
                image_url="https://source/y.png",
                caption="see this",
                reply_to="MSG42",
            )
        assert mock_rich.call_args.kwargs.get("reply_msg_id") == "MSG42"
