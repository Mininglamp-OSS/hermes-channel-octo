"""
Tests for hermes_octo_plugin.adapter — adapter initialization and config parsing.
"""

import pytest
from unittest.mock import MagicMock
from hermes_octo_plugin.adapter import (
    LRUCache,
    check_octo_requirements,
    MAX_MESSAGE_LENGTH,
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_HISTORY_PROMPT_TEMPLATE,
)
from hermes_octo_plugin.types import MessagePayload, MessageType
from tests.conftest import make_bare_adapter


class TestLRUCache:
    def test_set_and_get(self):
        cache = LRUCache(max_size=3)
        cache.set("a", "1")
        assert cache.get("a") == "1"

    def test_miss_returns_none(self):
        cache = LRUCache(max_size=3)
        assert cache.get("nonexistent") is None

    def test_eviction(self):
        cache = LRUCache(max_size=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == "2"
        assert cache.get("c") == "3"

    def test_access_refreshes_order(self):
        cache = LRUCache(max_size=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.get("a")  # Access "a" to refresh it
        cache.set("c", "3")  # Should evict "b" (not "a")
        assert cache.get("a") == "1"
        assert cache.get("b") is None
        assert cache.get("c") == "3"

    def test_update_existing(self):
        cache = LRUCache(max_size=3)
        cache.set("a", "1")
        cache.set("a", "2")
        assert cache.get("a") == "2"
        assert len(cache) == 1

    def test_contains(self):
        cache = LRUCache(max_size=3)
        cache.set("a", "1")
        assert "a" in cache
        assert "b" not in cache

    def test_len(self):
        cache = LRUCache(max_size=10)
        assert len(cache) == 0
        cache.set("a", "1")
        cache.set("b", "2")
        assert len(cache) == 2


class TestOctoAdapterConfig:
    def _make_config(self, **extra):
        config = MagicMock()
        config.extra = {
            "api_url": "https://api.example.com",
            "bot_token": "test-token-123",
            **extra,
        }
        config.token = "test-token-123"
        return config

    def test_config_defaults(self):
        """Verify default configuration values."""
        assert MAX_MESSAGE_LENGTH == 5000
        assert DEFAULT_HISTORY_LIMIT == 20
        assert "{count}" in DEFAULT_HISTORY_PROMPT_TEMPLATE
        assert "{messages}" in DEFAULT_HISTORY_PROMPT_TEMPLATE


class TestResolveContent:
    """Test the _resolve_content method using a mock adapter."""

    def test_text_message(self):
        payload = MessagePayload(type=MessageType.Text, content="hello world")
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert result == "hello world"

    def test_image_message(self):
        payload = MessagePayload(type=MessageType.Image, url="https://example.com/img.png")
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert "[图片]" in result
        assert "https://example.com/img.png" in result

    def test_voice_message(self):
        payload = MessagePayload(type=MessageType.Voice, url="https://example.com/voice.ogg")
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert "[语音消息]" in result

    def test_file_message(self):
        payload = MessagePayload(type=MessageType.File, name="doc.pdf", url="https://example.com/doc.pdf")
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert "[文件: doc.pdf]" in result

    def test_video_message(self):
        payload = MessagePayload(type=MessageType.Video, url="https://example.com/video.mp4")
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert "[视频]" in result

    def test_location_message(self):
        payload = MessagePayload(type=MessageType.Location)
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert "[位置信息]" in result

    def test_card_message(self):
        payload = MessagePayload(type=MessageType.Card, name="Alice")
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert "[名片: Alice]" in result

    def test_empty_text(self):
        payload = MessagePayload(type=MessageType.Text, content="")
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert result == ""

    def test_forward_message(self):
        payload = MessagePayload(type=MessageType.MultipleForward)
        adapter = make_bare_adapter()
        result = adapter._resolve_content(payload)
        assert "[合并转发]" in result


class TestCheckOctoRequirements:
    def test_deps_available(self):
        # check_fn reflects runtime dep availability only — not user config.
        # The octo extra is installed in this test env, so this must be True
        # regardless of OCTO_API_URL / OCTO_BOT_TOKEN.
        assert check_octo_requirements() is True


class TestHistoryRecording:
    """Test the group history recording logic."""

    def test_record_history_entry(self):
        adapter = make_bare_adapter()
        adapter._history_limit = 5

        adapter._record_history_entry("group1", "user1", "hello")
        adapter._record_history_entry("group1", "user2", "world")

        assert len(adapter._group_histories["group1"]) == 2
        assert adapter._group_histories["group1"][0]["sender"] == "user1"
        assert adapter._group_histories["group1"][1]["body"] == "world"

    def test_history_limit(self):
        adapter = make_bare_adapter()
        adapter._history_limit = 3

        for i in range(10):
            adapter._record_history_entry("group1", f"user{i}", f"msg{i}")

        assert len(adapter._group_histories["group1"]) == 3
        # Should keep the last 3
        assert adapter._group_histories["group1"][0]["body"] == "msg7"


class TestGroupMdHandling:
    """Test GROUP.md event handling."""

    def test_handle_group_md_deleted(self):
        adapter = make_bare_adapter()
        adapter._group_md_cache = {"group1": {"content": "test", "version": 1}}
        adapter._group_md_checked = {"group1"}

        adapter._handle_group_md_event("group1", "group_md_deleted")

        assert "group1" not in adapter._group_md_cache
        assert "group1" not in adapter._group_md_checked

    def test_handle_group_md_updated(self):
        adapter = make_bare_adapter()
        adapter._group_md_cache = {"group1": {"content": "old", "version": 1}}
        adapter._group_md_checked = {"group1"}

        adapter._handle_group_md_event("group1", "group_md_updated")

        # Should force re-fetch
        assert "group1" not in adapter._group_md_checked


class TestMdDirPathValidation:
    """Path-segment validation defends against an Octo server returning a
    crafted group_no / short_id that escapes the workspace cache root."""

    def test_md_dir_rejects_path_traversal_in_key(self):
        from hermes_octo_plugin.adapter import _validate_octo_path_segment
        with pytest.raises(ValueError):
            _validate_octo_path_segment("../../etc/passwd", "group_key")

    def test_md_dir_returns_none_on_malformed_key(self, monkeypatch):
        adapter = make_bare_adapter()
        adapter._owner_uid = "owner1"

        # Stub get_hermes_home so _md_dir doesn't short-circuit on None home.
        import hermes_constants
        monkeypatch.setattr(
            hermes_constants, "get_hermes_home", lambda: "/tmp/octo-test-home",
            raising=False,
        )

        # Malformed key with traversal characters must be refused.
        assert adapter._md_dir("../../etc/passwd") is None
        # Well-formed key still works.
        assert adapter._md_dir("group123") is not None
