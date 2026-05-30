"""Tests for Batch 1: MultipleForward expansion + inbound file inline/download."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_octo_plugin.adapter import OctoAdapter, _format_size
from hermes_octo_plugin.types import MessagePayload, MessageType
from tests.conftest import make_bare_adapter


def _make_adapter() -> OctoAdapter:
    a = make_bare_adapter()
    a._api_url = "https://example.test"
    a._bot_token = "tok"
    return a


# ─── _format_size ────────────────────────────────────────────────────────────


class TestFormatSize:
    def test_none(self):
        assert _format_size(None) == "?"

    def test_bytes(self):
        assert "B" in _format_size(500)

    def test_kilobytes(self):
        s = _format_size(2048)
        assert "KB" in s and "2" in s

    def test_megabytes(self):
        s = _format_size(5 * 1024 * 1024)
        assert "MB" in s


# ─── MultipleForward expansion ───────────────────────────────────────────────


class TestMultipleForwardExpansion:
    def test_simple_forward_renders_each_inner_message(self):
        a = _make_adapter()
        payload = MessagePayload(
            type=MessageType.MultipleForward,
            extra={
                "users": [
                    {"uid": "u1", "name": "Alice"},
                    {"uid": "u2", "name": "Bob"},
                ],
                "msgs": [
                    {"from_uid": "u1", "payload": {"type": int(MessageType.Text), "content": "hello"}},
                    {"from_uid": "u2", "payload": {"type": int(MessageType.Text), "content": "world"}},
                ],
            },
        )
        out = a._resolve_multiple_forward_text(payload)
        assert "[合并转发: 聊天记录]" in out
        assert "Alice: hello" in out
        assert "Bob: world" in out

    def test_unknown_sender_falls_back_to_uid(self):
        a = _make_adapter()
        payload = MessagePayload(
            type=MessageType.MultipleForward,
            extra={
                "users": [],
                "msgs": [
                    {"from_uid": "u_ghost", "payload": {"type": int(MessageType.Text), "content": "hi"}},
                ],
            },
        )
        out = a._resolve_multiple_forward_text(payload)
        assert "u_ghost: hi" in out

    def test_inner_image_renders_with_url(self):
        a = _make_adapter()
        payload = MessagePayload(
            type=MessageType.MultipleForward,
            extra={
                "users": [{"uid": "u1", "name": "Alice"}],
                "msgs": [
                    {"from_uid": "u1", "payload": {"type": int(MessageType.Image), "url": "file/abc.png"}},
                ],
            },
        )
        out = a._resolve_multiple_forward_text(payload)
        assert "[图片]" in out
        assert "abc.png" in out  # Full URL appended

    def test_inner_file_renders_with_name_and_url(self):
        a = _make_adapter()
        payload = MessagePayload(
            type=MessageType.MultipleForward,
            extra={
                "users": [{"uid": "u1", "name": "Alice"}],
                "msgs": [
                    {"from_uid": "u1", "payload": {"type": int(MessageType.File), "url": "file/doc.pdf", "name": "report.pdf"}},
                ],
            },
        )
        out = a._resolve_multiple_forward_text(payload)
        assert "[文件: report.pdf]" in out
        assert "doc.pdf" in out

    def test_nested_forward_recurses(self):
        a = _make_adapter()
        inner_forward = {
            "from_uid": "u1",
            "payload": {
                "type": int(MessageType.MultipleForward),
                "users": [{"uid": "u2", "name": "Bob"}],
                "msgs": [
                    {"from_uid": "u2", "payload": {"type": int(MessageType.Text), "content": "nested msg"}},
                ],
            },
        }
        payload = MessagePayload(
            type=MessageType.MultipleForward,
            extra={
                "users": [{"uid": "u1", "name": "Alice"}],
                "msgs": [inner_forward],
            },
        )
        out = a._resolve_multiple_forward_text(payload)
        assert "Alice: [合并转发]" in out
        assert "Bob: nested msg" in out

    def test_users_pollinate_uid_to_name_map(self):
        a = _make_adapter()
        assert a._uid_to_name == {}
        payload = MessagePayload(
            type=MessageType.MultipleForward,
            extra={
                "users": [{"uid": "new_u", "name": "Carol"}],
                "msgs": [],
            },
        )
        a._resolve_multiple_forward_text(payload)
        assert a._uid_to_name == {"new_u": "Carol"}

    def test_resolve_content_dispatches_to_forward_expander(self):
        a = _make_adapter()
        payload = MessagePayload(
            type=MessageType.MultipleForward,
            extra={
                "users": [{"uid": "u1", "name": "Alice"}],
                "msgs": [
                    {"from_uid": "u1", "payload": {"type": int(MessageType.Text), "content": "from resolve_content"}},
                ],
            },
        )
        out = a._resolve_content(payload)
        assert "Alice: from resolve_content" in out
