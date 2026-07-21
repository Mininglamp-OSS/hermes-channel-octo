import json
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from hermes_octo_plugin import api
from hermes_octo_plugin.agent_tools import (
    OWNER_ONLY_ACTIONS,
    _confine_secret_path,
    _handle_write_secret,
)


PLAINTEXT = "sk-test-plaintext"


class _Resp:
    def __init__(self, status: int, payload=None, ok: bool | None = None):
        self.status = status
        self.ok = (200 <= status < 300) if ok is None else ok
        self._payload = payload
        self.request_info = None
        self.history = ()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def json(self, content_type=None):
        return self._payload


class _Session:
    def __init__(self, resp: _Resp):
        self.resp = resp
        self.post_calls = []

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        return self.resp


class TestResolveSecret:
    @pytest.mark.asyncio
    async def test_resolved_current_contract(self):
        session = _Session(_Resp(200, {"secret_id": "sec_1", "value": PLAINTEXT}))

        result = await api.resolve_secret(session, "https://api.example", "tok", "openai key")

        assert result == {"status": "resolved", "value": PLAINTEXT, "secret_id": "sec_1"}
        args, kwargs = session.post_calls[0]
        assert args[0] == "https://api.example/v1/bot/secrets/resolve"
        assert kwargs["json"] == {"query": "openai key"}
        assert kwargs["headers"]["Authorization"] == "Bearer tok"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status,expected", [(404, "not_found"), (429, "rate_limited")])
    async def test_status_outcomes_without_reading_body(self, status, expected):
        session = _Session(_Resp(status, {"value": PLAINTEXT}, ok=False))

        result = await api.resolve_secret(session, "https://api.example", "tok", "missing")

        assert result == {"status": expected}

    @pytest.mark.asyncio
    async def test_ambiguous_copies_labels_only(self):
        session = _Session(_Resp(422, {
            "error": {"details": {"candidates": [
                {"display_name": "openai prod", "secret_id": "a", "value": PLAINTEXT, "masked": "sk-***"},
                {"display_name": "", "secret_id": "dropped"},
            ]}}
        }, ok=False))

        result = await api.resolve_secret(session, "https://api.example", "tok", "openai")

        assert result == {
            "status": "ambiguous",
            "candidates": [{"display_name": "openai prod", "secret_id": "a"}],
        }
        assert PLAINTEXT not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_500_error_does_not_echo_body(self):
        session = _Session(_Resp(500, {"value": PLAINTEXT}, ok=False))

        with pytest.raises(Exception) as exc:
            await api.resolve_secret(session, "https://api.example", "tok", "openai")

        assert "500" in str(exc.value)
        assert PLAINTEXT not in str(exc.value)


class TestWriteSecret:
    @pytest.fixture(autouse=True)
    def secrets_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OCTO_SECRETS_FILE_ROOT", str(tmp_path))
        return tmp_path

    @pytest.mark.asyncio
    async def test_owner_only_action_registered(self):
        assert "write-secret" in OWNER_ONLY_ACTIONS

    @pytest.mark.asyncio
    async def test_resolves_and_writes_without_returning_plaintext(self, secrets_root, monkeypatch):
        resolve = AsyncMock(return_value={
            "status": "resolved",
            "value": PLAINTEXT,
            "secret_id": "sec_1",
            "display_name": "openai key",
        })
        monkeypatch.setattr(api, "resolve_secret", resolve)

        out = await _handle_write_secret(
            None,
            "https://api.example",
            "tok",
            alias="openai key",
            file_path="secrets/.env",
            template="OPENAI_API_KEY={{secret}}\n",
            mode="overwrite",
        )
        data = json.loads(out)

        assert data["ok"] is True
        assert data["data"] == {
            "written": True,
            "path": os.path.join("secrets", ".env"),
            "mode": "overwrite",
            "display_name": "openai key",
        }
        assert PLAINTEXT not in out
        assert (secrets_root / "secrets" / ".env").read_text(encoding="utf-8") == f"OPENAI_API_KEY={PLAINTEXT}\n"
        mode = stat.S_IMODE((secrets_root / "secrets" / ".env").stat().st_mode)
        assert mode == 0o600
        resolve.assert_awaited_once_with(None, "https://api.example", "tok", "openai key")

    @pytest.mark.asyncio
    async def test_append_mode(self, secrets_root, monkeypatch):
        monkeypatch.setattr(api, "resolve_secret", AsyncMock(return_value={"status": "resolved", "value": PLAINTEXT}))
        target = secrets_root / ".env"
        target.write_text("EXISTING=1\n", encoding="utf-8")

        out = await _handle_write_secret(
            None, "https://api.example", "tok",
            alias="k", file_path=".env", template="KEY={{secret}}\n", mode="append",
        )

        assert json.loads(out)["data"]["mode"] == "append"
        assert target.read_text(encoding="utf-8") == f"EXISTING=1\nKEY={PLAINTEXT}\n"

    @pytest.mark.asyncio
    async def test_template_without_placeholder_rejected_before_resolve(self, monkeypatch):
        resolve = AsyncMock(return_value={"status": "resolved", "value": PLAINTEXT})
        monkeypatch.setattr(api, "resolve_secret", resolve)

        out = await _handle_write_secret(
            None, "https://api.example", "tok",
            alias="k", file_path=".env", template="OPENAI_API_KEY=", mode="overwrite",
        )

        assert "{{secret}} placeholder" in json.loads(out)["error"]
        resolve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_path_escape_rejected_before_resolve(self, monkeypatch):
        resolve = AsyncMock(return_value={"status": "resolved", "value": PLAINTEXT})
        monkeypatch.setattr(api, "resolve_secret", resolve)

        out = await _handle_write_secret(
            None, "https://api.example", "tok",
            alias="k", file_path="../escape.txt", mode="overwrite",
        )

        assert "outside the allowed directory" in json.loads(out)["error"]
        assert PLAINTEXT not in out
        resolve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_symlink_escape_rejected_before_resolve(self, secrets_root, tmp_path, monkeypatch):
        outside = Path(tempfile.mkdtemp(prefix="octo-secret-outside-"))
        (secrets_root / "link").symlink_to(outside, target_is_directory=True)
        resolve = AsyncMock(return_value={"status": "resolved", "value": PLAINTEXT})
        monkeypatch.setattr(api, "resolve_secret", resolve)

        out = await _handle_write_secret(
            None, "https://api.example", "tok",
            alias="k", file_path="link/key.txt", mode="overwrite",
        )

        assert "outside the allowed directory" in json.loads(out)["error"] or "symlink" in json.loads(out)["error"]
        assert not (outside / "key.txt").exists()
        resolve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ambiguous_result_is_label_only(self, monkeypatch):
        monkeypatch.setattr(api, "resolve_secret", AsyncMock(return_value={
            "status": "ambiguous",
            "candidates": [{"display_name": "openai prod", "secret_id": "a"}],
        }))

        out = await _handle_write_secret(
            None, "https://api.example", "tok",
            alias="openai", file_path=".env", mode="overwrite",
        )
        data = json.loads(out)

        assert data["ok"] is True
        assert data["data"]["ambiguous"] is True
        assert PLAINTEXT not in out
