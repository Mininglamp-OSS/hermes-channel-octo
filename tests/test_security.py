"""Security boundary tests for octo plugin.

Covers:
  - C2 / NEW-I1: ``_validate_octo_url`` SSRF guard (scheme, host, private/metadata IPs).
  - C2: ``_OCTO_CHAT_ID_RE`` chat_id character / length validation.
  - S6 / NEW-I3: ``redact_log`` always pulls force=True from agent.redact so
    secrets are masked even when HERMES_REDACT_SECRETS is disabled.
"""

import os
from unittest import mock

import pytest

from hermes_octo_plugin.adapter import (
    _OCTO_CHAT_ID_RE,
    _is_private_or_metadata_host,
    _validate_octo_url,
    redact_log,
)


# ─── NEW-I1: _validate_octo_url ─────────────────────────────────────────────

class TestValidateOctoUrl:
    @pytest.mark.parametrize("url", [
        "https://api.octo.example.com/",
        "http://api.octo.example.com:8080/v1",
        "https://octo.example.cn",
    ])
    def test_accepts_public_http_https(self, url):
        assert _validate_octo_url(url, "OCTO_API_URL") == url

    @pytest.mark.parametrize("url", ["ftp://octo.example/", "file:///etc/passwd", "javascript:alert(1)"])
    def test_rejects_non_http_scheme(self, url):
        with pytest.raises(ValueError, match="must be http"):
            _validate_octo_url(url, "OCTO_API_URL")

    def test_rejects_missing_host(self):
        with pytest.raises(ValueError, match="missing host"):
            _validate_octo_url("http://", "OCTO_API_URL")

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1/",
        "http://localhost:8080/",
        "https://10.0.0.1/v1",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://octo.internal/",
        "http://[::1]/",
    ])
    def test_rejects_private_and_metadata(self, url):
        # Make sure OCTO_ALLOW_PRIVATE_HOSTS is not set so the SSRF guard runs.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCTO_ALLOW_PRIVATE_HOSTS", None)
            with pytest.raises(ValueError, match="SSRF guard"):
                _validate_octo_url(url, "OCTO_API_URL")

    def test_allow_private_hosts_env_bypass(self):
        with mock.patch.dict(os.environ, {"OCTO_ALLOW_PRIVATE_HOSTS": "true"}):
            assert _validate_octo_url("http://127.0.0.1/", "OCTO_API_URL") == "http://127.0.0.1/"

    def test_error_message_names_the_env_var(self):
        # Operators reading the error need to know which env var to fix.
        try:
            _validate_octo_url("ftp://x/", "OCTO_CDN_URL")
        except ValueError as e:
            assert "OCTO_CDN_URL" in str(e)
        else:
            pytest.fail("expected ValueError")


class TestIsPrivateOrMetadataHost:
    @pytest.mark.parametrize("host", [
        "127.0.0.1", "::1", "localhost", "10.20.30.40",
        "172.16.0.1", "192.168.10.10", "169.254.169.254",
        "metadata.google.internal", "metadata", "100.100.100.200",
        "foo.local", "bar.internal", "",
    ])
    def test_blocks(self, host):
        assert _is_private_or_metadata_host(host) is True

    @pytest.mark.parametrize("host", [
        "api.octo.example.com", "1.1.1.1", "8.8.8.8", "octo.cn",
    ])
    def test_allows_public(self, host):
        assert _is_private_or_metadata_host(host) is False


# ─── C2: _OCTO_CHAT_ID_RE ───────────────────────────────────────────────────

class TestChatIdRegex:
    @pytest.mark.parametrize("cid", [
        "user_abc123", "group-42", "uid@octo", "chan:topic1",
        "A" * 128,  # exactly at the boundary
    ])
    def test_accepts_legal(self, cid):
        assert _OCTO_CHAT_ID_RE.fullmatch(cid) is not None

    @pytest.mark.parametrize("cid", [
        "",                       # empty
        "A" * 129,                # over limit
        "../etc/passwd",          # path traversal
        "user'; DROP TABLE",      # SQL meta
        "<script>alert(1)",       # HTML
        "user space",             # whitespace
        "user\nwithnewline",      # CRLF injection
        "user\x00null",           # NUL byte
        "user%2Fpath",            # URL-encoded slash
    ])
    def test_rejects_illegal(self, cid):
        assert _OCTO_CHAT_ID_RE.fullmatch(cid) is None


# ─── S6 / NEW-I3: redact_log forces redaction ───────────────────────────────

class TestRedactLog:
    def test_redacts_bearer_token(self):
        s = "request failed: Authorization: Bearer sk-abc123xyz789defgh"
        out = redact_log(s)
        # Either the raw secret is masked, or redact module fell back to noop
        # (ImportError branch). The latter only triggers when agent.redact is
        # not installed — in our env it is, so this should mask.
        assert "sk-abc123xyz789defgh" not in out

    def test_redacts_when_global_disabled(self):
        # NEW-I3: even when HERMES_REDACT_SECRETS=false the octo wrapper must
        # still redact because it calls _redact_raw(..., force=True).
        with mock.patch.dict(os.environ, {"HERMES_REDACT_SECRETS": "false"}):
            out = redact_log("Authorization: Bearer sk-leaked-token-abc12345")
            assert "sk-leaked-token-abc12345" not in out

    def test_passthrough_on_benign_text(self):
        s = "octo: connection refused after 3 retries"
        assert redact_log(s) == s
