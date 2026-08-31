"""Regression tests: an expected "bot is not a group member" refusal must not
be logged as an ERROR.

A bot that gets @mentioned in a group it has not joined triggers a deliberate
server-side refusal on every single such mention.  That is an expected
condition, but it used to be logged at ERROR, which buried genuine faults.

The load-bearing detail is *how* the condition is recognised.  The server
returns HTTP **400** and reports the actual condition in a machine-readable
``error.code`` field in the body.  Matching on the HTTP status would be wrong
twice over: 403 never appears at this layer, and 400 is far too broad to
downgrade wholesale.  These tests pin the code-based judgement so a future
refactor cannot silently regress to a status-based one.
"""

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hermes_octo_plugin import api
from hermes_octo_plugin.adapter import OctoAdapter
from tests.conftest import make_bare_adapter


# Verbatim body captured from a real server response, kept byte-for-byte so
# these tests fail if the wire format the fix depends on ever changes.
REAL_ERROR_BODY = (
    '{"error":{"code":"err.server.bot_api.not_group_member","details":{},'
    '"http_status":403,"message":"\u673a\u5668\u4eba\u4e0d\u662f\u8be5\u7fa4'
    '\u6210\u5458\u3002"},"msg":"\u673a\u5668\u4eba\u4e0d\u662f\u8be5\u7fa4'
    '\u6210\u5458\u3002","status":400}'
)


class _FakeResponse:
    """Minimal stand-in for the parts of aiohttp's response we consume.

    Mirrors ``response.content.read(n)`` rather than ``response.text()``:
    the helper reads a size-capped number of bytes so a gzip-bombed error
    body cannot be buffered in full, and the fake has to expose the same
    surface or it would stop testing the real call.
    """

    def __init__(self, status: int, body: str):
        self.status = status
        self._body = body
        self.content = self._Content(body)

    class _Content:
        def __init__(self, body: str):
            self._raw = body.encode("utf-8")
            self._pos = 0

        async def read(self, n: int = -1) -> bytes:
            # Model aiohttp's StreamReader: reads advance a cursor and a
            # spent stream returns b"" (EOF).  A fake that replayed the whole
            # body on every call would let a read-loop spin forever.
            if self._pos >= len(self._raw):
                return b""
            take = len(self._raw) - self._pos if n < 0 else n
            chunk = self._raw[self._pos : self._pos + take]
            self._pos += len(chunk)
            return chunk


class _ChunkedFakeResponse:
    """A response whose body arrives in several stream chunks.

    This is the ordinary case on the wire, not an exotic one: aiohttp's
    ``StreamReader.read(n)`` returns whatever is *currently buffered* up to
    ``n`` bytes, so any body split across TCP segments (or sent with chunked
    transfer encoding) is delivered over multiple reads.  ``_FakeResponse``
    hands back the whole body on the first read and therefore cannot catch a
    single-read implementation; this fake can.
    """

    def __init__(self, status: int, body: str, chunk_size: int = 16):
        self.status = status
        self.content = self._Content(body, chunk_size)

    class _Content:
        def __init__(self, body: str, chunk_size: int):
            self._raw = body.encode("utf-8")
            self._pos = 0
            self._chunk_size = chunk_size

        async def read(self, n: int = -1) -> bytes:
            if self._pos >= len(self._raw):
                return b""  # EOF
            take = len(self._raw) - self._pos if n < 0 else min(n, self._chunk_size)
            chunk = self._raw[self._pos : self._pos + take]
            self._pos += len(chunk)
            return chunk


async def test_error_code_survives_a_body_split_across_chunks():
    """A multi-chunk error body must still yield the code.

    Regression lock for the read-loop.  With a single ``content.read(cap)``
    the helper sees only the first chunk, ``json.loads`` fails on the
    truncated prefix, ``code`` comes back ``None``, and the ERROR-level log
    spam this change exists to remove returns.  The body here is identical to
    the real one -- only its arrival pattern differs.
    """

    err = await api._response_error(
        "/v1/bot/messages/sync",
        cast(Any, _ChunkedFakeResponse(400, REAL_ERROR_BODY, chunk_size=16)),
    )

    assert err.status == 400
    assert err.code == api.ERR_NOT_GROUP_MEMBER, (
        "a single stream read returns only the first buffered chunk, so the "
        "code is lost when the envelope spans several chunks"
    )


async def test_capped_read_stops_at_the_limit_for_an_oversized_body():
    """The read loop must still refuse to buffer an unbounded body."""

    oversized = "x" * (api._ERROR_BODY_READ_LIMIT + 4096)
    response = _ChunkedFakeResponse(500, oversized, chunk_size=4096)

    raw = await api._read_capped_body(cast(Any, response))

    assert len(raw) == api._ERROR_BODY_READ_LIMIT


async def test_redirect_bodies_are_still_never_read_by_the_loop():
    """3xx responses must not be drained, preserving the SSRF posture."""

    class _ExplodingContent:
        async def read(self, n: int = -1) -> bytes:
            raise AssertionError("a redirect body must never be read")

    response = SimpleNamespace(status=302, content=_ExplodingContent())

    err = await api._response_error("/v1/bot/messages/sync", cast(Any, response))

    assert err.status == 302
    assert err.code is None


def test_error_code_extracted_from_real_server_body():
    """The machine-readable code survives extraction from the real payload."""
    assert api._extract_error_code(REAL_ERROR_BODY) == api.ERR_NOT_GROUP_MEMBER


async def test_outer_http_status_is_400_not_403():
    """Pin the fact that motivates code-based matching.

    The body carries an inner ``http_status: 403``, but the response the
    client actually sees is a 400.  Any fix keyed on ``status == 403`` would
    never fire in production.

    This drives the real ``_response_error`` helper rather than asserting a
    property of this file's own fixture constant: an assertion about
    ``REAL_ERROR_BODY`` alone cannot fail for any change to the source, so it
    would pin nothing.  What must hold is that the *exception the production
    code sees* carries status 400 and the machine-readable code — i.e. that
    the inner 403 is invisible at the boundary where the judgement is made.
    """

    err = await api._response_error(
        "/v1/bot/messages/sync", cast(Any, _FakeResponse(400, REAL_ERROR_BODY))
    )

    assert err.status == 400, "a status==403 match would never fire"
    assert err.code == api.ERR_NOT_GROUP_MEMBER


@pytest.mark.parametrize(
    "body",
    [
        "not json at all",
        "[]",
        '{"error":"plain string"}',
        '{"error":{}}',
        '{"error":{"code":123}}',
        # Free-form text must never be lifted out of the body.
        '{"error":{"code":"contains spaces and stuff"}}',
        # Over-long values are rejected outright.
        '{"error":{"code":"' + "a" * 81 + '"}}',
    ],
)
def test_malformed_or_unsafe_bodies_yield_no_code(body):
    """A hostile or malformed body can never widen what gets exposed."""
    assert api._extract_error_code(body) is None


@pytest.mark.asyncio
async def test_response_error_carries_code_and_status():
    err = await api._response_error(
        "/v1/bot/messages/sync", cast(Any, _FakeResponse(400, REAL_ERROR_BODY))
    )
    assert err.status == 400
    assert err.code == api.ERR_NOT_GROUP_MEMBER
    # The human-readable message must still not leak the response body.
    assert "\u673a\u5668\u4eba" not in str(err)
    assert "not_group_member" not in str(err)


@pytest.mark.asyncio
async def test_response_error_survives_unreadable_body():
    """Failing to drain the body must not mask the original status."""

    class _Exploding(_FakeResponse):
        def __init__(self, status: int, body: str):
            super().__init__(status, body)

            class _Boom:
                async def read(self, n: int = -1) -> bytes:
                    raise RuntimeError("connection reset")

            self.content = _Boom()

    err = await api._response_error("/v1/bot/messages/sync", cast(Any, _Exploding(500, "")))
    assert err.status == 500
    assert err.code is None


@pytest.mark.asyncio
async def test_redirect_body_is_never_read():
    """A redirect response body must not be touched, even to look for a code.

    Draining a redirect target is exactly how an SSRF probe gets its answer
    back, so the surrounding helpers refuse redirects without reading them.
    Code extraction must not weaken that.
    """

    class _Tracking(_FakeResponse):
        def __init__(self, status):
            super().__init__(status, REAL_ERROR_BODY)
            self.body_read = False
            outer = self

            class _Tracked:
                def __init__(self):
                    self._raw = REAL_ERROR_BODY.encode("utf-8")
                    self._pos = 0

                async def read(self, n: int = -1) -> bytes:
                    outer.body_read = True
                    if self._pos >= len(self._raw):
                        return b""
                    take = len(self._raw) - self._pos if n < 0 else n
                    chunk = self._raw[self._pos : self._pos + take]
                    self._pos += len(chunk)
                    return chunk

            self.content = _Tracked()

    redirect = _Tracking(302)
    err = await api._response_error("/v1/bot/example", cast(Any, redirect))
    assert redirect.body_read is False
    assert err.status == 302
    assert err.code is None

    # A real error response is still inspected.
    error = _Tracking(400)
    err = await api._response_error("/v1/bot/example", cast(Any, error))
    assert error.body_read is True
    assert err.code == api.ERR_NOT_GROUP_MEMBER


async def _history_with_error(monkeypatch, caplog, exc: Exception):
    """Drive the history path so the given error escapes the API call."""
    adapter = make_bare_adapter()
    adapter._http_session = cast(Any, object())  # truthy: enables API fallback
    adapter._group_histories = {}

    async def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(api, "get_channel_messages", _boom)

    with caplog.at_level(logging.DEBUG):
        result = await adapter._build_history_context("chan-1", "bot-uid")

    assert result == ""  # the failure is still absorbed, not propagated
    return [r for r in caplog.records if "HISTORY" in r.getMessage()]


@pytest.mark.asyncio
async def test_not_group_member_is_downgraded_to_debug(monkeypatch, caplog):
    exc = api.OctoApiError(
        "/v1/bot/messages/sync", status=400, code=api.ERR_NOT_GROUP_MEMBER
    )
    records = await _history_with_error(monkeypatch, caplog, exc)

    assert not [r for r in records if r.levelno >= logging.ERROR], (
        "an expected non-membership refusal must not be logged at ERROR"
    )
    assert [
        r
        for r in records
        if r.levelno == logging.DEBUG and "not a member" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_same_status_different_code_still_errors(monkeypatch, caplog):
    """The downgrade keys on the code, not the 400 status.

    This is the test that distinguishes a correct fix from one that simply
    silences every HTTP 400 -- a genuine fault sharing the status must remain
    at ERROR.
    """
    exc = api.OctoApiError(
        "/v1/bot/messages/sync", status=400, code="err.server.bot_api.rate_limited"
    )
    records = await _history_with_error(monkeypatch, caplog, exc)

    assert [r for r in records if r.levelno >= logging.ERROR], (
        "a real failure sharing HTTP 400 must still be logged at ERROR"
    )


@pytest.mark.asyncio
async def test_unrelated_exception_still_errors(monkeypatch, caplog):
    records = await _history_with_error(
        monkeypatch, caplog, RuntimeError("network exploded")
    )

    assert [r for r in records if r.levelno >= logging.ERROR], (
        "an unrelated failure must still be logged at ERROR"
    )


async def _read_tool_with_error(monkeypatch, caplog, exc: Exception):
    """Drive the read_channel_messages tool path so the error escapes the API."""
    from unittest.mock import AsyncMock

    adapter = make_bare_adapter()
    adapter._http_session = cast(Any, object())

    async def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(api, "get_channel_messages", _boom)
    monkeypatch.setattr(
        adapter,
        "check_read_permission",
        AsyncMock(return_value=(SimpleNamespace(allowed=True, reason=None), "chan-1", 2)),
    )

    with caplog.at_level(logging.DEBUG):
        result = await adapter.read_channel_messages(
            requester_uid="uid-1", target="group:chan-1", limit=10
        )

    assert result["ok"] is False  # the failure is still reported to the caller
    return [r for r in caplog.records if "read_channel_messages" in r.getMessage()]


@pytest.mark.asyncio
async def test_read_tool_not_group_member_is_downgraded(monkeypatch, caplog):
    """The sibling call site gets the same treatment as the HISTORY path.

    Both reviewers flagged that the commit subject ("stop logging the expected
    not_group_member refusal at ERROR") was broader than the diff, which only
    covered the mention-driven history fetch.  Narrowing the subject would
    have been the other option; extending the downgrade is the honest one,
    since the refusal means exactly the same thing on this path.
    """
    exc = api.OctoApiError(
        "/v1/bot/messages/sync", status=400, code=api.ERR_NOT_GROUP_MEMBER
    )
    records = await _read_tool_with_error(monkeypatch, caplog, exc)

    assert not [r for r in records if r.levelno >= logging.ERROR], (
        "an expected non-membership refusal must not be logged at ERROR"
    )
    assert [r for r in records if r.levelno == logging.DEBUG]


@pytest.mark.asyncio
async def test_read_tool_other_failure_still_errors(monkeypatch, caplog):
    """The narrowing must not silence genuine faults on this path either."""
    records = await _read_tool_with_error(
        monkeypatch, caplog, RuntimeError("network exploded")
    )

    assert [r for r in records if r.levelno >= logging.ERROR], (
        "an unrelated failure must still be logged at ERROR"
    )
