"""Media protocol safety and fidelity tests."""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace
from pathlib import Path

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import logging


from hermes_octo_plugin import adapter as adapter_module, api, transport as transport_module
from hermes_octo_plugin.adapter import OctoAdapter
from hermes_octo_plugin.transport import (
    SSRFGuardConnector as _SSRFGuardConnector,
    SSRFGuardResolver as _SSRFGuardResolver,
    TransportPolicy,
)
from hermes_octo_plugin.types import ChannelType, MessagePayload, MessageType
from tests.conftest import make_bare_adapter


class _NotFoundResponse:
    ok = False
    status = 404

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _UnexpectedBody:
    def __init__(self):
        self.called = False

    def iter_chunked(self, _size):
        self.called = True
        raise AssertionError("redirect body must not be consumed")


class _RedirectResponse:
    ok = True
    status = 302

    def __init__(self):
        self.content = _UnexpectedBody()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _ChunkedBody:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    def iter_chunked(self, _size: int):
        async def chunks():
            for chunk in self._chunks:
                yield chunk

        return chunks()


class _ChunkedResponse:
    status = 200

    def __init__(self, *chunks: bytes) -> None:
        self.content = _ChunkedBody(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

def test_bearer_auth_is_limited_to_exact_configured_api_or_cdn_origins():
    adapter = make_bare_adapter()
    adapter._api_url = "https://api.octo.example/v1"
    adapter._cdn_url = "https://cdn.octo.example/assets"
    adapter._bot_token = "test-token"

    assert adapter._inbound_media_headers("https://api.octo.example/file/a.png") == {
        "Authorization": "Bearer test-token"
    }
    assert adapter._inbound_media_headers("https://cdn.octo.example/a.png") == {
        "Authorization": "Bearer test-token"
    }
    assert adapter._inbound_media_headers("https://api.octo.example.evil/a.png") == {}
    assert adapter._inbound_media_headers("https://storage.example/a.png") == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "expected_headers"),
    [
        (
            "https://api.octo.example/file/a.png",
            {"Authorization": "Bearer test-token"},
        ),
        ("https://storage.example/a.png", {}),
    ],
)
async def test_inbound_media_download_passes_only_origin_scoped_auth(
    url: str, expected_headers: dict[str, str]
):
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._api_url = "https://api.octo.example/v1"
    adapter._cdn_url = "https://cdn.octo.example/assets"
    adapter._bot_token = "test-token"
    adapter._http_session = MagicMock()
    adapter._http_session.get.return_value = _NotFoundResponse()

    assert await adapter._download_inbound_media_to_local(url, "image/png") is None
    assert adapter._http_session.get.call_args.kwargs["headers"] == expected_headers


@pytest.mark.asyncio
async def test_inbound_media_rejects_redirect_without_reading_body():
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._api_url = "https://api.octo.example/v1"
    adapter._bot_token = "test-token"
    adapter._http_session = MagicMock()
    response = _RedirectResponse()
    adapter._http_session.get.return_value = response

    assert (
        await adapter._download_inbound_media_to_local(
            "https://api.octo.example/file/a.png",
            "image/png",
        )
        is None
    )

    assert response.content.called is False

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://127.0.0.1:8080/admin",
        "http://10.0.0.8/private.png",
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
)
async def test_inbound_media_rejects_private_and_metadata_urls_before_io(url: str):
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._api_url = "https://api.octo.example/v1"
    adapter._bot_token = "test-token"
    adapter._http_session = MagicMock()
    adapter._http_session.get.return_value = _NotFoundResponse()

    assert await adapter._download_inbound_media_to_local(url, "image/png") is None
    adapter._http_session.get.assert_not_called()



@pytest.mark.asyncio
async def test_inbound_media_rejects_opposite_scheme_trusted_endpoint_before_io():
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._http_session = MagicMock()
    adapter._http_session.transport_policy = TransportPolicy({
        "https://api.octo.example:8443",
    })

    assert (
        await adapter._download_inbound_media_to_local(
            "http://api.octo.example:8443/private.png",
            "image/png",
        )
        is None
    )
    adapter._http_session.get.assert_not_called()

def test_inbound_private_media_requires_opt_in_for_exact_configured_origin(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = make_bare_adapter()
    adapter._api_url = "http://api.internal:8080/v1"
    adapter._http_session = MagicMock()

    monkeypatch.delenv("OCTO_ALLOW_PRIVATE_HOSTS", raising=False)
    adapter._http_session.transport_policy = TransportPolicy()
    assert not adapter._inbound_media_url_allowed(
        "http://api.internal:8080/download/report.pdf"
    )

    monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "true")
    adapter._http_session.transport_policy = TransportPolicy({adapter._api_url})
    assert adapter._inbound_media_url_allowed(
        "http://api.internal:8080/download/report.pdf"
    )
    assert not adapter._inbound_media_url_allowed(
        "http://api.internal:8081/download/report.pdf"
    )


@pytest.mark.asyncio
async def test_ssrf_resolver_rejects_private_dns_answers_but_allows_trusted_origin():
    resolver = _SSRFGuardResolver(
        trusted_origins={"https://api.octo.example"}
    )
    resolver._delegate.resolve = AsyncMock(
        return_value=[
            {
                "hostname": "storage.example",
                "host": "127.0.0.1",
                "port": 443,
                "family": 2,
                "proto": 6,
                "flags": 0,
            }
        ]
    )

    with pytest.raises(OSError, match="unsafe address"):
        await resolver.resolve("storage.example", 443)

    trusted = await resolver.resolve("api.octo.example", 443)
    assert trusted[0]["host"] == "127.0.0.1"
    with pytest.raises(OSError, match="unsafe"):
        await resolver.resolve("api.octo.example", 8443)

    resolver._delegate.resolve = AsyncMock(
        return_value=[
            {
                "hostname": "api.octo.example",
                "host": "169.254.169.254",
                "port": 443,
                "family": 2,
                "proto": 6,
                "flags": 0,
            }
        ]
    )
    with pytest.raises(OSError, match="unsafe address"):
        await resolver.resolve("api.octo.example", 443)
    await resolver.close()


@pytest.mark.asyncio
async def test_http_session_private_trust_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = make_bare_adapter()
    adapter._api_url = "https://api.octo.example/v1"
    adapter._cdn_url = "https://cdn.octo.example/assets"

    for enabled, expected in (
        (False, frozenset()),
        (
            True,
            frozenset({
                ("https", "api.octo.example", 443),
                ("https", "cdn.octo.example", 443),
            }),
        ),
    ):
        if enabled:
            monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "true")
        else:
            monkeypatch.delenv("OCTO_ALLOW_PRIVATE_HOSTS", raising=False)
        connector = MagicMock()
        session = MagicMock()
        with (
            patch(
                "hermes_octo_plugin.transport.SSRFGuardConnector",
                return_value=connector,
            ) as connector_cls,
            patch("hermes_octo_plugin.transport.aiohttp.ClientSession", return_value=session),
        ):
            assert adapter._new_http_session() is session

        resolver = connector_cls.call_args.kwargs["resolver"]
        assert resolver.policy.trusted_download_origins() == expected
        await resolver.close()



def test_transport_policy_normalizes_public_idn_and_private_host_aliases(
    monkeypatch: pytest.MonkeyPatch,
):
    public_policy = TransportPolicy({"https://bücher.example"})

    assert public_policy.is_download_url_trusted(
        "https://xn--bcher-kva.example/report.bin"
    )

    self_hosted_policy = TransportPolicy({"http://127.0.0.1:8443"})

    assert self_hosted_policy.is_download_url_trusted(
        "http://①②⑦.0.0.1:8443/report.bin"
    )
    assert not self_hosted_policy.is_download_url_trusted(
        "https://①②⑦.0.0.1:8443/report.bin"
    )


@pytest.mark.asyncio
async def test_ssrf_resolver_rejects_metadata_hostname_before_dns():
    resolver = _SSRFGuardResolver(
        trusted_origins={"http://metadata.google.internal"}
    )
    resolver._delegate.resolve = AsyncMock()

    with pytest.raises(OSError, match="unsafe host"):
        await resolver.resolve("metadata.google.internal", 80)

    resolver._delegate.resolve.assert_not_awaited()
    await resolver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "①②⑦.0.0.1", "2130706433", "127.1", "0177.0.0.1"],
)
async def test_ssrf_connector_blocks_literal_and_legacy_loopback_before_aiohttp_bypass(
    host: str,
):
    resolver = _SSRFGuardResolver()
    resolver._delegate.resolve = AsyncMock()
    connector = _SSRFGuardConnector(resolver=resolver)
    try:
        with pytest.raises(OSError, match="unsafe"):
            await connector._resolve_host(host, 80)
        resolver._delegate.resolve.assert_not_awaited()
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_guarded_connector_owns_and_closes_explicit_resolver():
    resolver = _SSRFGuardResolver()
    resolver.close = AsyncMock()
    connector = _SSRFGuardConnector(resolver=resolver)
    session = aiohttp.ClientSession(connector=connector)

    await session.close()

    resolver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_resolver_close_remains_retryable():
    resolver = _SSRFGuardResolver()
    close_entered = asyncio.Event()
    close_completed = asyncio.Event()
    close_calls = 0

    async def cancellable_close():
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            close_entered.set()
            await asyncio.Event().wait()
        close_completed.set()

    resolver.close = cancellable_close  # type: ignore[method-assign]
    connector = _SSRFGuardConnector(resolver=resolver)
    first_close = asyncio.create_task(connector.close())
    await close_entered.wait()
    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close

    assert connector._ssrf_resolver_closed is False
    await connector.close()

    assert close_calls == 2
    assert close_completed.is_set()
    assert connector._ssrf_resolver_closed is True


@pytest.mark.asyncio
async def test_guarded_connector_allows_opted_in_private_literal_origin():
    resolver = _SSRFGuardResolver(
        trusted_origins={"http://127.0.0.1:8080"}
    )
    connector = _SSRFGuardConnector(resolver=resolver)
    try:
        records = await connector._resolve_host("127.0.0.1", 8080)
    finally:
        await connector.close()

    assert records[0]["host"] == "127.0.0.1"
    assert records[0]["family"] in {socket.AF_UNSPEC, socket.AF_INET}



@pytest.mark.asyncio
async def test_guarded_resolver_allows_opted_in_ipv6_loopback_origin():
    resolver = _SSRFGuardResolver(
        trusted_origins={"ws://[::1]:9000"}
    )
    resolver._delegate.resolve = AsyncMock(
        return_value=[
            {
                "hostname": "::1",
                "host": "::1",
                "port": 9000,
                "family": socket.AF_INET6,
                "proto": socket.IPPROTO_TCP,
                "flags": 0,
            }
        ]
    )
    try:
        records = await resolver.resolve("::1", 9000, family=socket.AF_UNSPEC)
    finally:
        await resolver.close()

    assert records[0]["host"] == "::1"
    assert records[0]["family"] == socket.AF_INET6



def test_transport_origin_preserves_explicit_zero_port() -> None:
    policy = TransportPolicy({"http://10.0.0.8"})

    assert policy.is_download_url_trusted("http://10.0.0.8/file") is True
    assert policy.is_download_url_trusted("http://10.0.0.8:80/file") is True
    assert policy.is_download_url_trusted("http://10.0.0.8:0/file") is False


@pytest.mark.asyncio
async def test_guarded_websocket_socket_connects_only_validated_numeric_address(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OCTO_ALLOW_PRIVATE_HOSTS", raising=False)
    resolver = MagicMock()
    resolver.resolve = AsyncMock(
        return_value=[
            {
                "hostname": "socket.example",
                "host": "93.184.216.34",
                "port": 443,
                "family": socket.AF_INET,
                "proto": socket.IPPROTO_TCP,
                "flags": 0,
            }
        ]
    )
    resolver.close = AsyncMock()
    resolver_factory = MagicMock(return_value=resolver)
    guarded_socket = MagicMock()
    socket_factory = MagicMock(return_value=guarded_socket)
    loop = MagicMock()
    loop.sock_connect = AsyncMock()

    assert hasattr(transport_module, "open_guarded_websocket_socket")
    with (
        patch.object(transport_module, "SSRFGuardResolver", resolver_factory),
        patch.object(transport_module.socket, "socket", socket_factory),
        patch.object(transport_module.asyncio, "get_running_loop", return_value=loop),
    ):
        result = await transport_module.open_guarded_websocket_socket(
            "wss://socket.example/ws"
        )

    assert result is guarded_socket
    policy = resolver_factory.call_args.kwargs["policy"]
    assert policy.trusted_download_origins() == frozenset()
    resolver.resolve.assert_awaited_once_with(
        "socket.example",
        443,
        family=socket.AF_UNSPEC,
    )
    resolver.close.assert_awaited_once()
    socket_factory.assert_called_once_with(
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
    )
    guarded_socket.setblocking.assert_called_once_with(False)
    loop.sock_connect.assert_awaited_once_with(
        guarded_socket,
        ("93.184.216.34", 443),
    )


@pytest.mark.asyncio
async def test_guarded_websocket_socket_stops_on_unsafe_dns_before_connect(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OCTO_ALLOW_PRIVATE_HOSTS", raising=False)
    timeout_active = False

    class _TimeoutMarker:
        async def __aenter__(self):
            nonlocal timeout_active
            timeout_active = True

        async def __aexit__(self, *_args):
            nonlocal timeout_active
            timeout_active = False

    async def reject_unsafe_dns(*_args, **_kwargs):
        assert timeout_active is True
        raise OSError("unsafe address")

    resolver = MagicMock()
    resolver.resolve = AsyncMock(side_effect=reject_unsafe_dns)
    resolver.close = AsyncMock()
    socket_factory = MagicMock()

    assert hasattr(transport_module, "open_guarded_websocket_socket")
    with (
        patch.object(
            transport_module,
            "SSRFGuardResolver",
            return_value=resolver,
        ),
        patch.object(transport_module.socket, "socket", socket_factory),
        pytest.raises(OSError, match="unsafe address"),
        patch.object(
            transport_module.asyncio,
            "timeout",
            return_value=_TimeoutMarker(),
        ),
    ):
        await transport_module.open_guarded_websocket_socket(
            "wss://socket.example/ws"
        )

    resolver.close.assert_awaited_once()
    socket_factory.assert_not_called()


@pytest.mark.asyncio
async def test_guarded_websocket_socket_never_trusts_target_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OCTO_ALLOW_PRIVATE_HOSTS", raising=False)
    resolver = MagicMock()
    resolver.resolve = AsyncMock(side_effect=OSError("stop after policy capture"))
    resolver.close = AsyncMock()
    resolver_factory = MagicMock(return_value=resolver)

    with (
        patch.object(transport_module, "SSRFGuardResolver", resolver_factory),
        pytest.raises(OSError, match="policy capture"),
    ):
        await transport_module.open_guarded_websocket_socket(
            "wss://socket.internal:9443/ws"
        )

    policy = resolver_factory.call_args.kwargs["policy"]
    assert policy.trusted_download_origins() == frozenset()
    resolver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_guarded_websocket_socket_trusts_exact_target_with_opt_in(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "true")
    resolver = MagicMock()
    resolver.resolve = AsyncMock(side_effect=OSError("stop after policy capture"))
    resolver.close = AsyncMock()
    resolver_factory = MagicMock(return_value=resolver)

    with (
        patch.object(transport_module, "SSRFGuardResolver", resolver_factory),
        pytest.raises(OSError, match="policy capture"),
    ):
        await transport_module.open_guarded_websocket_socket(
            "wss://socket.internal:9443/ws"
        )

    policy = resolver_factory.call_args.kwargs["policy"]
    assert policy.trusted_download_origins() == frozenset({
        ("wss", "socket.internal", 9443),
    })
    resolver.close.assert_awaited_once()


def test_private_host_policy_rejects_ipv4_mapped_link_local() -> None:
    with pytest.raises(RuntimeError, match="unsafe presigned upload URL"):
        api._validate_presigned_upload_origin(
            TransportPolicy(),
            "http://[::ffff:169.254.169.254]/latest/meta-data/",
        )


def test_private_host_policy_rejects_ipv4_mapped_metadata_literal() -> None:
    with pytest.raises(RuntimeError, match="unsafe presigned upload URL"):
        api._validate_presigned_upload_origin(
            TransportPolicy(),
            "http://[::ffff:6464:64c8]/latest/meta-data/",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "expected_headers"),
    [
        (
            "https://cdn.octo.example/download/report.pdf",
            {"Authorization": "Bearer test-token"},
        ),
        ("https://storage.example/report.pdf", {}),
    ],
)
async def test_inbound_file_download_passes_only_origin_scoped_auth(
    url: str, expected_headers: dict[str, str]
):
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._api_url = "https://api.octo.example/v1"
    adapter._cdn_url = "https://cdn.octo.example/assets"
    adapter._bot_token = "test-token"
    adapter._http_session = MagicMock()
    adapter._http_session.get.return_value = _NotFoundResponse()

    await adapter._resolve_inbound_file(url, "report.pdf", None)
    assert adapter._http_session.get.call_args.kwargs["headers"] == expected_headers


@pytest.mark.asyncio
async def test_inbound_file_rejects_redirect_without_reading_body():
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._api_url = "https://api.octo.example/v1"
    adapter._bot_token = "test-token"
    adapter._http_session = MagicMock()
    response = _RedirectResponse()
    adapter._http_session.get.return_value = response

    result = await adapter._resolve_inbound_file(
        "https://api.octo.example/download/report.pdf",
        "report.pdf",
        None,
    )

    assert result.content == "[文件: report.pdf - 下载失败 HTTP 302]"
    assert response.content.called is False


@pytest.mark.parametrize(
    ("raw_filename", "expected_filename"),
    [
        ("report[preview].txt", "report(preview).txt"),
        ("report\nspoofed.txt", "未知文件"),
        ("report\x1fspoofed.txt", "未知文件"),
        ("report\u2066spoofed.txt", "report_spoofed.txt"),
        ("../private-report.txt", "未知文件"),
        (r"..\private-report.txt", "未知文件"),
    ],
)
def test_inbound_file_display_name_never_serializes_unsafe_metadata(
    raw_filename: str,
    expected_filename: str,
) -> None:
    adapter = make_bare_adapter()

    content = adapter._resolve_content(
        MessagePayload(type=MessageType.File, name=raw_filename)
    )

    assert content == f"[文件: {expected_filename}]"
    display_name = content[len("[文件: "):-1]
    for unsafe in ("[", "]", "\n", "\x1f", "\u2066", "/", "\\"):
        assert unsafe not in display_name


def test_inbound_long_filename_is_bounded_without_losing_its_identity() -> None:
    raw_filename = f"{'报告' * 200}.pdf"

    display_name = adapter_module._inbound_file_display_name(raw_filename)

    assert display_name != "未知文件"
    assert display_name.endswith(".pdf")
    assert len(display_name.encode("utf-8")) <= 255


def test_inbound_megabyte_filename_truncation_has_bounded_work() -> None:
    raw_filename = f"{'a' * 1_000_000}.pdf"

    display_name = adapter_module._inbound_file_display_name(raw_filename)

    assert display_name.endswith(".pdf")
    assert len(display_name.encode("utf-8")) <= 255



def test_inbound_long_extension_is_bounded() -> None:
    display_name = adapter_module._inbound_file_display_name(f"a.{('x' * 300)}")

    assert len(display_name.encode("utf-8")) <= 255


def test_inbound_temp_filename_reserves_uuid_prefix_budget() -> None:
    safe_name = adapter_module._truncate_utf8_filename("报告" * 200, 255 - 33)

    assert len(f"{'0' * 32}-{safe_name}".encode("utf-8")) <= 255


class _RaisingRequest:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, *_args):
        return None


@pytest.mark.asyncio
async def test_inbound_media_failure_logs_omit_signed_query_and_exception_detail(
    caplog,
):
    signed_url = (
        "https://api.octo.example/download/photo.png?"
        "signature=signed-query-secret"
    )
    exception_detail = "upstream media diagnostic secret"
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._api_url = "https://api.octo.example/v1"
    adapter._bot_token = "test-token"
    adapter._http_session = MagicMock()
    adapter._http_session.get.return_value = _RaisingRequest(
        RuntimeError(exception_detail)
    )
    caplog.set_level(logging.WARNING, logger="hermes_octo_plugin.adapter")

    assert (
        await adapter._download_inbound_media_to_local(signed_url, "image/png")
        is None
    )

    assert "signed-query-secret" not in caplog.text
    assert exception_detail not in caplog.text
    assert "inbound media download failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("msg_type", "kwargs", "expected_payload"),
    [
        (
            MessageType.GIF,
            {"width": 120, "height": 80},
            {"type": MessageType.GIF, "url": "https://cdn.example/a.gif", "width": 120, "height": 80},
        ),
        (
            MessageType.Voice,
            {"duration": 3},
            {"type": MessageType.Voice, "url": "https://cdn.example/a.gif", "duration": 3},
        ),
        (
            MessageType.Video,
            {"width": 1280, "height": 720, "duration": 12},
            {
                "type": MessageType.Video,
                "url": "https://cdn.example/a.gif",
                "width": 1280,
                "height": 720,
                "duration": 12,
            },
        ),
    ],
)
async def test_media_api_preserves_the_server_defined_fields_and_reply(
    msg_type: MessageType,
    kwargs: dict[str, int],
    expected_payload: dict[str, object],
):
    post_json = AsyncMock(
        return_value={"message_id": "media-1", "message_seq": 7}
    )
    with patch.object(api, "post_json", post_json):
        result = await api.send_media_message(
            MagicMock(),
            "https://api.example.invalid",
            "test-token",
            "user-1",
            ChannelType.DM,
            msg_type,
            "https://cdn.example/a.gif",
            reply_msg_id="parent-message",
            client_msg_no="media-dedup-1",
            on_behalf_of="grantor-1",
            **kwargs,
        )
    assert result.message_id == "media-1"
    assert result.message_seq == 7
    assert result.client_msg_no == "media-dedup-1"

    assert post_json.await_args.args[3] == "/v1/bot/sendMessage"
    assert post_json.await_args.args[4] == {
        "channel_id": "user-1",
        "channel_type": ChannelType.DM,
        "client_msg_no": "media-dedup-1",
        "on_behalf_of": "grantor-1",
        "payload": {
            **expected_payload,
            "reply": {"message_id": "parent-message"},
        },
    }


@pytest.mark.asyncio
async def test_outbound_media_uses_space_dm_target_and_preserves_reply_metadata_and_captions():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._chat_kind = {"s14_user-1": ChannelType.DM}
    adapter._space_dm_targets = {"s14_user-1": "user-1"}

    get_members = AsyncMock()
    with (
        patch.object(
            api,
            "download_file",
            AsyncMock(return_value=(b"media", "application/octet-stream", "source.bin")),
        ),
        patch.object(api, "parse_image_dimensions", return_value=None),
        patch.object(
            api,
            "upload_and_get_url",
            AsyncMock(return_value="https://cdn.example/uploaded"),
        ),
        patch.object(api, "get_group_members", get_members),
        patch.object(api, "send_media_message", AsyncMock()) as send_media,
        patch.object(api, "send_message", AsyncMock()) as send_text,
    ):
        assert (await adapter.send_image(
            "s14_user-1", "https://source.example/image.webp",
            caption="@[u1:Alice] image caption", reply_to="parent-message",
        )).success
        assert (await adapter.send_document(
            "s14_user-1", "https://source.example/report.pdf",
            caption="@[u1:Alice] file caption", reply_to="parent-message",
        )).success
        assert (await adapter.send_voice(
            "s14_user-1", "https://source.example/voice.amr",
            caption="@[u1:Alice] voice caption", reply_to="parent-message",
            duration=3,
        )).success
        assert (await adapter.send_video(
            "s14_user-1", "https://source.example/video.mp4",
            caption="@[u1:Alice] video caption", reply_to="parent-message",
            width=1280, height=720, duration=12,
        )).success

    assert send_media.await_count == 4
    for call in send_media.await_args_list:
        assert call.kwargs["channel_id"] == "user-1"
        assert call.kwargs["channel_type"] == ChannelType.DM
        assert call.kwargs["reply_msg_id"] == "parent-message"

    voice_call = next(
        call for call in send_media.await_args_list
        if call.kwargs["msg_type"] == MessageType.Voice
    )
    assert voice_call.kwargs["duration"] == 3
    video_call = next(
        call for call in send_media.await_args_list
        if call.kwargs["msg_type"] == MessageType.Video
    )
    assert video_call.kwargs["width"] == 1280
    assert video_call.kwargs["height"] == 720
    assert video_call.kwargs["duration"] == 12

    get_members.assert_not_awaited()
    assert [call.kwargs["content"] for call in send_text.await_args_list] == [
        "@Alice image caption", "@Alice file caption",
        "@Alice voice caption", "@Alice video caption",
    ]
    for call in send_text.await_args_list:
        assert call.kwargs["channel_id"] == "user-1"
        assert call.kwargs["channel_type"] == ChannelType.DM
        assert call.kwargs["reply_msg_id"] == "parent-message"
        assert call.kwargs["mention_uids"] == []
        assert call.kwargs["mention_entities"] == []


@pytest.mark.asyncio
async def test_group_image_caption_uses_adapter_session_for_fresh_mention_roster():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    adapter._chat_kind = {"group-1": ChannelType.Group}
    get_members = AsyncMock(
        return_value=[SimpleNamespace(uid="member-1", name="Member")]
    )
    send_media = AsyncMock(
        return_value=SimpleNamespace(
            message_id="media-1",
            message_seq=None,
            client_msg_no=None,
        )
    )
    send_text = AsyncMock()

    with (
        patch.object(
            api,
            "upload_and_get_url",
            AsyncMock(return_value="https://cdn.example/uploaded"),
        ),
        patch.object(api, "parse_image_dimensions", return_value=None),
        patch.object(api, "get_group_members", get_members),
        patch.object(api, "send_media_message", send_media),
        patch.object(api, "send_message", send_text),
    ):
        result = await adapter.send_image(
            "group-1",
            "data:image/png;base64,bWVkaWE=",
            caption="@[member-1:Member]",
        )

    assert result.success is True
    get_members.assert_awaited_once_with(
        adapter._http_session,
        "https://api.example.invalid",
        "test-token",
        "group-1",
    )
    caption = send_text.await_args.kwargs
    assert caption["content"] == "@Member"
    assert caption["mention_uids"] == ["member-1"]
    assert [entity.uid for entity in caption["mention_entities"]] == ["member-1"]


@pytest.mark.asyncio
async def test_outbound_image_rejects_unsupported_media_metadata_before_io():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"

    with (
        patch.object(api, "download_file", AsyncMock()) as download,
        patch.object(api, "send_media_message", AsyncMock()) as send_media,
    ):
        result = await adapter.send_image(
            "group-1",
            "https://source.example/image.png",
            metadata={"duration": 3},
        )

    assert result.success is False
    assert "duration" in (result.error or "")
    download.assert_not_awaited()
    send_media.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name",
    ["send_document", "send_voice", "send_video"],
)
@pytest.mark.parametrize("as_file_url", [False, True])
async def test_native_media_accepts_local_paths(
    tmp_path,
    method_name: str,
    as_file_url: bool,
):
    source = tmp_path / "media.bin"
    source.write_bytes(b"local media")
    source_value = source.as_uri() if as_file_url else str(source)
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    upload = AsyncMock(return_value="https://cdn.example/uploaded")
    send = AsyncMock()

    with (
        patch.object(api, "authorize_local_media_path", return_value=str(source)),
        patch.object(api, "upload_and_get_url", upload),
        patch.object(api, "send_media_message", send),
    ):
        result = await getattr(adapter, method_name)("group-1", source_value)

    assert result.success is True
    assert upload.await_args.args[3] == "media.bin"
    assert upload.await_args.args[4] == b"local media"
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_media_accepts_data_urls() -> None:
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    upload = AsyncMock(return_value="https://cdn.example/uploaded")
    send = AsyncMock()

    with (
        patch.object(api, "upload_and_get_url", upload),
        patch.object(api, "send_media_message", send),
    ):
        result = await adapter.send_document(
            "group-1",
            "data:text/plain;base64,bG9jYWwgbWVkaWE=",
        )

    assert result.success is True
    assert upload.await_args.args[3:6] == (
        "file.txt",
        b"local media",
        "text/plain",
    )
    send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_name",
    [
        "../report.txt",
        "nested/report.txt",
        "report\nname.txt",
        f"{'x' * 256}.txt",
        r"..\report.txt",
        r"nested\report.txt",
        ".",
        "..",
    ],
)
async def test_native_document_rejects_unsafe_explicit_filename(
    file_name: str,
) -> None:
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    upload = AsyncMock(return_value="https://cdn.example/uploaded")
    send = AsyncMock()

    with (
        patch.object(
            adapter,
            "_load_outbound_media",
            AsyncMock(
                return_value=(
                    b"local media",
                    "application/octet-stream",
                    "report.txt",
                )
            ),
        ) as load,
        patch.object(api, "upload_and_get_url", upload),
        patch.object(api, "send_media_message", send),
    ):
        result = await adapter.send_document(
            "group-1",
            "/authorized/report.txt",
            file_name=file_name,
        )

    assert result.success is False
    assert result.error == "media filename is invalid"
    upload.assert_not_awaited()
    load.assert_not_awaited()

    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_document_rejects_unsafe_source_derived_filename() -> None:
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    upload = AsyncMock(return_value="https://cdn.example/uploaded")
    send = AsyncMock()

    with (
        patch.object(
            adapter,
            "_load_outbound_media",
            AsyncMock(
                return_value=(
                    b"remote media",
                    "application/pdf",
                    "../derived.pdf",
                )
            ),
        ),
        patch.object(api, "upload_and_get_url", upload),
        patch.object(api, "send_media_message", send),
    ):
        result = await adapter.send_document(
            "group-1",
            "https://source.example/report.pdf",
        )

    assert result.success is False
    assert result.error == "media filename is invalid"
    upload.assert_not_awaited()
    send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name",
    ["send_image", "send_voice", "send_video"],
)
async def test_native_media_rejects_unsafe_source_derived_filename(
    method_name: str,
) -> None:
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    upload = AsyncMock(return_value="https://cdn.example/uploaded")
    send = AsyncMock()

    with (
        patch.object(
            adapter,
            "_load_outbound_media",
            AsyncMock(
                return_value=(
                    b"remote media",
                    "application/octet-stream",
                    "../derived.bin",
                )
            ),
        ),
        patch.object(api, "upload_and_get_url", upload),
        patch.object(api, "send_media_message", send),
    ):
        result = await getattr(adapter, method_name)(
            "group-1",
            "https://source.example/media.bin",
        )

    assert result.success is False
    assert result.error == "media filename is invalid"
    upload.assert_not_awaited()
    send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name",
    ["send_document", "send_voice", "send_video"],
)
async def test_native_remote_media_uses_the_server_upload_limit(method_name: str):
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    async def guarded_download(session, *_args, **kwargs):
        assert isinstance(session.connector, _SSRFGuardConnector)
        assert kwargs["policy"] is session.transport_policy
        return b"media", "application/octet-stream", "source.bin"

    download = AsyncMock(side_effect=guarded_download)

    with (
        patch.object(api, "download_file", download),
        patch.object(
            api,
            "upload_and_get_url",
            AsyncMock(return_value="https://cdn.example/uploaded"),
        ),
        patch.object(api, "send_media_message", AsyncMock()),
    ):
        result = await getattr(adapter, method_name)(
            "group-1",
            "https://source.example/media.bin",
        )

    assert result.success is True
    assert download.await_args.kwargs["max_size"] == api.MAX_OUTBOUND_MEDIA_BYTES
    assert "enforce_host_safety" not in download.await_args.kwargs





def test_local_media_uses_current_hermes_static_path_validator(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.platforms.base import BasePlatformAdapter

    source = tmp_path / "source.bin"
    authorized = tmp_path / "authorized.bin"
    calls: list[str] = []

    def validate_media_delivery_path(path: str) -> str:
        calls.append(path)
        return str(authorized)

    monkeypatch.setattr(
        BasePlatformAdapter,
        "validate_media_delivery_path",
        staticmethod(validate_media_delivery_path),
        raising=False,
    )

    assert api.authorize_local_media_path(str(source)) == str(authorized)
    assert calls == [str(source)]


def test_local_media_fails_closed_when_hermes_014_has_no_path_validator(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.platforms.base import BasePlatformAdapter

    monkeypatch.delattr(
        BasePlatformAdapter,
        "validate_media_delivery_path",
        raising=False,
    )

    assert api.authorize_local_media_path(str(tmp_path / "source.bin")) is None
@pytest.mark.asyncio
async def test_native_local_media_uses_hermes_authorized_path_before_read(tmp_path):
    adapter = make_bare_adapter()
    requested = tmp_path / "requested.bin"
    authorized = tmp_path / "authorized.bin"
    requested.write_bytes(b"requested")
    authorized.write_bytes(b"authorized")

    with patch.object(
        api,
        "authorize_local_media_path",
        return_value=str(authorized),
    ) as validate:
        data, content_type, filename = await adapter._load_outbound_media(str(requested))

    validate.assert_called_once_with(str(requested))
    assert data == b"authorized"
    assert content_type == "application/octet-stream"
    assert filename == "authorized.bin"


@pytest.mark.asyncio
async def test_native_local_media_rejects_hermes_denied_path_before_read(tmp_path):
    adapter = make_bare_adapter()
    requested = tmp_path / "denied.bin"
    requested.write_bytes(b"must not be read")

    with (
        patch.object(api, "authorize_local_media_path", return_value=None),
        patch.object(
            api,
            "read_local_media",
            side_effect=AssertionError("denied path was read"),
        ) as read_local,
    ):
        with pytest.raises(PermissionError, match="not authorized"):
            await adapter._load_outbound_media(str(requested))

    read_local.assert_not_called()


def _inbound_event_adapter() -> OctoAdapter:
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._robot_id = "bot-1"
    adapter._aes_key = b"key"
    adapter._aes_iv = b"iv"
    adapter._resolve_sender_name = AsyncMock(return_value="Alice")
    adapter._send_typing_safe = AsyncMock()
    adapter._send_read_receipt_safe = AsyncMock()
    adapter.build_source = MagicMock(return_value=SimpleNamespace())
    adapter.handle_message = AsyncMock()
    return adapter


async def _deliver_inbound_payload(adapter: OctoAdapter, raw: bytes) -> object:
    recv = SimpleNamespace(
        message_id="message-1",
        message_seq=1,
        from_uid="user-1",
        channel_id="bot-1",
        channel_type=ChannelType.DM,
        timestamp=1,
        encrypted_payload=raw,
    )
    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(recv)
    return adapter.handle_message.await_args.args[0]


@pytest.mark.asyncio
async def test_inbound_binary_file_downloads_once_and_delivers_the_same_local_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    adapter = _inbound_event_adapter()
    session = MagicMock()
    session.get.return_value = _ChunkedResponse(b"%PDF")
    adapter._http_session = session
    monkeypatch.setattr("hermes_octo_plugin.adapter.FILE_TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(
        "hermes_octo_plugin.adapter.MEDIA_TEMP_DIR", str(tmp_path / "media")
    )

    event = await _deliver_inbound_payload(
        adapter,
        b'{"type": 8, "name": "report.pdf", '
        b'"url": "https://files.example/report.pdf"}',
    )

    assert session.get.call_count == 1
    assert event.media_types == ["application/octet-stream"]
    assert len(event.media_urls) == 1
    local_path = event.media_urls[0]
    assert Path(local_path).read_bytes() == b"%PDF"
    assert set(tmp_path.iterdir()) == {Path(local_path)}
    assert local_path in event.text
    assert "https://files.example/report.pdf" not in event.text


@pytest.mark.asyncio
async def test_inbound_small_text_file_inlines_from_one_request_without_media(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    adapter = _inbound_event_adapter()
    session = MagicMock()
    session.get.return_value = _ChunkedResponse(b"hello")
    adapter._http_session = session
    file_temp_dir = tmp_path / "files"
    media_temp_dir = tmp_path / "media"
    monkeypatch.setattr("hermes_octo_plugin.adapter.FILE_TEMP_DIR", str(file_temp_dir))
    monkeypatch.setattr("hermes_octo_plugin.adapter.MEDIA_TEMP_DIR", str(media_temp_dir))

    event = await _deliver_inbound_payload(
        adapter,
        b'{"type": 8, "name": "notes.txt", '
        b'"url": "https://files.example/notes.txt"}',
    )

    assert session.get.call_count == 1
    assert event.text == (
        "[文件: notes.txt]\n\n--- 文件内容 ---\nhello\n--- 文件结束 ---"
    )
    assert event.media_urls == []
    assert event.media_types == []
    assert event.message_type.value == "text"
    assert "https://files.example/notes.txt" not in event.text
    assert list(file_temp_dir.iterdir()) == []
    assert not media_temp_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_type", "mime", "label", "url"),
    [
        (MessageType.Image, "image/jpeg", "[图片]", "https://public.example/photo.jpg"),
        (
            MessageType.GIF,
            "image/gif",
            "[GIF]",
            "https://files.example/animation.gif?X-Amz-Signature=signed-secret",
        ),
        (
            MessageType.Voice,
            "audio/ogg",
            "[语音消息]",
            "http://169.254.169.254/latest/meta-data/token",
        ),
        (
            MessageType.Video,
            "video/mp4",
            "[视频]",
            "https://files.example/video.mp4?signature=another-secret",
        ),
    ],
)
async def test_inbound_media_text_never_includes_remote_url(
    message_type, mime, label, url
):
    adapter = _inbound_event_adapter()
    adapter._download_inbound_media_to_local = AsyncMock(return_value="/tmp/local-media")

    event = await _deliver_inbound_payload(
        adapter,
        f'{{"type": {int(message_type)}, "url": "{url}"}}'.encode(),
    )

    assert event.text == label
    assert event.media_urls == ["/tmp/local-media"]
    assert event.media_types == [mime]
    assert url not in event.text


@pytest.mark.asyncio
async def test_inbound_rich_text_localizes_each_image_and_keeps_success_order():
    adapter = _inbound_event_adapter()
    adapter._download_inbound_media_to_local = AsyncMock(
        side_effect=["/tmp/first.jpg", None, "/tmp/third.gif"]
    )

    event = await _deliver_inbound_payload(
        adapter,
        b'{"type": 14, "content": ['
        b'{"type": "image", "url": "https://files.example/first.jpg"},'
        b'{"type": "text", "text": "between"},'
        b'{"type": "image", "url": "https://files.example/rejected.png"},'
        b'{"type": "image", "url": "https://files.example/third.gif"}'
        b']}',
    )

    assert [
        call.args
        for call in adapter._download_inbound_media_to_local.await_args_list
    ] == [
        ("https://files.example/first.jpg", "image/jpeg"),
        ("https://files.example/rejected.png", "image/png"),
        ("https://files.example/third.gif", "image/gif"),
    ]
    assert event.media_urls == ["/tmp/first.jpg", "/tmp/third.gif"]
    assert event.media_types == ["image/jpeg", "image/gif"]


@pytest.mark.asyncio
async def test_inbound_file_download_failure_never_forwards_remote_url():
    adapter = _inbound_event_adapter()
    session = MagicMock()
    session.get.return_value = _NotFoundResponse()
    adapter._http_session = session

    event = await _deliver_inbound_payload(
        adapter,
        b'{"type": 8, "name": "rejected.pdf", '
        b'"url": "https://files.example/rejected.pdf"}',
    )

    assert session.get.call_count == 1
    assert event.text == "[文件: rejected.pdf - 下载失败 HTTP 404]"
    assert event.media_urls == []
    assert event.media_types == []
    assert "https://files.example/rejected.pdf" not in event.text




def test_nested_file_previews_never_include_remote_urls():
    adapter = make_bare_adapter()
    adapter._api_url = "https://api.octo.example"
    url = "https://files.example/private.pdf"

    quoted = adapter._resolve_quoted_message_text(
        {"type": 8, "name": "private.pdf", "url": url}
    )
    forwarded = adapter._resolve_inner_message_text(
        {"payload": {"type": 8, "name": "private.pdf", "url": url}}
    )

    assert quoted == "[文件: private.pdf]"
    assert forwarded == "[文件: private.pdf]"


@pytest.mark.asyncio
async def test_inbound_file_retry_logs_no_filename_or_raw_exception(caplog, monkeypatch):
    adapter = make_bare_adapter()
    filename = "payroll-2026.pdf"
    url = "https://files.example/private.pdf?X-Amz-Signature=signed-secret"
    adapter._http_session = MagicMock()
    adapter._http_session.get = MagicMock(
        side_effect=RuntimeError(f"download failed for {filename}: {url}")
    )
    monkeypatch.setattr("hermes_octo_plugin.adapter.asyncio.sleep", AsyncMock())

    result = await adapter._resolve_inbound_file(url, filename, None)

    assert result.content == f"[文件: {filename} - 下载失败]"
    assert adapter._http_session.get.call_count == 3
    assert filename not in caplog.text
    assert url not in caplog.text
    assert "RuntimeError" in caplog.text
@pytest.mark.asyncio
async def test_inbound_rich_text_rejected_url_never_reaches_media_urls():
    adapter = _inbound_event_adapter()
    adapter._download_inbound_media_to_local = AsyncMock(return_value=None)

    event = await _deliver_inbound_payload(
        adapter,
        b'{"type": 14, "content": ['
        b'{"type": "image", "url": "http://127.0.0.1/private.png"}'
        b']}',
    )

    adapter._download_inbound_media_to_local.assert_awaited_once_with(
        "http://127.0.0.1/private.png",
        "image/png",
    )
    assert event.media_urls == []
    assert event.media_types == []
    assert "http://127.0.0.1/private.png" not in event.text


@pytest.mark.asyncio
async def test_native_image_download_failure_does_not_fall_back_to_remote_url():
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    send = AsyncMock()

    with (
        patch.object(
            api,
            "download_file",
            AsyncMock(side_effect=RuntimeError("download rejected")),
        ),
        patch.object(api, "send_media_message", send),
    ):
        result = await adapter.send_image(
            "group-1",
            "https://source.example/image.png",
        )

    assert result.success is False
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_private_media_is_not_forwarded_after_local_download_rejection():
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._robot_id = "bot-1"
    adapter._aes_key = b"key"
    adapter._aes_iv = b"iv"
    adapter._resolve_sender_name = AsyncMock(return_value="Alice")
    adapter._send_typing_safe = AsyncMock()
    adapter.build_source = MagicMock(return_value=SimpleNamespace())
    adapter.handle_message = AsyncMock()
    raw = b'{"type": 2, "url": "http://127.0.0.1/private.png"}'
    recv = SimpleNamespace(
        message_id="message-1",
        message_seq=1,
        from_uid="user-1",
        channel_id="bot-1",
        channel_type=ChannelType.DM,
        timestamp=1,
        encrypted_payload=raw,
    )

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(recv)

    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == []
    assert event.media_types == []


@pytest.mark.asyncio
async def test_guarded_download_failure_never_forwards_remote_media_url():
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._robot_id = "bot-1"
    adapter._aes_key = b"key"
    adapter._aes_iv = b"iv"
    adapter._resolve_sender_name = AsyncMock(return_value="Alice")
    adapter._send_typing_safe = AsyncMock()
    adapter.build_source = MagicMock(return_value=SimpleNamespace())
    adapter.handle_message = AsyncMock()
    adapter._download_inbound_media_to_local = AsyncMock(return_value=None)
    raw = b'{"type": 2, "url": "https://attacker.example/image.png"}'
    recv = SimpleNamespace(
        message_id="message-1",
        message_seq=1,
        from_uid="user-1",
        channel_id="bot-1",
        channel_type=ChannelType.DM,
        timestamp=1,
        encrypted_payload=raw,
    )

    with patch("hermes_octo_plugin.adapter.aes_decrypt", return_value=raw):
        await adapter._handle_recv(recv)

    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == []
    assert event.media_types == []
    adapter._download_inbound_media_to_local.assert_awaited_once_with(
        "https://attacker.example/image.png",
        "image/jpeg",
    )


@pytest.mark.asyncio
async def test_native_media_failure_redacts_signed_source_from_result_and_logs(caplog):
    adapter = make_bare_adapter()
    adapter.platform = SimpleNamespace(value="octo")
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    source = "https://source.example/image.png?token=signed-secret"
    download = AsyncMock(side_effect=RuntimeError(f"rejected {source}"))

    with (
        patch.object(api, "download_file", download),
        patch.object(api, "send_media_message", AsyncMock()),
    ):
        result = await adapter.send_image("group-1", source)

    assert result.success is False
    assert "rejected https://source.example/image.png" in (result.error or "")
    assert source not in (result.error or "")
    assert "signed-secret" not in (result.error or "")
    assert source not in caplog.text
    assert "signed-secret" not in caplog.text
    download.assert_awaited_once()


@pytest.mark.asyncio
async def test_normal_send_failure_redacts_signed_error_and_keeps_retryability(caplog):
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    source = "https://source.example/message?SessionToken=signed-secret"
    send = AsyncMock(side_effect=RuntimeError(f"timeout while sending {source}"))

    with patch.object(api, "send_message", send):
        result = await adapter.send("group-1", "message")

    assert result.success is False
    assert result.retryable is True
    assert "timeout while sending https://source.example/message" in (result.error or "")
    assert source not in (result.error or "")
    assert "signed-secret" not in (result.error or "")
    assert source not in caplog.text
    assert "signed-secret" not in caplog.text
    send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["send_document", "send_voice", "send_video"])
async def test_native_media_send_failures_redact_signed_errors(
    caplog,
    method_name: str,
) -> None:
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    source = "https://source.example/media?Credentials=signed-secret"
    load = AsyncMock(side_effect=RuntimeError(f"rejected {source}"))

    with patch.object(adapter, "_load_outbound_media", load):
        result = await getattr(adapter, method_name)("group-1", source)

    assert result.success is False
    assert "rejected https://source.example/media" in (result.error or "")
    assert source not in (result.error or "")
    assert "signed-secret" not in (result.error or "")
    assert source not in caplog.text
    assert "signed-secret" not in caplog.text
    load.assert_awaited_once_with(source)
