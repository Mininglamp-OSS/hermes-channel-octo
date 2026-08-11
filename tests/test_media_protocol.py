"""Media protocol safety and fidelity tests."""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from hermes_octo_plugin import api, transport as transport_module
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
@pytest.mark.parametrize(
    ("allow_private", "expected_trusted"),
    [
        (False, set()),
        (
            True,
            {
                ("https", "api.octo.example", 443),
                ("https", "cdn.octo.example", 443),
            },
        ),
    ],
)
async def test_http_session_trusts_configured_private_origins_only_with_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    allow_private: bool,
    expected_trusted: set[tuple[str, str, int]],
):
    adapter = make_bare_adapter()
    adapter._api_url = "https://api.octo.example/v1"
    adapter._cdn_url = "https://cdn.octo.example/assets"
    if allow_private:
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
    assert resolver.policy.trusted_download_origins() == expected_trusted
    await resolver.close()


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
    ["127.0.0.1", "2130706433", "127.1", "0177.0.0.1"],
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

@pytest.mark.asyncio
async def test_private_upload_origin_does_not_expand_shared_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "1")
    policy = TransportPolicy({"https://api.example"})
    policy.trust_validated_upload_origin(
        "http://storage.example:8080/upload"
    )
    resolver = _SSRFGuardResolver(policy=policy)
    resolver._delegate.resolve = AsyncMock(
        return_value=[
            {
                "hostname": "storage.example",
                "host": "10.0.0.8",
                "port": 8080,
                "family": socket.AF_INET,
                "proto": 6,
                "flags": 0,
            }
        ]
    )

    try:
        with pytest.raises(OSError, match="unsafe address"):
            await resolver.resolve("storage.example", 8080)
    finally:
        await resolver.close()


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
async def test_guarded_websocket_socket_trusts_exact_private_origin_only_with_opt_in(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "1")
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
    resolver.resolve.assert_awaited_once_with(
        "socket.internal",
        9443,
        family=socket.AF_UNSPEC,
    )
    resolver.close.assert_awaited_once()


def test_private_host_policy_still_rejects_ipv4_mapped_link_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "1")
    policy = TransportPolicy()

    with pytest.raises(RuntimeError, match="unsafe presigned upload URL"):
        policy.trust_validated_upload_origin(
            "http://[::ffff:169.254.169.254]/latest/meta-data/"
        )


def test_private_host_policy_rejects_ipv4_mapped_metadata_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "1")
    policy = TransportPolicy()

    with pytest.raises(RuntimeError, match="unsafe presigned upload URL"):
        policy.trust_validated_upload_origin(
            "http://[::ffff:6464:64c8]/latest/meta-data/"
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

    assert result == "[文件: report.pdf - 下载失败 HTTP 302]"
    assert response.content.called is False


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

    assert [call.kwargs["content"] for call in send_text.await_args_list] == [
        "@Alice image caption", "@Alice file caption",
        "@Alice voice caption", "@Alice video caption",
    ]
    for call in send_text.await_args_list:
        assert call.kwargs["channel_id"] == "user-1"
        assert call.kwargs["channel_type"] == ChannelType.DM
        assert call.kwargs["reply_msg_id"] == "parent-message"
        assert call.kwargs["mention_uids"] == ["u1"]
        assert [entity.uid for entity in call.kwargs["mention_entities"]] == ["u1"]


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
        ),
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
    assert download.await_args.kwargs["enforce_host_safety"] is True



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
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    source = "https://source.example/image.png?token=signed-secret"

    with (
        patch.object(
            api,
            "download_file",
            AsyncMock(side_effect=RuntimeError(f"rejected {source}")),
        ),
        patch.object(api, "send_media_message", AsyncMock()),
    ):
        result = await adapter.send_image("group-1", source)

    assert result.success is False
    assert source not in (result.error or "")
    assert "signed-secret" not in caplog.text
