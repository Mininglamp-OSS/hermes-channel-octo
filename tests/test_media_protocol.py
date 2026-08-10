"""Media protocol safety and fidelity tests."""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from hermes_octo_plugin import api
from hermes_octo_plugin.adapter import (
    OctoAdapter,
    _SSRFGuardConnector,
    _SSRFGuardResolver,
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
    resolver = _SSRFGuardResolver(trusted_hosts={"api.octo.example"})
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
        (True, {"api.octo.example", "cdn.octo.example"}),
    ],
)
async def test_http_session_trusts_configured_private_hosts_only_with_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    allow_private: bool,
    expected_trusted: set[str],
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
            "hermes_octo_plugin.adapter._SSRFGuardConnector",
            return_value=connector,
        ) as connector_cls,
        patch("hermes_octo_plugin.adapter.aiohttp.ClientSession", return_value=session),
    ):
        assert adapter._new_http_session() is session

    resolver = connector_cls.call_args.kwargs["resolver"]
    assert resolver._trusted_hosts == expected_trusted
    await resolver.close()


@pytest.mark.asyncio
async def test_ssrf_resolver_rejects_metadata_hostname_before_dns():
    resolver = _SSRFGuardResolver(trusted_hosts={"metadata.google.internal"})
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
    resolver = _SSRFGuardResolver(trusted_hosts={"127.0.0.1"})
    connector = _SSRFGuardConnector(resolver=resolver)
    try:
        records = await connector._resolve_host("127.0.0.1", 8080)
    finally:
        await connector.close()

    assert records[0]["host"] == "127.0.0.1"
    assert records[0]["family"] in {socket.AF_UNSPEC, socket.AF_INET}


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
            caption="image caption", reply_to="parent-message",
        )).success
        assert (await adapter.send_document(
            "s14_user-1", "https://source.example/report.pdf",
            caption="file caption", reply_to="parent-message",
        )).success
        assert (await adapter.send_voice(
            "s14_user-1", "https://source.example/voice.amr",
            caption="voice caption", reply_to="parent-message",
            duration=3,
        )).success
        assert (await adapter.send_video(
            "s14_user-1", "https://source.example/video.mp4",
            caption="video caption", reply_to="parent-message",
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
        "image caption", "file caption", "voice caption", "video caption",
    ]
    for call in send_text.await_args_list:
        assert call.kwargs["channel_id"] == "user-1"
        assert call.kwargs["channel_type"] == ChannelType.DM
        assert call.kwargs["reply_msg_id"] == "parent-message"


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
    "method_name",
    ["send_document", "send_voice", "send_video"],
)
async def test_native_remote_media_uses_the_server_upload_limit(method_name: str):
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "test-token"
    download = AsyncMock(
        return_value=(b"media", "application/octet-stream", "source.bin")
    )

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
    assert download.await_args.kwargs["enforce_host_safety"] is False

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
