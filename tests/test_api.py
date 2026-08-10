"""
Tests for hermes_octo_plugin.api — API function signatures and parameter checks (mock).
"""

import pytest
import asyncio
import json
from unittest.mock import ANY, AsyncMock, MagicMock, patch
import aiohttp
from hermes_octo_plugin import api
from hermes_octo_plugin.api import (
    post_json,
    register_bot,
    send_message,
    send_typing,
    send_media_message,
    send_read_receipt,
    stream_start,
    stream_end,
    get_upload_presign,
    upload_file_to_presigned_url,
    upload_and_get_url,
    download_file,
    get_channel_messages,
    fetch_bot_groups,
    get_group_members,
    get_group_info,
    fetch_user_info,
    get_group_md,
    update_group_md,
    update_group,
    list_threads,
    list_thread_members,
    infer_content_type,
    parse_image_dimensions,
)
from hermes_octo_plugin.types import ChannelType, MentionEntity, MessageType


class _FailedApiResponse:
    """Small async-response double for API failure-contract tests."""

    ok = False
    status = 503
    reason = "Service Unavailable"
    request_info = MagicMock()
    history = ()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return "backend unavailable: Bearer secret-token-from-backend"

    async def json(self, **_kwargs):
        return {"error": "backend unavailable"}


class _FailedApiSession:
    def __init__(self):
        self.get = MagicMock(return_value=_FailedApiResponse())
        self.post = MagicMock(return_value=_FailedApiResponse())
        self.put = MagicMock(return_value=_FailedApiResponse())


class _NetworkFailedSession:
    def __init__(self):
        self.get = MagicMock(side_effect=aiohttp.ClientConnectionError("offline"))


class _RedirectResponse:
    ok = True
    status = 302
    headers = {"Location": "http://127.0.0.1/admin"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _EmptyDownloadContent:
    async def iter_any(self):
        if False:
            yield b""


class _SuccessfulDownloadResponse:
    ok = True
    status = 200
    headers = {"Content-Type": "application/octet-stream"}
    content = _EmptyDownloadContent()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _UserInfoResponse:
    ok = True
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return {"uid": "returned", "name": "User"}


class TestApiFailureTruth:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("call", "kwargs"),
        [
            (fetch_bot_groups, {}),
            (get_group_members, {"group_no": "group-1"}),
            (get_group_md, {"group_no": "group-1"}),
            (
                get_channel_messages,
                {"channel_id": "group-1", "channel_type": ChannelType.Group},
            ),
        ],
    )
    async def test_backend_failure_is_not_normalized_to_an_empty_result(self, call, kwargs):
        """Only protocol-defined absence may become ``None``/an empty list."""
        with pytest.raises(api.OctoApiError, match="HTTP 503"):
            await call(
                _FailedApiSession(),
                "https://api.example.invalid",
                "test-token",
                **kwargs,
            )

    @pytest.mark.asyncio
    async def test_network_failure_is_not_normalized_to_an_empty_member_roster(self):
        with pytest.raises(aiohttp.ClientConnectionError, match="offline"):
            await get_group_members(
                _NetworkFailedSession(),
                "https://api.example.invalid",
                "test-token",
                "group-1",
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("call", "kwargs"),
        [
            (
                get_upload_presign,
                {
                    "filename": "report.pdf",
                    "file_size": 7,
                    "content_type": "application/pdf",
                },
            ),
            (get_group_info, {"group_no": "group-1"}),
        ],
    )
    async def test_authenticated_api_errors_never_expose_response_bodies(self, call, kwargs):
        with pytest.raises(api.OctoApiError, match="HTTP 503") as exc_info:
            await call(
                _FailedApiSession(),
                "https://api.example.invalid",
                "test-token",
                **kwargs,
            )

        assert "secret-token-from-backend" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_presigned_upload_error_never_exposes_response_body_or_url(self):
        upload_url = "https://storage.example/upload?signature=secret"
        with pytest.raises(RuntimeError, match="HTTP 503") as exc_info:
            await upload_file_to_presigned_url(
                _FailedApiSession(),
                upload_url=upload_url,
                download_url="https://cdn.example/file",
                file_data=b"payload",
                content_type="application/octet-stream",
            )

        assert "secret-token-from-backend" not in str(exc_info.value)
        assert upload_url not in str(exc_info.value)
        assert "signature=secret" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_error_never_exposes_signed_source_url(self):
        signed_url = "https://files.example.invalid/report?token=signed-secret"

        with pytest.raises(RuntimeError, match="HTTP 503") as exc_info:
            await download_file(_FailedApiSession(), signed_url)

        assert signed_url not in str(exc_info.value)
        assert "signed-secret" not in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "unsafe_url",
        [
            "http://127.0.0.1/private",
            "http://2130706433/private",
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
        ],
    )
    async def test_download_rejects_unsafe_initial_url_before_io(self, unsafe_url: str):
        session = MagicMock()
        session.get = MagicMock(side_effect=AssertionError("network I/O attempted"))

        with pytest.raises(RuntimeError, match="unsafe download URL"):
            await download_file(session, unsafe_url)

        session.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_download_rejects_unsafe_redirect_before_following_it(self):
        session = MagicMock()
        session.get = MagicMock(return_value=_RedirectResponse())

        with pytest.raises(RuntimeError, match="unsafe download URL"):
            await download_file(session, "https://files.example.invalid/report")

        session.get.assert_called_once()
        assert session.get.call_args.kwargs["allow_redirects"] is False

    @pytest.mark.asyncio
    async def test_download_allows_exact_guarded_private_origin(self):
        session = MagicMock()
        session.connector._ssrf_resolver._trusted_hosts = {"10.0.0.8"}
        session.get = MagicMock(return_value=_SuccessfulDownloadResponse())

        data, content_type, filename = await download_file(
            session, "http://10.0.0.8/report.bin"
        )

        assert data == b""
        assert content_type == "application/octet-stream"
        assert filename == "report.bin"
        session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_outbound_download_policy_accepts_private_media_sources(self):
        session = MagicMock()
        session.get = MagicMock(return_value=_SuccessfulDownloadResponse())

        data, content_type, filename = await download_file(
            session,
            "http://10.0.0.8/report.bin",
            enforce_host_safety=False,
        )

        assert data == b""
        assert content_type == "application/octet-stream"
        assert filename == "report.bin"
        session.get.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://100.100.100.200/latest/meta-data/",
            "http://[fd00:ec2::254]/latest/meta-data/",
        ],
    )
    async def test_outbound_download_policy_never_allows_metadata_hosts(
        self,
        url: str,
    ):
        session = MagicMock()

        with pytest.raises(RuntimeError, match="unsafe download URL"):
            await download_file(
                session,
                url,
                enforce_host_safety=False,
            )

        session.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_user_info_passes_uid_as_query_param(self):
        session = MagicMock()
        session.get = MagicMock(return_value=_UserInfoResponse())
        uid = "person&admin=true#fragment"

        result = await fetch_user_info(session, "https://octo.test", "token", uid)

        assert result == {"uid": "returned", "name": "User", "avatar": ""}
        args, kwargs = session.get.call_args
        assert args == ("https://octo.test/v1/bot/user/info",)
        assert kwargs["params"] == {"uid": uid}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("call", "kwargs"),
        [
            (api.get_group_members, {"group_no": "../space/members?limit=500#"}),
            (
                api.get_thread,
                {"group_no": "group-1", "short_id": "../members"},
            ),
        ],
    )
    async def test_api_rejects_malformed_path_segments_before_io(self, call, kwargs):
        session = MagicMock()
        session.get = MagicMock(side_effect=AssertionError("network I/O attempted"))

        with pytest.raises(ValueError, match="invalid Octo"):
            await call(
                session,
                "https://api.example.invalid",
                "test-token",
                **kwargs,
            )

        session.get.assert_not_called()


class TestUpdateGroupRoute:
    @pytest.mark.asyncio
    async def test_uses_current_put_info_route_and_omits_unsupplied_fields(self):
        put_json = AsyncMock(return_value=None)
        post_json = AsyncMock(return_value=None)
        with (
            patch.object(api, "put_json", put_json),
            patch.object(api, "post_json", post_json),
        ):
            await update_group(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                group_no="group-1",
                name="Renamed group",
            )

        put_json.assert_awaited_once_with(
            ANY,
            "https://api.example.invalid",
            "test-token",
            "/v1/bot/groups/group-1/info",
            {"name": "Renamed group"},
        )
        post_json.assert_not_awaited()


class TestSendIdentity:
    @pytest.mark.asyncio
    async def test_normalizes_int64_message_identity_without_fabricating_fields(self):
        with patch.object(
            api,
            "post_json",
            AsyncMock(
                return_value={
                    "message_id": 9223372036854775807,
                    "message_seq": "7",
                    "client_msg_no": "client-1",
                }
            ),
        ):
            result = await send_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                "group-1",
                ChannelType.Group,
                "hello",
            )

        assert result.message_id == "9223372036854775807"
        assert result.message_seq == 7
        assert result.client_msg_no == "client-1"

    @pytest.mark.asyncio
    async def test_text_send_accepts_and_returns_the_dedup_key_when_server_omits_it(self):
        post_json = AsyncMock(return_value={"message_id": "message-1"})
        with patch.object(api, "post_json", post_json):
            result = await send_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                "group-1",
                ChannelType.Group,
                "hello",
                client_msg_no="dedup-1",
                on_behalf_of="grantor-1",
            )

        assert post_json.await_args.args[4]["client_msg_no"] == "dedup-1"
        assert post_json.await_args.args[4]["on_behalf_of"] == "grantor-1"
        assert result.client_msg_no == "dedup-1"

    @pytest.mark.asyncio
    async def test_typing_indicator_uses_the_configured_sending_identity(self):
        with patch.object(api, "post_json", AsyncMock()) as post_json:
            await send_typing(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                "group-1",
                ChannelType.Group,
                on_behalf_of="grantor-1",
            )

        assert post_json.await_args.args[4] == {
            "channel_id": "group-1",
            "channel_type": ChannelType.Group,
            "on_behalf_of": "grantor-1",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("response", [None, {}, {"message_seq": 7}])
    async def test_send_rejects_success_response_without_message_id(self, response):
        with patch.object(api, "post_json", AsyncMock(return_value=response)):
            with pytest.raises(api.OctoApiError, match="missing message_id"):
                await send_message(
                    MagicMock(),
                    "https://api.example.invalid",
                    "test-token",
                    "group-1",
                    ChannelType.Group,
                    "hello",
                )

    @pytest.mark.asyncio
    async def test_native_text_edit_uses_the_proven_octo_envelope(self):
        with patch.object(api, "post_json", AsyncMock(return_value=None)) as post_json:
            await api.edit_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                channel_id="group-1",
                channel_type=ChannelType.Group,
                message_id="9223372036854775807",
                content="updated",
                finalize=True,
                on_behalf_of="grantor-1",
            )

        path = post_json.await_args.args[3]
        body = post_json.await_args.args[4]
        assert path == "/v1/bot/message/edit"
        assert body == {
            "message_id": "9223372036854775807",
            "channel_id": "group-1",
            "channel_type": ChannelType.Group,
            "content_edit": '{"type": 1, "content": "updated"}',
            "on_behalf_of": "grantor-1",
        }
        assert "finalize" not in body

    @pytest.mark.asyncio
    async def test_native_text_edit_serializes_mention_entities(self):
        entity = MentionEntity(uid="user-1", offset=6, length=4)
        with patch.object(api, "post_json", AsyncMock(return_value=None)) as post_json:
            await api.edit_message(
                MagicMock(),
                "https://api.example.invalid",
                "test-token",
                channel_id="group-1",
                channel_type=ChannelType.Group,
                message_id="message-1",
                content="hello @董振兴",
                mention_uids=["user-1"],
                mention_entities=[entity],
            )

        frame = json.loads(post_json.await_args.args[4]["content_edit"])
        assert frame == {
            "type": 1,
            "content": "hello @董振兴",
            "mention": {
                "uids": ["user-1"],
                "entities": [{"uid": "user-1", "offset": 6, "length": 4}],
            },
        }


class TestHeartbeatApi:
    @pytest.mark.asyncio
    async def test_posts_authenticated_empty_heartbeat_envelope(self):
        with patch.object(api, "post_json", AsyncMock(return_value=None)) as post_json:
            await api.send_heartbeat(
                MagicMock(), "https://api.example.invalid", "test-token"
            )

        post_json.assert_awaited_once_with(
            ANY,
            "https://api.example.invalid",
            "test-token",
            "/v1/bot/heartbeat",
            {},
        )


class TestInferContentType:
    def test_jpeg(self):
        assert infer_content_type("photo.jpg") == "image/jpeg"
        assert infer_content_type("photo.jpeg") == "image/jpeg"

    def test_png(self):
        assert infer_content_type("image.png") == "image/png"

    def test_mp4(self):
        assert infer_content_type("video.mp4") == "video/mp4"

    def test_mp3(self):
        assert infer_content_type("audio.mp3") == "audio/mpeg"

    def test_pdf(self):
        assert infer_content_type("doc.pdf") == "application/pdf"

    def test_unknown(self):
        assert infer_content_type("file.xyz") == "application/octet-stream"

    def test_case_insensitive(self):
        assert infer_content_type("Photo.JPG") == "image/jpeg"

    def test_no_extension(self):
        assert infer_content_type("noext") == "application/octet-stream"


class TestParseImageDimensions:
    def test_png(self):
        # Minimal PNG header (IHDR chunk)
        png_header = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk
            b"\x00\x00\x01\x00"    # width = 256
            b"\x00\x00\x00\x80"    # height = 128
            b"\x08\x02\x00\x00\x00"  # bit depth, color type, etc.
        )
        result = parse_image_dimensions(png_header, "image/png")
        assert result == (256, 128)

    def test_gif(self):
        # Minimal GIF header
        gif_header = (
            b"GIF89a"
            b"\x40\x01"  # width = 320 (LE)
            b"\xf0\x00"  # height = 240 (LE)
            b"\x00\x00"  # extra
        )
        result = parse_image_dimensions(gif_header, "image/gif")
        assert result == (320, 240)

    def test_too_small(self):
        result = parse_image_dimensions(b"\x89PNG", "image/png")
        assert result is None

    def test_unknown_mime(self):
        result = parse_image_dimensions(b"\x00" * 100, "application/octet-stream")
        assert result is None


class TestFunctionSignatures:
    """Verify that API functions accept the expected parameters."""

    def test_register_bot_signature(self):
        """register_bot should accept session, api_url, bot_token, force_refresh."""
        import inspect
        sig = inspect.signature(register_bot)
        params = list(sig.parameters.keys())
        assert "session" in params
        assert "api_url" in params
        assert "bot_token" in params
        assert "force_refresh" in params

    def test_send_message_signature(self):
        sig = inspect.signature(send_message)
        params = list(sig.parameters.keys())
        assert "session" in params
        assert "channel_id" in params
        assert "channel_type" in params
        assert "content" in params
        assert "mention_uids" in params
        assert "mention_entities" in params
        assert "mention_all" in params
        assert "stream_no" in params
        assert "reply_msg_id" in params

    def test_send_read_receipt_signature(self):
        sig = inspect.signature(send_read_receipt)
        params = list(sig.parameters.keys())
        assert "channel_id" in params
        assert "channel_type" in params
        assert "message_ids" in params

    def test_stream_start_signature(self):
        sig = inspect.signature(stream_start)
        params = list(sig.parameters.keys())
        assert "channel_id" in params
        assert "channel_type" in params
        assert "initial_content" in params

    def test_stream_end_signature(self):
        sig = inspect.signature(stream_end)
        params = list(sig.parameters.keys())
        assert "stream_no" in params
        assert "channel_id" in params
        assert "channel_type" in params

    def test_get_upload_presign_signature(self):
        params = inspect.signature(get_upload_presign).parameters
        assert {"filename", "file_size", "content_type"} <= set(params)

    def test_upload_file_to_presigned_url_signature(self):
        params = inspect.signature(upload_file_to_presigned_url).parameters
        assert {
            "upload_url",
            "download_url",
            "file_data",
            "content_type",
            "content_disposition",
        } <= set(params)

    def test_get_channel_messages_signature(self):
        sig = inspect.signature(get_channel_messages)
        params = list(sig.parameters.keys())
        assert "channel_id" in params
        assert "channel_type" in params
        assert "limit" in params

    def test_get_group_md_signature(self):
        sig = inspect.signature(get_group_md)
        params = list(sig.parameters.keys())
        assert "group_no" in params

    def test_update_group_md_signature(self):
        sig = inspect.signature(update_group_md)
        params = list(sig.parameters.keys())
        assert "group_no" in params
        assert "content" in params

    def test_fetch_user_info_signature(self):
        sig = inspect.signature(fetch_user_info)
        params = list(sig.parameters.keys())
        assert "uid" in params


import inspect


class TestSendReadReceipt:
    """Test send_read_receipt parameter building."""

    @pytest.mark.asyncio
    async def test_sends_correct_payload(self):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.text = AsyncMock(return_value="null")
        mock_response.json = AsyncMock(return_value=None)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_response)

        await send_read_receipt(
            mock_session,
            "https://api.example.com",
            "token",
            "channel1",
            ChannelType.Group,
            ["msg1", "msg2"],
        )

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert "json" in call_kwargs.kwargs or len(call_kwargs.args) > 1


class TestStreamAPI:
    """Test stream API parameter building."""

    @pytest.mark.asyncio
    async def test_stream_start_encodes_payload(self):
        """stream_start should base64-encode the initial payload."""
        import base64
        import json

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.text = AsyncMock(return_value='{"stream_no": "s123"}')
        mock_response.json = AsyncMock(return_value={"stream_no": "s123"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_response)

        result = await stream_start(
            mock_session, "https://api.example.com", "token",
            "channel1", ChannelType.Group, "Hello!",
        )

        assert result == "s123"
        mock_session.post.assert_called_once()


class TestGetChannelMessages:
    """Test channel messages API."""

    @pytest.mark.asyncio
    async def test_parses_base64_payload(self):
        import base64
        import json

        encoded_payload = base64.b64encode(
            json.dumps({"type": 1, "content": "hello"}).encode()
        ).decode()

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.text = AsyncMock(return_value=json.dumps({
            "messages": [
                {"from_uid": "user1", "payload": encoded_payload, "timestamp": 1000},
            ]
        }))
        mock_response.json = AsyncMock(return_value={
            "messages": [
                {"from_uid": "user1", "payload": encoded_payload, "timestamp": 1000},
            ]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_response)

        messages = await get_channel_messages(
            mock_session, "https://api.example.com", "token",
            "channel1", ChannelType.Group, limit=10,
        )

        assert len(messages) == 1
        assert messages[0]["content"] == "hello"
        assert messages[0]["from_uid"] == "user1"


class TestThreadApi:
    """Thread endpoints have returned both wrapped dicts and bare arrays."""

    @pytest.mark.asyncio
    async def test_list_threads_accepts_bare_array_response(self):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.text = AsyncMock(return_value='[{"short_id":"t1","name":"测试 octo"}]')
        mock_response.json = AsyncMock(return_value=[{"short_id": "t1", "name": "测试 octo"}])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_response)

        threads = await list_threads(
            mock_session, "https://api.example.com", "token", group_no="g1",
        )

        assert threads == [{"short_id": "t1", "name": "测试 octo"}]

    @pytest.mark.asyncio
    async def test_list_thread_members_accepts_bare_array_response(self):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.text = AsyncMock(return_value='[{"uid":"u1","name":"董振兴"}]')
        mock_response.json = AsyncMock(return_value=[{"uid": "u1", "name": "董振兴"}])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_response)

        members = await list_thread_members(
            mock_session, "https://api.example.com", "token", group_no="g1", short_id="t1",
        )

        assert members == [{"uid": "u1", "name": "董振兴"}]
