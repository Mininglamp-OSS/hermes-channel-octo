"""Backend-agnostic presigned media upload and filename decoding contracts."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import unquote

from hermes_octo_plugin.api import (
    download_file,
    get_upload_presign,
    upload_and_get_url,
    upload_file_to_presigned_url,
)




# ---------------------------------------------------------------------------
# Filename decoding in download_file URL path fallback
# ---------------------------------------------------------------------------
class TestFilenameDecoding:
    """Test that the URL path fallback in download_file decodes percent-encoding."""

    def test_unquote_chinese(self):
        """Verify urllib.parse.unquote decodes Chinese characters."""
        assert unquote("%E5%AE%A1%E6%9F%A5.xlsx") == "审查.xlsx"

    def test_unquote_spaces(self):
        assert unquote("my%20report.xlsx") == "my report.xlsx"

    def test_unquote_malformed_sequence(self):
        """Python's unquote returns malformed sequences unchanged."""
        assert unquote("file%GG.txt") == "file%GG.txt"

    def test_unquote_plain_ascii(self):
        assert unquote("report.xlsx") == "report.xlsx"

class TestPresignedUpload:
    @pytest.mark.asyncio
    async def test_get_presign_sends_exact_size_and_normalizes_signed_headers(self):
        response = AsyncMock()
        response.ok = True
        response.json = AsyncMock(
            return_value={
                "method": "PUT",
                "uploadUrl": "https://storage.example/upload?signature=secret",
                "downloadUrl": "https://cdn.example/report.txt",
                "contentDisposition": 'attachment; filename="report.txt"',
            }
        )
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.get.return_value = response

        result = await get_upload_presign(
            session,
            "https://api.example",
            "bot-token",
            filename="report.txt",
            file_size=4,
            content_type="text/plain",
        )

        url = session.get.call_args.args[0]
        assert "/v1/bot/upload/presigned?" in url
        assert "filename=report.txt" in url
        assert "fileSize=4" in url
        assert "contentType=text%2Fplain" in url
        assert result == {
            "uploadUrl": "https://storage.example/upload?signature=secret",
            "downloadUrl": "https://cdn.example/report.txt",
            "contentType": "application/octet-stream",
            "contentDisposition": 'attachment; filename="report.txt"',
        }

    @pytest.mark.asyncio
    async def test_get_presign_preserves_backend_signed_headers(self):
        response = AsyncMock()
        response.ok = True
        response.json = AsyncMock(
            return_value={
                "method": "PUT",
                "uploadUrl": "https://storage.example/upload?signature=secret",
                "downloadUrl": "https://cdn.example/report.txt",
                "headers": {
                    "content-type": "text/plain; charset=utf-8",
                    "x-amz-meta-scope": "octo",
                },
            }
        )
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.get.return_value = response

        result = await get_upload_presign(
            session,
            "https://api.example",
            "bot-token",
            filename="report.txt",
            file_size=4,
        )

        assert result["headers"] == {
            "content-type": "text/plain; charset=utf-8",
            "x-amz-meta-scope": "octo",
        }
        assert result["contentType"] == "text/plain; charset=utf-8"


    @pytest.mark.asyncio
    async def test_presigned_put_replays_server_headers_and_returns_download_url(self):
        response = AsyncMock()
        response.ok = True
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.put.return_value = response

        result = await upload_file_to_presigned_url(
            session,
            upload_url="https://storage.example/upload?signature=secret",
            download_url="https://cdn.example/report.txt",
            file_data=b"data",
            content_type="text/plain; charset=utf-8",
            content_disposition='attachment; filename="report.txt"',
        )

        assert result == "https://cdn.example/report.txt"
        assert session.put.call_args.kwargs["headers"] == {
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": "4",
            "Content-Disposition": 'attachment; filename="report.txt"',
        }

    @pytest.mark.asyncio
    async def test_presigned_put_replays_arbitrary_server_signed_headers(self):
        response = AsyncMock()
        response.ok = True
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.put.return_value = response

        await upload_file_to_presigned_url(
            session,
            upload_url="https://storage.example/upload?signature=secret",
            download_url="https://cdn.example/report.txt",
            file_data=b"data",
            content_type="application/octet-stream",
            headers={
                "content-type": "text/plain; charset=utf-8",
                "x-amz-meta-scope": "octo",
            },
        )

        assert session.put.call_args.kwargs["headers"] == {
            "content-type": "text/plain; charset=utf-8",
            "x-amz-meta-scope": "octo",
            "Content-Length": "4",
        }

    @pytest.mark.asyncio
    async def test_presigned_put_rejects_mismatched_signed_content_length(self):
        session = MagicMock()

        with pytest.raises(ValueError, match="Content-Length"):
            await upload_file_to_presigned_url(
                session,
                upload_url="https://storage.example/upload?signature=secret",
                download_url="https://cdn.example/report.txt",
                file_data=b"data",
                content_type="application/octet-stream",
                headers={"Content-Length": "5"},
            )

        session.put.assert_not_called()


    @pytest.mark.asyncio
    async def test_upload_and_get_url_uses_server_presign_without_cos_credentials(self):
        presign = {
            "uploadUrl": "https://storage.example/upload?signature=secret",
            "downloadUrl": "https://cdn.example/report.txt",
            "contentType": "text/plain",
            "contentDisposition": 'attachment; filename="report.txt"',
        }
        with (
            patch(
                "hermes_octo_plugin.api.get_upload_presign",
                AsyncMock(return_value=presign),
            ) as get_presign,
            patch(
                "hermes_octo_plugin.api.upload_file_to_presigned_url",
                AsyncMock(return_value=presign["downloadUrl"]),
            ) as put_file,
        ):
            result = await upload_and_get_url(
                MagicMock(),
                "https://api.example",
                "bot-token",
                "report.txt",
                b"data",
                "text/plain",
            )

        assert result == presign["downloadUrl"]
        get_presign.assert_awaited_once_with(
            get_presign.await_args.args[0],
            "https://api.example",
            "bot-token",
            filename="report.txt",
            file_size=4,
            content_type="text/plain",
        )
        put_file.assert_awaited_once()


    @pytest.mark.asyncio
    async def test_private_presigned_origin_is_rejected_without_transport_policy(self):
        presign = {
            "uploadUrl": "http://127.0.0.1/upload",
            "downloadUrl": "https://cdn.example/report.txt",
            "contentType": "text/plain",
        }
        put_file = AsyncMock()
        with (
            patch(
                "hermes_octo_plugin.api.get_upload_presign",
                AsyncMock(return_value=presign),
            ),
            patch(
                "hermes_octo_plugin.api.upload_file_to_presigned_url",
                put_file,
            ),
        ):
            with pytest.raises(RuntimeError, match="unsafe presigned upload URL"):
                await upload_and_get_url(
                    MagicMock(),
                    "https://api.example",
                    "bot-token",
                    "report.txt",
                    b"private data",
                    "text/plain",
                )

        put_file.assert_not_awaited()


    @pytest.mark.asyncio
    async def test_presign_private_storage_origin_is_narrowly_trusted_with_opt_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "true")
        session = MagicMock()
        from hermes_octo_plugin.transport import TransportPolicy

        policy = TransportPolicy({"api.internal"})
        presign = {
            "uploadUrl": "http://minio.internal/upload?signature=secret",
            "downloadUrl": "http://cdn.internal/report.txt",
            "contentType": "text/plain",
        }

        async def assert_trusted(active_session, **_kwargs):
            assert active_session is session
            assert policy.trusted_hosts() == frozenset({
                "api.internal",
                "minio.internal",
            })
            return presign["downloadUrl"]

        with (
            patch(
                "hermes_octo_plugin.api.get_upload_presign",
                AsyncMock(return_value=presign),
            ),
            patch(
                "hermes_octo_plugin.api.upload_file_to_presigned_url",
                side_effect=assert_trusted,
            ),
        ):
            result = await upload_and_get_url(
                session,
                "http://api.internal",
                "bot-token",
                "report.txt",
                b"data",
                "text/plain",
                policy=policy,
            )

        assert result == presign["downloadUrl"]

    @pytest.mark.asyncio
    async def test_presign_metadata_upload_origin_is_never_trusted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "true")
        session = MagicMock()
        from hermes_octo_plugin.transport import TransportPolicy

        policy = TransportPolicy({"api.internal"})
        presign = {
            "uploadUrl": "http://169.254.169.254/latest/meta-data/",
            "downloadUrl": "http://cdn.internal/report.txt",
            "contentType": "text/plain",
        }
        with patch(
            "hermes_octo_plugin.api.get_upload_presign",
            AsyncMock(return_value=presign),
        ):
            with pytest.raises(RuntimeError, match="unsafe presigned upload URL"):
                await upload_and_get_url(
                    session,
                    "http://api.internal",
                    "bot-token",
                    "report.txt",
                    b"data",
                    "text/plain",
                    policy=policy,
                )

        assert policy.trusted_hosts() == frozenset({"api.internal"})
