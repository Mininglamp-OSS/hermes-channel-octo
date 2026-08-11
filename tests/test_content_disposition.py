"""Backend-agnostic presigned media upload and filename decoding contracts."""

from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import unquote

import pytest

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

    @pytest.mark.parametrize(
        "filename",
        ["\u202egnp.exe", "a\u200bb.txt", "a\u0085b.txt", "report.txt.", "report.txt "],
    )
    def test_safe_media_filename_rejects_format_controls_and_trailing_spoofing(
        self,
        filename,
    ):
        from hermes_octo_plugin.api import safe_media_filename

        assert safe_media_filename(filename) is None

    def test_content_disposition_checks_decoded_filename_and_keeps_legacy_percent(self):
        from hermes_octo_plugin.api import _content_disposition_filename

        assert _content_disposition_filename(
            "attachment; filename*=UTF-8''%0d%0aX-Injected:%201"
        ) is None
        assert _content_disposition_filename(
            'attachment; filename="100%20.txt"'
        ) == "100%20.txt"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_disposition", "url", "expected_filename"),
    [
        (
            "attachment; filename*=UTF-8''..%2Foutside.txt",
            "https://files.example/report.txt",
            "report.txt",
        ),
        (
            "attachment; filename*=UTF-8''%00awkward.txt",
            "https://files.example/%2E%2E%2Foutside.txt",
            "file",
        ),
        (
            "attachment; filename=..%2Foutside.txt",
            "https://files.example/report.txt",
            "..%2Foutside.txt",
        ),
    ],
)
async def test_download_uses_only_safe_decoded_filename_candidates(
    content_disposition: str,
    url: str,
    expected_filename: str,
):
    class Content:
        async def iter_any(self):
            yield b"media"

    response = AsyncMock()
    response.status = 200
    response.headers = {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": content_disposition,
    }
    response.content = Content()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get.return_value = response

    data, _, filename = await download_file(session, url)

    assert data == b"media"
    assert filename == expected_filename

class TestPresignedUpload:
    @pytest.mark.asyncio
    async def test_get_presign_sends_exact_size_and_normalizes_signed_headers(self):
        response = AsyncMock()
        response.ok = True
        response.status = 200
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
        response.status = 200
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
        response.status = 200
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        guarded_session = MagicMock()
        guarded_session.put.return_value = response
        guarded_session.close = AsyncMock()

        with patch(
            "hermes_octo_plugin.api.new_guarded_http_session",
            return_value=guarded_session,
        ):
            result = await upload_file_to_presigned_url(
                upload_url="https://storage.example/upload?signature=secret",
                download_url="https://cdn.example/report.txt",
                file_data=b"data",
                content_type="text/plain; charset=utf-8",
                content_disposition='attachment; filename="report.txt"',
            )

        assert result == "https://cdn.example/report.txt"
        assert guarded_session.put.call_args.kwargs["headers"] == {
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": "4",
            "Content-Disposition": 'attachment; filename="report.txt"',
        }
        guarded_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_presigned_put_replays_arbitrary_server_signed_headers(self):
        response = AsyncMock()
        response.ok = True
        response.status = 200
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        guarded_session = MagicMock()
        guarded_session.put.return_value = response
        guarded_session.close = AsyncMock()

        with patch(
            "hermes_octo_plugin.api.new_guarded_http_session",
            return_value=guarded_session,
        ):
            await upload_file_to_presigned_url(
                upload_url="https://storage.example/upload?signature=secret",
                download_url="https://cdn.example/report.txt",
                file_data=b"data",
                content_type="application/octet-stream",
                headers={
                    "content-type": "text/plain; charset=utf-8",
                    "x-amz-meta-scope": "octo",
                },
            )

        assert guarded_session.put.call_args.kwargs["headers"] == {
            "content-type": "text/plain; charset=utf-8",
            "x-amz-meta-scope": "octo",
            "Content-Length": "4",
        }
        guarded_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_presigned_put_rejects_mismatched_signed_content_length(self):
        session = MagicMock()

        with pytest.raises(ValueError, match="Content-Length"):
            await upload_file_to_presigned_url(
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
    @pytest.mark.parametrize(
        "upload_url",
        [
            "http://127.0.0.1:9000/upload?X-Amz-Signature=top-secret",
            "http://10.0.0.8:9000/upload?X-Amz-Signature=top-secret",
        ],
    )
    async def test_private_presign_rejects_unconfigured_origin(
        self,
        monkeypatch: pytest.MonkeyPatch,
        upload_url: str,
    ) -> None:
        monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "true")
        from hermes_octo_plugin.transport import TransportPolicy

        policy = TransportPolicy({"http://api.internal"})
        presign = {
            "uploadUrl": upload_url,
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
                    "http://api.internal",
                    "bot-token",
                    "report.txt",
                    b"private data",
                    "text/plain",
                    policy=policy,
                )

        assert policy.trusted_download_origins() == frozenset({
            ("http", "api.internal", 80),
        })
        put_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unicode_loopback_presign_requires_private_host_opt_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OCTO_ALLOW_PRIVATE_HOSTS", raising=False)
        from hermes_octo_plugin.transport import TransportPolicy

        upload_url = "http://①②⑦.0.0.1:9000/upload?X-Amz-Signature=top-secret"
        presign = {
            "uploadUrl": upload_url,
            "downloadUrl": "https://cdn.example/report.txt",
            "contentType": "text/plain",
        }
        session = MagicMock()
        policy = TransportPolicy({"https://api.example"})

        with patch(
            "hermes_octo_plugin.api.get_upload_presign",
            AsyncMock(return_value=presign),
        ):
            with pytest.raises(RuntimeError, match="unsafe presigned upload URL"):
                await upload_and_get_url(
                    session,
                    "https://api.example",
                    "bot-token",
                    "report.txt",
                    b"private data",
                    "text/plain",
                    policy=policy,
                )

        assert policy.is_download_url_trusted(upload_url) is False
        session.put.assert_not_called()


    @pytest.mark.asyncio
    async def test_presign_accepts_configured_private_exact_origin(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OCTO_ALLOW_PRIVATE_HOSTS", raising=False)
        session = MagicMock()
        from hermes_octo_plugin.transport import TransportPolicy
        policy = TransportPolicy({"http://127.0.0.1:9000"})
        presign = {
            "uploadUrl": "http://127.0.0.1:9000/upload?signature=secret",
            "downloadUrl": "http://cdn.example/report.txt",
            "contentType": "text/plain",
        }

        async def assert_trusted(**_kwargs):
            assert policy.trusted_download_origins() == frozenset({
                ("http", "127.0.0.1", 9000),
            })
            assert _kwargs["policy"] is policy
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
                "http://127.0.0.1:9000",
                "bot-token",
                "report.txt",
                b"data",
                "text/plain",
                policy=policy,
            )

        assert result == presign["downloadUrl"]

    @pytest.mark.asyncio
    async def test_presign_rejects_opposite_scheme_on_trusted_endpoint(self) -> None:
        from hermes_octo_plugin.transport import TransportPolicy

        policy = TransportPolicy({"https://storage.example:8443"})
        presign = {
            "uploadUrl": "http://storage.example:8443/upload?signature=secret",
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
                    b"data",
                    "text/plain",
                    policy=policy,
                )

        put_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_presigned_put_uses_isolated_exact_origin_session(self) -> None:
        from hermes_octo_plugin.transport import TransportPolicy

        upload_url = "http://127.0.0.1:9000/upload?signature=secret"
        policy = TransportPolicy({"http://127.0.0.1:9000"})
        response = AsyncMock()
        response.ok = True
        response.status = 200
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        main_session = MagicMock()
        isolated_session = MagicMock()
        isolated_session.put.return_value = response
        isolated_session.close = AsyncMock()

        with patch(
            "hermes_octo_plugin.api.new_guarded_http_session",
            return_value=isolated_session,
            create=True,
        ) as session_factory:
            result = await upload_file_to_presigned_url(
                upload_url=upload_url,
                download_url="http://cdn.example/report.txt",
                file_data=b"data",
                content_type="text/plain",
                policy=policy,
            )

        assert result == "http://cdn.example/report.txt"
        session_factory.assert_called_once_with(policy=policy)
        main_session.put.assert_not_called()
        isolated_session.put.assert_called_once()
        isolated_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_public_presigned_put_uses_guarded_session(self) -> None:
        response = AsyncMock()
        response.status = 200
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.put.return_value = response
        guarded_session = MagicMock()
        guarded_session.put.return_value = response
        guarded_session.close = AsyncMock()

        with patch(
            "hermes_octo_plugin.api.new_guarded_http_session",
            return_value=guarded_session,
        ) as session_factory:
            result = await upload_file_to_presigned_url(
                upload_url="https://storage.example/upload?signature=secret",
                download_url="https://cdn.example/report.txt",
                file_data=b"data",
                content_type="text/plain",
            )

        assert result == "https://cdn.example/report.txt"
        session_factory.assert_called_once()
        assert session_factory.call_args.kwargs["policy"].trusted_download_origins() == frozenset()
        session.put.assert_not_called()
        guarded_session.put.assert_called_once()
        guarded_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_presign_metadata_upload_origin_is_never_trusted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OCTO_ALLOW_PRIVATE_HOSTS", "true")
        session = MagicMock()
        from hermes_octo_plugin.transport import TransportPolicy

        policy = TransportPolicy({"http://api.internal"})
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

        assert policy.trusted_download_origins() == frozenset({
            ("http", "api.internal", 80),
        })
