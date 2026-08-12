from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_octo_plugin import api
from hermes_octo_plugin.adapter import COMMAND_MENU_SYNC_INTERVAL_S, OctoAdapter
from hermes_octo_plugin.command_menu import CommandMenuManifest
from tests.conftest import make_bare_adapter


def _manifest(command: str, digest: str) -> CommandMenuManifest:
    return CommandMenuManifest(
        commands=({"command": command, "description": "Description"},),
        digest=digest,
        source_counts={
            "core": 1,
            "quick": 0,
            "plugin": 0,
            "bundle": 0,
            "skill": 0,
        },
        collected_count=1,
        omitted_count=0,
        payload_chars=52,
        max_chars=1000,
    )


def _adapter() -> OctoAdapter:
    adapter = make_bare_adapter()
    adapter._http_session = MagicMock()
    adapter._http_session.close = AsyncMock()
    adapter._api_url = "https://api.example.invalid"
    adapter._bot_token = "bot-token"
    adapter._command_menu_max_chars_config = 1000
    adapter.gateway_runner = MagicMock(config=MagicMock(quick_commands={}))
    return adapter


@pytest.mark.asyncio
async def test_reconcile_publishes_first_snapshot_and_skips_unchanged_snapshot():
    adapter = _adapter()
    manifest = _manifest("/new", "digest-one")

    with (
        patch(
            "hermes_octo_plugin.command_menu.collect_runtime_command_menu",
            return_value=manifest,
        ) as collect,
        patch("hermes_octo_plugin.adapter.api.set_commands", AsyncMock()) as publish,
    ):
        assert await adapter._reconcile_command_menu(force=True) is True
        assert await adapter._reconcile_command_menu() is False

    collect.assert_called_with({}, max_chars=1000)
    assert collect.call_count == 2
    publish.assert_awaited_once_with(
        adapter._http_session,
        adapter._api_url,
        adapter._bot_token,
        list(manifest.commands),
    )
    assert adapter._command_menu_published_digest == "digest-one"


@pytest.mark.asyncio
async def test_reconcile_retries_failed_publish_without_marking_digest():
    adapter = _adapter()
    manifest = _manifest("/new", "digest-one")
    publish = AsyncMock(side_effect=[RuntimeError("offline"), None])

    with (
        patch(
            "hermes_octo_plugin.command_menu.collect_runtime_command_menu",
            return_value=manifest,
        ),
        patch("hermes_octo_plugin.adapter.api.set_commands", publish),
    ):
        assert await adapter._reconcile_command_menu(force=True) is False
        assert adapter._command_menu_published_digest is None
        assert await adapter._reconcile_command_menu() is True

    assert publish.await_count == 2
    assert adapter._command_menu_published_digest == "digest-one"


@pytest.mark.asyncio
async def test_failed_reconnect_force_retries_even_when_digest_matches_prior_success():
    adapter = _adapter()
    adapter._command_menu_published_digest = "digest-one"
    manifest = _manifest("/new", "digest-one")
    publish = AsyncMock(side_effect=[RuntimeError("offline"), None])

    with (
        patch(
            "hermes_octo_plugin.command_menu.collect_runtime_command_menu",
            return_value=manifest,
        ),
        patch("hermes_octo_plugin.adapter.api.set_commands", publish),
    ):
        assert await adapter._reconcile_command_menu(force=True) is False
        assert await adapter._reconcile_command_menu() is True

    assert publish.await_count == 2


@pytest.mark.asyncio
async def test_reconcile_never_overwrites_server_after_collection_failure():
    adapter = _adapter()

    with (
        patch(
            "hermes_octo_plugin.command_menu.collect_runtime_command_menu",
            side_effect=RuntimeError("registry unavailable"),
        ),
        patch("hermes_octo_plugin.adapter.api.set_commands", AsyncMock()) as publish,
    ):
        assert await adapter._reconcile_command_menu(force=True) is False

    publish.assert_not_awaited()
    assert adapter._command_menu_published_digest is None

@pytest.mark.parametrize(
    ("failure", "safe_detail"),
    [
        (RuntimeError("printf super-secret"), "RuntimeError"),
        (
            api.OctoApiError("/v1/bot/setCommands/secret", status=503),
            "OctoApiError (HTTP 503)",
        ),
    ],
)
@pytest.mark.asyncio
async def test_reconcile_logs_only_safe_failure_metadata(
    caplog, failure, safe_detail
):
    adapter = _adapter()

    with (
        caplog.at_level(logging.WARNING, logger="hermes_octo_plugin.adapter"),
        patch(
            "hermes_octo_plugin.command_menu.collect_runtime_command_menu",
            side_effect=failure,
        ),
        patch("hermes_octo_plugin.adapter.api.set_commands", AsyncMock()) as publish,
    ):
        assert await adapter._reconcile_command_menu(force=True) is False

    publish.assert_not_awaited()
    assert safe_detail in caplog.messages[-1]
    assert "super-secret" not in caplog.text
    assert "/v1/bot/setCommands/secret" not in caplog.text

@pytest.mark.asyncio
async def test_reconcile_does_not_overwrite_server_for_invalid_budget():
    adapter = _adapter()
    adapter._command_menu_max_chars_config = "bad"

    with (
        patch(
            "hermes_octo_plugin.command_menu.collect_runtime_command_menu"
        ) as collect,
        patch("hermes_octo_plugin.adapter.api.set_commands", AsyncMock()) as publish,
    ):
        assert await adapter._reconcile_command_menu(force=True) is False

    collect.assert_not_called()
    publish.assert_not_awaited()
    assert adapter._command_menu_published_digest is None


@pytest.mark.asyncio
async def test_sync_loop_forces_on_wakeup_and_polls_without_force():
    adapter = _adapter()
    adapter._wait_for_command_menu_sync = AsyncMock(
        side_effect=[True, False, asyncio.CancelledError()]
    )
    adapter._reconcile_command_menu = AsyncMock(return_value=True)

    with pytest.raises(asyncio.CancelledError):
        await adapter._command_menu_sync_loop()

    assert adapter._reconcile_command_menu.await_args_list[0].kwargs == {"force": True}
    assert adapter._reconcile_command_menu.await_args_list[1].kwargs == {}


def test_live_quick_commands_falls_back_to_gateway_runner_weakref():
    adapter = _adapter()
    del adapter.gateway_runner
    runner = MagicMock(config=MagicMock(quick_commands={"ship": {"type": "exec"}}))

    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        assert adapter._live_quick_commands() == {"ship": {"type": "exec"}}


def test_reconnect_wakes_existing_singleton_sync_task():
    adapter = _adapter()
    adapter._command_menu_force_event.clear()
    running_task = MagicMock()
    running_task.done.return_value = False
    adapter._command_menu_task = running_task

    with patch("hermes_octo_plugin.adapter.asyncio.create_task") as create_task:
        adapter._wake_command_menu_sync()

    assert adapter._command_menu_force_event.is_set()
    create_task.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_cleanup_cancels_command_menu_task_before_session_close():
    adapter = _adapter()
    adapter._mark_disconnected = MagicMock()
    session = adapter._http_session
    cancelled = asyncio.Event()

    async def command_menu_loop() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    adapter._command_menu_task = asyncio.create_task(command_menu_loop())
    await asyncio.sleep(0)
    await adapter._finalize_disconnect_resources()

    assert cancelled.is_set()
    assert adapter._command_menu_task is None
    session.close.assert_awaited_once()
