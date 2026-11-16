"""Adapter resolution for multi-profile gateways.

Tools and slash commands must answer through the adapter that owns the
trusted route of the calling turn.  A turn with no trustworthy source, or
one from another platform, fails closed instead of borrowing the
process-wide default adapter.  Older single-profile Hermes keeps using its
per-platform adapter map.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_octo_plugin import agent_tools


def _bind_session_route(monkeypatch, *, route, session_key: str) -> None:
    """Expose Hermes' trusted route/session APIs without depending on its version."""
    gateway = importlib.import_module("gateway")
    session_context = importlib.import_module("gateway.session_context")
    monkeypatch.setattr(
        session_context,
        "bound_session_key",
        lambda: session_key,
        raising=False,
    )

    plugin_messages = ModuleType("gateway.plugin_messages")
    plugin_messages.bound_route = lambda: route
    monkeypatch.setitem(sys.modules, "gateway.plugin_messages", plugin_messages)
    monkeypatch.setattr(gateway, "plugin_messages", plugin_messages, raising=False)


def _bind_runner(monkeypatch, runner) -> None:
    gateway_run = ModuleType("gateway.run")
    gateway_run._gateway_runner_ref = lambda: runner
    monkeypatch.setitem(sys.modules, "gateway.run", gateway_run)


def _octo_platform():
    from hermes_octo_plugin.adapter import _octo_platform as platform

    return platform()


def _runner_with_route_resolution(*, platform, delivery_adapter_for, **kwargs):
    return SimpleNamespace(
        adapters={platform: object()},
        session_store=SimpleNamespace(lookup_by_session_key=Mock(return_value=None)),
        _delivery_adapter_for=delivery_adapter_for,
        **kwargs,
    )


@pytest.mark.parametrize("profile", ["profile-a", "profile-b"])
def test_live_profile_route_uses_delivery_adapter_for_that_source(
    monkeypatch, profile: str
) -> None:
    platform = _octo_platform()
    adapters_by_profile = {
        "profile-a": object(),
        "profile-b": object(),
    }
    source = SimpleNamespace(
        platform=platform,
        chat_id=f"room-{profile}",
        profile=profile,
    )
    delivery_adapter_for = Mock(
        side_effect=lambda routed_source: adapters_by_profile[routed_source.profile]
    )
    session_key = f"octo:{profile}:room"
    runner = _runner_with_route_resolution(
        platform=platform,
        delivery_adapter_for=delivery_adapter_for,
        _session_key_for_source=Mock(return_value=session_key),
    )
    _bind_session_route(monkeypatch, route=source, session_key=session_key)
    _bind_runner(monkeypatch, runner)

    assert agent_tools._resolve_adapter() is adapters_by_profile[profile]
    delivery_adapter_for.assert_called_once_with(source)


def test_restored_session_source_uses_delivery_adapter(monkeypatch) -> None:
    platform = _octo_platform()
    expected_adapter = object()
    entry = SimpleNamespace(origin=object())
    source = SimpleNamespace(
        platform=platform, chat_id="restored-room", profile="profile-b"
    )
    lookup = Mock(return_value=entry)
    restore_source = Mock(return_value=source)
    delivery_adapter_for = Mock(return_value=expected_adapter)
    session_key = "octo:profile-b:restored-room"
    runner = _runner_with_route_resolution(
        platform=platform,
        delivery_adapter_for=delivery_adapter_for,
        _restored_source=restore_source,
    )
    runner.session_store.lookup_by_session_key = lookup
    _bind_session_route(monkeypatch, route=None, session_key=session_key)

    gateway = importlib.import_module("gateway")
    session_identity = ModuleType("gateway.session_identity")
    session_identity.replace_source = lambda restored: restored
    monkeypatch.setitem(sys.modules, "gateway.session_identity", session_identity)
    monkeypatch.setattr(gateway, "session_identity", session_identity, raising=False)
    _bind_runner(monkeypatch, runner)

    assert agent_tools._resolve_adapter() is expected_adapter
    lookup.assert_called_once_with(session_key)
    restore_source.assert_called_once_with(entry)
    delivery_adapter_for.assert_called_once_with(source)


def test_disconnected_profile_does_not_fall_back_to_default_adapter(
    monkeypatch,
) -> None:
    platform = _octo_platform()
    source = SimpleNamespace(platform=platform, chat_id="room-b", profile="profile-b")
    session_key = "octo:profile-b:room-b"
    delivery_adapter_for = Mock(return_value=None)
    runner = _runner_with_route_resolution(
        platform=platform,
        delivery_adapter_for=delivery_adapter_for,
        _session_key_for_source=Mock(return_value=session_key),
    )
    _bind_session_route(monkeypatch, route=source, session_key=session_key)
    _bind_runner(monkeypatch, runner)

    assert agent_tools._resolve_adapter() is None
    delivery_adapter_for.assert_called_once_with(source)


def test_bound_route_must_match_the_bound_session_key(monkeypatch) -> None:
    platform = _octo_platform()
    source = SimpleNamespace(platform=platform, chat_id="room-b", profile="profile-b")
    delivery_adapter_for = Mock(return_value=object())
    runner = _runner_with_route_resolution(
        platform=platform,
        delivery_adapter_for=delivery_adapter_for,
        _session_key_for_source=Mock(return_value="octo:profile-a:room-a"),
    )
    _bind_session_route(
        monkeypatch, route=source, session_key="octo:profile-b:room-b"
    )
    _bind_runner(monkeypatch, runner)

    assert agent_tools._resolve_adapter() is None
    delivery_adapter_for.assert_not_called()


def test_non_octo_route_never_selects_an_adapter(monkeypatch) -> None:
    from gateway.config import Platform

    platform = _octo_platform()
    source = SimpleNamespace(
        platform=Platform.DISCORD, chat_id="discord-room", profile="profile-b"
    )
    delivery_adapter_for = Mock(return_value=object())
    runner = _runner_with_route_resolution(
        platform=platform,
        delivery_adapter_for=delivery_adapter_for,
        _session_key_for_source=Mock(return_value="discord:profile-b:room"),
    )
    _bind_session_route(
        monkeypatch, route=source, session_key="discord:profile-b:room"
    )
    _bind_runner(monkeypatch, runner)

    assert agent_tools._resolve_adapter() is None
    delivery_adapter_for.assert_not_called()


def test_unbound_turn_without_a_stored_session_fails_closed(monkeypatch) -> None:
    platform = _octo_platform()
    delivery_adapter_for = Mock(return_value=object())
    runner = _runner_with_route_resolution(
        platform=platform,
        delivery_adapter_for=delivery_adapter_for,
    )
    _bind_session_route(monkeypatch, route=None, session_key="")
    _bind_runner(monkeypatch, runner)

    assert agent_tools._resolve_adapter() is None
    delivery_adapter_for.assert_not_called()


@pytest.mark.parametrize("missing_api", ["plugin_messages", "bound_session_key"])
def test_route_aware_runner_without_trusted_route_api_fails_closed(
    monkeypatch, missing_api: str
) -> None:
    """A runner with the delivery seam must not borrow the default adapter."""
    platform = _octo_platform()
    delivery_adapter_for = Mock(return_value=object())
    runner = _runner_with_route_resolution(
        platform=platform,
        delivery_adapter_for=delivery_adapter_for,
    )
    if missing_api == "plugin_messages":
        monkeypatch.setitem(sys.modules, "gateway.plugin_messages", None)
    else:
        _bind_session_route(monkeypatch, route=None, session_key="")
        session_context = importlib.import_module("gateway.session_context")
        monkeypatch.delattr(session_context, "bound_session_key", raising=False)
    _bind_runner(monkeypatch, runner)

    assert agent_tools._resolve_adapter() is None
    delivery_adapter_for.assert_not_called()


def test_legacy_single_profile_runner_uses_adapters_map(monkeypatch) -> None:
    platform = _octo_platform()
    legacy_adapter = object()
    runner = SimpleNamespace(adapters={platform: legacy_adapter})
    _bind_session_route(monkeypatch, route=None, session_key="")
    _bind_runner(monkeypatch, runner)

    assert agent_tools._resolve_adapter() is legacy_adapter
