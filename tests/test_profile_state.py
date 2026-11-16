from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from hermes_octo_plugin.adapter import OctoAdapter
from hermes_octo_plugin.card_sessions import CardSession
from hermes_octo_plugin.identity import CardBindingStore, IdentityMetadata, PlannedRoute, PHASE_SINGLE
from hermes_octo_plugin.types import ChannelType


def test_profile_state_remains_owned_after_home_scope_changes(tmp_path, monkeypatch):
    constants = pytest.importorskip("hermes_constants")
    if not hasattr(constants, "set_hermes_home_override"):
        pytest.skip("Hermes has no context-local profile home")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    adapters = []
    for name in ("alpha", "bravo"):
        token = constants.set_hermes_home_override(tmp_path / name)
        try:
            adapters.append(OctoAdapter(PlatformConfig(extra={"bot_token": f"token-{name}"})))
        finally:
            constants.reset_hermes_home_override(token)

    for name, adapter in zip(("alpha", "bravo"), adapters):
        state = adapter.identity_state
        state.save_metadata(IdentityMetadata(phase=PHASE_SINGLE, legacy_robot_id=name))
        state.save_plan((PlannedRoute(session_key=name, chat_type="dm", chat_id=name, channel_type=int(ChannelType.DM)),))
        state.routes.bind(robot_id=name, session_key=name, chat_type="dm", chat_id=name, channel_type=int(ChannelType.DM))
        state.routes.flush(force=True)
        runtime = adapter.runtimes[0]
        runtime._registration = SimpleNamespace(robot_id="shared-bot")
        runtime._attach_card_binding_store()
        runtime._card_sessions.register(CardSession(
            message_id=name, binding_id=name, session_key=name, chat_id=name,
            channel_id=name, channel_type=ChannelType.DM, requester_uid=name,
            card={"type": "AdaptiveCard", "body": []}, plain=name,
            action_labels={}, input_ids=(),
        ))

    for name, adapter in zip(("alpha", "bravo"), adapters):
        root = tmp_path / name / "workspace" / "octo"
        assert adapter.identity_state.load_metadata().legacy_robot_id == name
        assert adapter.identity_state.load_plan()[0].session_key == name
        assert adapter.identity_state.routes.path == root / "identity" / "identity-routes.json"
        assert adapter.identity_state.routes.path.is_file()
        assert adapter.identity_state.routes.lookup_chat_id(name).robot_id == name
        assert (root / "shared-bot" / "card-sessions.json").is_file()
        records = CardBindingStore(robot_id="shared-bot", base_dir=root).load()
        assert [record["message_id"] for record in records] == [name]
