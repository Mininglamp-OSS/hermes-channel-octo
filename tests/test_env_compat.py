from hermes_octo_plugin import env
from hermes_octo_plugin.adapter import _env_enablement


def test_legacy_hermes_reads_process_environment(monkeypatch):
    monkeypatch.setattr(env, "_host_scoped_secret", None)
    monkeypatch.setenv("OCTO_BOT_TOKEN", "legacy-token")
    monkeypatch.setenv("OCTO_API_URL", "https://legacy.example.test")
    monkeypatch.delenv("OCTO_CDN_URL", raising=False)
    monkeypatch.delenv("OCTO_HOME_CHANNEL", raising=False)

    assert _env_enablement() == {
        "bot_token": "legacy-token",
        "api_url": "https://legacy.example.test",
    }
    assert env.scoped_env("OCTO_CDN_URL", "missing") == "missing"
