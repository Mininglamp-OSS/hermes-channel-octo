"""
Shared pytest configuration and fixtures.
"""

import pytest


def pytest_configure(config):
    """Trigger plugin discovery once so ``Platform("octo")`` resolves via the
    dynamic ``_missing_`` hook. Without this, any test that instantiates a
    real ``OctoAdapter`` explodes with ``'octo' is not a valid Platform``.
    Moved out of module-import scope so collecting these tests no longer
    mutates the global ``Platform`` enum as a side effect of import.
    """
    try:
        from hermes_cli.plugins import discover_plugins  # type: ignore
        discover_plugins()
    except Exception:  # pragma: no cover — running tests without hermes is OK
        pass


def make_bare_adapter():
    """Build an ``OctoAdapter`` bypassing ``__init__`` and seed the in-memory
    state fields **currently exercised by the test suite** — not a faithful
    mirror of ``__init__`` (crypto handles, async tasks, registration, CDN
    config, etc. are intentionally omitted).

    Tests historically used ``object.__new__(OctoAdapter)`` plus ad-hoc attr
    assignment, which (a) drifted as new fields were added to ``__init__`` and
    (b) leaked into production code via defensive ``getattr`` shims.
    Centralising the bare init here keeps production code free of test-only
    defensives; extend the field list below as new tests need it.
    """
    from hermes_octo_plugin.adapter import (
        DEFAULT_HISTORY_LIMIT,
        DEFAULT_HISTORY_PROMPT_TEMPLATE,
        HEARTBEAT_INTERVAL,
        LRUCache,
        NAME_CACHE_MAX_SIZE,
        OctoAdapter,
        PING_MAX_RETRY,
    )

    a = object.__new__(OctoAdapter)
    # Name resolution / membership maps
    a._uid_to_name = {}
    a._base_uid_to_name = {}
    a._member_map = {}
    a._name_cache = LRUCache(max_size=NAME_CACHE_MAX_SIZE)
    a._user_group_index = {}
    a._group_names = {}
    a._known_group_ids = set()
    # Per-channel caches
    a._chat_kind = {}
    a._group_md_cache = {}
    a._group_md_checked = set()
    a._group_histories = {}
    a._group_cache_timestamps = {}
    a._cache_activity = {}
    a._active_streams = {}
    # Connection / lifecycle state
    a._ws = None
    a._http_session = None
    a._temp_buffer = bytearray()
    a._connected = False
    a._need_reconnect = False
    a._reconnect_attempts = 0
    a._reconnect_in_progress = False
    a._last_token_refresh = 0.0
    a._ping_retry_count = 0
    # Identity / config (callers override as needed)
    a._api_url = ""
    a._bot_token = ""
    a._robot_id = ""
    a._owner_uid = ""
    a._history_limit = DEFAULT_HISTORY_LIMIT
    a._require_mention = True
    a._ignore_mention_all = False
    a._history_prompt_template = DEFAULT_HISTORY_PROMPT_TEMPLATE
    a._stream_threshold = 500
    a._heartbeat_interval_s = float(HEARTBEAT_INTERVAL)
    a._ping_max_retry = int(PING_MAX_RETRY)
    return a


@pytest.fixture
def sample_message_payload():
    """A sample text message payload dict."""
    return {
        "type": 1,
        "content": "Hello, world!",
    }


@pytest.fixture
def sample_mention_payload():
    """A sample message payload with mentions."""
    return {
        "type": 1,
        "content": "@Alice @Bob hello everyone",
        "mention": {
            "uids": ["uid1", "uid2"],
            "entities": [
                {"uid": "uid1", "offset": 0, "length": 6},
                {"uid": "uid2", "offset": 7, "length": 4},
            ],
        },
    }


@pytest.fixture
def sample_reply_payload():
    """A sample message payload with reply context."""
    return {
        "type": 1,
        "content": "This is a reply",
        "reply": {
            "from_uid": "user_original",
            "from_name": "OriginalSender",
            "payload": {
                "content": "Original message text",
            },
        },
    }
