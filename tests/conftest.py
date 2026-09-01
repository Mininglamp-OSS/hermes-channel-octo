"""
Shared pytest configuration and fixtures.
"""

import asyncio
from collections import OrderedDict

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


def make_bare_octo_adapter(runtime=None, *, runtimes=None):
    """Build an ``OctoAdapter`` bypassing ``__init__``, in single-identity mode.

    Only the fields the adapter needs to orchestrate identities and answer
    Hermes are seeded.  Pass ``runtimes`` to exercise the multi-identity paths;
    the default keeps the historical single-token fast path where no route
    lookup happens and no identity state is written to disk.
    """
    from types import SimpleNamespace

    from hermes_octo_plugin.adapter import OctoAdapter, SharedOctoConfig, _octo_platform

    owned = tuple(runtimes) if runtimes is not None else (
        () if runtime is None else (runtime,)
    )
    adapter = object.__new__(OctoAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter.platform = _octo_platform()
    adapter._message_handler = None
    adapter._running = False
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._shared = SharedOctoConfig()
    adapter._api_url = ""
    adapter._bot_tokens = tuple("token" for _ in owned) or ("token",)
    adapter._state_base_dir = None
    adapter._runtimes = owned
    adapter._runtimes_by_robot_id = {}
    adapter._sole_identity_mode = len(owned) <= 1
    adapter._startup = None
    adapter._legacy_plan = ()
    adapter._gateway_loop = None
    adapter._disconnecting = False
    adapter._route_flush_tasks = set()
    return adapter


def make_bare_adapter():
    """Build a bare ``IdentityRuntime`` (plus its owning ``OctoAdapter``) and
    seed the in-memory state fields **currently exercised by the test suite** —
    not a faithful mirror of ``__init__`` (crypto handles, async tasks,
    registration, CDN config, etc. are intentionally omitted).

    One Octo identity owns the connection, caches, inbound handling and the
    send implementations, so a bare runtime is what most tests want.  It is
    wired to a bare ``OctoAdapter`` in single-identity mode, which is the
    historical behaviour: no route lookup, no identity state on disk.

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
        IdentityRuntime,
        LRUCache,
        NAME_CACHE_MAX_SIZE,
        OctoAdapter,
        PING_MAX_RETRY,
        UNKNOWN_MESSAGE_TYPE_TELEMETRY_CAP,
    )
    from hermes_octo_plugin import cards
    from hermes_octo_plugin.card_events import CardSessionRegistry

    a = object.__new__(IdentityRuntime)
    a._adapter = make_bare_octo_adapter(a)
    a._state_base_dir = None
    a._card_store_robot_id = ""
    # Name resolution / membership maps
    a._uid_to_name = {}
    a._base_uid_to_name = {}
    a._member_map = {}
    a._group_member_rosters = {}
    a._group_robot_map = {}
    a._name_cache = LRUCache(max_size=NAME_CACHE_MAX_SIZE)
    a._user_group_index = {}
    a._group_names = {}
    a._known_group_ids = set()
    # Per-channel caches
    a._chat_kind = {}
    a._space_dm_targets = {}
    a._group_md_cache = {}
    a._group_md_checked = set()
    a._group_scope_generations = {}
    a._group_histories = {}
    a._group_cache_timestamps = {}
    a._cache_activity = {}
    a._progress_tasks = set()
    a._gateway_loop = None
    a._event_poller = None
    a._event_task = None
    a._card_sessions = CardSessionRegistry()
    a._card_profile_cache = cards.CardProfileCache()
    a._native_clarify_enabled = False
    a._unknown_message_type_counts = OrderedDict()
    a._unknown_message_type_log_budget = UNKNOWN_MESSAGE_TYPE_TELEMETRY_CAP
    a._disconnecting = False
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
    a._http_heartbeat_disabled = False
    a._http_heartbeat_task = None
    a._heartbeat_task = None
    a._recv_task = None
    a._cache_cleanup_task = None
    a._reconnect_task = None
    a._prefetch_task = None
    a._command_menu_task = None
    a._command_menu_force_event = asyncio.Event()
    a._command_menu_force_pending = False
    a._command_menu_published_digest = None
    a._command_menu_max_chars_config = 1000
    a._lifecycle_lock = asyncio.Lock()
    # Identity / config (callers override as needed)
    a._api_url = ""
    a._cdn_url = ""
    a._ws_url = ""
    a._bot_token = ""
    a._on_behalf_of = ""
    a._robot_id = ""
    a._owner_uid = ""
    a._history_limit = DEFAULT_HISTORY_LIMIT
    a._require_mention = True
    a._ignore_mention_all = False
    a._history_prompt_template = DEFAULT_HISTORY_PROMPT_TEMPLATE
    a._progress_card_renderer = "local"
    a._heartbeat_interval_s = float(HEARTBEAT_INTERVAL)
    a._ping_max_retry = int(PING_MAX_RETRY)
    a._event_poll_interval_s = 2.0
    a._event_poll_wait_s = 25
    a._event_poll_limit = 50
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
