"""Profile-scoped ``OCTO_*`` configuration reads for multiplexed Hermes gateways.

Hermes with ``gateway.multiplex_profiles`` serves several profiles from one
gateway process. Each profile's ``.env`` lives in a context-local secret scope
(``agent.secret_scope``) instead of ``os.environ``, so a secondary profile's
``OCTO_*`` values are invisible to ``os.getenv`` and a scoped miss must stay a
miss rather than borrow the default (launch) profile's process env value.

Every read goes through the host's scope-aware reader
(``gateway.platforms._shared.get_scoped_secret``, the same helper the bundled
platform plugins use): an installed profile scope is authoritative, single-
profile and default-profile startup still reads ``os.environ``. Hermes
installations without multiplexing have no such helper; the guarded import
keeps them on plain ``os.getenv``.

Explicit ``PlatformConfig.extra`` values still win at every call site; this
module only replaces the env rung.
"""

from __future__ import annotations

import os

try:  # Hermes with multiplexed profiles
    from gateway.platforms._shared import get_scoped_secret as _host_scoped_secret
except ImportError:  # older Hermes without the scope-aware reader
    _host_scoped_secret = None


def scoped_env(name: str, default: str | None = None) -> str | None:
    """Value of *name* for the profile this call runs for, else *default*."""

    if _host_scoped_secret is None:
        return os.getenv(name, default)
    return _host_scoped_secret(name, default)


def scoped_flag(name: str) -> bool:
    """Boolean gate read for a truthy env flag (``1`` / ``true`` / ``yes``)."""

    return str(scoped_env(name, "") or "").strip().lower() in {"1", "true", "yes"}
