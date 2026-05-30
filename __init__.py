"""Bundled-plugin entry shim for ``hermes plugins install``.

This file lets the bundled-plugin protocol (clone the repo into
``HERMES_HOME/plugins/<name>/``, expect ``plugin.yaml`` + ``__init__.py``
at the root) discover the real package which lives at
``src/hermes_octo_plugin/``.

For pip-installed mode this file is irrelevant — entry-points point
directly at ``hermes_octo_plugin``, and ``src/hermes_octo_plugin/__init__.py``
exposes ``register`` the normal way.

NOTE on deps: ``hermes plugins install`` does NOT pip-install pyproject
deps. Bundled-mode users must install ``websockets`` / ``aiohttp`` /
``cryptography`` / ``python-socks`` manually (see README). Pip-mode users
get them automatically via the pyproject dependency list.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hermes_octo_plugin import register  # noqa: E402,F401

__all__ = ["register"]
