"""Smoke tests for the bundled-plugin root-level shim.

These guard the ``hermes plugins install`` code path, which clones this
repo into ``HERMES_HOME/plugins/<name>/`` and looks for ``plugin.yaml`` +
``__init__.py`` at the root.
"""
import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_root_init_exposes_register():
    """Root __init__.py must expose register() for bundled-plugin loader."""
    init_file = REPO_ROOT / "__init__.py"
    assert init_file.exists(), "Root __init__.py shim missing"
    spec = importlib.util.spec_from_file_location(
        "_bundled_shim_test",
        init_file,
        submodule_search_locations=[str(REPO_ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, "register", None))


def test_root_plugin_yaml_matches_src():
    """Root plugin.yaml must not drift from the packaged copy under src/."""
    root_yaml = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    src_yaml = yaml.safe_load(
        (REPO_ROOT / "src/hermes_octo_plugin/plugin.yaml").read_text(encoding="utf-8")
    )
    assert root_yaml == src_yaml, "Root plugin.yaml drifted from src copy"
