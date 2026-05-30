"""Regression: data files must ship inside the built wheel.

setuptools' src-layout package finder silently excludes non-``*.py`` files
unless ``[tool.setuptools.package-data]`` lists them. If that stanza ever
gets dropped or a new data file slips in unguarded, ``pip install`` users
get a broken package — ``plugin.yaml`` disappears (breaking
``hermes config`` env-var prompts) and ``skills/octo-bot-api/SKILL.md``
disappears (breaking ``register_skill`` at gateway boot).

This test builds a wheel from the repo root and asserts the expected
files are inside the zip. It is intentionally slow (subprocess + pip
wheel build); accept the cost — it is the only thing that catches
packaging-config drift before users do.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every non-``*.py`` file under src/hermes_octo_plugin/ that the runtime
# resolves via ``Path(__file__).parent / ...`` must appear here.
EXPECTED_DATA_FILES = (
    "hermes_octo_plugin/plugin.yaml",
    "hermes_octo_plugin/skills/octo-bot-api/SKILL.md",
)


@pytest.mark.timeout(180)
def test_data_files_ship_in_wheel(tmp_path):
    """Build the wheel and assert every runtime-loaded data file is inside."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel",
            "--no-deps", "--no-build-isolation",
            str(REPO_ROOT), "-w", str(tmp_path),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        # Fall back to isolated build if the active env lacks build deps.
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps",
             str(REPO_ROOT), "-w", str(tmp_path)],
            capture_output=True, text=True, check=False,
        )
    assert result.returncode == 0, (
        f"pip wheel failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    wheels = list(tmp_path.glob("hermes_channel_octo-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())

    missing = [f for f in EXPECTED_DATA_FILES if f not in names]
    assert not missing, (
        f"wheel is missing required data files: {missing}\n"
        f"wheel contents:\n  " + "\n  ".join(sorted(names))
    )
