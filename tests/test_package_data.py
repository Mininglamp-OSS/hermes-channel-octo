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
import shutil
import subprocess
import sys
import zipfile
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version



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
    uv = shutil.which("uv")
    if uv:
        command = [
            uv,
            "build",
            "--wheel",
            "--out-dir",
            str(tmp_path),
            str(REPO_ROOT),
        ]
        builder = "uv build"
    else:
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            str(REPO_ROOT),
            "-w",
            str(tmp_path),
        ]
        builder = "pip wheel"
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{builder} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    wheels = list(tmp_path.glob("hermes_channel_octo-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = zf.read(metadata_name).decode("utf-8")

    missing = [f for f in EXPECTED_DATA_FILES if f not in names]
    assert not missing, (
        f"wheel is missing required data files: {missing}\n"
        f"wheel contents:\n  " + "\n  ".join(sorted(names))
    )
    assert "Requires-Dist: hermes-agent<0.22,>=0.14" in metadata
    assert "Requires-Dist: cryptography<49,>=46.0" in metadata


def test_cryptography_dependency_covers_46_and_48_with_documented_rationale():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    cryptography = next(
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("cryptography")
    )
    requirement = Requirement(cryptography)

    assert Version("46.0") in requirement.specifier
    assert Version("48.0.1") in requirement.specifier
    assert Version("49.0") not in requirement.specifier
    readme = (REPO_ROOT / "README.md").read_text()
    assert "cryptography 46.x" in readme
    assert "48.x:" in readme
    assert "cryptography 48.0.1" in readme
