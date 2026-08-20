"""Tests for the repository release-version update helper."""

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
UPDATE_SCRIPT = REPOSITORY_ROOT / ".github" / "scripts" / "update_release_version.py"
pytestmark = pytest.mark.skipif(
    not UPDATE_SCRIPT.is_file(),
    reason="release updater is repository automation and is not included in installed wheels",
)


def write_version_files(root, version="0.2.5"):
    (root / "docs" / "source").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "example"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "docs" / "source" / "conf.py").write_text(
        f'release = "{version}"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "Dockerfile.pip-install").write_text(
        "FROM python:3.12-slim\n"
        f"ARG PACKAGE_SPEC=torchsparsegradutils[all]=={version}\n"
        "RUN python - <<'PY'\n"
        f'assert dist.version == "{version}", dist.version\n'
        "PY\n",
        encoding="utf-8",
    )


def run_updater(root, version, *extra_args):
    return subprocess.run(
        [sys.executable, str(UPDATE_SCRIPT), version, *extra_args, "--root", str(root)],
        capture_output=True,
        check=False,
        text=True,
    )


def test_update_release_version_updates_every_location(tmp_path):
    write_version_files(tmp_path)

    result = run_updater(tmp_path, "0.2.6")

    assert result.returncode == 0, result.stderr
    for path in (
        tmp_path / "pyproject.toml",
        tmp_path / "docs" / "source" / "conf.py",
        tmp_path / "Dockerfile.pip-install",
    ):
        contents = path.read_text(encoding="utf-8")
        assert "0.2.5" not in contents
        assert "0.2.6" in contents

    check_result = run_updater(tmp_path, "0.2.6", "--check")
    assert check_result.returncode == 0, check_result.stderr


def test_update_release_version_rejects_inconsistent_metadata(tmp_path):
    write_version_files(tmp_path)
    conf_path = tmp_path / "docs" / "source" / "conf.py"
    conf_path.write_text(
        conf_path.read_text(encoding="utf-8").replace('version = "0.2.5"', 'version = "0.2.4"'),
        encoding="utf-8",
    )

    result = run_updater(tmp_path, "0.2.6")

    assert result.returncode != 0
    assert "Release version metadata is inconsistent" in result.stderr


def test_update_release_version_rejects_non_increasing_version(tmp_path):
    write_version_files(tmp_path)

    result = run_updater(tmp_path, "0.2.5")

    assert result.returncode != 0
    assert "must be greater than current version" in result.stderr


def test_update_release_version_rejects_invalid_stable_version(tmp_path):
    write_version_files(tmp_path)

    result = run_updater(tmp_path, "v0.2.6")

    assert result.returncode != 0
    assert "without a leading 'v'" in result.stderr
