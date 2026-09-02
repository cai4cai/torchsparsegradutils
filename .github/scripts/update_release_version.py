#!/usr/bin/env python3
"""Validate or update every repository location containing the release version."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

STABLE_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


class ReleaseVersionError(ValueError):
    """Raised when release version metadata is invalid or inconsistent."""


@dataclass(frozen=True)
class VersionFiles:
    pyproject: Path
    sphinx_conf: Path
    pip_install_dockerfile: Path

    @classmethod
    def from_root(cls, root: Path) -> "VersionFiles":
        return cls(
            pyproject=root / "pyproject.toml",
            sphinx_conf=root / "docs" / "source" / "conf.py",
            pip_install_dockerfile=root / "Dockerfile.pip-install",
        )


def parse_stable_version(value: str) -> tuple[int, int, int]:
    match = STABLE_VERSION.fullmatch(value)
    if match is None:
        raise ReleaseVersionError(
            f"Expected a stable release version in X.Y.Z form without a leading 'v'; got {value!r}"
        )
    return tuple(int(part) for part in match.groups())


def _single_match(pattern: str, text: str, path: Path, description: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ReleaseVersionError(f"Expected exactly one {description} in {path}, found {len(matches)}")
    return matches[0]


def read_versions(files: VersionFiles) -> dict[str, str]:
    pyproject_text = files.pyproject.read_text(encoding="utf-8")
    sphinx_text = files.sphinx_conf.read_text(encoding="utf-8")
    dockerfile_text = files.pip_install_dockerfile.read_text(encoding="utf-8")

    project_section = _single_match(
        r"^\[project\]\s*$([\s\S]*?)(?=^\[|\Z)",
        pyproject_text,
        files.pyproject,
        "[project] section",
    )
    package_version = _single_match(
        r'^version\s*=\s*"([^"]+)"\s*$',
        project_section,
        files.pyproject,
        "project.version value",
    )

    return {
        "pyproject.toml project.version": package_version,
        "docs/source/conf.py release": _single_match(
            r'^release\s*=\s*"([^"]+)"\s*$',
            sphinx_text,
            files.sphinx_conf,
            "Sphinx release value",
        ),
        "docs/source/conf.py version": _single_match(
            r'^version\s*=\s*"([^"]+)"\s*$',
            sphinx_text,
            files.sphinx_conf,
            "Sphinx version value",
        ),
        "Dockerfile.pip-install PACKAGE_SPEC": _single_match(
            r"^ARG PACKAGE_SPEC=torchsparsegradutils\[all\]==([^\s]+)\s*$",
            dockerfile_text,
            files.pip_install_dockerfile,
            "PACKAGE_SPEC version",
        ),
        "Dockerfile.pip-install installed-version assertion": _single_match(
            r'^assert dist\.version == "([^"]+)", dist\.version\s*$',
            dockerfile_text,
            files.pip_install_dockerfile,
            "installed distribution version assertion",
        ),
    }


def validate_versions(files: VersionFiles, expected: str | None = None) -> str:
    versions = read_versions(files)
    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        details = "\n".join(f"- {location}: {version}" for location, version in versions.items())
        raise ReleaseVersionError(f"Release version metadata is inconsistent:\n{details}")

    current = unique_versions.pop()
    parse_stable_version(current)
    if expected is not None and current != expected:
        raise ReleaseVersionError(f"Expected all release metadata to be {expected}, found {current}")
    return current


def _replace_once(text: str, old: str, new: str, path: Path, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ReleaseVersionError(f"Expected exactly one {description} in {path}, found {count}")
    return text.replace(old, new, 1)


def update_versions(files: VersionFiles, new_version: str) -> str:
    new_tuple = parse_stable_version(new_version)
    current = validate_versions(files)
    if new_tuple <= parse_stable_version(current):
        raise ReleaseVersionError(f"New version {new_version} must be greater than current version {current}")

    pyproject_text = _replace_once(
        files.pyproject.read_text(encoding="utf-8"),
        f'version = "{current}"',
        f'version = "{new_version}"',
        files.pyproject,
        "project version",
    )
    sphinx_text = files.sphinx_conf.read_text(encoding="utf-8")
    sphinx_text = _replace_once(
        sphinx_text,
        f'release = "{current}"',
        f'release = "{new_version}"',
        files.sphinx_conf,
        "Sphinx release value",
    )
    sphinx_text = _replace_once(
        sphinx_text,
        f'version = "{current}"',
        f'version = "{new_version}"',
        files.sphinx_conf,
        "Sphinx version value",
    )
    dockerfile_text = files.pip_install_dockerfile.read_text(encoding="utf-8")
    dockerfile_text = _replace_once(
        dockerfile_text,
        f"ARG PACKAGE_SPEC=torchsparsegradutils[all]=={current}",
        f"ARG PACKAGE_SPEC=torchsparsegradutils[all]=={new_version}",
        files.pip_install_dockerfile,
        "PACKAGE_SPEC version",
    )
    dockerfile_text = _replace_once(
        dockerfile_text,
        f'assert dist.version == "{current}", dist.version',
        f'assert dist.version == "{new_version}", dist.version',
        files.pip_install_dockerfile,
        "installed distribution version assertion",
    )

    files.pyproject.write_text(pyproject_text, encoding="utf-8")
    files.sphinx_conf.write_text(sphinx_text, encoding="utf-8")
    files.pip_install_dockerfile.write_text(dockerfile_text, encoding="utf-8")
    validate_versions(files, expected=new_version)
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Stable release version in X.Y.Z form, without a leading 'v'")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that every version location already matches VERSION without changing files",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    files = VersionFiles.from_root(args.root.resolve())

    try:
        parse_stable_version(args.version)
        if args.check:
            validate_versions(files, expected=args.version)
            print(f"Release version metadata is consistent at {args.version}")
        else:
            previous = update_versions(files, args.version)
            print(f"Updated release version metadata from {previous} to {args.version}")
    except (OSError, ReleaseVersionError) as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
