"""Regression checks for deprecated upstream PyTorch APIs."""

import ast
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "torchsparsegradutils"
COMPATIBILITY_MODULE = Path("_compat.py")
DEPRECATED_DOCUMENTATION_NORM = re.compile(r"(?<![\w.])torch\.norm\s*\(")


def attribute_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_deprecated_torch_calls_are_not_reintroduced():
    violations = []
    compatibility_calls = []

    for path in PACKAGE_ROOT.rglob("*.py"):
        relative_path = path.relative_to(PACKAGE_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            location = f"{relative_path}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.Attribute) and attribute_name(node) == "torch.jit.script":
                violations.append(location)

            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue

            call_name = attribute_name(node.func)
            if call_name == "torch.norm":
                violations.append(location)
            elif call_name == "torch.triangular_solve":
                if relative_path == COMPATIBILITY_MODULE:
                    compatibility_calls.append(location)
                else:
                    violations.append(location)

    assert not violations, "Deprecated PyTorch APIs found:\n" + "\n".join(violations)
    assert len(compatibility_calls) == 1, "Expected one isolated sparse triangular compatibility call"


def test_documentation_does_not_use_deprecated_norm_apis():
    violations = []
    documentation_paths = [REPOSITORY_ROOT / "README.md"]
    documentation_paths.extend((REPOSITORY_ROOT / "docs").rglob("*.md"))
    documentation_paths.extend((REPOSITORY_ROOT / "docs").rglob("*.rst"))

    for path in documentation_paths:
        if DEPRECATED_DOCUMENTATION_NORM.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert not violations, "Deprecated norm examples found:\n" + "\n".join(violations)
