"""Gate 1 — smoke (spec/testing.md "Gates & ordering": "one tiny input per
op — fail fast"; spec/commit.md Phase 2 #11).

Only ``tsgu::_smoke`` (the bring-up op from commit 10) exists end-to-end
today — the nine real ops (map.md routing table) get their own gate-1
coverage as each lands its kernel commit (Phase 3). This module is the
whole of gate 1's registry right now; it demonstrates the marker mechanism
(``pytest -m gate1``, run first by ``tests/gpu_gates.py``) rather than
covering real kernels.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate1


def test_smoke_op_schema_registered():
    """The schema-registration half of the smoke op is pure Python (no CUDA
    needed) -- runs unconditionally, GPU or not."""
    assert hasattr(torch.ops.tsgu, "_smoke"), "tsgu::_smoke is not registered"
    schema = torch.ops.tsgu._smoke.default._schema
    assert schema.name == "tsgu::_smoke"


@pytest.mark.skipif(
    not (torch.cuda.is_available() and backend_available()),
    reason=(
        "tsgu::_smoke has no CPU implementation by design (spec/commit.md "
        "Phase 2 #10) -- it exists only to prove the torchsparsegradutils_cuda "
        "backend end to end and needs both a CUDA device and a loaded, "
        "version-matched backend, neither of which is available on this "
        "machine right now. This SKIP (not a failure) is gate 1 degrading "
        "cleanly without a GPU -- the expected outcome for `tox -e gpu` here."
    ),
)
def test_smoke_op_runs_on_cuda():
    x = torch.zeros(4, device="cuda")
    out = torch.ops.tsgu._smoke(x)
    assert torch.allclose(out, torch.ones(4, device="cuda"))
