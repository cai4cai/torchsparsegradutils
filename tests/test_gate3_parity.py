"""Gate 3 — parity (spec/testing.md "Gates & ordering": "Oracle A + B, full
matrix"; spec/commit.md Phase 2 #11).

The full Oracle-A-vs-Oracle-B parity matrix (layout x batched/unbatched x
value dtype x index dtype x p-sweep, per map.md's Contract column) is
populated per op as each op's kernel lands (Phase 3, commits 12-19) — there
is no CUDA implementation to run parity against yet. What this commit adds
is the one item testing.md names explicitly as CPU-runnable: "CPU CI (no
GPU): ... oracle self-consistency (A vs B on CPU)".

- Oracle A: ``tests/oracle`` — the frozen pure-PyTorch implementation
  extracted from git history (testing.md Pillar 1).
- Oracle B: an independent fp64-dense reference, computed inline here (per
  testing.md: "Compute the op densely in fp64, cast to the test dtype" —
  there is no separate Oracle-B module; every parity test builds its own
  dense reference).

A discrepancy here would mean Oracle A itself is suspect (e.g. a wrong
adjoint formula the rewrite would inherit), which Oracle-A-only testing
could never catch (testing.md's whole reason for having two oracles). This
also doubles as a standing regression check on the oracle extraction
(commit 6) itself.
"""

from __future__ import annotations

import pytest
import torch

from tests._tolerances import assert_close
from tests.oracle import oracle_sparse_mm
from torchsparsegradutils.utils.random_sparse import rand_sparse

pytestmark = pytest.mark.gate3

_N, _M, _P, _NNZ = 6, 5, 3, 12


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
def test_oracle_a_vs_oracle_b_sparse_mm(layout, value_dtype):
    torch.manual_seed(0)
    A = rand_sparse((_N, _M), _NNZ, layout=layout, values_dtype=value_dtype)
    B = torch.randn(_M, _P, dtype=value_dtype)

    oracle_a = oracle_sparse_mm(A, B)

    # Oracle B: dense fp64 reference, cast down to the test dtype.
    a_dense_f64 = A.to_dense().to(torch.float64)
    oracle_b = (a_dense_f64 @ B.to(torch.float64)).to(value_dtype)

    # reduction_length: terms summed per output element ~= average nnz per
    # row (the SpMM contraction dimension's sparse occupancy).
    reduction_length = max(1, _NNZ // _N)
    assert_close(oracle_a, oracle_b, reduction_length=reduction_length)
