"""Gate 2 — opcheck for tsgu::coo2csr (spec/testing.md "Gates & ordering":
"opcheck (all ops x representative inputs)"; spec/commit.md Phase 3 commit
19).

tsgu::coo2csr gets its real CUDA implementation in this commit AND is wired
into ``convert_coo_to_csr_indices_values`` / ``BatchedCSR.from_torch``'s COO
paths (map.md "Kernel routing": ``convert_coo_to_csr*`` -> ``tsgu::coo2csr``)
-- so opcheck here exercises schema and fake-kernel consistency for real.

**There is deliberately NO gate-4 (gradcheck) file for this op**: coo2csr is
index-only (map.md routing table: "— (index-only, no grad)") — it only
rearranges integer index tensors, which carry no gradient, so it has no
``register_autograd`` and nothing for gradcheck to prove. Every other gate
(2, 3, 5, 6) applies.

Schema note: the op returns ``List[Tensor]`` (length 3 —
``[rowptr, col_sorted, permutation]``), not a single tensor; opcheck's
real-vs-fake comparison covers all three.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate2

_SKIP_REASON = (
    "tsgu::coo2csr has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)


def _coo2csr_cuda_ready() -> bool:
    # Identical in effect to the other gate files' machine-level condition
    # once commit 19's kernel is registered; the extra dispatch-key check
    # keeps this file skipping (not erroring) when the front-package wiring
    # is present but the cuda/ tree has not yet been rebuilt with the
    # convert kernel (commit 19 is developed in two concurrent lanes).
    return (
        torch.cuda.is_available()
        and backend_available()
        and torch._C._dispatch_has_kernel_for_dispatch_key("tsgu::coo2csr", "CUDA")
    )


requires_cuda_backend = pytest.mark.skipif(not _coo2csr_cuda_ready(), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)


def _shuffled_coords(index_dtype, *, B=2, n=3, m=4, seed=0):
    """Representative scattered coordinates: unsorted, with empty rows and
    both batch items occupied. Duplicate-free (a valid pattern has none)."""
    gen = torch.Generator().manual_seed(seed)
    coords = [(b, r, c) for b in range(B) for r in range(n) for c in range(m)]
    keep = torch.randperm(len(coords), generator=gen)[: (B * n * m) // 2]
    chosen = [coords[i] for i in keep.tolist()]
    batch = torch.tensor([b for b, _r, _c in chosen], dtype=index_dtype, device="cuda")
    row = torch.tensor([r for _b, r, _c in chosen], dtype=index_dtype, device="cuda")
    col = torch.tensor([c for _b, _r, c in chosen], dtype=index_dtype, device="cuda")
    return batch, row, col, B, n


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_opcheck_coo2csr_representative(index_dtype):
    batch, row, col, B, n = _shuffled_coords(index_dtype)
    torch.library.opcheck(torch.ops.tsgu.coo2csr.default, (batch, row, col, B, n))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_opcheck_coo2csr_b_equals_one(index_dtype):
    """B=1 encodes unbatched (naming.md §2) -- batch is all-zero."""
    row = torch.tensor([2, 0, 1, 0], dtype=index_dtype, device="cuda")
    col = torch.tensor([1, 2, 0, 0], dtype=index_dtype, device="cuda")
    batch = torch.zeros_like(row)
    torch.library.opcheck(torch.ops.tsgu.coo2csr.default, (batch, row, col, 1, 3))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_opcheck_coo2csr_nse_zero(index_dtype):
    """No specified entries at all -- rowptr must still be a well-formed
    all-zero (B * n + 1,) pointer."""
    empty = torch.zeros(0, dtype=index_dtype, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.coo2csr.default, (empty, empty.clone(), empty.clone(), 2, 3))


@requires_cuda_backend
def test_opcheck_coo2csr_empty_rows_and_empty_batch_item():
    """Batch item 0 entirely empty; occupied item 1 leaves rows 0 and 2
    empty (adversarial structure, testing.md Pillar 2)."""
    batch = torch.tensor([1, 1], dtype=torch.int64, device="cuda")
    row = torch.tensor([1, 1], dtype=torch.int64, device="cuda")
    col = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.coo2csr.default, (batch, row, col, 2, 3))


@requires_cuda_backend
def test_opcheck_coo2csr_noncontiguous_inputs():
    """testing.md Pillar 2: "strided and non-contiguous dense operands" --
    every input built as a stride-2 slice so none is contiguous; the
    launcher's host-side .contiguous() must make this a real, correct
    execution for opcheck's real-vs-fake comparison to pass."""
    base = torch.tensor([0, 9, 0, 9, 1, 9, 1, 9], dtype=torch.int64, device="cuda")
    batch = base[::2]
    row = torch.tensor([1, 9, 0, 9, 1, 9, 0, 9], dtype=torch.int64, device="cuda")[::2]
    col = torch.tensor([0, 9, 1, 9, 1, 9, 0, 9], dtype=torch.int64, device="cuda")[::2]
    assert not (batch.is_contiguous() and row.is_contiguous() and col.is_contiguous())
    torch.library.opcheck(torch.ops.tsgu.coo2csr.default, (batch, row, col, 2, 2))
