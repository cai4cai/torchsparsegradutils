"""Gate 6 — contract conformance for ``convert_coo_to_csr_indices_values`` /
tsgu::coo2csr (spec/testing.md "Gates & ordering": "contract conformance";
map.md invariant 7: "raise, never silently accept"; spec/commit.md Phase 3
commit 19).

Two sections:

- **Wrapper validation (device-independent).** The wrapper's shape/value
  checks (torchsparsegradutils/utils/convert.py) raise BEFORE any op
  dispatch, so these tests run on CPU tensors everywhere -- no CUDA, no
  backend needed. Includes the commit-19 regression: batched input with
  ragged nse per item (an empty batch item among occupied ones included)
  now raises a clear ValueError instead of silently mis-reshaping (the
  legacy ``torch.unique``-based num_batches derivation dropped empty
  items). Messages state the accepted logical form alongside the received
  one (naming.md §1's error template).
- **Kernel-side contract (CUDA-gated).** A sanity valid call plus the
  launcher's defensive STD_TORCH_CHECKs (device, 1-D shapes / equal
  lengths, one shared index dtype), following the same conventions as the
  other kernels' launchers (e.g. cuda/csrc/kernels/spmm/spmm.cu).

No workspace assertion here: coo2csr's outputs ARE O(nse + B*n) by
definition (rowptr + col_sorted + permutation), and its sort scratch is the
one measured by the memory harness in benchmarks/bench_coo2csr.py.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available
from torchsparsegradutils.utils.convert import convert_coo_to_csr_indices_values

pytestmark = pytest.mark.gate6

_SKIP_REASON = (
    "tsgu::coo2csr has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)


def _coo2csr_cuda_ready() -> bool:
    # Dispatch-key check: skip (not error) until the cuda/ tree is rebuilt
    # with the convert kernel (commit 19 is developed in two concurrent lanes).
    return (
        torch.cuda.is_available()
        and backend_available()
        and torch._C._dispatch_has_kernel_for_dispatch_key("tsgu::coo2csr", "CUDA")
    )


requires_cuda_backend = pytest.mark.skipif(not _coo2csr_cuda_ready(), reason=_SKIP_REASON)


# ---------------------------------------------------------------------------
# Wrapper validation -- device-independent (CPU tensors; raises before any
# op dispatch, so no backend is needed).
# ---------------------------------------------------------------------------


def test_wrapper_rejects_too_few_index_rows():
    with pytest.raises(ValueError, match=r"at least 2 rows"):
        convert_coo_to_csr_indices_values(torch.zeros(1, 4, dtype=torch.int64), 4)


def test_wrapper_rejects_too_many_index_rows():
    with pytest.raises(ValueError, match=r"at most 3 rows"):
        convert_coo_to_csr_indices_values(torch.zeros(4, 4, dtype=torch.int64), 4)


def test_wrapper_rejects_row_indices_out_of_range():
    indices = torch.tensor([[0, 5], [1, 0]])
    with pytest.raises(ValueError, match=r"less than num_rows"):
        convert_coo_to_csr_indices_values(indices, 4)


def test_wrapper_rejects_values_length_mismatch():
    indices = torch.tensor([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match=r"does not match number of indices"):
        convert_coo_to_csr_indices_values(indices, 2, torch.zeros(3))


def test_wrapper_rejects_ragged_nse_per_batch_item():
    """Commit-19 regression: pre-fix this silently produced misaligned
    (num_batches, -1) reshapes; the message must state the constraint and
    the received (ragged) nse per item."""
    indices = torch.tensor([[0, 1, 1], [0, 0, 1], [1, 0, 1]])
    with pytest.raises(ValueError, match=r"equal nse per batch item.*\[1, 2\]"):
        convert_coo_to_csr_indices_values(indices, 2, torch.zeros(3))


def test_wrapper_rejects_empty_batch_item_as_ragged():
    """An empty batch item BETWEEN occupied ones is ragged nse. Pre-fix,
    torch.unique dropped item 1 entirely (wrong num_batches, silently wrong
    shapes); now the item is counted (num_batches = max(batch) + 1) and the
    raggedness raises."""
    indices = torch.tensor([[0, 0, 2, 2], [0, 1, 0, 1], [1, 0, 1, 0]])
    with pytest.raises(ValueError, match=r"equal nse per batch item.*\[2, 0, 2\]"):
        convert_coo_to_csr_indices_values(indices, 2, torch.zeros(4))


# ---------------------------------------------------------------------------
# Kernel-side contract -- CUDA-gated (defensive launcher checks).
# ---------------------------------------------------------------------------


def _valid_inputs():
    batch = torch.tensor([0, 0, 1], dtype=torch.int64, device="cuda")
    row = torch.tensor([1, 0, 1], dtype=torch.int64, device="cuda")
    col = torch.tensor([0, 1, 1], dtype=torch.int64, device="cuda")
    return batch, row, col


@requires_cuda_backend
def test_sanity_valid_call_does_not_raise():
    batch, row, col = _valid_inputs()
    rowptr, col_sorted, permutation = torch.ops.tsgu.coo2csr(batch, row, col, 2, 2)
    assert rowptr.shape == (5,)
    assert col_sorted.shape == (3,) and permutation.shape == (3,)


@requires_cuda_backend
def test_rejects_cpu_tensors():
    batch, row, col = _valid_inputs()
    with pytest.raises(RuntimeError, match=r"CUDA"):
        torch.ops.tsgu.coo2csr(batch.cpu(), row, col, 2, 2)


@requires_cuda_backend
def test_rejects_mismatched_coordinate_lengths():
    batch, row, col = _valid_inputs()
    with pytest.raises(RuntimeError, match=r"nse_total"):
        torch.ops.tsgu.coo2csr(batch, row[:-1], col, 2, 2)


@requires_cuda_backend
def test_rejects_wrong_ndim_coordinates():
    batch, row, col = _valid_inputs()
    with pytest.raises(RuntimeError, match=r"nse_total"):
        torch.ops.tsgu.coo2csr(batch, row.unsqueeze(0), col, 2, 2)


@requires_cuda_backend
def test_rejects_mismatched_index_dtypes():
    batch, row, col = _valid_inputs()
    with pytest.raises(RuntimeError, match=r"index dtype"):
        torch.ops.tsgu.coo2csr(batch, row, col.to(torch.int32), 2, 2)


@requires_cuda_backend
def test_error_message_states_accepted_and_received_shape():
    """naming.md §1 template: "state the accepted logical forms and the
    received shape" -- spot-check one launcher message contains both."""
    batch, row, col = _valid_inputs()
    with pytest.raises(RuntimeError) as exc_info:
        torch.ops.tsgu.coo2csr(batch, row.unsqueeze(0), col, 2, 2)
    message = str(exc_info.value)
    assert "(nse_total,)" in message  # accepted logical form
    assert "(1, 2, 1)" in message  # received dims
