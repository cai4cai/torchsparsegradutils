"""e2e rsample memory-regression pin, pytest-sized (spec/commit.md Phase 4
commit 20; spec/benchmarks.md §3 memory bars, verbatim): "e2e rsample: the
encoder-CSR backward blow-up documented in the old README is a pinned
regression case — peak_bwd <= 1.2x the COO path's, or the suite fails."

benchmarks/bench_e2e_rsample.py measures the full-size composite; this is
the small, always-runnable version of the same pin (2-channel 24^3 encoder
volume, well under the 4 GB card's budget, seconds not minutes): build the
same PairwiseEncoder -> SparseMultivariateNormal rsample composite in both
layouts, run forward + backward under ``reset_peak_memory_stats`` /
``max_memory_allocated`` (benchmarks.md §1's reset -> forward -> backward ->
record protocol), and assert peak_bwd(CSR) <= 1.2 x peak_bwd(COO).

Plain test, no gateN marker: the six gates are per-``tsgu::``-op suites
(pyproject markers); this pins a composite regression across the public API.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

_SKIP_REASON = (
    "the rsample composite routes through tsgu::spmm/tsgu::sddmm, which are CUDA-only "
    "(architecture.md §4) -- needs a CUDA device and a loaded, version-matched backend."
)


def _e2e_cuda_ready() -> bool:
    # Same machine-level condition as the gate files' guard (e.g.
    # tests/test_gate2_opcheck_grouped_gemm.py): CUDA device + handshaken
    # backend + the kernels this composite dispatches to actually registered.
    return (
        torch.cuda.is_available()
        and backend_available()
        and torch._C._dispatch_has_kernel_for_dispatch_key("tsgu::spmm", "CUDA")
        and torch._C._dispatch_has_kernel_for_dispatch_key("tsgu::sddmm", "CUDA")
    )


requires_cuda_backend = pytest.mark.skipif(not _e2e_cuda_ready(), reason=_SKIP_REASON)

# Small on purpose: 2 x 24^3 -> n = 27,648, nse ~ 270k; runs in seconds and
# peaks far below the 4 GB card's budget (hard resource guard).
CHANNELS, SIDE = 2, 24
RADIUS = 1.5
NUM_SAMPLES = 16
VALUE_DTYPE = torch.float32
INDEX_DTYPE = torch.int64  # COO coalesce coerces int32 -> int64; keep the two layouts symmetric
MEM_BAR_RATIO = 1.2


def _peak_bwd_mb(layout: torch.layout) -> float:
    """peak_bwd for one layout: reset -> forward rsample -> backward -> record."""
    from torchsparsegradutils.distributions import SparseMultivariateNormal
    from torchsparsegradutils.encoders import PairwiseEncoder

    device = torch.device("cuda")
    volume_shape = (CHANNELS, SIDE, SIDE, SIDE)
    encoder = PairwiseEncoder(
        radius=RADIUS,
        volume_shape=volume_shape,
        diag=True,  # LL^T: diagonal lives in the sparse factor
        upper=False,
        channel_voxel_relation="indep",
        layout=layout,
        indices_dtype=INDEX_DTYPE,
        device=device,
    )
    torch.manual_seed(42)
    params = torch.randn(len(encoder.offsets), *volume_shape, dtype=VALUE_DTYPE, device=device)
    with torch.no_grad():
        params.mul_(0.1)
        for i, offset in enumerate(encoder.offsets):
            if all(o == 0 for o in offset):
                params[i] = params[i].abs() + 0.5  # positive diagonal for a valid Cholesky factor
                break
    params.requires_grad_(True)
    loc = torch.zeros(encoder.volume_numel, dtype=VALUE_DTYPE, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    scale_tril = encoder(params)
    samples = SparseMultivariateNormal(loc=loc, scale_tril=scale_tril).rsample((NUM_SAMPLES,))
    torch.cuda.synchronize(device)
    samples.sum().backward()
    torch.cuda.synchronize(device)
    peak_bwd = torch.cuda.max_memory_allocated(device)

    del encoder, params, loc, scale_tril, samples
    torch.cuda.empty_cache()
    return peak_bwd / 2**20


@requires_cuda_backend
def test_encoder_csr_rsample_peak_bwd_within_1p2x_coo():
    peak_coo = _peak_bwd_mb(torch.sparse_coo)
    peak_csr = _peak_bwd_mb(torch.sparse_csr)
    ratio = peak_csr / peak_coo
    assert peak_csr <= MEM_BAR_RATIO * peak_coo, (
        f"encoder-CSR rsample backward memory regression (benchmarks.md §3 pinned case): "
        f"peak_bwd(CSR)={peak_csr:.1f}MB > {MEM_BAR_RATIO}x peak_bwd(COO)={peak_coo:.1f}MB "
        f"(ratio {ratio:.3f}x)"
    )
