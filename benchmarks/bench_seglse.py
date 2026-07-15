"""Op-level acceptance-bar benchmark for tsgu::seglse (spec/commit.md Phase 3
commit 12; spec/benchmarks.md §3: seglse baselines = pytorch_scatter's
scatter_logsumexp on equivalent index arrays, and the old pure-PyTorch path
(tests/oracle's frozen Oracle A)).

    uv run python -m benchmarks.bench_seglse

Runs benchmarks.md §1's protocol (do_bench CUDA-event windowing, memory
reset/peak/workspace) for one representative CSR configuration on the
synthetic tier (benchmarks.md §2), and writes one JSON per baseline under
benchmarks/results/ (backend="custom" -- a real tsgu:: kernel exists to
attribute "ours" to, unlike the oracle-only rows benchmarks/run.py's
docstring describes for pre-kernel commits).

pytorch_scatter is optional: if it isn't importable, that row is skipped and
the run says so loudly instead of fabricating a baseline number.
"""

from __future__ import annotations

import torch

from benchmarks import harness, memory
from benchmarks.env import fingerprint
from benchmarks.results import Result, write_result
from tests.oracle import oracle_sparse_logsumexp

try:
    import torch_scatter

    _HAS_TORCH_SCATTER = True
except ImportError:
    _HAS_TORCH_SCATTER = False


# Synthetic CSR config (benchmarks.md §2 "synthetic batched/ragged" tier,
# unbatched slice): DLMC-shaped sparsity (~90% sparse), n_rows x n_cols with
# a p sweep not applicable to seglse (no rhs width -- p column is null per
# the acceptance-bar table's own seglse rows in benchmarks.md §5).
_N, _M = 4096, 4096
_DENSITY = 0.10  # ~90% sparse, DLMC transformer regime


def _make_csr(device, value_dtype, index_dtype, seed=0):
    torch.manual_seed(seed)
    dense = torch.randn(_N, _M, dtype=value_dtype, device=device)
    mask = torch.rand(_N, _M, device=device) < _DENSITY
    dense = dense * mask
    csr = dense.to_sparse_csr()
    crow = csr.crow_indices().to(index_dtype)
    col = csr.col_indices().to(index_dtype)
    vals = csr.values().clone().requires_grad_(True)
    return dense, crow, col, vals


def _bench_row(*, baseline_name, baseline_ms, ours_ms, bar, matrix, nse, fp, mem=None, backward=False):
    speedup = baseline_ms / ours_ms if (baseline_ms and ours_ms) else None
    bar_met = speedup is not None and speedup >= 1.0
    variant = f"csr·B=1·f32/i32{'·bwd' if backward else ''}"
    result = Result.build(
        op="seglse",
        family="segmented-logsumexp",
        backend="custom",
        variant=variant,
        matrix=matrix,
        n=_N,
        m=_M,
        nse=nse,
        p=None,
        fp=fp,
        baseline_name=baseline_name,
        baseline_ms=baseline_ms,
        ours_ms=ours_ms,
        speedup=speedup,
        peak_fwd_mb=mem.peak_fwd_mb if mem else None,
        workspace_mb=mem.workspace_mb if mem else None,
        bar=bar,
        bar_met=bar_met,
    )
    path = write_result(result)
    print(f"[{baseline_name}] baseline={baseline_ms:.4f}ms ours={ours_ms:.4f}ms speedup={speedup:.2f}x "
          f"bar={bar} met={bar_met} -> {path}")
    return result


def main() -> None:
    device = torch.device("cuda")
    value_dtype, index_dtype = torch.float32, torch.int32
    fp = fingerprint()

    dense, crow, col, vals = _make_csr(device, value_dtype, index_dtype)
    nse = vals.numel()
    print(f"matrix: dense {_N}x{_M}, nse={nse} ({nse / (_N * _M):.1%} density)")

    # --- ours: tsgu::seglse (dim=1, include_zeros=True; via the public wrapper) ---
    import torchsparsegradutils as tsgu

    def _ours_fwd():
        sp = torch.sparse_csr_tensor(crow, col, vals.detach(), (_N, _M))
        return tsgu.sparse_logsumexp(sp, dim=1, include_zeros=True)

    ours_timing = harness.do_bench(_ours_fwd, device=device)
    print(f"ours (tsgu::seglse) fwd: median={ours_timing.median_ms:.4f}ms "
          f"p10={ours_timing.p10_ms:.4f} p90={ours_timing.p90_ms:.4f} degraded={ours_timing.degraded}")

    io_bytes = nse * (vals.element_size() + col.element_size()) + crow.numel() * crow.element_size()
    mem = memory.measure(_ours_fwd, io_bytes=io_bytes, device=device)
    print(f"ours memory: peak_fwd={mem.peak_fwd_mb}MB workspace={mem.workspace_mb}MB")

    # --- baseline 1: pytorch_scatter.scatter_logsumexp on equivalent index arrays ---
    if _HAS_TORCH_SCATTER:
        row_idx = torch.repeat_interleave(
            torch.arange(_N, device=device, dtype=torch.int64), (crow[1:] - crow[:-1]).long()
        )

        def _scatter_fwd():
            return torch_scatter.scatter_logsumexp(vals.detach(), row_idx, dim=0, dim_size=_N)

        scatter_timing = harness.do_bench(_scatter_fwd, device=device)
        print(f"pytorch_scatter fwd: median={scatter_timing.median_ms:.4f}ms")
        _bench_row(
            baseline_name="pytorch_scatter.scatter_logsumexp",
            baseline_ms=scatter_timing.median_ms,
            ours_ms=ours_timing.median_ms,
            bar="win",
            matrix="synth-dlmc90",
            nse=nse,
            fp=fp,
            mem=mem,
        )
    else:
        print("pytorch_scatter not installed -- SKIPPING that acceptance-bar row (not fabricated).")

    # --- baseline 2: old pure-PyTorch path (tests/oracle Oracle A) ---
    sp_dense_layout = torch.sparse_csr_tensor(crow, col, vals.detach(), (_N, _M))

    def _oracle_fwd():
        return oracle_sparse_logsumexp(sp_dense_layout, dim=1, include_zeros=True)

    oracle_timing = harness.do_bench(_oracle_fwd, device=device)
    print(f"oracle (old pure-PyTorch) fwd: median={oracle_timing.median_ms:.4f}ms")
    _bench_row(
        baseline_name="oracle-A (tests/oracle, frozen pure-PyTorch, old sparse_logsumexp)",
        baseline_ms=oracle_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        bar="win",
        matrix="synth-dlmc90",
        nse=nse,
        fp=fp,
        mem=mem,
    )


if __name__ == "__main__":
    main()
