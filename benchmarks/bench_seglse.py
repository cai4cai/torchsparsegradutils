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


def _bench_row(
    *, filename, baseline_name, baseline_ms, ours_ms, bar, matrix, nse, fp, mem=None, bw=None, backward=False
):
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
        peak_bwd_mb=mem.peak_bwd_mb if mem else None,
        workspace_mb=mem.workspace_mb if mem else None,
        bw_pct_peak=bw,
        bar=bar,
        bar_met=bar_met,
        meta={
            "workspace_fwd_mb": mem.workspace_fwd_mb,
            "workspace_bwd_mb": mem.workspace_bwd_mb,
            "workspace_bound_met": mem.workspace_bound_met,
        }
        if mem
        else None,
    )
    path = write_result(result, filename=filename)
    print(
        f"[{baseline_name}] baseline={baseline_ms:.4f}ms ours={ours_ms:.4f}ms speedup={speedup:.2f}x "
        f"bar={bar} met={bar_met} -> {path}"
    )
    return result


def main() -> None:
    device = torch.device("cuda")
    value_dtype, index_dtype = torch.float32, torch.int32
    fp = fingerprint()

    dense, crow, col, vals = _make_csr(device, value_dtype, index_dtype)
    nse = vals.numel()
    print(f"matrix: dense {_N}x{_M}, nse={nse} ({nse / (_N * _M):.1%} density)")
    # Free the dense source matrix (n*m*4 bytes of generator scaffolding):
    # anything left resident is counted by the allocator peak and would
    # pollute the §1 memory measurement below.
    del dense

    # --- ours: tsgu::seglse (dim=1, include_zeros=True; via the public wrapper) ---
    import torchsparsegradutils as tsgu

    def _ours_fwd():
        sp = torch.sparse_csr_tensor(crow, col, vals.detach(), (_N, _M))
        return tsgu.sparse_logsumexp(sp, dim=1, include_zeros=True)

    ours_timing = harness.do_bench(_ours_fwd, device=device)
    print(
        f"ours (tsgu::seglse) fwd: median={ours_timing.median_ms:.4f}ms "
        f"p10={ours_timing.p10_ms:.4f} p90={ours_timing.p90_ms:.4f} degraded={ours_timing.degraded}"
    )

    # --- memory: forward under grad + backward (benchmarks.md §1), on the
    # raw op (torch.ops.tsgu.seglse — the same kernel the wrapper timing
    # above hits, minus the sparse-ctor plumbing) so the registered autograd
    # (values grad via tsgu::seglse_bwd) is what gets measured; loss =
    # out.sum(). io_bytes = every resident matrix tensor (vals, col, crow —
    # col is matrix data even though seglse itself reads only vals+rowptr) +
    # the differentiable vals copy + fwd output/saved lse + upstream grad +
    # grad_vals (backward's output).
    vals_leaf = vals.detach().clone().requires_grad_(True)

    def _grad_fwd():
        return torch.ops.tsgu.seglse(vals_leaf, crow, 1, _N, _M, True)

    def _grad_bwd(out):
        # Sum-style loss, with the upstream gradient materialized explicitly:
        # autograd's own sum() backward delivers an expanded stride-0 grad,
        # which the raw kernels' backward primitives read as a contiguous
        # buffer (the landmine _seglse_backward documents) -- ones_like is
        # the same mathematical gradient, contiguous by construction.
        out.backward(torch.ones_like(out))

    v = vals.element_size()
    op_bytes = nse * v + crow.numel() * crow.element_size()
    io_bytes = op_bytes + nse * col.element_size() + 2 * nse * v + 2 * _N * v
    mem = None
    if memory.budget_guard(4 * io_bytes, label="seglse grad memory pass"):
        mem = memory.measure(_grad_fwd, _grad_bwd, io_bytes=io_bytes, bound_bytes=op_bytes, device=device)
        print(
            f"ours memory: peak_fwd={mem.peak_fwd_mb}MB peak_bwd={mem.peak_bwd_mb}MB "
            f"workspace={mem.workspace_mb}MB bound_met={mem.workspace_bound_met}"
        )

    # Bytes-moved model (benchmarks.md §1, compulsory traffic — see
    # benchmarks/memory.py): vals + rowptr read + per-segment output write
    # (seglse reads no col array).
    bytes_moved = op_bytes + _N * v
    bw = memory.bw_pct_peak(bytes_moved, ours_timing.median_ms)
    print(f"achieved bandwidth (compulsory-traffic model): {bw:.1f}% of {memory.PEAK_DRAM_BANDWIDTH_GB_S:.0f} GB/s")

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
            filename="seglse_vs_pytorch_scatter.json",
            baseline_name="pytorch_scatter.scatter_logsumexp",
            baseline_ms=scatter_timing.median_ms,
            ours_ms=ours_timing.median_ms,
            bar="win",
            matrix="synth-dlmc90",
            nse=nse,
            fp=fp,
            mem=mem,
            bw=bw,
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
        filename="seglse_vs_oracle.json",
        baseline_name="oracle-A (tests/oracle, frozen pure-PyTorch, old sparse_logsumexp)",
        baseline_ms=oracle_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        bar="win",
        matrix="synth-dlmc90",
        nse=nse,
        fp=fp,
        mem=mem,
        bw=bw,
    )


if __name__ == "__main__":
    main()
