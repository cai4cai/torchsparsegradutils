"""Op-level acceptance-bar benchmark for tsgu::seglse_bidir (spec/commit.md
Phase 3 commit 13; spec/benchmarks.md §3 seglse_bidir row): three bars --
(a) beat pytorch_scatter.scatter_logsumexp (two separate calls, its only
comparable CUDA code), (b) beat the old pure-PyTorch bidir path (tests/oracle
Oracle A), (c) >=1.5x over two separate tsgu::seglse calls (else the fused op
has no reason to exist -- kernels.md Family 2 "Bidirectional").

    uv run python -m benchmarks.bench_seglse_bidir

Same protocol as bench_seglse.py (benchmarks.md §1: do_bench CUDA-event
windowing, memory reset/peak/workspace), synthetic CSR tier (benchmarks.md
§2). Each row gets an explicit, distinct filename (bench_seglse.py's default
naming collides when multiple _bench_row calls share one run timestamp).

pytorch_scatter is optional: if it isn't importable, that row is skipped and
the run says so loudly instead of fabricating a baseline number.
"""

from __future__ import annotations

import torch

from benchmarks import harness, memory
from benchmarks.env import fingerprint
from benchmarks.results import Result, write_result
from tests.oracle import oracle_sparse_bidir_logsumexp

try:
    import torch_scatter

    _HAS_TORCH_SCATTER = True
except ImportError:
    _HAS_TORCH_SCATTER = False


# Synthetic CSR config (benchmarks.md §2 "synthetic batched/ragged" tier,
# unbatched slice) -- same shape/density as bench_seglse.py so the seglse_bidir
# vs 2x seglse comparison is apples-to-apples.
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


def _bench_row(*, filename, baseline_name, baseline_ms, ours_ms, bar, matrix, nse, fp, mem=None, bw=None):
    speedup = baseline_ms / ours_ms if (baseline_ms and ours_ms) else None
    bar_met = speedup is not None and speedup >= bar
    variant = "csr·B=1·f32/i32"
    result = Result.build(
        op="seglse_bidir",
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
        bar=f">= {bar}x" if bar != 1.0 else "win",
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

    import torchsparsegradutils as tsgu

    def _sp():
        return torch.sparse_csr_tensor(crow, col, vals.detach(), (_N, _M))

    # --- ours: tsgu::seglse_bidir (via the public wrapper) ---
    def _ours_fwd():
        return tsgu.sparse_bidir_logsumexp(_sp(), include_zeros=True)

    ours_timing = harness.do_bench(_ours_fwd, device=device)
    print(
        f"ours (tsgu::seglse_bidir) fwd: median={ours_timing.median_ms:.4f}ms "
        f"p10={ours_timing.p10_ms:.4f} p90={ours_timing.p90_ms:.4f} degraded={ours_timing.degraded}"
    )

    # --- memory: forward under grad + backward (benchmarks.md §1), on the
    # raw op (torch.ops.tsgu.seglse_bidir — the same kernel the wrapper
    # timing above hits, minus the sparse-ctor plumbing) so the registered
    # autograd (values grad via tsgu::seglse_bidir_bwd) is what gets
    # measured; loss = out.sum(). Output is the padded (2, B, G) buffer,
    # G = max(n, m). io_bytes = inputs (vals + rowptr + col) + the
    # differentiable vals copy + fwd output/saved padded + upstream grad +
    # grad_vals (backward's output).
    vals_leaf = vals.detach().clone().requires_grad_(True)

    def _grad_fwd():
        return torch.ops.tsgu.seglse_bidir(vals_leaf, crow, col, 1, _N, _M, True)

    def _grad_bwd(out):
        # Sum-style loss, with the upstream gradient materialized explicitly:
        # autograd's own sum() backward delivers an expanded stride-0 grad,
        # which the raw kernels' backward primitives read as a contiguous
        # buffer (the landmine _seglse_backward documents) -- ones_like is
        # the same mathematical gradient, contiguous by construction.
        out.backward(torch.ones_like(out))

    v = vals.element_size()
    op_bytes = nse * (v + col.element_size()) + crow.numel() * crow.element_size()
    out_bytes = 2 * max(_N, _M) * v
    io_bytes = op_bytes + 2 * nse * v + 2 * out_bytes  # + vals_leaf + grad_vals
    mem = None
    if memory.budget_guard(4 * io_bytes, label="seglse_bidir grad memory pass"):
        mem = memory.measure(_grad_fwd, _grad_bwd, io_bytes=io_bytes, bound_bytes=op_bytes, device=device)
        print(
            f"ours memory: peak_fwd={mem.peak_fwd_mb}MB peak_bwd={mem.peak_bwd_mb}MB "
            f"workspace={mem.workspace_mb}MB bound_met={mem.workspace_bound_met}"
        )

    # Bytes-moved model (benchmarks.md §1, compulsory traffic — see
    # benchmarks/memory.py): vals + col + rowptr read + padded output write.
    bytes_moved = op_bytes + out_bytes
    bw = memory.bw_pct_peak(bytes_moved, ours_timing.median_ms)
    print(f"achieved bandwidth (compulsory-traffic model): {bw:.1f}% of {memory.PEAK_DRAM_BANDWIDTH_GB_S:.0f} GB/s")

    # --- baseline (a): pytorch_scatter.scatter_logsumexp, two separate calls ---
    if _HAS_TORCH_SCATTER:
        row_idx = torch.repeat_interleave(
            torch.arange(_N, device=device, dtype=torch.int64), (crow[1:] - crow[:-1]).long()
        )
        col_idx = col.long()

        def _scatter_fwd():
            row_lse = torch_scatter.scatter_logsumexp(vals.detach(), row_idx, dim=0, dim_size=_N)
            col_lse = torch_scatter.scatter_logsumexp(vals.detach(), col_idx, dim=0, dim_size=_M)
            return col_lse, row_lse

        scatter_timing = harness.do_bench(_scatter_fwd, device=device)
        print(f"pytorch_scatter (2x scatter_logsumexp) fwd: median={scatter_timing.median_ms:.4f}ms")
        _bench_row(
            filename="seglse_bidir_vs_pytorch_scatter.json",
            baseline_name="pytorch_scatter.scatter_logsumexp (2x, row+col)",
            baseline_ms=scatter_timing.median_ms,
            ours_ms=ours_timing.median_ms,
            bar=1.0,
            matrix="synth-dlmc90",
            nse=nse,
            fp=fp,
            mem=mem,
            bw=bw,
        )
    else:
        print("pytorch_scatter not installed -- SKIPPING that acceptance-bar row (not fabricated).")

    # --- baseline (b): old pure-PyTorch bidir path (tests/oracle Oracle A) ---
    def _oracle_fwd():
        return oracle_sparse_bidir_logsumexp(_sp(), include_zeros=True)

    oracle_timing = harness.do_bench(_oracle_fwd, device=device)
    print(f"oracle (old pure-PyTorch bidir) fwd: median={oracle_timing.median_ms:.4f}ms")
    _bench_row(
        filename="seglse_bidir_vs_oracle.json",
        baseline_name="oracle-A (tests/oracle, frozen pure-PyTorch, old sparse_bidir_logsumexp)",
        baseline_ms=oracle_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        bar=1.0,
        matrix="synth-dlmc90",
        nse=nse,
        fp=fp,
        mem=mem,
        bw=bw,
    )

    # --- baseline (c): two separate tsgu::seglse calls -- the fusion-win bar ---
    def _two_call_fwd():
        sp = _sp()
        return (
            tsgu.sparse_logsumexp(sp, dim=0, include_zeros=True),
            tsgu.sparse_logsumexp(sp, dim=1, include_zeros=True),
        )

    two_call_timing = harness.do_bench(_two_call_fwd, device=device)
    print(f"2x tsgu::seglse fwd: median={two_call_timing.median_ms:.4f}ms")
    _bench_row(
        filename="seglse_bidir_vs_2x_seglse.json",
        baseline_name="2x tsgu::seglse (dim=0 + dim=1)",
        baseline_ms=two_call_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        bar=1.5,
        matrix="synth-dlmc90",
        nse=nse,
        fp=fp,
        mem=mem,
        bw=bw,
    )


if __name__ == "__main__":
    main()
