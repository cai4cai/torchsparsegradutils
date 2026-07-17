"""Op-level acceptance-bar benchmark for tsgu::spmm (spec/commit.md Phase 3
commit 15; spec/benchmarks.md §3: SpMM baselines = cuSPARSE `cusparseSpMM`
(CSR) unbatched, block-diag + cuSPARSE for batched -- "Beat block-diag
decisively; >= parity vs raw cuSPARSE unbatched"; this commit's own T3/T5
instructions add a p=1 SpMV row vs cuSPARSE SpMV explicitly).

    uv run python -m benchmarks.bench_spmm

Runs benchmarks.md §1's protocol (do_bench CUDA-event windowing, memory
reset/peak/workspace) for three representative configurations on the
synthetic tier (benchmarks.md §2: migration-period rule, synthetic-only) and
writes one JSON per row under benchmarks/results/ (backend="custom").

Row 1 -- unbatched CSR vs cuSPARSE SpMM: `torch.sparse.mm` on a CUDA
sparse_csr tensor is documented (torch.sparse docs) to dispatch to
cusparseSpMM -- this *is* the vendor primitive kernels.md/benchmarks.md name
as the SpMM baseline, reached through PyTorch's own public API.

Row 2 -- batched vs block-diag + cuSPARSE: no vendor batched SpMM primitive
exists (kernels.md), so the comparison is the pre-rewrite block-diag pattern
from tests/oracle/sparse_matmul.py's `sparse_mm` (block-diagonalises the
batch of sparse matrices, then a single `torch.sparse.mm` call -- literally
the oracle's own forward path).

Row 3 -- unbatched CSR, p=1 (SpMV) vs cuSPARSE SpMV: same `torch.sparse.mm`
call with a single-column dense operand -- kernels.md: "spmv = spmm with
p = 1, no separate op", so the vendor baseline for SpMV is the same
`torch.sparse.mm`/cusparseSpMM entry point at p=1 (cuSPARSE internally
selects its SpMV path for a single right-hand-side column).
"""

from __future__ import annotations

import torch

from benchmarks import harness, memory
from benchmarks.env import fingerprint
from benchmarks.results import Result, write_result

# --- Row 1 / Row 3: unbatched CSR, DLMC-shaped sparsity (benchmarks.md §2) --
_N1, _M1 = 4096, 4096
_DENSITY1 = 0.10  # ~90% sparse, DLMC transformer regime (matches bench_sddmm.py)

# --- Row 2: batched, synthetic-ragged workhorse (benchmarks.md §2) ---------
_B2, _N2, _M2, _P2 = 8, 1024, 1024, 32
_DENSITY2 = 0.005  # benchmarks.md §2 REFERENCE_DENSITY


def _make_unbatched_csr(device, value_dtype, index_dtype, seed=0):
    """Returns only the folded pattern + values — the dense source matrix
    and the intermediate torch CSR tensor are dropped on return so no
    generator scaffolding stays resident to pollute the §1 allocator-peak
    measurement (the baseline's CSR tensor is rebuilt from these arrays
    after the memory pass)."""
    torch.manual_seed(seed)
    dense = torch.randn(_N1, _M1, dtype=value_dtype, device=device)
    mask = torch.rand(_N1, _M1, device=device) < _DENSITY1
    csr = (dense * mask).to_sparse_csr()
    rowptr = csr.crow_indices().to(index_dtype)
    col = csr.col_indices().to(index_dtype)
    vals = csr.values()
    return rowptr, col, vals


def _make_batched_csr_tensor(device, value_dtype, index_dtype, seed=1):
    """Equal-nse-per-item batched CSR (torch's own batched-CSR storage
    constraint, naming.md §1) -- one sparse_csr tensor, straight from
    ``rand_sparse``, matching what the oracle's block-diag path (and our own
    wrapper) both accept directly."""
    from torchsparsegradutils.utils import rand_sparse

    torch.manual_seed(seed)
    nnz_per_item = max(1, int(_N2 * _M2 * _DENSITY2))
    return rand_sparse(
        (_B2, _N2, _M2),
        nnz_per_item,
        torch.sparse_csr,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )


def _bench_row(
    *, filename, baseline_name, baseline_ms, ours_ms, bar, variant, matrix, n, m, nse, p, fp, mem=None, bw=None
):
    speedup = baseline_ms / ours_ms if (baseline_ms and ours_ms) else None
    bar_met = speedup is not None and speedup >= 1.0
    result = Result.build(
        op="spmm",
        family="spmm",
        backend="custom",
        variant=variant,
        matrix=matrix,
        n=n,
        m=m,
        nse=nse,
        p=p,
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


def _bench_unbatched_vs_cusparse(device, fp, *, p) -> None:
    value_dtype, index_dtype = torch.float32, torch.int32
    rowptr, col, vals = _make_unbatched_csr(device, value_dtype, index_dtype)
    nse = col.numel()
    print(f"unbatched matrix: dense {_N1}x{_M1}, nse={nse} ({nse / (_N1 * _M1):.1%} density), p={p}")

    import torchsparsegradutils  # noqa: F401  (registers tsgu:: ops)

    dense = torch.randn(1, _M1, p, dtype=value_dtype, device=device)

    def _ours_fwd():
        return torch.ops.tsgu.spmm(vals, rowptr, col, dense, 1, _N1, _M1)

    ours_timing = harness.do_bench(_ours_fwd, device=device)
    print(f"ours (tsgu::spmm) fwd p={p}: median={ours_timing.median_ms:.4f}ms")

    # --- memory: forward under grad + backward (benchmarks.md §1). vals and
    # dense are the differentiable operands (tsgu::spmm's registered
    # autograd); loss = out.sum(). io_bytes = resident inputs (pattern +
    # vals + dense + their differentiable copies) + fwd output + upstream
    # grad + gradient buffers (grad_vals, grad_dense — backward's outputs).
    v = vals.element_size()
    vals_leaf = vals.detach().clone().requires_grad_(True)
    dense_leaf = dense.clone().requires_grad_(True)

    def _grad_fwd():
        return torch.ops.tsgu.spmm(vals_leaf, rowptr, col, dense_leaf, 1, _N1, _M1)

    def _grad_bwd(out):
        # Sum-style loss, with the upstream gradient materialized explicitly:
        # autograd's own sum() backward delivers an expanded stride-0 grad,
        # which the raw kernels' backward primitives read as a contiguous
        # buffer (the landmine _seglse_backward documents) -- ones_like is
        # the same mathematical gradient, contiguous by construction.
        out.backward(torch.ones_like(out))

    sparse_bytes = nse * (col.element_size() + v) + rowptr.numel() * rowptr.element_size()
    out_bytes = _N1 * p * v
    io_bytes = sparse_bytes + 2 * nse * v + 3 * dense.numel() * v + 2 * out_bytes
    mem = None
    if memory.budget_guard(4 * io_bytes, label=f"spmm unbatched p={p} grad memory pass"):
        mem = memory.measure(_grad_fwd, _grad_bwd, io_bytes=io_bytes, bound_bytes=sparse_bytes, device=device)
        print(
            f"ours memory: peak_fwd={mem.peak_fwd_mb}MB peak_bwd={mem.peak_bwd_mb}MB "
            f"workspace={mem.workspace_mb}MB bound_met={mem.workspace_bound_met}"
        )
    # Bytes-moved model (benchmarks.md §1, compulsory traffic — see
    # benchmarks/memory.py): sparse operand read + dense operand read once +
    # output write.
    bytes_moved = sparse_bytes + dense.numel() * v + out_bytes
    bw = memory.bw_pct_peak(bytes_moved, ours_timing.median_ms)
    print(f"achieved bandwidth (compulsory-traffic model): {bw:.1f}% of {memory.PEAK_DRAM_BANDWIDTH_GB_S:.0f} GB/s")

    # Baseline CSR tensor rebuilt only now, after the memory pass (see
    # _make_unbatched_csr's docstring).
    csr = torch.sparse_csr_tensor(rowptr.to(torch.int64), col.to(torch.int64), vals, (_N1, _M1))
    dense2 = dense[0]  # (m, p), for torch.sparse.mm(csr, dense2)

    def _cusparse_fwd():
        return torch.sparse.mm(csr, dense2)

    cusparse_timing = harness.do_bench(_cusparse_fwd, device=device)
    print(f"cuSPARSE (torch.sparse.mm) fwd p={p}: median={cusparse_timing.median_ms:.4f}ms")

    label = "SpMV" if p == 1 else "SpMM"
    _bench_row(
        filename=f"spmm_unbatched_p{p}_vs_cusparse.json",
        baseline_name=f"cuSPARSE cusparse{label} (via torch.sparse.mm)",
        baseline_ms=cusparse_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        bar="≥1.0×",
        variant=f"csr·B=1·f32/i32·p={p}",
        matrix="synth-dlmc90",
        n=_N1,
        m=_M1,
        nse=nse,
        p=p,
        fp=fp,
        mem=mem,
        bw=bw,
    )


def _bench_batched_vs_blockdiag(device, fp) -> None:
    from tests.oracle.sparse_matmul import sparse_mm as oracle_sparse_mm

    value_dtype, index_dtype = torch.float64, torch.int64
    A = _make_batched_csr_tensor(device, value_dtype, index_dtype)
    nse = A.col_indices().numel()
    print(f"batched matrix: {_B2}x{_N2}x{_M2}, nse={nse} (~{nse / _B2}/item), p={_P2}")

    import torchsparsegradutils  # noqa: F401

    dense = torch.randn(_B2, _M2, _P2, dtype=value_dtype, device=device)

    from torchsparsegradutils._batched import BatchedCSR

    csr_desc = BatchedCSR.from_torch(A)

    def _ours_fwd():
        return torch.ops.tsgu.spmm(csr_desc.values, csr_desc.rowptr, csr_desc.col, dense, _B2, _N2, _M2)

    ours_timing = harness.do_bench(_ours_fwd, device=device)
    print(f"ours (tsgu::spmm) fwd: median={ours_timing.median_ms:.4f}ms")

    # Baseline timed before the memory pass so the batched COO scaffolding
    # tensor A (the oracle's input) can be freed and not pollute the §1
    # allocator-peak measurement below.
    def _blockdiag_fwd(A=A, dense=dense):  # bound as defaults so `del A` below fully releases it
        return oracle_sparse_mm(A, dense)

    blockdiag_timing = harness.do_bench(_blockdiag_fwd, device=device)
    print(f"block-diag + cuSPARSE (tests/oracle) fwd: median={blockdiag_timing.median_ms:.4f}ms")
    del A, _blockdiag_fwd

    # --- memory: forward under grad + backward (see the unbatched row's
    # comment for the io_bytes / bound / bytes-moved accounting; f64 here).
    v = csr_desc.values.element_size()
    vals_leaf = csr_desc.values.detach().clone().requires_grad_(True)
    dense_leaf = dense.clone().requires_grad_(True)

    def _grad_fwd():
        return torch.ops.tsgu.spmm(vals_leaf, csr_desc.rowptr, csr_desc.col, dense_leaf, _B2, _N2, _M2)

    def _grad_bwd(out):
        # Sum-style loss, with the upstream gradient materialized explicitly:
        # autograd's own sum() backward delivers an expanded stride-0 grad,
        # which the raw kernels' backward primitives read as a contiguous
        # buffer (the landmine _seglse_backward documents) -- ones_like is
        # the same mathematical gradient, contiguous by construction.
        out.backward(torch.ones_like(out))

    sparse_bytes = nse * (csr_desc.col.element_size() + v) + csr_desc.rowptr.numel() * csr_desc.rowptr.element_size()
    out_bytes = _B2 * _N2 * _P2 * v
    io_bytes = sparse_bytes + 2 * nse * v + 3 * dense.numel() * v + 2 * out_bytes
    mem = None
    if memory.budget_guard(4 * io_bytes, label="spmm batched grad memory pass"):
        mem = memory.measure(_grad_fwd, _grad_bwd, io_bytes=io_bytes, bound_bytes=sparse_bytes, device=device)
        print(
            f"ours memory: peak_fwd={mem.peak_fwd_mb}MB peak_bwd={mem.peak_bwd_mb}MB "
            f"workspace={mem.workspace_mb}MB bound_met={mem.workspace_bound_met}"
        )

    bytes_moved = sparse_bytes + dense.numel() * v + out_bytes
    bw = memory.bw_pct_peak(bytes_moved, ours_timing.median_ms)
    print(f"achieved bandwidth (compulsory-traffic model): {bw:.1f}% of {memory.PEAK_DRAM_BANDWIDTH_GB_S:.0f} GB/s")

    _bench_row(
        filename="spmm_batched_vs_blockdiag.json",
        baseline_name="block-diag + cuSPARSE (tests/oracle/sparse_matmul.py pattern)",
        baseline_ms=blockdiag_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        bar="win",
        variant=f"csr·B={_B2}·f64/i64",
        matrix="synth-ragged",
        n=_N2,
        m=_M2,
        nse=nse,
        p=_P2,
        fp=fp,
        mem=mem,
        bw=bw,
    )


def main() -> None:
    device = torch.device("cuda")
    fp = fingerprint()
    _bench_unbatched_vs_cusparse(device, fp, p=128)
    _bench_unbatched_vs_cusparse(device, fp, p=1)  # SpMV row
    _bench_batched_vs_blockdiag(device, fp)


if __name__ == "__main__":
    main()
