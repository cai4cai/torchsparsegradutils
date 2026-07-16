"""Op-level acceptance-bar benchmark for tsgu::sddmm (spec/commit.md Phase 3
commit 14; spec/benchmarks.md §3: SDDMM baselines = cuSPARSE `cusparseSDDMM`
unbatched, and the current pure-PyTorch chain (git history) as secondary --
"no vendor batched baseline exists -- report absolute + vs block-diag").

    uv run python -m benchmarks.bench_sddmm

Runs benchmarks.md §1's protocol (do_bench CUDA-event windowing, memory
reset/peak/workspace) for two representative configurations on the synthetic
tier (benchmarks.md §2: migration-period rule, synthetic-only) and writes one
JSON per baseline row under benchmarks/results/ (backend="custom").

Row 1 -- unbatched CSR vs cuSPARSE SDDMM: torch.sparse.sampled_addmm is
documented (torch.sparse docs) to dispatch to cusparseSDDMM for CUDA CSR
sparse `input` + dense `mat1`/`mat2` -- this *is* the vendor primitive
kernels.md/benchmarks.md name as the SDDMM baseline, reached through
PyTorch's own public API rather than a hand-rolled cuSPARSE handle. Verified
numerically equal to tsgu::sddmm at negate=False during this commit's
development (both compute dot(g[row,:], mat[col,:]) at the same CSR
pattern -- sampled_addmm's mat2 is (k, n), i.e. our `mat`, transposed).

Row 2 -- batched vs the old pure-PyTorch chain: no vendor batched SDDMM
primitive exists (kernels.md), so the comparison is the pre-rewrite
block-diag/index_select pattern from tests/oracle/sparse_matmul.py's
`SparseMatMul.backward` gradA computation, reimplemented here directly at
the folded-row level (`_oracle_index_select_sddmm`) -- algorithmically
identical to what block-diagonalising A and running the oracle's backward
would do (uncompress rowptr -> row_indices, index_select g/mat, row-dot),
without paying to actually materialise a block-diagonal sparse tensor just
to time it.
"""

from __future__ import annotations

import torch

from benchmarks import harness, memory
from benchmarks.env import fingerprint
from benchmarks.results import Result, write_result

# --- Row 1: unbatched CSR, DLMC-shaped sparsity (benchmarks.md §2) ----------
_N1, _M1, _P1 = 4096, 4096, 128
_DENSITY1 = 0.10  # ~90% sparse, DLMC transformer regime (matches bench_seglse.py)

# --- Row 2: batched, synthetic-ragged workhorse (benchmarks.md §2) ---------
_B2, _N2, _M2, _P2 = 8, 1024, 1024, 32
_DENSITY2 = 0.005  # benchmarks.md §2 REFERENCE_DENSITY


def _make_unbatched_csr(device, value_dtype, index_dtype, seed=0):
    torch.manual_seed(seed)
    dense = torch.randn(_N1, _M1, dtype=value_dtype, device=device)
    mask = torch.rand(_N1, _M1, device=device) < _DENSITY1
    csr = (dense * mask).to_sparse_csr()
    rowptr = csr.crow_indices().to(index_dtype)
    col = csr.col_indices().to(index_dtype)
    return rowptr, col


def _make_batched_csr(device, value_dtype, index_dtype, seed=1):
    """Ragged-free (equal nse/item) batched CSR, folded per naming.md §2 --
    B independent (_N2, _M2) items at the same density, concatenated into one
    (B * _N2 + 1,) rowptr with column indices kept local per naming.md's
    BatchedCSR contract."""
    torch.manual_seed(seed)
    rowptrs = []
    cols = []
    offset = 0
    for b in range(_B2):
        dense = torch.randn(_N2, _M2, dtype=value_dtype, device=device)
        mask = torch.rand(_N2, _M2, device=device) < _DENSITY2
        csr = (dense * mask).to_sparse_csr()
        crow = csr.crow_indices().to(index_dtype)
        col = csr.col_indices().to(index_dtype)
        rowptrs.append(crow[:-1] + offset)
        cols.append(col)
        offset += int(col.shape[0])
    rowptrs.append(torch.tensor([offset], dtype=index_dtype, device=device))
    rowptr = torch.cat(rowptrs)
    col = torch.cat(cols)
    return rowptr, col


def _oracle_index_select_sddmm(rowptr, col, g, mat, B, n, m, negate):
    """The pre-rewrite pure-PyTorch chain (tests/oracle/sparse_matmul.py
    `SparseMatMul.backward`'s gradA computation), applied directly at the
    folded-row level -- what running that oracle on a block-diagonalised A
    computes, without materialising the block-diagonal sparse tensor."""
    row_g = torch.repeat_interleave(
        torch.arange(B * n, device=rowptr.device, dtype=torch.int64), (rowptr[1:] - rowptr[:-1]).long()
    )
    batch = row_g // n
    col_global = batch * m + col.long()
    g_flat = g.reshape(B * n, -1)
    mat_flat = mat.reshape(B * m, -1)
    grad_select = g_flat.index_select(0, row_g)
    mat_select = mat_flat.index_select(0, col_global)
    out = (grad_select * mat_select).sum(dim=1)
    return -out if negate else out


def _bench_row(*, baseline_name, baseline_ms, ours_ms, bar, variant, matrix, n, m, nse, p, fp, mem=None):
    speedup = baseline_ms / ours_ms if (baseline_ms and ours_ms) else None
    bar_met = speedup is not None and speedup >= 1.0
    result = Result.build(
        op="sddmm",
        family="sddmm",
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
        workspace_mb=mem.workspace_mb if mem else None,
        bar=bar,
        bar_met=bar_met,
    )
    path = write_result(result)
    print(
        f"[{baseline_name}] baseline={baseline_ms:.4f}ms ours={ours_ms:.4f}ms speedup={speedup:.2f}x "
        f"bar={bar} met={bar_met} -> {path}"
    )
    return result


def _bench_unbatched_vs_cusparse(device, fp) -> None:
    value_dtype, index_dtype = torch.float32, torch.int32
    rowptr, col = _make_unbatched_csr(device, value_dtype, index_dtype)
    nse = col.numel()
    print(f"row1 matrix: dense {_N1}x{_M1}, nse={nse} ({nse / (_N1 * _M1):.1%} density), p={_P1}")

    import torchsparsegradutils  # noqa: F401  (registers tsgu:: ops)

    g = torch.randn(1, _N1, _P1, dtype=value_dtype, device=device)
    mat = torch.randn(1, _M1, _P1, dtype=value_dtype, device=device)

    def _ours_fwd():
        return torch.ops.tsgu.sddmm(rowptr, col, g, mat, 1, _N1, _M1, False)

    ours_timing = harness.do_bench(_ours_fwd, device=device)
    print(f"ours (tsgu::sddmm) fwd: median={ours_timing.median_ms:.4f}ms")

    io_bytes = nse * (col.element_size() + 4) + rowptr.numel() * rowptr.element_size() + g.numel() * 4 + mat.numel() * 4
    mem = memory.measure(_ours_fwd, io_bytes=io_bytes, device=device)
    print(f"ours memory: peak_fwd={mem.peak_fwd_mb}MB workspace={mem.workspace_mb}MB")

    # --- baseline: cuSPARSE cusparseSDDMM, reached via torch.sparse.sampled_addmm ---
    input_csr = torch.sparse_csr_tensor(
        rowptr.to(torch.int64), col.to(torch.int64), torch.zeros(nse, dtype=value_dtype, device=device), (_N1, _M1)
    )
    mat2 = mat[0].t().contiguous()  # sampled_addmm wants mat2 (k, n) = mat^T

    def _cusparse_fwd():
        return torch.sparse.sampled_addmm(input_csr, g[0], mat2, beta=0.0, alpha=1.0)

    cusparse_timing = harness.do_bench(_cusparse_fwd, device=device)
    print(f"cuSPARSE (torch.sparse.sampled_addmm) fwd: median={cusparse_timing.median_ms:.4f}ms")

    _bench_row(
        baseline_name="cuSPARSE cusparseSDDMM (via torch.sparse.sampled_addmm)",
        baseline_ms=cusparse_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        bar="≥1.0×",
        variant="csr·B=1·f32/i32",
        matrix="synth-dlmc90",
        n=_N1,
        m=_M1,
        nse=nse,
        p=_P1,
        fp=fp,
        mem=mem,
    )


def _bench_batched_vs_oracle_chain(device, fp) -> None:
    value_dtype, index_dtype = torch.float64, torch.int64
    rowptr, col = _make_batched_csr(device, value_dtype, index_dtype)
    nse = col.numel()
    print(f"row2 matrix: {_B2}x{_N2}x{_M2} batched, nse={nse} (~{nse / _B2} / item), p={_P2}")

    import torchsparsegradutils  # noqa: F401

    g = torch.randn(_B2, _N2, _P2, dtype=value_dtype, device=device)
    mat = torch.randn(_B2, _M2, _P2, dtype=value_dtype, device=device)

    def _ours_fwd():
        return torch.ops.tsgu.sddmm(rowptr, col, g, mat, _B2, _N2, _M2, False)

    ours_timing = harness.do_bench(_ours_fwd, device=device)
    print(f"ours (tsgu::sddmm) fwd: median={ours_timing.median_ms:.4f}ms")

    def _oracle_fwd():
        return _oracle_index_select_sddmm(rowptr, col, g, mat, _B2, _N2, _M2, False)

    oracle_timing = harness.do_bench(_oracle_fwd, device=device)
    print(f"pure-PyTorch chain (block-diag equivalent) fwd: median={oracle_timing.median_ms:.4f}ms")

    io_bytes = nse * (col.element_size() + 8) + rowptr.numel() * rowptr.element_size() + g.numel() * 8 + mat.numel() * 8
    mem = memory.measure(_ours_fwd, io_bytes=io_bytes, device=device)
    print(f"ours memory: peak_fwd={mem.peak_fwd_mb}MB workspace={mem.workspace_mb}MB")

    _bench_row(
        baseline_name="pure-PyTorch index_select chain (tests/oracle/sparse_matmul.py pattern, block-diag equivalent)",
        baseline_ms=oracle_timing.median_ms,
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
    )


def main() -> None:
    device = torch.device("cuda")
    fp = fingerprint()
    _bench_unbatched_vs_cusparse(device, fp)
    _bench_batched_vs_oracle_chain(device, fp)


if __name__ == "__main__":
    main()
