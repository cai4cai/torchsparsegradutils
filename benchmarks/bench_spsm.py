"""Op-level acceptance-bar benchmark for tsgu::spsm (spec/commit.md Phase 3
commit 16; spec/benchmarks.md §3: SpSM baseline = cuSPARSE `cusparseSpSM`
"incl. its analysis cost amortisation" -- "≥ parity cold; win warm (plan
cached on descriptor)").

    uv run python -m benchmarks.bench_spsm

Baseline: ``torch.triangular_solve(B, A, upper=..., unitriangular=...)`` on
a CUDA ``sparse_csr`` tensor is PyTorch's own sparse triangular-solve entry
point (the same one ``tests/oracle/sparse_solve.py``'s legacy implementation
used) and is documented to dispatch to cuSPARSE's `cusparseSpSM` on CUDA CSR
input -- this *is* the vendor primitive kernels.md/benchmarks.md name as the
SpSM baseline. It exposes no separate analysis/solve split at the Python
level (unlike cuSPARSE's own C API, which does), so every call pays
whatever amortisation PyTorch's own internal caching gives it -- the same
number is used for both the cold and warm baseline rows below (this mirrors
benchmarks.md §5's own dummy table: the SpSM cold/warm rows share one
``baseline_ms``).

Row 1 -- COLD: first tsgu::spsm call on a FRESH BatchedCSR (built fresh from
a newly-constructed sparse tensor each timed iteration, so the plan cache
never gets a chance to warm up) vs. cuSPARSE.

Row 2 -- WARM: repeated tsgu::spsm calls reusing the SAME BatchedCSR (and
therefore the SAME cached plan -- architecture.md §3) vs. the same cuSPARSE
baseline. The whole point of this op (kernels.md Family 3: "Analysis-phase
reuse... amortised wins in iterative/rsample loops") is that row 2 beats
row 1 by roughly the analysis cost.

Row 3 -- batched vs. a per-item Python loop over the same cuSPARSE call (no
vendor *batched* SpSM primitive exists, kernels.md) -- the batched/COO win
case this library exists for.
"""

from __future__ import annotations

import torch

from benchmarks import harness, memory
from benchmarks.env import fingerprint
from benchmarks.results import Result, write_result

# --- Row 1/2: unbatched, banded lower-triangular (DLMC-shaped low-degree
# rows -- benchmarks.md §2's synthetic-tier convention, adapted for a
# triangular pattern: bandwidth-limited rather than uniform density, since a
# uniform-density triangular fill at this n would be near-dense). ----------
_N1, _BW1, _P1 = 4096, 8, 8


def _make_banded_csr(n, bw, value_dtype, index_dtype, device, seed=0):
    """Pure-Python row/col construction -- expensive relative to a kernel
    launch (O(n*bw) Python-level appends). Called once per matrix, never
    inside a timed do_bench closure (that would time Python list-building,
    not tsgu::spsm) -- the COLD benchmark below instead ``.clone()``s the
    canonical pattern for a fresh tensor identity each iteration."""
    torch.manual_seed(seed)
    rows, cols = [], []
    for r in range(n):
        lo = max(0, r - bw)
        for c in range(lo, r + 1):
            rows.append(r)
            cols.append(c)
    row_t = torch.tensor(rows, dtype=torch.int64)
    col_t = torch.tensor(cols, dtype=index_dtype, device=device)
    counts = torch.bincount(row_t, minlength=n)
    rowptr = torch.zeros(n + 1, dtype=index_dtype, device=device)
    rowptr[1:] = torch.cumsum(counts, dim=0).to(index_dtype)
    vals = (torch.rand(len(cols), dtype=value_dtype) * 0.1).to(device)
    is_diag = (row_t == torch.tensor(cols)).to(device)
    vals[is_diag] = vals[is_diag].abs() + 4.0
    return rowptr, col_t, vals


def _bench_row(*, op_suffix, baseline_name, baseline_ms, ours_ms, bar, variant, matrix, n, m, nse, p, fp, mem=None):
    speedup = baseline_ms / ours_ms if (baseline_ms and ours_ms) else None
    bar_met = speedup is not None and speedup >= 1.0
    result = Result.build(
        op=f"spsm{op_suffix}",
        family="spsm",
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
    path = write_result(result)
    print(
        f"[{baseline_name}] baseline={baseline_ms:.4f}ms ours={ours_ms:.4f}ms speedup={speedup:.2f}x "
        f"bar={bar} met={bar_met} -> {path}"
    )
    return result


def _bench_cold_vs_warm(device, fp) -> None:
    value_dtype, index_dtype = torch.float32, torch.int32

    import torchsparsegradutils  # noqa: F401  (registers tsgu:: ops)
    from torchsparsegradutils._batched import BatchedCSR

    rowptr0, col0, vals0 = _make_banded_csr(_N1, _BW1, value_dtype, index_dtype, device)
    nse = col0.numel()
    print(f"unbatched banded matrix: n={_N1}, bandwidth={_BW1}, nse={nse}, p={_P1}")

    rhs = torch.randn(1, _N1, _P1, dtype=value_dtype, device=device)

    # --- COLD: a FRESH (rowptr, col, vals) tensor triple every timed
    # iteration, so the plan cache (keyed on tensor identity, plan.h) never
    # gets a chance to warm up -- built via .clone() of the canonical
    # pattern (a fast GPU-side copy with a NEW data pointer/storage) rather
    # than re-running the Python-level pattern generator per iteration,
    # which would time Python list-building instead of tsgu::spsm.
    def _ours_cold():
        rowptr, col, vals = rowptr0.clone(), col0.clone(), vals0.clone()
        return torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, _N1, False, False, False)

    cold_timing = harness.do_bench(_ours_cold, device=device)
    print(f"ours (tsgu::spsm) COLD: median={cold_timing.median_ms:.4f}ms")

    # --- WARM: one BatchedCSR built once, reused (cache hit) every call.
    csr_desc = BatchedCSR(values=vals0, rowptr=rowptr0, col=col0, shape=(1, _N1, _N1))
    # Prime the plan cache once before timing (this call itself is the
    # "cold" one from the plan's perspective -- excluded from the warm
    # median, same as any do_bench warmup).
    torch.ops.tsgu.spsm(csr_desc.values, csr_desc.rowptr, csr_desc.col, rhs, 1, _N1, False, False, False)

    def _ours_warm():
        return torch.ops.tsgu.spsm(csr_desc.values, csr_desc.rowptr, csr_desc.col, rhs, 1, _N1, False, False, False)

    warm_timing = harness.do_bench(_ours_warm, device=device)
    print(f"ours (tsgu::spsm) WARM: median={warm_timing.median_ms:.4f}ms")

    # --- memory (warm): forward under grad + backward (benchmarks.md §1).
    # vals and rhs are the differentiable operands (tsgu::spsm's registered
    # autograd: grad_rhs = transposed re-solve, grad_vals = negated sddmm at
    # A's pattern); loss = out.sum(). The plan cache is primed with the same
    # tensors under no_grad first, so this measures the WARM path like the
    # timing above. io_bytes = inputs + fwd output + upstream grad +
    # gradient buffers (grad_vals, grad_rhs — backward's outputs).
    v = vals0.element_size()
    vals_leaf = vals0.clone().requires_grad_(True)
    rhs_leaf = rhs.clone().requires_grad_(True)
    with torch.no_grad():
        torch.ops.tsgu.spsm(vals_leaf, rowptr0, col0, rhs_leaf, 1, _N1, False, False, False)

    def _grad_fwd():
        return torch.ops.tsgu.spsm(vals_leaf, rowptr0, col0, rhs_leaf, 1, _N1, False, False, False)

    def _grad_bwd(out):
        # Sum-style loss, with the upstream gradient materialized explicitly:
        # autograd's own sum() backward delivers an expanded stride-0 grad,
        # which the raw kernels' backward primitives read as a contiguous
        # buffer (the landmine _seglse_backward documents) -- ones_like is
        # the same mathematical gradient, contiguous by construction.
        out.backward(torch.ones_like(out))

    sparse_bytes = nse * (col0.element_size() + v) + rowptr0.numel() * rowptr0.element_size()
    rhs_bytes = rhs.numel() * v
    io_bytes = sparse_bytes + 2 * nse * v + 5 * rhs_bytes  # +vals_leaf+grad_vals; rhs+rhs_leaf+out+grad_out+grad_rhs
    mem = None
    if memory.budget_guard(4 * io_bytes, label="spsm warm grad memory pass"):
        mem = memory.measure(_grad_fwd, _grad_bwd, io_bytes=io_bytes, bound_bytes=sparse_bytes, device=device)
        print(
            f"ours memory (warm): peak_fwd={mem.peak_fwd_mb}MB peak_bwd={mem.peak_bwd_mb}MB "
            f"workspace={mem.workspace_mb}MB bound_met={mem.workspace_bound_met}"
        )

    A_csr = torch.sparse_csr_tensor(rowptr0, col0, vals0, (_N1, _N1))
    rhs2 = rhs[0]

    def _cusparse_fwd():
        return torch.triangular_solve(rhs2, A_csr, upper=False, unitriangular=False).solution

    cusparse_timing = harness.do_bench(_cusparse_fwd, device=device)
    print(f"cuSPARSE (torch.triangular_solve) fwd: median={cusparse_timing.median_ms:.4f}ms")

    _bench_row(
        op_suffix="_cold",
        baseline_name="cuSPARSE cusparseSpSM (via torch.triangular_solve)",
        baseline_ms=cusparse_timing.median_ms,
        ours_ms=cold_timing.median_ms,
        bar="≥1.0×",
        variant=f"csr·B=1·f32/i32·p={_P1}·cold",
        matrix="synth-banded",
        n=_N1,
        m=_N1,
        nse=nse,
        p=_P1,
        fp=fp,
    )
    _bench_row(
        op_suffix="_warm",
        baseline_name="cuSPARSE cusparseSpSM (via torch.triangular_solve)",
        baseline_ms=cusparse_timing.median_ms,
        ours_ms=warm_timing.median_ms,
        bar="win",
        variant=f"csr·B=1·f32/i32·p={_P1}·warm",
        matrix="synth-banded",
        n=_N1,
        m=_N1,
        nse=nse,
        p=_P1,
        fp=fp,
        mem=mem,
    )


# --- Row 3: batched vs. a per-item cuSPARSE loop (no vendor batched SpSM). -
_B2, _N2, _BW2, _P2 = 8, 1024, 6, 8


def _bench_batched_vs_looped_cusparse(device, fp) -> None:
    import torchsparsegradutils  # noqa: F401
    from torchsparsegradutils._batched import BatchedCSR

    value_dtype, index_dtype = torch.float64, torch.int64
    per_item = [_make_banded_csr(_N2, _BW2, value_dtype, index_dtype, device, seed=b) for b in range(_B2)]
    csrs = [torch.sparse_csr_tensor(rp, c, v, (_N2, _N2)) for rp, c, v in per_item]
    A_batched = torch.stack([csr.to_sparse_coo() for csr in csrs])
    nse = sum(c.numel() for _, c, _ in per_item)
    print(f"batched matrix: B={_B2}, n={_N2}, bandwidth={_BW2}, nse={nse} (~{nse / _B2}/item), p={_P2}")

    rhs = torch.randn(_B2, _N2, _P2, dtype=value_dtype, device=device)
    csr_desc = BatchedCSR.from_torch(A_batched)
    torch.ops.tsgu.spsm(csr_desc.values, csr_desc.rowptr, csr_desc.col, rhs, _B2, _N2, False, False, False)

    def _ours_fwd():
        return torch.ops.tsgu.spsm(csr_desc.values, csr_desc.rowptr, csr_desc.col, rhs, _B2, _N2, False, False, False)

    ours_timing = harness.do_bench(_ours_fwd, device=device)
    print(f"ours (tsgu::spsm) batched fwd: median={ours_timing.median_ms:.4f}ms")

    # Baseline timed before the memory pass so the per-item CSR / batched
    # COO scaffolding can be freed and not pollute the §1 allocator-peak
    # measurement below.
    def _looped_cusparse_fwd(csrs=csrs, rhs=rhs):  # bound as defaults so `del csrs` below fully releases them
        return torch.stack([torch.triangular_solve(rhs[b], csrs[b], upper=False).solution for b in range(_B2)])

    looped_timing = harness.do_bench(_looped_cusparse_fwd, device=device)
    print(f"looped cuSPARSE (per-item torch.triangular_solve) fwd: median={looped_timing.median_ms:.4f}ms")
    del per_item, csrs, A_batched, _looped_cusparse_fwd

    # --- memory: forward under grad + backward (see the warm row's comment
    # for the io_bytes / bound accounting; f64 here, plan primed above).
    v = csr_desc.values.element_size()
    vals_leaf = csr_desc.values.detach().clone().requires_grad_(True)
    rhs_leaf = rhs.clone().requires_grad_(True)
    with torch.no_grad():
        torch.ops.tsgu.spsm(vals_leaf, csr_desc.rowptr, csr_desc.col, rhs_leaf, _B2, _N2, False, False, False)

    def _grad_fwd():
        return torch.ops.tsgu.spsm(vals_leaf, csr_desc.rowptr, csr_desc.col, rhs_leaf, _B2, _N2, False, False, False)

    def _grad_bwd(out):
        # Sum-style loss, with the upstream gradient materialized explicitly:
        # autograd's own sum() backward delivers an expanded stride-0 grad,
        # which the raw kernels' backward primitives read as a contiguous
        # buffer (the landmine _seglse_backward documents) -- ones_like is
        # the same mathematical gradient, contiguous by construction.
        out.backward(torch.ones_like(out))

    sparse_bytes = nse * (csr_desc.col.element_size() + v) + csr_desc.rowptr.numel() * csr_desc.rowptr.element_size()
    rhs_bytes = rhs.numel() * v
    io_bytes = sparse_bytes + 2 * nse * v + 5 * rhs_bytes  # +vals_leaf+grad_vals; rhs+rhs_leaf+out+grad_out+grad_rhs
    mem = None
    if memory.budget_guard(4 * io_bytes, label="spsm batched grad memory pass"):
        mem = memory.measure(_grad_fwd, _grad_bwd, io_bytes=io_bytes, bound_bytes=sparse_bytes, device=device)
        print(
            f"ours memory: peak_fwd={mem.peak_fwd_mb}MB peak_bwd={mem.peak_bwd_mb}MB "
            f"workspace={mem.workspace_mb}MB bound_met={mem.workspace_bound_met}"
        )

    _bench_row(
        op_suffix="_batched",
        baseline_name="looped cuSPARSE cusparseSpSM (per-item torch.triangular_solve)",
        baseline_ms=looped_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        bar="win",
        variant=f"csr·B={_B2}·f64/i64·p={_P2}",
        matrix="synth-banded",
        n=_N2,
        m=_N2,
        nse=nse,
        p=_P2,
        fp=fp,
        mem=mem,
    )


def main() -> None:
    device = torch.device("cuda")
    fp = fingerprint()
    _bench_cold_vs_warm(device, fp)
    _bench_batched_vs_looped_cusparse(device, fp)


if __name__ == "__main__":
    main()
