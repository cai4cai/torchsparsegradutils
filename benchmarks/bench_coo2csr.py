"""Op-level acceptance-bar benchmark for tsgu::coo2csr (spec/commit.md Phase
3 commit 19; map.md §3: ``convert_coo_to_csr_indices_values`` is 🟠 VENDOR,
"Native sort+compress kernel, baseline cuSPARSE Xcoo2csr").

    uv run python -m benchmarks.bench_coo2csr

Runs benchmarks.md §1's protocol (do_bench CUDA-event windowing, memory
reset/peak/workspace) over an nse x B x index-dtype sweep and writes one
JSON per row under benchmarks/results/ (backend="custom").

Baseline recorded here: the **pure-torch fallback path** this kernel
replaces — a two-pass stable ``torch.argsort`` (secondary key ``col``, then
primary key ``row_global``; each argsort is itself a CUB-backed device
radix sort) followed by a bincount+cumsum compress. That composition is
exactly what ``convert_coo_to_csr_indices_values`` / ``BatchedCSR``'s COO
paths execute when the kernel is absent, so the row's speedup is the
end-user-visible win of commit 19's internal switch. (The raw cuSPARSE
``Xcoo2csr`` primitive is not reachable from Python without a bespoke
binding — it only does the compress step, not the sort — so the vendor
head-to-head lives in the kernel lane's NVBench target, not here.)

VRAM budget (benchmarks/corpus.peak_budget_gb — this machine has a 4 GB
card): configurations whose estimated peak exceeds the budget are skipped
WITH a printed note (benchmarks.md: no silent caps).
"""

from __future__ import annotations

import torch

from benchmarks import harness, memory
from benchmarks.corpus import dtype_short, peak_budget_gb
from benchmarks.env import fingerprint
from benchmarks.results import Result, write_result

# Sweep axes (this commit's T3/T5 instructions).
SWEEP_NSE = (100_000, 1_000_000, 5_000_000)
SWEEP_B = (1, 16)
INDEX_DTYPES = (torch.int32, torch.int64)

# Fixed logical matrix shape per item; nse sweeps independently (duplicate
# coordinates may occur at the highest densities — the op's contract keeps
# them, no dedup, so they are a legitimate sort workload).
_N = _M = 4096
_SEED = 20260717


def _estimate_peak_bytes(nse: int, B: int, index_dtype: torch.dtype) -> int:
    """Rough peak-bytes model: 3 input coordinate arrays + 3 outputs
    (rowptr, col_sorted, permutation) + the baseline's argsort temporaries
    (each stable argsort materialises keys + an int64 permutation). A
    documented fudge factor stands in for allocator overhead, like
    benchmarks/corpus.estimate_peak_bytes does for the sparse ops."""
    idx = torch.tensor([], dtype=index_dtype).element_size()
    coords = 3 * nse * idx
    outputs = (B * _N + 1) * idx + 2 * nse * idx
    sort_scratch = 4 * nse * 8  # int64 row_global/keys + two argsort permutations
    return int(2.0 * (coords + outputs + sort_scratch))


def _make_inputs(nse: int, B: int, index_dtype: torch.dtype, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(_SEED)
    batch = torch.randint(0, B, (nse,), generator=gen).to(index_dtype).to(device)
    row = torch.randint(0, _N, (nse,), generator=gen).to(index_dtype).to(device)
    col = torch.randint(0, _M, (nse,), generator=gen).to(index_dtype).to(device)
    return batch, row, col


def _pure_torch_coo2csr(batch: torch.Tensor, row: torch.Tensor, col: torch.Tensor, B: int, n: int):
    """The pure-torch fallback composition (see module docstring) — the
    identical ordering contract as tsgu::coo2csr: stable (row_global, col)
    lexicographic sort + compress, no dedup."""
    row_global = batch.long() * n + row.long()
    permutation_secondary = torch.argsort(col, stable=True)
    permutation = permutation_secondary[torch.argsort(row_global[permutation_secondary], stable=True)]
    counts = torch.bincount(row_global.index_select(0, permutation), minlength=B * n)
    rowptr = torch.zeros(B * n + 1, dtype=row.dtype, device=row.device)
    rowptr[1:] = counts.cumsum(0).to(row.dtype)
    return rowptr, col.index_select(0, permutation), permutation


def _bench_config(nse: int, B: int, index_dtype: torch.dtype, device: torch.device, fp) -> None:
    budget_bytes = peak_budget_gb() * 1024**3
    estimate = _estimate_peak_bytes(nse, B, index_dtype)
    if estimate > budget_bytes:
        # benchmarks.md: no silent caps -- over-budget configs are skipped
        # loudly, never shrunk.
        print(
            f"[skip] coo2csr nse={nse} B={B} {dtype_short(index_dtype)}: estimated peak "
            f"{estimate / 1024**3:.2f} GB exceeds VRAM budget {budget_bytes / 1024**3:.2f} GB "
            "(benchmarks/corpus.peak_budget_gb)"
        )
        return

    batch, row, col = _make_inputs(nse, B, index_dtype, device)

    def _ours():
        return torch.ops.tsgu.coo2csr(batch, row, col, B, _N)

    def _baseline():
        return _pure_torch_coo2csr(batch, row, col, B, _N)

    ours_timing = harness.do_bench(_ours, device=device)
    baseline_timing = harness.do_bench(_baseline, device=device)

    idx_bytes = batch.element_size()
    io_bytes = 3 * nse * idx_bytes + (B * _N + 1) * idx_bytes + 2 * nse * idx_bytes  # inputs + 3 outputs
    # Index-only op: no backward pass exists (the op produces integer index
    # structures, nothing differentiable), so backward_fn stays None and
    # peak_bwd is legitimately absent (None) in the JSON — benchmarks.md
    # §5's "— means not applicable". The O(nse) workspace bound (sort
    # scratch is O(nse) too) is still asserted, against the three input
    # coordinate arrays + rowptr.
    bound_bytes = 3 * nse * idx_bytes + (B * _N + 1) * idx_bytes
    mem = memory.measure(_ours, io_bytes=io_bytes, bound_bytes=bound_bytes, device=device)
    print(
        f"ours memory: peak_fwd={mem.peak_fwd_mb}MB peak_bwd={mem.peak_bwd_mb} (no backward: index-only op) "
        f"workspace={mem.workspace_mb}MB bound_met={mem.workspace_bound_met}"
    )

    speedup = baseline_timing.median_ms / ours_timing.median_ms if ours_timing.median_ms else None
    bar = "win"  # replace the pure-torch composition it routes around (benchmarks.md §3 batched/composed rule)
    bar_met = speedup is not None and speedup >= 1.0
    result = Result.build(
        op="coo2csr",
        family="convert",
        backend="custom",
        variant=f"coo·B={B}·{dtype_short(index_dtype)}·nse={nse}",
        matrix="synth-uniform",
        n=_N,
        m=_M,
        nse=nse,
        p=None,
        fp=fp,
        baseline_name="pure-torch two-pass stable argsort + compress (the fallback path)",
        baseline_ms=baseline_timing.median_ms,
        ours_ms=ours_timing.median_ms,
        speedup=speedup,
        peak_fwd_mb=mem.peak_fwd_mb,
        peak_bwd_mb=mem.peak_bwd_mb,  # None: index-only op, no backward (see comment above)
        workspace_mb=mem.workspace_mb,
        bar=bar,
        bar_met=bar_met,
        meta={
            "workspace_fwd_mb": mem.workspace_fwd_mb,
            "workspace_bwd_mb": mem.workspace_bwd_mb,
            "workspace_bound_met": mem.workspace_bound_met,
        },
    )
    path = write_result(result, filename=f"coo2csr_B{B}_{dtype_short(index_dtype)}_nse{nse}.json")
    print(
        f"[coo2csr nse={nse} B={B} {dtype_short(index_dtype)}] baseline={baseline_timing.median_ms:.4f}ms "
        f"ours={ours_timing.median_ms:.4f}ms speedup={speedup:.2f}x bar={bar} met={bar_met} -> {path}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        print(
            "[abort] tsgu::coo2csr is CUDA-only (architecture.md §4) -- no CUDA device "
            "visible, so there is no 'ours' to measure. No rows written."
        )
        return

    import torchsparsegradutils  # noqa: F401  (registers tsgu:: ops)
    from torchsparsegradutils._dispatch import backend_available

    if not (backend_available() and torch._C._dispatch_has_kernel_for_dispatch_key("tsgu::coo2csr", "CUDA")):
        print(
            "[abort] tsgu::coo2csr has no CUDA kernel registered (backend absent, version-mismatched, "
            "or the cuda/ tree predates commit 19's convert kernel). No rows written."
        )
        return

    device = torch.device("cuda")
    fp = fingerprint()
    for index_dtype in INDEX_DTYPES:
        for B in SWEEP_B:
            for nse in SWEEP_NSE:
                _bench_config(nse, B, index_dtype, device, fp)


if __name__ == "__main__":
    main()
