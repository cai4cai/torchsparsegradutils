"""Op-level acceptance-bar benchmark for tsgu::grouped_gemm (spec/commit.md
Phase 3 commit 18; spec/benchmarks.md §3: Grouped GEMM baselines = cuBLAS
``cublasGemmGroupedBatched``, "DGL segment_mm if installable", rule
">= parity").

    uv run python -m benchmarks.bench_grouped_gemm

Runs benchmarks.md §1's protocol (do_bench CUDA-event windowing, memory
reset/peak/workspace) over an (N, D, R) sweep for both public wrappers,
forward and backward, and writes one JSON per row under benchmarks/results/
(backend="custom").

Baselines recorded here (the rewrite is dependency-light -- map.md -- so
neither DGL nor a raw cuBLAS grouped-batched binding is a dependency we can
call from Python; the two recorded baselines are the strongest available
in-repo comparisons):

- **legacy nested-tensor implementation** (tests/oracle/indexed_matmul.py --
  the exact pre-rewrite ``segment_mm``/``gather_mm`` bodies this commit
  deletes from the shipped package), and
- **per-segment torch.mm loop** (one cuBLAS GEMM per group -- the naive
  vendor-composed counterfactual; each ``torch.mm`` IS a cuBLAS call, so
  beating this loop is beating "R separate cuBLAS launches").

TF32 is disabled for every row (benchmarks.md §1: "TF32 off for every
parity-relevant number (a labelled TF32-on column is allowed for grouped
GEMM only)" -- the optional TF32-on column is future work, not recorded
here).

VRAM budget (benchmarks/corpus.peak_budget_gb -- this machine has a 4 GB
card): configurations whose estimated peak exceeds the budget are skipped
WITH a printed note (benchmarks.md: no silent caps).

Set TSGU_BENCH_GROUPED_GEMM_QUICK=1 to run only the smallest configuration
per op (a plumbing smoke run); the full sweep is the deliverable run.
"""

from __future__ import annotations

import os

import torch

from benchmarks import harness, memory
from benchmarks.corpus import peak_budget_gb
from benchmarks.env import fingerprint
from benchmarks.results import Result, write_result
from tests.oracle.indexed_matmul import gather_mm as legacy_gather_mm
from tests.oracle.indexed_matmul import segment_mm as legacy_segment_mm
from torchsparsegradutils import gather_mm, segment_mm

# (N, D, R) sweep -- this commit's T3/T5 instructions. D1 = D2 = D.
SWEEP_N = (4096, 16384, 65536)
SWEEP_D = (64, 128, 256)
SWEEP_R = (4, 16, 64)

_VALUE_DTYPE = torch.float32
_INDEX_DTYPE = torch.int64
_SEED = 20260717


def _estimate_peak_bytes(N: int, D: int, R: int, *, backward: bool) -> int:
    """Rough peak-bytes model for one (op, config) measurement, sized to the
    *worst* participant: the legacy nested-tensor baseline copies a and the
    output several times (as_nested_tensor + cat), and backward adds
    grad_a/grad_b/gout plus the argsort permutation. A documented fudge
    factor stands in for allocator overhead, like
    benchmarks/corpus.estimate_peak_bytes does for the sparse ops."""
    v = torch.tensor([], dtype=_VALUE_DTYPE).element_size()
    idx_bytes = torch.tensor([], dtype=_INDEX_DTYPE).element_size()
    base = N * D * v * 2 + R * D * D * v + N * idx_bytes  # a + out + b + idx
    fudge = 6.0 if backward else 4.0
    return int(fudge * base)


def _make_inputs(N: int, D: int, R: int, op_name: str, device: torch.device):
    torch.manual_seed(_SEED)
    a = torch.randn(N, D, dtype=_VALUE_DTYPE, device=device)
    b = torch.randn(R, D, D, dtype=_VALUE_DTYPE, device=device)
    if op_name == "segment_mm":
        # Uniform segments (N divisible by every swept R).
        third = torch.full((R,), N // R, dtype=_INDEX_DTYPE, device=device)
    else:
        third = torch.randint(0, R, (N,), dtype=_INDEX_DTYPE, device=device)
    return a, b, third


def _loop_segment_mm(a: torch.Tensor, b: torch.Tensor, seglen_a: torch.Tensor) -> torch.Tensor:
    """Per-segment torch.mm loop baseline: one cuBLAS GEMM per group."""
    parts = []
    off = 0
    for i, length in enumerate(seglen_a.tolist()):
        parts.append(torch.mm(a[off : off + length], b[i]))
        off += length
    return torch.cat(parts, dim=0)


def _loop_gather_mm(a: torch.Tensor, b: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    """Per-group torch.mm loop baseline: rows regrouped by boolean mask,
    one cuBLAS GEMM per group, results scattered back to input order."""
    out = torch.zeros(a.shape[0], b.shape[-1], dtype=a.dtype, device=a.device)
    for k in range(b.shape[0]):
        mask = idx_b == k
        out[mask] = torch.mm(a[mask], b[k])
    return out


_OURS = {"segment_mm": segment_mm, "gather_mm": gather_mm}
_BASELINES = {
    "segment_mm": {
        "legacy nested-tensor (tests/oracle/indexed_matmul.py)": legacy_segment_mm,
        "per-segment torch.mm loop (one cuBLAS GEMM per group)": _loop_segment_mm,
    },
    "gather_mm": {
        "legacy nested-tensor (tests/oracle/indexed_matmul.py)": legacy_gather_mm,
        "per-segment torch.mm loop (one cuBLAS GEMM per group)": _loop_gather_mm,
    },
}


def _time_fwd_bwd(fn, a, b, third, device):
    """Median forward ms and median backward ms for out = fn(a, b, third)."""
    fwd_timing = harness.do_bench(lambda: fn(a, b, third), device=device)

    a_g = a.detach().clone().requires_grad_(True)
    b_g = b.detach().clone().requires_grad_(True)
    out = fn(a_g, b_g, third)
    gout = torch.randn_like(out)
    bwd_timing = harness.do_bench(lambda: torch.autograd.grad(out, (a_g, b_g), gout, retain_graph=True), device=device)
    return fwd_timing.median_ms, bwd_timing.median_ms


def _write_row(*, op_name, direction, baseline_name, baseline_ms, ours_ms, N, D, R, fp, mem=None):
    speedup = baseline_ms / ours_ms if (baseline_ms and ours_ms) else None
    bar = "≥1.0×"  # benchmarks.md §3 grouped GEMM rule: ">= parity"
    bar_met = speedup is not None and speedup >= 1.0
    result = Result.build(
        op="grouped_gemm",
        family="grouped_gemm",
        backend="custom",
        variant=f"{op_name}·{direction}·N={N}·D={D}·R={R}·f32/i64·TF32-off",
        matrix="synth-uniform-seg" if op_name == "segment_mm" else "synth-random-idx",
        n=N,
        m=D,
        nse=N,  # dense op: one "entry" (row) per output row; no sparse nse exists
        p=D,
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
    )
    path = write_result(result)
    print(
        f"[{op_name} {direction} N={N} D={D} R={R}] vs {baseline_name}: "
        f"baseline={baseline_ms:.4f}ms ours={ours_ms:.4f}ms speedup={speedup:.2f}x met={bar_met} -> {path}"
    )


def _bench_config(op_name: str, N: int, D: int, R: int, device: torch.device, fp) -> None:
    budget_bytes = peak_budget_gb() * 1024**3
    estimate = _estimate_peak_bytes(N, D, R, backward=True)
    if estimate > budget_bytes:
        # benchmarks.md: no silent caps -- over-budget configs are skipped
        # loudly, never shrunk.
        print(
            f"[skip] {op_name} N={N} D={D} R={R}: estimated peak "
            f"{estimate / 1024**3:.2f} GB exceeds VRAM budget {budget_bytes / 1024**3:.2f} GB "
            "(benchmarks/corpus.peak_budget_gb)"
        )
        return

    a, b, third = _make_inputs(N, D, R, op_name, device)
    ours_fn = _OURS[op_name]

    ours_fwd_ms, ours_bwd_ms = _time_fwd_bwd(ours_fn, a, b, third, device)
    print(f"{op_name} N={N} D={D} R={R}: ours fwd={ours_fwd_ms:.4f}ms bwd={ours_bwd_ms:.4f}ms")

    # Memory (benchmarks.md §1: measured alongside time, always) -- attached
    # to the ours-vs-legacy rows below.
    io_bytes = (
        a.numel() * a.element_size()
        + b.numel() * b.element_size()
        + third.numel() * third.element_size()
        + a.shape[0] * b.shape[-1] * a.element_size()  # output
    )
    a_m = a.detach().clone().requires_grad_(True)
    b_m = b.detach().clone().requires_grad_(True)
    mem = memory.measure(
        lambda: ours_fn(a_m, b_m, third),
        lambda out: out.backward(torch.ones_like(out)),
        io_bytes=io_bytes,
        device=device,
    )
    print(f"  ours memory: peak_fwd={mem.peak_fwd_mb}MB peak_bwd={mem.peak_bwd_mb}MB workspace={mem.workspace_mb}MB")

    for baseline_name, baseline_fn in _BASELINES[op_name].items():
        base_fwd_ms, base_bwd_ms = _time_fwd_bwd(baseline_fn, a, b, third, device)
        is_legacy = "legacy" in baseline_name
        _write_row(
            op_name=op_name,
            direction="fwd",
            baseline_name=baseline_name,
            baseline_ms=base_fwd_ms,
            ours_ms=ours_fwd_ms,
            N=N,
            D=D,
            R=R,
            fp=fp,
            mem=mem if is_legacy else None,
        )
        _write_row(
            op_name=op_name,
            direction="bwd",
            baseline_name=baseline_name,
            baseline_ms=base_bwd_ms,
            ours_ms=ours_bwd_ms,
            N=N,
            D=D,
            R=R,
            fp=fp,
        )


def main() -> None:
    if not torch.cuda.is_available():
        print(
            "[abort] tsgu::grouped_gemm is CUDA-only (architecture.md §4) -- no CUDA device "
            "visible, so there is no 'ours' to measure. No rows written."
        )
        return

    # benchmarks.md §1: TF32 off for every parity-relevant number.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda")
    fp = fingerprint()
    quick = os.getenv("TSGU_BENCH_GROUPED_GEMM_QUICK", "").strip().lower() in {"1", "true", "yes", "on"}

    for op_name in ("segment_mm", "gather_mm"):
        for N in SWEEP_N:
            for D in SWEEP_D:
                for R in SWEEP_R:
                    _bench_config(op_name, N, D, R, device, fp)
                    if quick:
                        break
                if quick:
                    break
            if quick:
                break


if __name__ == "__main__":
    main()
