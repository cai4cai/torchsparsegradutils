"""benchmarks/harness.py — timing protocol.

spec/benchmarks.md §1 (non-negotiable, applies to every real GPU number):
CUDA-event do_bench-style windowing, >= 25 warmup iterations, then measure
>= 100 iterations or 2000 ms (whichever is longer), report median + p10/p90.
L2 flush between iterations (sparse kernels are memory-bound; cache-hot
numbers are fiction). Small-shape guard: back-to-back launch batching so the
end event isn't CPU-bound (the do_bench small-kernel trap).

**Degraded mode:** with no CUDA device visible to torch (this machine's GPU
is currently down — spec/commit.md #11 worker context), :func:`do_bench`
falls back to a plain ``time.perf_counter`` loop. This is explicitly *not*
benchmarks.md §1's protocol (no CUDA events, no L2 flush — there's no
addressable L2 to flush the same way from Python on CPU) — it exists purely
so ``benchmarks.run`` can execute end-to-end and prove the harness ->
memory -> fingerprint -> JSON pipeline today. Every :class:`TimingResult`
carries ``degraded`` so callers/writers never present a CPU wall-clock
number as if it were a real §1 measurement. The moment
``torch.cuda.is_available()`` is ``True`` again, the real protocol runs with
no code change.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

# --- real (CUDA) protocol constants, benchmarks.md §1 -----------------------
MIN_WARMUP_ITERS = 25
MIN_ITERS = 100
MIN_WINDOW_MS = 2000.0
# Larger than any consumer/datacenter GPU's L2 as of 2026; zeroed every
# iteration to defeat cache-hot timing (benchmarks.md §1).
L2_FLUSH_BYTES = 64 * 1024 * 1024
# Small-shape guard: number of launches batched between one pair of timing
# events, so the measured interval isn't dominated by CPU launch overhead.
LAUNCH_BATCH = 8

# --- degraded (CPU, no CUDA) protocol constants -----------------------------
# Deliberately much lighter than the real protocol above — this is a
# plumbing proof, not a §1-compliant measurement (see module docstring).
_DEGRADED_WARMUP_ITERS = 10
_DEGRADED_MIN_ITERS = 30


@dataclass(frozen=True)
class TimingResult:
    median_ms: float
    p10_ms: float
    p90_ms: float
    n_iters: int
    degraded: bool  # True: CPU perf_counter fallback, not benchmarks.md §1's protocol.


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _make_l2_flush_buffer(device: torch.device) -> Optional[torch.Tensor]:
    if device.type != "cuda":
        return None
    return torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def do_bench(fn: Callable[[], object], *, device: Optional[torch.device] = None) -> TimingResult:
    """Time ``fn()`` (called with no arguments) per benchmarks.md §1's
    protocol; degrades to a perf_counter loop with no CUDA (see module
    docstring). ``device`` defaults to CUDA if available, else CPU."""
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if device.type == "cuda":
        return _do_bench_cuda(fn, device)
    return _do_bench_cpu_degraded(fn)


def _do_bench_cuda(fn: Callable[[], object], device: torch.device) -> TimingResult:
    flush_buf = _make_l2_flush_buffer(device)

    for _ in range(MIN_WARMUP_ITERS):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    t0 = time.perf_counter()

    while len(times_ms) < MIN_ITERS or (time.perf_counter() - t0) * 1000.0 < MIN_WINDOW_MS:
        if flush_buf is not None:
            flush_buf.zero_()
        start.record()
        for _ in range(LAUNCH_BATCH):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        times_ms.append(start.elapsed_time(end) / LAUNCH_BATCH)

    times_ms.sort()
    return TimingResult(
        median_ms=statistics.median(times_ms),
        p10_ms=_percentile(times_ms, 0.10),
        p90_ms=_percentile(times_ms, 0.90),
        n_iters=len(times_ms),
        degraded=False,
    )


def _do_bench_cpu_degraded(fn: Callable[[], object]) -> TimingResult:
    for _ in range(_DEGRADED_WARMUP_ITERS):
        fn()

    times_ms: list[float] = []
    for _ in range(_DEGRADED_MIN_ITERS):
        s = time.perf_counter()
        fn()
        e = time.perf_counter()
        times_ms.append((e - s) * 1000.0)

    times_ms.sort()
    return TimingResult(
        median_ms=statistics.median(times_ms),
        p10_ms=_percentile(times_ms, 0.10),
        p90_ms=_percentile(times_ms, 0.90),
        n_iters=len(times_ms),
        degraded=True,
    )
