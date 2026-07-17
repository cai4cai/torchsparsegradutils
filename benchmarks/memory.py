"""benchmarks/memory.py — memory measurement protocol.

spec/benchmarks.md §1: "Memory is measured alongside time, always ... Per
measurement: ``reset_peak_memory_stats()`` -> forward -> record peak ->
backward -> record peak, in a fresh allocator state (``empty_cache()``
between configs; caching-allocator reuse would understate). Reported:
``peak_fwd``, ``peak_bwd``, and workspace = peak minus tensor
inputs/outputs."

Workspace is computed per phase (``workspace_fwd_mb`` from ``peak_fwd``,
``workspace_bwd_mb`` from ``peak_bwd``); the §5 schema's single
``workspace_mb`` is the larger of the two. ``io_bytes`` is the caller's
accounting of every tensor that is legitimately an input or output of the
measured computation — for a run with a backward pass that includes the
gradient buffers (they are the backward's outputs) and the upstream
gradient, not just the forward operands.

**O(nse) workspace bound** (benchmarks.md §1: workspace "asserted against
kernels.md's O(nse)/O(n_rows) bound"): callers pass ``bound_bytes`` — the
byte size of the op's O(nse)+O(n_rows) state (stored values + indices +
rowptr for the sparse ops; the coordinate arrays for index-only ops) — and
the assertion checks ``workspace <= WORKSPACE_BOUND_FACTOR * bound_bytes``.
The verdict is *recorded* (``workspace_bound_met``, persisted per-row in
the result JSON), not hard-failed: during migration the suite reports
honestly and the spec table carries the ✅/❌ — promoting it to a suite
failure is the post-migration step benchmarks.md §1 describes.

**Achieved bandwidth** (benchmarks.md §1, memory-bound families SpMM /
SDDMM / seglse): :func:`bw_pct_peak` turns a caller-supplied bytes-moved
model (compulsory traffic: values+indices read once + each dense operand
read once + output written once — cache-reuse of gathered rows is *not*
modelled, so for gather-heavy ops like SDDMM this is a lower bound on the
true achieved bandwidth) plus the §1 median time into a % of
:data:`PEAK_DRAM_BANDWIDTH_GB_S`.

**Degraded mode:** with no CUDA device visible to torch, there is no peak
allocator counter to read (``torch.cuda.max_memory_allocated`` requires
CUDA). Rather than substitute a CPU RSS number that isn't apples-to-apples
with the GPU protocol this module exists to run, every field is ``None``
(benchmarks.md §5's schema already treats these fields as nullable) and
``degraded=True`` is set so callers/writers never present it as a real
measurement. `forward_fn`/`backward_fn` still execute — this remains an
end-to-end plumbing proof, just without memory numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

# Peak DRAM bandwidth of the migration dev machine's GPU, for benchmarks.md
# §1's achieved-bandwidth metric: NVIDIA RTX A1000 Laptop GPU — 64-bit bus,
# GDDR6 @ 11 Gbps effective -> 176 GB/s (matches the value verified locally
# on this machine; every result JSON's header pins the GPU name so a future
# runner change can't silently reuse this constant).
PEAK_DRAM_BANDWIDTH_GB_S = 176.0

# Generous small-constant multiplier for the O(nse)/O(n_rows) workspace
# assertion (benchmarks.md §1). Chosen so the kernels that genuinely scale
# with nse pass with headroom — the largest measured passing ratio (2026-07-17
# per-op runs) is the unbatched spmm fwd+bwd at ~9.8x its vals+indices+rowptr
# bound (the backward's transposed-CSC build is int64-heavy) — while a
# dense-materialising regression (n*m scale, 100x+ the bound at the swept
# shapes) still trips it decisively. The spsm warm row genuinely fails at
# this factor (a ~13 MB per-call internal solver allocation vs a 0.3 MB
# nse-bound at its small banded shape) and is recorded false, not excused.
WORKSPACE_BOUND_FACTOR = 16


@dataclass(frozen=True)
class MemoryResult:
    peak_fwd_mb: Optional[float]
    peak_bwd_mb: Optional[float]  # None when the op has no backward pass (e.g. index-only coo2csr)
    workspace_fwd_mb: Optional[float]
    workspace_bwd_mb: Optional[float]  # None when the op has no backward pass
    workspace_mb: Optional[float]  # max over measured phases — benchmarks.md §5's single field
    workspace_bound_met: Optional[bool]  # None when no bound_bytes given (or degraded)
    degraded: bool


def _bytes_to_mb(n: float) -> float:
    return n / (1024 * 1024)


def bw_pct_peak(bytes_moved: int, median_ms: float) -> Optional[float]:
    """Achieved bandwidth as % of :data:`PEAK_DRAM_BANDWIDTH_GB_S`:
    ``bytes_moved`` (the caller's documented bytes-moved model) over the §1
    median time. Returns ``None`` for a degenerate time."""
    if median_ms is None or median_ms <= 0:
        return None
    achieved_gb_s = bytes_moved / (median_ms * 1e6)  # bytes / (ms * 1e-3 s) / 1e9
    return 100.0 * achieved_gb_s / PEAK_DRAM_BANDWIDTH_GB_S


def budget_guard(estimated_peak_bytes: int, *, label: str) -> bool:
    """VRAM guard for the 4 GB dev card: ``True`` if ``estimated_peak_bytes``
    fits ``benchmarks.corpus.peak_budget_gb()``; otherwise prints a loud
    skip note (benchmarks.md: no silent caps) and returns ``False``."""
    from benchmarks.corpus import peak_budget_gb  # local import: keep this module import-light

    budget_bytes = peak_budget_gb() * 1024**3
    if estimated_peak_bytes > budget_bytes:
        print(
            f"[skip] {label}: estimated peak {estimated_peak_bytes / 1024**3:.2f} GB exceeds "
            f"VRAM budget {budget_bytes / 1024**3:.2f} GB (benchmarks/corpus.peak_budget_gb)"
        )
        return False
    return True


def measure(
    forward_fn: Callable[[], torch.Tensor],
    backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    *,
    io_bytes: int = 0,
    bound_bytes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> MemoryResult:
    """Run ``forward_fn()`` (and, if given, ``backward_fn(output)``)
    following the reset -> forward -> record -> backward -> record protocol.

    ``io_bytes`` is the size (bytes) of the tensor inputs/outputs the caller
    already knows about — including, when ``backward_fn`` is given, the
    gradient buffers backward produces and the upstream gradient;
    ``workspace = peak - io_bytes`` per phase (benchmarks.md §1), clamped at
    zero so an underestimated ``io_bytes`` can't produce a negative
    workspace number. ``bound_bytes`` (optional) enables the O(nse)
    workspace-bound check (module docstring); its verdict is recorded on the
    result, never raised.

    An op with no backward pass (index-only, e.g. coo2csr) passes
    ``backward_fn=None`` and legitimately gets ``peak_bwd_mb=None`` — absent
    is honest there, per benchmarks.md §5's nullable schema.
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if device.type != "cuda":
        out = forward_fn()
        if backward_fn is not None:
            backward_fn(out)
        return MemoryResult(
            peak_fwd_mb=None,
            peak_bwd_mb=None,
            workspace_fwd_mb=None,
            workspace_bwd_mb=None,
            workspace_mb=None,
            workspace_bound_met=None,
            degraded=True,
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    out = forward_fn()
    torch.cuda.synchronize(device)
    peak_fwd = torch.cuda.max_memory_allocated(device)

    peak_bwd: Optional[int] = None
    if backward_fn is not None:
        backward_fn(out)
        torch.cuda.synchronize(device)
        peak_bwd = torch.cuda.max_memory_allocated(device)

    workspace_fwd = max(0, peak_fwd - io_bytes)
    workspace_bwd = max(0, peak_bwd - io_bytes) if peak_bwd is not None else None
    workspace = workspace_fwd if workspace_bwd is None else max(workspace_fwd, workspace_bwd)

    bound_met: Optional[bool] = None
    if bound_bytes is not None:
        bound_met = workspace <= WORKSPACE_BOUND_FACTOR * bound_bytes

    return MemoryResult(
        peak_fwd_mb=_bytes_to_mb(peak_fwd),
        peak_bwd_mb=_bytes_to_mb(peak_bwd) if peak_bwd is not None else None,
        workspace_fwd_mb=_bytes_to_mb(workspace_fwd),
        workspace_bwd_mb=_bytes_to_mb(workspace_bwd) if workspace_bwd is not None else None,
        workspace_mb=_bytes_to_mb(workspace),
        workspace_bound_met=bound_met,
        degraded=False,
    )
