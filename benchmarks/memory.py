"""benchmarks/memory.py — memory measurement protocol.

spec/benchmarks.md §1: "Memory is measured alongside time, always ... Per
measurement: ``reset_peak_memory_stats()`` -> forward -> record peak ->
backward -> record peak, in a fresh allocator state (``empty_cache()``
between configs; caching-allocator reuse would understate). Reported:
``peak_fwd``, ``peak_bwd``, and workspace = peak minus tensor
inputs/outputs."

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


@dataclass(frozen=True)
class MemoryResult:
    peak_fwd_mb: Optional[float]
    peak_bwd_mb: Optional[float]
    workspace_mb: Optional[float]
    degraded: bool


def _bytes_to_mb(n: float) -> float:
    return n / (1024 * 1024)


def measure(
    forward_fn: Callable[[], torch.Tensor],
    backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    *,
    io_bytes: int = 0,
    device: Optional[torch.device] = None,
) -> MemoryResult:
    """Run ``forward_fn()`` (and, if given, ``backward_fn(output)``)
    following the reset -> forward -> record -> backward -> record protocol.

    ``io_bytes`` is the size (bytes) of the tensor inputs/outputs the caller
    already knows about; ``workspace = peak - io_bytes`` (benchmarks.md §1),
    clamped at zero so an underestimated ``io_bytes`` can't produce a
    negative workspace number.
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if device.type != "cuda":
        out = forward_fn()
        if backward_fn is not None:
            backward_fn(out)
        return MemoryResult(peak_fwd_mb=None, peak_bwd_mb=None, workspace_mb=None, degraded=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    out = forward_fn()
    torch.cuda.synchronize(device)
    peak_fwd = torch.cuda.max_memory_allocated(device)

    peak_bwd = peak_fwd
    if backward_fn is not None:
        backward_fn(out)
        torch.cuda.synchronize(device)
        peak_bwd = torch.cuda.max_memory_allocated(device)

    workspace = max(0, peak_bwd - io_bytes)
    return MemoryResult(
        peak_fwd_mb=_bytes_to_mb(peak_fwd),
        peak_bwd_mb=_bytes_to_mb(peak_bwd),
        workspace_mb=_bytes_to_mb(workspace),
        degraded=False,
    )
