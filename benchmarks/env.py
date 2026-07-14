"""benchmarks/env.py — machine fingerprint.

spec/benchmarks.md §1: "Hardware state: locked clocks (``nvidia-smi -lgc`` +
persistence mode), recorded in every result file header (GPU, driver, CUDA,
torch versions, clock pin). A result without a header is not a result."

**Clock-lock check:** actually *locking* clocks (``nvidia-smi -lgc``) needs
root, which this harness does not assume or request. Instead this module
*queries* the current clock state (no root needed) and records it — a run
with unlocked clocks still produces a result, just one whose header says so
loudly, per spec/commit.md #11: "warns loudly if clocks aren't pinned ...
record clock state in the fingerprint rather than failing."

This dev machine's GPU is known to intermittently fall off the PCIe bus
(spec/commit.md #11 worker context) — every ``nvidia-smi`` call here is
defensive (timeout + broad exception handling) and degrades to ``None``
fields plus a warning rather than raising.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import torch

_GIT_COMMIT_CACHE: Optional[str] = None


def _git_commit() -> str:
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is not None:
        return _GIT_COMMIT_CACHE
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        _GIT_COMMIT_CACHE = out.stdout.strip()
    except Exception:
        _GIT_COMMIT_CACHE = "unknown"
    return _GIT_COMMIT_CACHE


def _nvidia_smi_query(fields: str) -> Optional[list[str]]:
    """``nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits``,
    first line only, as a list of stripped strings. Never raises: returns
    ``None`` if ``nvidia-smi`` is missing, errors, or times out."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        return None


def _to_float(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class ClockState:
    """Current (queried, not necessarily locked) GPU clocks."""

    sm_clock_mhz: Optional[float]
    mem_clock_mhz: Optional[float]
    max_sm_clock_mhz: Optional[float]
    locked: Optional[bool]  # best-effort proxy (sm == max_sm); -lgc pinning itself is never verified (needs root)
    note: str

    def as_dict(self) -> dict:
        return {
            "sm_clock_mhz": self.sm_clock_mhz,
            "mem_clock_mhz": self.mem_clock_mhz,
            "max_sm_clock_mhz": self.max_sm_clock_mhz,
            "locked": self.locked,
            "note": self.note,
        }


def clock_state() -> ClockState:
    current = _nvidia_smi_query("clocks.sm,clocks.mem")
    maxes = _nvidia_smi_query("clocks.max.sm")
    if current is None or maxes is None:
        warnings.warn(
            "benchmarks.env: could not query GPU clock state via nvidia-smi "
            "(not installed, GPU not visible/off the bus, or the query "
            "failed). Clocks are recorded as unknown in this result's "
            "fingerprint rather than assumed locked -- do not trust timing "
            "numbers from this run for cross-machine or cross-run "
            "comparison (benchmarks.md §1).",
            stacklevel=2,
        )
        return ClockState(
            sm_clock_mhz=None,
            mem_clock_mhz=None,
            max_sm_clock_mhz=None,
            locked=None,
            note="nvidia-smi unavailable (no GPU / driver error / not installed)",
        )

    sm = _to_float(current[0]) if len(current) > 0 else None
    mem = _to_float(current[1]) if len(current) > 1 else None
    max_sm = _to_float(maxes[0]) if maxes else None
    locked = sm is not None and max_sm is not None and abs(sm - max_sm) < 1.0

    if not locked:
        warnings.warn(
            f"benchmarks.env: GPU SM clock ({sm} MHz) is below its max "
            f"({max_sm} MHz) -- clocks do not appear pinned. `nvidia-smi -lgc` "
            "needs root; this run proceeds anyway with the actual clock "
            "state recorded in the result header (spec/benchmarks.md §1, "
            "spec/commit.md #11).",
            stacklevel=2,
        )
    return ClockState(
        sm_clock_mhz=sm,
        mem_clock_mhz=mem,
        max_sm_clock_mhz=max_sm,
        locked=locked,
        note="queried via nvidia-smi (no root; -lgc pinning itself not enforced by this harness)",
    )


_NO_GPU_CLOCK_STATE = ClockState(
    sm_clock_mhz=None,
    mem_clock_mhz=None,
    max_sm_clock_mhz=None,
    locked=None,
    note="no CUDA device visible to torch -- degraded CPU-only run",
)


@dataclass(frozen=True)
class Fingerprint:
    gpu: Optional[str]
    driver: Optional[str]
    clocks: dict
    torch_version: str
    cuda_version: Optional[str]
    commit: str
    date: str
    hostname: str
    platform: str
    degraded_no_gpu: bool

    def as_dict(self) -> dict:
        return {
            "gpu": self.gpu,
            "driver": self.driver,
            "clocks": self.clocks,
            "torch": self.torch_version,
            "cuda": self.cuda_version,
            "commit": self.commit,
            "date": self.date,
            "hostname": self.hostname,
            "platform": self.platform,
            "degraded_no_gpu": self.degraded_no_gpu,
        }


def fingerprint() -> Fingerprint:
    """Build this run's machine fingerprint. Never raises: every GPU-facing
    query degrades to ``None``/a warning rather than an exception, because a
    benchmark run must survive this dev machine's GPU being down."""
    has_cuda = torch.cuda.is_available()

    gpu_name: Optional[str] = None
    if has_cuda:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None

    driver_info = _nvidia_smi_query("driver_version")
    driver = driver_info[0] if driver_info else None

    clocks = clock_state().as_dict() if has_cuda else _NO_GPU_CLOCK_STATE.as_dict()

    if not has_cuda:
        warnings.warn(
            "benchmarks.env: torch.cuda.is_available() is False -- this "
            "result is a degraded, no-GPU, plumbing-proof run (perf_counter "
            "timing, null memory fields). Not a benchmarks.md §1-compliant "
            "measurement.",
            stacklevel=2,
        )

    return Fingerprint(
        gpu=gpu_name,
        driver=driver,
        clocks=clocks,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        commit=_git_commit(),
        date=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform=platform.platform(),
        degraded_no_gpu=not has_cuda,
    )
