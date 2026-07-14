"""benchmarks/results.py — result schema + JSON writer.

spec/benchmarks.md §4: "During migration: JSON only, no dashboards. Every
run persists one JSON file (fields = §5 schema) under
``benchmarks/results/``; the viz script reads those." §5's schema, verbatim::

    op, family, backend(custom|vendor-scaffold), variant(layout/batch/dtypes),
    matrix, n, m, nse, p, baseline_name, baseline_ms, ours_ms, speedup,
    peak_fwd_mb, peak_bwd_mb, workspace_mb, bw_pct_peak, mem_bar_met, bar,
    bar_met, gpu, clocks, torch, cuda, commit, date

Every one of those fields is present on every :class:`Result`
(:data:`SCHEMA_FIELDS`, asserted in :meth:`Result.as_dict`). A ``meta`` block
carries the machine fingerprint's non-schema detail (hostname, platform, the
degraded-no-GPU flag) plus timing/memory harness detail (p10/p90, iteration
count, whether timing/memory measurement ran in degraded mode) so nothing
about the run is lost while the §5 field set stays exactly as specified.

``benchmarks/results/`` is gitignored (spec/commit.md #11: "results/*.json
from your test run are NOT committed").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from benchmarks.env import Fingerprint

RESULTS_DIR = Path(__file__).parent / "results"

# benchmarks.md §5, verbatim field order.
SCHEMA_FIELDS: tuple[str, ...] = (
    "op",
    "family",
    "backend",
    "variant",
    "matrix",
    "n",
    "m",
    "nse",
    "p",
    "baseline_name",
    "baseline_ms",
    "ours_ms",
    "speedup",
    "peak_fwd_mb",
    "peak_bwd_mb",
    "workspace_mb",
    "bw_pct_peak",
    "mem_bar_met",
    "bar",
    "bar_met",
    "gpu",
    "clocks",
    "torch",
    "cuda",
    "commit",
    "date",
)


@dataclass
class Result:
    op: str
    family: str
    # "custom" | "vendor-scaffold" | None. None is not one of benchmarks.md
    # §5's two provenance values -- it's what a run with no `tsgu::` kernel
    # to attribute "ours" to gets (see benchmarks/run.py's module docstring):
    # this commit's oracle-baseline row proves the plumbing before any
    # kernel exists (Phase 3). Once a kernel lands, its rows use the real
    # custom/vendor-scaffold values.
    backend: Optional[str]
    variant: str
    matrix: str
    n: int
    m: int
    nse: int
    p: Optional[int]
    baseline_name: Optional[str]
    baseline_ms: Optional[float]
    ours_ms: Optional[float]
    speedup: Optional[float]
    peak_fwd_mb: Optional[float]
    peak_bwd_mb: Optional[float]
    workspace_mb: Optional[float]
    bw_pct_peak: Optional[float]
    mem_bar_met: Optional[bool]
    bar: Optional[str]
    bar_met: Optional[bool]
    gpu: Optional[str]
    clocks: dict
    torch: str
    cuda: Optional[str]
    commit: str
    date: str
    meta: dict

    @classmethod
    def build(
        cls,
        *,
        op: str,
        family: str,
        backend: Optional[str],
        variant: str,
        matrix: str,
        n: int,
        m: int,
        nse: int,
        p: Optional[int],
        fp: Fingerprint,
        baseline_name: Optional[str] = None,
        baseline_ms: Optional[float] = None,
        ours_ms: Optional[float] = None,
        speedup: Optional[float] = None,
        peak_fwd_mb: Optional[float] = None,
        peak_bwd_mb: Optional[float] = None,
        workspace_mb: Optional[float] = None,
        bw_pct_peak: Optional[float] = None,
        mem_bar_met: Optional[bool] = None,
        bar: Optional[str] = None,
        bar_met: Optional[bool] = None,
        meta: Optional[dict] = None,
    ) -> "Result":
        fp_dict = fp.as_dict()
        return cls(
            op=op,
            family=family,
            backend=backend,
            variant=variant,
            matrix=matrix,
            n=n,
            m=m,
            nse=nse,
            p=p,
            baseline_name=baseline_name,
            baseline_ms=baseline_ms,
            ours_ms=ours_ms,
            speedup=speedup,
            peak_fwd_mb=peak_fwd_mb,
            peak_bwd_mb=peak_bwd_mb,
            workspace_mb=workspace_mb,
            bw_pct_peak=bw_pct_peak,
            mem_bar_met=mem_bar_met,
            bar=bar,
            bar_met=bar_met,
            gpu=fp_dict["gpu"],
            clocks=fp_dict["clocks"],
            torch=fp_dict["torch"],
            cuda=fp_dict["cuda"],
            commit=fp_dict["commit"],
            date=fp_dict["date"],
            meta={
                "hostname": fp_dict["hostname"],
                "platform": fp_dict["platform"],
                "degraded_no_gpu": fp_dict["degraded_no_gpu"],
                **(meta or {}),
            },
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        missing = [f for f in SCHEMA_FIELDS if f not in d]
        assert not missing, f"Result is missing benchmarks.md §5 schema fields: {missing}"
        return d


def write_result(result: Result, *, filename: Optional[str] = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = result.date.translate(str.maketrans("", "", ":-."))
        filename = f"{result.op}_{result.backend or 'baseline'}_{ts}.json"
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(result.as_dict(), indent=2, default=str))
    return path
