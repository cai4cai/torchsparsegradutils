"""benchmarks/run.py — the benchmark entry point.

    uv run python -m benchmarks.run --op sparse_mm --backend oracle

Benchmarks the parity Oracle A implementation (``tests/oracle``) as the
baseline path -- proving the harness -> memory -> fingerprint -> JSON
pipeline end to end against a *real* implementation, not a stub, before any
``tsgu::`` kernel exists (Phase 3, commits 12-19; spec/commit.md #11:
"Baseline rows runnable against the oracle today").

On a CUDA-visible machine this runs the real timing/memory protocol
(spec/benchmarks.md §1). On this machine today (GPU down) it runs in the
documented degraded CPU mode: perf_counter timing
(:mod:`benchmarks.harness`), null memory fields (:mod:`benchmarks.memory`),
fingerprint noting no-gpu (:mod:`benchmarks.env`) -- GPU mode activates
automatically the moment ``torch.cuda.is_available()`` is ``True`` again, no
code change required.

Only ``--op sparse_mm --backend oracle`` is wired up today, matching this
commit's scope. Future kernel commits add ``--backend custom``/
``vendor-scaffold`` (once a `tsgu::` kernel exists to attribute "ours" to)
and the other eight ops' oracle baselines as their own benchmarks land. The
written result's ``backend`` field is ``None`` (not one of
benchmarks.md §5's two provenance values) for exactly that reason -- see
``benchmarks/results.py``'s ``Result.backend`` docstring.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from benchmarks import corpus, harness, memory
from benchmarks.env import fingerprint
from benchmarks.results import Result, write_result
from tests.oracle import oracle_sparse_mm


def _bench_sparse_mm_oracle(cfg: corpus.SweepConfig) -> Result:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    a = corpus.make_sparse_batch(cfg, device=device)
    b_dense = torch.randn(cfg.B, cfg.m, cfg.p, dtype=cfg.value_dtype, device=device)

    def _fwd():
        return oracle_sparse_mm(a, b_dense)

    timing = harness.do_bench(_fwd, device=device)

    # Rough io_bytes estimate for the workspace = peak - io computation
    # (benchmarks.md §1): sparse operand (values + 2 index streams) + dense
    # rhs + dense output.
    io_bytes = (
        a._nnz() * (a.values().element_size() + 2 * corpus.bytes_per_index(cfg.index_dtype))
        + b_dense.numel() * b_dense.element_size()
        + cfg.B * cfg.n * cfg.p * b_dense.element_size()
    )
    mem = memory.measure(_fwd, io_bytes=io_bytes, device=device)

    fp = fingerprint()
    variant = f"{corpus.layout_name(cfg.layout)}·B={cfg.B}·{corpus.dtype_short(cfg.value_dtype)}/{corpus.dtype_short(cfg.index_dtype)}"

    return Result.build(
        op="sparse_mm",
        family="SpMM",
        backend=None,  # no tsgu:: kernel yet -- this run IS the baseline (module docstring)
        variant=variant,
        matrix=cfg.name,
        n=cfg.n,
        m=cfg.m,
        nse=cfg.nse_per_item * cfg.B,
        p=cfg.p,
        baseline_name="oracle-A (tests/oracle, frozen pure-PyTorch)",
        baseline_ms=timing.median_ms,
        ours_ms=None,
        speedup=None,
        peak_fwd_mb=mem.peak_fwd_mb,
        peak_bwd_mb=mem.peak_bwd_mb,
        workspace_mb=mem.workspace_mb,
        bw_pct_peak=None,
        mem_bar_met=None,
        bar=None,
        bar_met=None,
        fp=fp,
        meta={
            "p10_ms": timing.p10_ms,
            "p90_ms": timing.p90_ms,
            "n_iters": timing.n_iters,
            "timing_degraded": timing.degraded,
            "memory_degraded": mem.degraded,
        },
    )


# Only sparse_mm x oracle is wired up in this commit -- see module docstring.
_OPS = {
    "sparse_mm": _bench_sparse_mm_oracle,
}
_BACKENDS = ("oracle",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True, choices=sorted(_OPS))
    parser.add_argument("--backend", required=True, choices=_BACKENDS)
    args = parser.parse_args(argv)

    cfg = corpus.smoke_config()
    result = _OPS[args.op](cfg)
    path = write_result(result)

    print(f"wrote {path}")
    print(json.dumps(result.as_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
