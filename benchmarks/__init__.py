"""benchmarks — beat-cuSPARSE evidence (spec/benchmarks.md, spec/commit.md
Phase 2 #11).

A full redo of the pre-rewrite ``benchmarks/`` suite (the old one's results
back the JOSS paper and are preserved in git history / a tagged release —
spec/benchmarks.md's header). New protocol, no compatibility constraint with
the old suite.

Modules:
    harness   CUDA-event do_bench-style timing windowing (§1); degrades to
              a perf_counter loop with no CUDA.
    memory    reset_peak_memory_stats/empty_cache protocol, peak_fwd /
              peak_bwd / workspace (§1); null fields with no CUDA.
    env       machine fingerprint (GPU, driver, clocks, torch/CUDA
              versions, git commit) + clock-lock check (§1, §4).
    results   JSON result schema + writer, one file per run under
              ``benchmarks/results/`` (§4, §5).
    corpus    seeded synthetic generators + the B x n x p sweep grid (§2),
              VRAM-auto-scaled for this dev machine.
    viz       reads result JSONs, renders the chart set (§6).
    run       the ``python -m benchmarks.run`` entry point.

During migration this is JSON-only (no dashboards, spec/benchmarks.md §4);
runner is this dev machine.
"""
