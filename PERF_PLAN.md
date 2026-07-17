# Performance Phase Playbook — cuda-rewrite-ptx

Successor to the 22-commit migration playbook. Migration is complete; this phase
implements [kernel_best_practices.md](kernel_best_practices.md) end to end.
Source of truth for *what* each commit does is that document's sections; this file
is the *who/when/where/gates*.

## Execution model

**Advisor: Fable (this session's model).** The advisor writes no kernel code by
default. The advisor: assembles each worker prompt (best-practices section +
spec contract refs + acceptance bar + baseline numbers), reviews every diff and
SASS-audit output, runs the gates, owns the benchmark ledger, and decides
pass/retry/escalate.

**Workers: Opus agents (`model: opus`), one commit each.** Every worker prompt
carries, verbatim:

- the relevant `kernel_best_practices.md` section(s) and hardware table,
- the acceptance bar and the baseline benchmark rows it must beat,
- the standing warning (recurring mistake, seen twice in the migration): **no
  fixed-width predicated unrolls, no serial per-entry scalar loads** — vectorize
  loads, verify in SASS,
- the gate list (below) the worker must run before reporting done.

**Escalation rule (amended from migration precedent):** if a worker misses its
perf bar, the advisor diagnoses (SASS + Nsight, not guesswork) and re-briefs a
fresh Opus worker with the diagnosis. Only after a second miss does the advisor
consider intervening directly — this phase is Opus-does-everything by design.

## Worktree

All work happens in a new worktree **`cuda-rewrite-ptx`**, branched off
`cuda-rewrite`. **Not created yet — creation is Step 0 at kickoff:**

```bash
# from the main checkout
git -C .worktrees/cuda-rewrite add kernel_best_practices.md PERF_PLAN.md
git -C .worktrees/cuda-rewrite commit -m "docs: kernel best-practices research + perf-phase playbook"
git worktree add .worktrees/cuda-rewrite-ptx -b cuda-rewrite-ptx cuda-rewrite
```

The docs commit must land on `cuda-rewrite` **before** the worktree is created —
both files are currently untracked, and untracked files do not follow a new
worktree. `cuda-rewrite` then stays frozen for the maintainer RFC; all perf
commits go to `cuda-rewrite-ptx`.

Machine constraints carry over unchanged: nix builds `--max-jobs 1 --cores 6`,
repoint both `.pth` files after every `nix build`, disk is ~90% full (a worktree
checkout is cheap but nix build artifacts are not — prune `result` symlinks and
old store paths between kernels if needed), GPU clocks cannot be locked (no
root) so every benchmark records the variance protocol from cross-cutting §10.

## Standing gates (every commit)

1. `tox -e gpu` green, including the run-twice determinism tests.
2. SASS audit script (commit 1) clean for the touched kernels: no `LDL`/`STL`
   spills, expected `LDG.E.128`/`LDGSTS` present, no `BAR` in spin loops.
3. Benchmark row(s) recorded with backend provenance (`custom` vs
   `vendor-scaffold`); result JSONs never committed.
4. A perf commit that regresses any other kernel's benchmark does not land.
5. Deterministic path exists and is exercised for every touched op
   (raise-don't-fallback, per spec).

## Commit sequence

Ordering logic: verification tooling first (nothing lands unverified), then the
open acceptance bar (grouped_gemm — the only release blocker), then cheap
cross-kernel wins that compound, then per-kernel passes, then
measurement-gated compiler flags, then the closing sweep.

| # | Commit | Scope (best-practices refs) | Bar |
|---|--------|------------------------------|-----|
| 0 | *(advisor, not a worker)* docs commit + worktree creation | see above | — |
| 1 | `perf(tools): sass audit + baseline ledger` | Implement the SASS audit checklist as `cuda/tools/sass_audit.py` (cuobjdump/nvdisasm wrappers; greps for LDG.E.128/LDS.128/LDGSTS/LDL/STL/BAR/REDUX; per-kernel report). Capture baseline benchmark JSONs for every op on this machine. | Script runs on all current kernels; baselines archived |
| 2 | `perf(cuda): grouped_gemm — cp.async staging` | cp.async.cg 16-byte multi-stage ring (2–3 stages), LDS.128 fragment loads, bank-conflict swizzle or padding; pyptx `gemm_highperf_ampere.py` is the annotated reference design. | LDGSTS verified in SASS; ≥0.8× cuBLAS geomean (from 0.5–0.8×) |
| 3 | `perf(cuda): grouped_gemm — scheduling` | Persistent work-stealing tile loop, K-descending problem sort, cache-aware ordering (tiles sharing a B back-to-back). Contingency if wave-quantization tail visible on 16 SMs: Stream-K with ordered (deterministic) fixup. | **cuBLAS parity (≥1.0× geomean) on segment shapes — closes the open migration bar** |
| 4 | `perf(cuda): cache-hint + vectorization pass (all kernels)` | `const __restrict__`/`__ldg` on reused dense-row gathers, `__ldcs` on stream-once values (probe-verified: `LDG.E.EF`), 128-bit loads with alignment handling everywhere the audit shows scalar runs. | Audit-verified per kernel; no benchmark regression; deltas logged |
| 5 | `perf(cuda): sddmm` | Sub-warp sizing (entries-per-warp = max(1, 32/p) for p≤32), serial-loop warp-per-entry for mid p, smem-tiled G-row path with empirical threshold p≈128–256, cp.async k-tiling in the tiled path. | Beat current 2.41× vs cuSPARSE on the suite; no small-p regression |
| 6 | `perf(cuda): spmm` | Row-length-stat cached dispatch (merge-path/bucketed vs row-split; rebalance trigger max/avg > 8), L2 access-policy window for index arrays in batched/Krylov reuse, cp.async double-buffer on the dense operand. | Hold ≥ vendor unbatched; measurable win on skewed-row suite; batched geomean improves |
| 7 | `perf(cuda): seglse` | Element-balanced VT assignment, warp-aggregated atomics (ballot/shuffle before any global atomic), integer-reinterp float atomicMax, rescale-per-partial not per-element; deterministic two-pass stays the det path. | Widen margin vs scatter_logsumexp; det path ≤2× fast path |
| 8 | `perf(cuda): seglse_bidir — CSC decision benchmark` | Implement cached-CSC-permutation second-pass variant (descriptor-cached, like the SpSM plan); benchmark vs fused-atomic single pass on skewed-column suites. Permutation variant becomes the det path regardless; winner becomes the fast path. **Resolves the flagged open design question.** | Decision recorded with benchmark rows; both paths gated |
| 9 | `perf(cuda): spsm — sync-free path` | Spin-flag variant: `red.release.gpu` producer counters, acquire-load consumer poll, `__nanosleep` backoff; dispatch on cached DAG stats (deep-narrow → sync-free, shallow-wide → level-set). Level-set stays the det path. | Audit: zero `BAR` in spin loop; win on deep-DAG suite; no shallow-DAG regression |
| 10 | `perf(cuda): coo2csr` | `begin_bit`/`end_bit` radix narrowing from nrows/ncols, sorted-input detection with early-exit to compress, duplicate-coalescing correctness per addendum (cuSPARSE does NOT dedup — verify our upstream handling). | Improve current 1.18× vendor bar; sorted-input fast path ≥5× blind sort |
| 11 | `perf(build): compiler gates` | seglse isolated in its own .cu; fast-math `expf` ULP accuracy suite (power-law + near-degenerate segments) — flag lands only if suite passes; `-ftz=true` and `-Xptxas -dlcm=cg` A/B'd the same way. Nothing lands on flags without an accuracy gate + measured win. | Every flag decision recorded (adopted or rejected-with-numbers) |
| 12 | `perf(bench): closing sweep + docs` | Full benchmark re-run, update `benchmarks.md` geomeans, annotate `kernel_best_practices.md` with measured outcomes (worked / didn't / rejected), close remaining open gates or convert them to issues. | Ledger complete; doc reflects reality |

**Deferred (explicitly not this phase):** fp16/bf16 tensor-core grouped GEMM and
seglse half-storage (v2, own phase — prototype in pyptx first per the lab-bench
note), SuiteSparse/DLMC corpora expansion, wheels, maintainer RFC.

## pyptx lab bench (advisor-managed, optional per commit)

`pip install pyptx` (pinned commit) into the dev venv only — never a package
dependency. Use: prototyping PTX hypotheses (swizzle variants, spin-flag
orderings, fragment layouts) against torch tensors in seconds before a worker
bakes them into C++; `print(k.ptx())` output attached to worker prompts as the
specification of intent.

## Advisor loop per commit

1. Pre-brief: baseline rows + SASS audit of the target kernel → prompt assembly.
2. Launch Opus worker; worker implements, runs gates, reports with numbers.
3. Advisor review: diff, audit output, benchmark rows, determinism tests.
4. Pass → commit on `cuda-rewrite-ptx`. Miss → diagnose, re-brief fresh worker
   (escalation rule above).
5. Ledger updated; next commit's baseline is this commit's result.
