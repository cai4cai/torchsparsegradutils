# Goal

## **GOAL IS TO BEAT CUSPARSE AND HAVE EVERYTHING NATIVE**

Every operation in this package becomes CUDA-native — forward *and* backward — with
**our own kernels**, preserving the public API and its sparsity-preserving-gradient
guarantee. cuSPARSE/cuBLAS/cuDSS are the **performance baselines to beat**, not the
implementation. Vendor calls are acceptable only as a bring-up scaffold and as
benchmark reference.

## Execution model — one advisor, Sonnet workers

The migration is executed by agents, orchestrated by **one long-running advisor
agent**. This section is the advisor's launch brief.

### The advisor (main agent)

Does **no implementation work itself**. Its whole job:

1. **Sequence** — walk [commit.md](commit.md) in order; a commit is eligible
   when its dependencies are merged (see lanes below).
2. **Dispatch** — launch one **Sonnet worker per commit**. The worker prompt is
   assembled mechanically, never freehand:
   - the commit's entry from commit.md, verbatim (title used as-is);
   - the spec sections that entry names (e.g. commit 16 → kernels.md Family 3
     SpSM row, architecture.md §3 plan-cache paragraph, benchmarks.md §3 SpSM
     bars, map.md routing row);
   - the four rules from commit.md's header, including: **if a needed decision
     is not in `spec/`, stop and report — never invent**;
   - the verification commands the worker must run and paste output from.
3. **Review** — check the worker's diff against the named spec sections before
   merging: naming (naming.md §3 grep checklist), routing, paths, contract
   column, provenance labels on benchmark JSONs. Reject and re-dispatch with
   the specific violation named; never fix silently.
4. **Track** — after each merge, tick map.md's routing table and commit.md;
   keep the tree green (rule 2) as the invariant it owns.
5. **Escalate** — two things go to the human, immediately and verbatim:
   decisions absent from spec/, and any acceptance bar that a finished kernel
   cannot meet (goal.md bars are not negotiable downward by agents).

### Workers (Sonnet)

One commit, one worker, one conversation. A worker gets everything it needs in
the prompt and the spec — it should never need to explore history or other
docs. Workers run the verification commands themselves and report results;
"it should pass" is not a report. Kernel-commit workers follow the T1–T6
template in commit.md exactly.

### Parallelisation map (from commit.md)

- **Serial spine:** commits 1 → 11 strictly in order (each builds the
  platform the next needs). One worker at a time.
- **Parallel kernel lanes — after commit 11 merges,** up to four workers at
  once, one per lane, each in its own worktree:
  - Lane A: 12 → 13 (seglse, then bidir)
  - Lane B: 14 → 15 → 16 → 17 (sddmm → spmm → spsm → solvers; strict order
    inside the lane)
  - Lane C: 18 (grouped_gemm)
  - Lane D: 19 (coo2csr)
  Within a lane, serial. Across lanes, only `registration.cpp` is shared —
  append-only, so merges are mechanical. The advisor merges lanes as they
  finish; benchmark runs are serialised on the GPU (one worker benchmarks at a
  time — timing on a shared GPU is otherwise invalid).
- **Serial closure:** 20 → 21 → 22 only after all four lanes are merged.
- Advisor may start Lane A's worker on commit 12 as the pipe-cleaner **alone**
  first, confirm the full chain works end-to-end once, then open lanes B–D.

### What "done" means

All 22 commits merged; map.md routing table fully ✅; testing.md's six gates
green; benchmarks.md acceptance bars met with `custom` provenance (scaffold
rows excluded); zero `_legacy_*` in the package; README rewritten. Anything
less is not done.
