# Commit sequence — the full migration playbook

Written so an implementing agent makes **zero design decisions**: every choice
is already in a spec doc, and each commit below names its inputs. The four
rules:

1. **Every decision is in `spec/`.** Names → [naming.md](naming.md) verbatim.
   Op routing → [map.md](map.md) Kernel-routing table verbatim. File paths →
   [architecture.md](architecture.md) §5 verbatim. Kernel design →
   [kernels.md](kernels.md). Gates → [testing.md](testing.md). Protocol →
   [benchmarks.md](benchmarks.md). If a needed decision is genuinely absent:
   **stop and surface it** — never invent.
2. **The tree is never red.** Every commit leaves `tox -e py312-torch-stable`
   (CPU) green; kernel commits additionally leave the local GPU gate
   (`tox -e gpu` = testing.md's six stages) green.
3. **One concern per commit.** Schema/API fixes are their own commit, never
   buried inside a kernel commit.
4. Commit titles below are used **verbatim** (conventional commits).

**Tooling decisions (agreed):** uv (`pyproject` + committed `uv.lock`,
dependency-groups) · tox via tox-uv (py310–313 × torch {min, stable, nightly}) ·
ruff (lint+format+imports, line 120) · clang-format/clang-tidy for CUDA/C++ ·
pyrefly (baseline & burn down) · no `requirements-ci.txt`.

---

## Phase 0 — platform

### 1 `docs(spec): add CUDA-rewrite spec`
Add the whole `spec/` folder as indexed in [index.md](index.md). No code.

### 2 `refactor!: kill legacy — drop CuPy/JAX bridges and deprecated APIs`
Everything ⚫ in map.md that nothing still depends on: delete
`torchsparsegradutils/{cupy,jax}/` + their 4 test files; remove
`cupy`/`jax`/`all` extras; drop `jax[cpu]` from `requirements-ci.txt`; delete
`pairwise_voxel_encoder.py`, deprecated aliases, `__getattr__` machinery in
`encoders/__init__.py`; purge from README/docs/doctests.
**Not here:** `sparse_block_diag`/`_split`, the 4 `autograd.Function` classes —
still load-bearing; they die in Phase 3/4.
Verify: suite green under old CI; `grep -ri "cupy\|jax\|c4t\|j4t" --exclude-dir=.git` clean.

### 3 `build: migrate packaging and CI installs to uv`
Dependency-groups; commit `uv.lock`; CI installs via `astral-sh/setup-uv` +
`uv sync`; delete `requirements-ci.txt` (fold surviving pins into `dev`);
add `tox.ini` (tox-uv, matrix above; CI calls tox envs); **delete `setup.py`**
(build.md — fix any dependent, don't keep the file).
Verify: `uv sync && tox -e py312-torch-stable` green; CI green.

### 4 `style: replace black/isort/flake8 with ruff; add clang-format/clang-tidy`
Drop `[tool.black]`/`[tool.isort]`; add `[tool.ruff]` (120, py310+, flake8
surface + `I`); `.clang-format`/`.clang-tidy` (pin clang version); pre-commit
hooks swapped; delete `black.yml`; lint steps → `ruff check` + `ruff format
--check`. One `ruff format` pass (separate commit if noisy).

### 5 `types: add pyrefly type checking`
`[tool.pyrefly]`; CI step; scope `torchsparsegradutils/`; record baseline error
count in the commit message; burn down in follow-ups, never blanket-ignore.

---

## Phase 1 — skeleton (pure Python; CPU CI green throughout)

### 6 `test: extract parity oracle; move tests to repo root; drop old benchmark suite`
Copy the current core-op implementations **frozen** into `tests/oracle/`
(header comment: "extracted from f19d7b4 — parity Oracle A, never shipped;
do not edit"). Move `torchsparsegradutils/tests/` → `tests/`. Delete the old
`benchmarks/` suite (git history + tagged release preserve it for the paper;
new harness arrives in commit 11).
Verify: `pytest tests/` green.

### 7 `refactor: restructure front package to architecture.md §5 hierarchy`
Create `ops/` (one module per public op; each wrapper calls its old
implementation body, moved in as a private `_legacy_*` function — marked
`# deleted by its kernel commit`), `solvers/`, split `utils/utils.py` per the
hierarchy. **Public API unchanged** (map.md contract). Pure moves — no logic
edits.
Verify: full suite green, `from torchsparsegradutils import *` surface identical.

### 8 `feat: BatchedCSR descriptor (_batched.py)`
Exactly architecture.md §3 with naming.md §2 vocabulary. COO→CSR internally via
the existing convert utils (kernel replaces the internals in commit 19). Unit +
hypothesis round-trip tests (`from_torch`/`to_torch(like=...)`, folded-row
math, ragged, B=1, int32 eligibility). Nothing else uses it yet.

### 9 `feat: tsgu:: op schemas, fake kernels, autograd registration, dispatch probe`
`_dispatch.py` (probe, `__backend_api_version__` handshake,
`TSGU_DISABLE_CUDA_BACKEND`). All nine op definitions from map.md's routing
table — schemas take plain dense tensors (architecture §2), fake kernels
value-independent, `register_autograd` per routing (backwards reference
`torch.ops.tsgu.*` — unresolved until kernels land; nothing calls them yet).
Wrappers stay on `_legacy_*`. Verify: opcheck fake/schema subsets green on CPU.

---

## Phase 2 — CUDA bring-up

### 10 `build: cuda package scaffold (kernel-builder) + smoke kernel`
`cuda/` per architecture §5: `build.toml` + `flake.nix` (kernel-builder,
Hub-only — build.md), `registration.cpp`, `common/` headers
(`batched_csr.cuh`, `reduce.cuh`, `dispatch.cuh`, `stream.cuh`),
`torchsparsegradutils_cuda/__init__.py` (load + handshake), one trivial
`tsgu::_smoke` op wired end-to-end, one dummy NVBench target, clang-tidy fed by
`compile_commands.json`.
Verify: `kernel-builder build` succeeds; `python -c` smoke test dispatches to
CUDA; front package still importable **without** the backend (probe error is
clear).

### 11 `test+bench: gate runner and benchmark harness`
Testing side: tolerance policy module (per-dtype, one place), hypothesis
profiles (`pr`=50 / `nightly`=1000, `deadline=None` in CI), the six-stage gate
runner as `tox -e gpu`. Bench side: benchmarks.md §1 protocol (clock-lock
check, L2 flush, do_bench-style windowing, memory + workspace measurement,
JSON writer with provenance + machine fingerprint), seeded synthetic
generators, viz script reading JSON. Baseline rows runnable against the oracle
today.

---

## Phase 3 — kernels (template × 8)

**Kernel-commit template** — every commit 12–19 does exactly this, nothing more:

- T1 implement per its [kernels.md](kernels.md) family section →
  `cuda/csrc/kernels/<name>/` (paths: architecture §5).
- T2 register in `registration.cpp` (append-only — lanes don't conflict).
- T3 NVBench target in `cuda/bench/` (axes = the kernel's tuning knobs).
- T4 switch the wrapper(s) in `ops/` from `_legacy_*` to the `tsgu::` op
  (routing: map.md), **delete that `_legacy_*` body** in the same commit.
- T5 run the full six-stage gate; add benchmark JSON (provenance `custom`, or
  `vendor-scaffold` where goal.md's scaffold rule is used — scaffold rows never
  enter beat-cuSPARSE claims).
- T6 tick the op's row in map.md's routing table (✅ live).

Dependency lanes — independent lanes may run as parallel agents (separate
worktrees; only `registration.cpp` is shared, append-only):

| Lane | Commits | Depends on |
|------|---------|------------|
| A | 12 → 13 | 11 |
| B | 14 → 15 → 16 → 17 | 11 |
| C | 18 | 11 |
| D | 19 | 11 |

### 12 `feat(cuda): tsgu::seglse + seglse_bwd — sparse_logsumexp live`
The pipe-cleaner: simplest kernel, no dependencies, exercises the whole chain.

### 13 `feat(cuda): tsgu::seglse_bidir + bwd — sparse_bidir_logsumexp live`
Must satisfy the bidir ≡ two-single-dim-calls equality (map contract) and the
≥1.5× fusion bar (benchmarks §3).

### 14 `feat(cuda): tsgu::sddmm`
Kernel only — no wrapper switch (nothing routes to it alone). Tested directly:
parity vs oracle SDDMM chain, gradcheck via a probe op, NVBench.

### 15 `feat(cuda): tsgu::spmm — sparse_mm live`
Fwd `spmm`, bwd `sddmm` + `spmm` on cached CSC (routing). Scaffold variant
permitted, provenance-labelled.

### 16 `feat(cuda): tsgu::spsm + descriptor plan cache — sparse_triangular_solve live`
Plan cache = lazy member on `BatchedCSR` (architecture §3). Cold/warm both
benchmarked (bars: benchmarks §3).

### 17 `refactor: batched Krylov solvers — sparse_generic_solve + lstsq live`
No new kernels: `solvers/` refactored to `(B,n,p)` iterates with per-batch
convergence masks, matvec = `tsgu::spmm` (Aᵀ via cached CSC); backwards via
`sddmm`. Pluggable user callables still honoured (map contract).

### 18 `feat(cuda): tsgu::grouped_gemm — segment_mm + gather_mm live`
DGL-exact semantics (map contract); bypasses BatchedCSR (architecture,
resolved); drops the nested-tensor implementation.

### 19 `feat(cuda): tsgu::coo2csr — convert utils live; BatchedCSR internal switch`
`convert_coo_to_csr*` route to the kernel; `BatchedCSR.from_torch` COO path
switches from the pure-torch conversion.

---

## Phase 4 — closure

### 20 `test: composite revalidation + e2e benchmarks`
Distributions/encoder statistical suites re-run (they inherit CUDA via public
ops — no code change expected); e2e `rsample` + CG-loop benchmarks; the
encoder-CSR ≤1.2×-COO memory regression pinned; real JSONs replace the dummy
tables/charts in benchmarks.md §5–6 (regenerate via `spec/images/` script
pattern).

### 21 `refactor!: final purge — no legacy left`
Delete `sparse_block_diag`/`_split`, any surviving `_legacy_*`, the 4 old
`autograd.Function` classes if any trace remains, dead imports/docs.
Verify: `grep -rn "_legacy\|block_diag" torchsparsegradutils/` empty; full GPU
gate + CPU matrix green.

### 22 `docs: rewrite README + docs for the CUDA-native package`
README around the new architecture and real benchmark numbers; merge
naming.md §2 into `docs/source/naming.rst`; spec/ stays as the design record.

---

## Post-migration backlog (explicitly NOT migration commits)
SuiteSparse-20 + DLMC corpora (pick & pin) · dedicated GPU runner + hosted GPU
CI · dashboards (JSON-only until then) · abi3 wheels for offline users ·
CUDA-version support matrix · fp16/bf16 tensor-core v2 (kernels.md) · encoder
gather kernel (benchmark-gated 🔴 candidate) · custom SpTRSV v2 (only if warm
SpSM loses).
