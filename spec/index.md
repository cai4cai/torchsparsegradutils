# CUDA-Native Rewrite — Spec Index

Baseline: `main` @ f19d7b4 (v0.2.3). Branch: `cuda-rewrite`.
Each document gets discussed and agreed one by one before any code is written.

| # | Document | Covers | Status |
|---|----------|--------|--------|
| 0 | [goal.md](goal.md) | The one goal: beat cuSPARSE, everything native + the execution model (advisor agent, Sonnet workers, parallel lanes) | ✅ agreed |
| 1 | [map.md](map.md) | Full API map — every public func/class tiered by CUDA need (🔴 kernel / 🟠 vendor-baseline / 🟡 composite / 🟢 host / ⚫ retire) | 🗣 ready to discuss |
| 1b | [naming.md](naming.md) | The backbone: naming.rst carried through the migration + rewrite vocabulary (BatchedCSR terms, kernel short names, tsgu:: op names) | 🗣 ready to discuss |
| 2 | [commit.md](commit.md) | The full migration playbook: 22 commits in 5 phases, kernel-commit template, parallel agent lanes, zero-decision rules | 🗣 ready to discuss |
| 3 | [architecture.md](architecture.md) | Two-package split, torch.library dispatch, BatchedCSR canonical format, full file hierarchy; CPU story open | 🗣 ready to discuss |
| 4 | [kernels.md](kernels.md) | All three kernel families in one doc — SDDMM (shared backward), segmented logsumexp (flagship), vendor-baseline forwards — plus shared variant-matrix rules | 🗣 ready to discuss |
| 5 | [build.md](build.md) | Tooling decisions (uv/ruff/clang/pyrefly) + open CUDA build questions; detail in commit.md | 🗣 ready to discuss |
| 6 | [testing.md](testing.md) | Two pillars: parity (dual oracle — old impls from git + fp64 dense reference) and legitimacy (opcheck, gradcheck, properties, adversarial numerics, determinism, contract) + CI gate ordering | 🗣 ready to discuss |
| 7 | [benchmarks.md](benchmarks.md) | Full suite redo: timing protocol, 4-tier corpus, per-family baselines, CI regression gate, result schema + example table (old suite preserved in git history for the JOSS paper) | 🗣 ready to discuss |

**Working agreement:** a document moves ⬜ → 🗣 when drafted, → ✅ when discussed
and agreed. Code for an area starts only after its kernel/architecture docs are ✅.
