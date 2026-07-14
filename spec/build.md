# Build & Tooling

Light on purpose — execution detail lives in [commit.md](commit.md).

## Decisions

- **uv** — packaging, lockfile (`uv.lock` committed), environments, CI installs.
  Dev/test/docs deps as `[dependency-groups]`; runtime extras (`cupy`, `jax`, `all`)
  stay extras.
- **ruff** — lint + format + import sorting. Replaces black, isort, flake8.
  Line length 120.
- **clang-format + clang-tidy** — format/lint for CUDA/C++ kernel code; config
  lands with the tooling commits, enforced from the first kernel line.
- **pyrefly** — type checking in CI, baseline-and-burn-down approach.
- **tox (`tox.ini`, via `tox-uv`)** — test-matrix orchestration: py310–py313 ×
  torch {minimum supported, stable, nightly}, envs created by uv (fast), same
  matrix locally and in CI so "works on my machine" and CI are the same command
  (`tox -e py312-torch-stable`). GPU envs included but non-default.
- **No `requirements-ci.txt`** — deleted; pins live in `uv.lock` / dependency groups.

## Candidate: HF kernel-builder (Nix) — checked 2026-07-14

Hugging Face's Nix-based kernel build system. Standalone repo archived 2026-04;
now lives in `github:huggingface/kernels` (flake refs must point there).
Docs: huggingface.co/docs/kernels.

- **What it gives us:** reproducible Nix builds of torch extensions across a
  variant matrix (torch versions × CUDA versions × ABIs), `check-abi` validation
  (manylinux_2_28, torch stable ABI), `init` scaffolding (`build.toml` +
  torch-ext layout), `devshell`/`testshell` for local iteration, and
  `build-and-upload` publishing to the HF Hub (`kernel` repo type).
- **Distribution model:** consumers load kernels at runtime via
  `kernels.get_kernel()` with lockable versions — **no wheel matrix to build or
  host**. Our front package could stay pure Python (uv-managed) and pull the
  compiled kernels from the Hub. Fits the two-package architecture directly.
- **Backends:** cuda, rocm, xpu, metal, cpu, neuron — ROCm/XPU doors open later
  without new build infra.
- **Requires:** torch ≥ 2.5 (matches us), Nix on build machines only.
- **Concerns to resolve in architecture.md:** runtime Hub dependency
  (offline/air-gapped users, HPC clusters), version pinning story
  (`kernels` lockfile vs uv.lock), and whether we also want optional
  self-contained wheels as a fallback.

## Resolved

- **Distribution: Hub-only via kernel-builder, for now.** No wheel building for
  `-cuda` during migration. Wheels (abi3) stay parked as a possible
  post-migration addition for offline/HPC users — revisit at the end.
- **CUDA version policy: local dev only during migration** — the kernels build
  for this machine's toolkit; the support matrix is decided at the end (mostly
  absorbed by kernel-builder's variant matrix anyway).
- **`setup.py`: delete** (commit.md, Commit 3) — if anything still needs it,
  that thing gets fixed, not the file kept.
