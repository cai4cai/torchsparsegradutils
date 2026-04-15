# Pull Request: JOSS Test Suite and 0.2.2 Release Readiness

## Summary

This PR prepares `torchsparsegradutils` for the JOSS review follow-up and the `0.2.2` patch release. It separates manual CUDA memory/performance experiments from the default pytest suite, makes stochastic tests deterministic, fixes the broken `all` extra, and updates CI/docs for the latest stable PyTorch target.

The default test suite now keeps normal CUDA functional coverage enabled when CUDA is visible, while CUDA memory/performance/OOM-style experiments are preserved behind an explicit opt-in marker.

## What Changed

- Added deterministic global pytest seeding in `torchsparsegradutils/tests/conftest.py`.
- Added seed opt-out environment variables:
  - `TSGU_UNLOCK_SEED=true`
  - legacy `UNLOCK_SEED=true`
- Added `--run-manual-cuda` pytest option.
- Added and registered `manual_cuda` marker for GPU memory/performance/OOM experiments.
- Added `torchsparsegradutils/tests/test_config.py` for shared device/dtype/layout constants, dtype-aware tolerances, confidence thresholds, and device comparison helpers.
- Removed `@pytest.mark.flaky(reruns=...)` markers after making tests deterministic.
- Marked all of `test_integration_pairwise_sparse_mvn.py` as `manual_cuda`.
- Marked `test_sparse_mm_memory_advantage` and `test_sparse_mm_memory_stability` as `manual_cuda`.
- Kept lightweight CUDA functional tests in the default suite when CUDA is available.
- Updated CI stable PyTorch matrix from `2.5.0` / `2.9.0` to `2.5.0` / `2.11.0`, with nightly still allowed to fail.
- Updated README badge text to `Tested 2.5 / 2.11 / nightly`.
- Bumped package/docs version from `0.2.1` to `0.2.2`.
- Fixed the `all` extra so it expands to concrete optional dependencies instead of self-referencing `torchsparsegradutils[cupy,jax]`.
- Added reviewer-oriented install/test commands to README and docs.

## Context Since `v0.2.1`

This branch builds on the post-`v0.2.1` mainline work:

- JOSS paper and documentation revisions.
- More robust PyTorch version comparison using `packaging.version`.
- Sparse matmul and sparse triangular solve reshape optimizations.
- `sparse_eye` optimization avoiding unnecessary `.coalesce()`.
- CuPy binding and sparse solve updates from the JOSS revision work.

## Reviewer / User-Facing Commands

After the `0.2.2` release:

```bash
uv venv --python 3.12 --seed --managed-python
pip install "torchsparsegradutils[all]==0.2.2"
python -m pytest
```

Manual CUDA memory/performance experiments:

```bash
python -m pytest --run-manual-cuda -m manual_cuda -s
```

## Verification

Run in a CUDA-visible workspace:

```bash
black --check .
isort --check-only --diff .
flake8 . --count --show-source --statistics
python -m pytest -q --ignore=torchsparsegradutils/tests/test_doctests.py
python -m pytest -q -m manual_cuda --collect-only
python -m build --wheel
```

Results:

- `black --check .`: passed
- `isort --check-only --diff .`: passed
- `flake8 . --count --show-source --statistics`: passed
- Default pytest excluding doctests: `4289 passed, 822 skipped, 324 warnings`
- Known flaky target sweep: `242 passed, 30 warnings`
- Manual CUDA collection: `676/5128 tests collected`, `4452 deselected`
- Wheel build: passed
- Wheel metadata confirmed:
  - `Version: 0.2.2`
  - `Provides-Extra: all`
  - `Requires-Dist: cupy-cuda12x>=13.0; extra == "all"`
  - `Requires-Dist: jax[cuda12]; extra == "all"`

## Notes

The full manual CUDA suite is intentionally not part of default CI. It remains available for local validation of memory advantage, performance, and OOM behavior on suitable GPU hardware.
