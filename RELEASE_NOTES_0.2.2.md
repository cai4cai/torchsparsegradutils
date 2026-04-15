# Release Notes: torchsparsegradutils 0.2.2

Patch release focused on JOSS review readiness, deterministic testing, CUDA test separation, packaging metadata, and refreshed PyTorch compatibility validation.

## Highlights

- Default pytest is now suitable for reviewers and CI on CPU or CUDA-visible machines.
- CUDA memory/performance/OOM experiments are preserved but moved behind an explicit manual marker.
- Stochastic tests are deterministic by default.
- The `all` optional dependency extra is fixed.
- CI now tests PyTorch `2.5.0`, `2.11.0`, and nightly.
- README badge updated to `Tested 2.5 / 2.11 / nightly`.

## Testing and CI

- Added global deterministic test seeding for:
  - Python `random`
  - NumPy
  - PyTorch CPU RNG
  - PyTorch CUDA RNGs
- Added seed opt-out support:
  - `TSGU_UNLOCK_SEED=true`
  - `UNLOCK_SEED=true`
- Added `--run-manual-cuda` pytest option.
- Added `manual_cuda` marker for CUDA memory/performance/OOM experiments.
- Added shared test constants and dtype-aware tolerances in `torchsparsegradutils/tests/test_config.py`.
- Removed flaky rerun markers after stabilizing seeding and tolerance behavior.
- Marked pairwise sparse MVN integration tests as manual CUDA.
- Marked sparse matrix multiplication memory advantage and memory stability tests as manual CUDA.
- Kept small CUDA functional tests in the default suite when CUDA is available.
- Updated GitHub Actions stable PyTorch matrix:
  - `2.5.0`
  - `2.11.0`
- Kept nightly CPU CI on Python `3.12` with `continue-on-error`.

## Packaging

- Bumped version from `0.2.1` to `0.2.2`.
- Fixed `torchsparsegradutils[all]`.
- The `all` extra now expands directly to:
  - `cupy-cuda12x>=13.0`
  - `jax[cuda12]`
- Wheel metadata now includes `Provides-Extra: all`.

## Documentation

- Added reviewer-oriented installation and test commands:

```bash
uv venv --python 3.12 --seed --managed-python
pip install "torchsparsegradutils[all]==0.2.2"
python -m pytest
```

- Documented manual CUDA validation command:

```bash
python -m pytest --run-manual-cuda -m manual_cuda -s
```

- Updated README badge to `Tested 2.5 / 2.11 / nightly`.
- Updated docs version metadata to `0.2.2`.

## Changes Since `v0.2.1`

In addition to this release-readiness work, the current release includes the post-`v0.2.1` mainline changes:

- JOSS paper revisions and rebuilt paper artifacts.
- Installation and benchmark documentation updates from reviewer feedback.
- More robust PyTorch version comparison using `packaging.version`.
- Sparse matrix multiplication reshape optimization.
- Sparse triangular solve reshape optimization.
- `sparse_eye` optimization using the `is_coalesced` flag instead of forcing `.coalesce()`.
- CuPy binding and sparse solve updates from JOSS review work.

## Verification

Verified in a CUDA-visible workspace:

```bash
black --check .
isort --check-only --diff .
flake8 . --count --show-source --statistics
python -m pytest -q --ignore=torchsparsegradutils/tests/test_doctests.py
python -m pytest -q -m manual_cuda --collect-only
python -m build --wheel
```

Observed results:

- Default pytest excluding doctests: `4289 passed, 822 skipped, 324 warnings`
- Known flaky target sweep: `242 passed, 30 warnings`
- Manual CUDA collection: `676/5128 tests collected`, `4452 deselected`
- Wheel build: passed
- Wheel metadata includes `Provides-Extra: all`

## Upgrade Notes

After publication:

```bash
pip install --upgrade "torchsparsegradutils[all]==0.2.2"
```

Users who do not need CuPy/JAX support can continue installing the base package:

```bash
pip install --upgrade torchsparsegradutils==0.2.2
```

The CuPy extra currently targets CUDA 12 via `cupy-cuda12x`. Users on a different CUDA runtime should install the appropriate CuPy package manually.
