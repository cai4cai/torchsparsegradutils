# Draft JOSS Reviewer Response

Thank you for flagging the test-suite issue. I agree that the previous state made the package harder to validate reproducibly, especially on CUDA-visible machines.

The issue had two parts:

1. The default pytest suite included CUDA memory/performance/OOM experiments that were useful during development but were not appropriate for normal CI or reviewer validation.
2. Several stochastic tests used random tensors without consistent global seeding, and some numerical/statistical assertions were too sensitive to CUDA float32 precision differences. This made a few tests appear flaky depending on the environment and random draw.

I have addressed this in a new release-readiness branch and prepared it for the `0.2.2` patch release.

The default test suite now:

- Uses deterministic global seeding for Python `random`, NumPy, PyTorch CPU RNG, and PyTorch CUDA RNGs.
- Keeps normal CPU tests and lightweight CUDA functional tests enabled when CUDA is available.
- Excludes CUDA memory/performance/OOM experiments unless explicitly requested.
- Removes the previous `pytest.mark.flaky(reruns=...)` markers.
- Uses shared dtype-aware tolerances for solver/backend tests.

The manual CUDA tests are still preserved for package development and memory-advantage validation. They can now be run explicitly with:

```bash
python -m pytest --run-manual-cuda -m manual_cuda -s
```

For normal reviewer validation after the `0.2.2` release, the intended clean-environment commands are:

```bash
uv venv --python 3.12 --seed --managed-python
pip install "torchsparsegradutils[all]==0.2.2"
python -m pytest
```

I also fixed the packaging issue you observed:

```text
WARNING: torchsparsegradutils 0.2.1 does not provide the extra 'all'
```

In `0.2.2`, the `all` extra now lists concrete optional dependencies directly instead of self-referencing the package. The built wheel metadata has been checked and now includes:

```text
Provides-Extra: all
Requires-Dist: cupy-cuda12x>=13.0; extra == "all"
Requires-Dist: jax[cuda12]; extra == "all"
```

I also updated CI to test the current supported stable PyTorch range:

- PyTorch `2.5.0`
- PyTorch `2.11.0`
- PyTorch nightly, allowed to fail

The README badge has been updated accordingly to:

```text
Tested 2.5 / 2.11 / nightly
```

Local verification on a CUDA-visible workspace now passes:

```text
4289 passed, 822 skipped, 324 warnings
```

I also repeated the known flaky target sweep, including:

- `test_sparse_batch_mv[batch_mv_test_data3]`
- `test_bicgstab.py::test_bicgstab_2d_rhs`
- representative linear CG, sparse solve, distribution sampling, CuPy, and JAX tests

That targeted sweep passed:

```text
242 passed, 30 warnings
```

The remaining skipped tests in the default run are intentional: they are the manual CUDA memory/performance/OOM experiments or tests skipped by existing backend/device capability checks. These are now documented and opt-in rather than part of the normal CI/reviewer command.

This should make the package reproducible for JOSS review while still preserving the GPU experiments I use to validate memory advantage over native PyTorch behavior.
