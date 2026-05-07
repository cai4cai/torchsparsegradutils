---
name: torchsparsegradutils-release
description: Use when preparing torchsparsegradutils PRs, releases, packaging metadata, test-stability changes, reviewer follow-up, or devcontainer-aware validation commands.
---

# torchsparsegradutils Release Workflow

## Environment

- Start by checking `pwd`, `git status --short`, and whether the shell is inside the devcontainer.
- If already in `/workspaces/torchsparsegradutils`, run commands directly.
- If outside the devcontainer, run project commands through the running devcontainer with `docker exec -w /workspaces/torchsparsegradutils <container> ...`. Find the container with `docker ps --format '{{.Names}} {{.Image}}'`.
- Prefer `python -m ...` commands so the active environment is explicit.

## Release Review

- Compare the working tree with the previous commit before writing PR or release notes:
  - `git diff --stat`
  - `git diff`
  - `git log -1 --stat`
- Keep release notes aligned with actual code changes, not stale planning text.
- Do not commit reviewer-response or release-draft files unless the user explicitly asks for them to be published.
- Public docs should contain durable installation and usage information. Put one-off reviewer validation wording in PR, release, or reviewer-response text.

## Version And Packaging

- Confirm version values match the intended release in `pyproject.toml` and `docs/source/conf.py`.
- For optional dependency changes, inspect `[project.optional-dependencies]` and verify wheel metadata after building.
- For this package, `all` should expand to concrete optional dependencies rather than self-reference another extra.

## Tests

- Preserve meaningful CUDA coverage in default pytest when CUDA is visible.
- Keep stochastic tests deterministic unless the user intentionally opts out.
- For sparse MVN integration changes, document known numerical or memory limits directly in the relevant tests.
- Recommended final validation:

```bash
python -m black --check .
python -m isort --check-only --diff .
python -m flake8 . --count --show-source --statistics
python -m pytest -q
python -m build
```

## Installed Wheel Validation

- After publishing or when validating package metadata, test the published artifact in a clean Docker image in addition to local CI.
- Use `Dockerfile.pip-install` to install the exact published package spec and run the local test suite against the installed wheel:

```bash
docker build --no-cache \
  -f Dockerfile.pip-install \
  -t torchsparsegradutils-pip-install:<version> \
  --build-arg PACKAGE_SPEC='torchsparsegradutils[all]==<version>' \
  .
docker run --rm torchsparsegradutils-pip-install:<version>
```

- The Dockerfile should install the exact `torchsparsegradutils[all]==<version>` package from PyPI, run `python -m pip check`, verify the `all` extra metadata, import the optional JAX and CuPy modules, copy `torchsparsegradutils/tests` into an isolated directory, confirm imports resolve from `site-packages`, and run `python -m pytest -q` from that isolated test directory.
- Keep CPU-only validation explicit with `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS=cpu` unless intentionally validating with a GPU.
- This Docker workflow can be promoted into a local CI CUDA runner in the future so installed published wheels are validated against GPU-visible tests as well as the CPU-only clean-image check.

## Publish Flow

- Merge the PR into `main` before creating the GitHub release.
- Create the release tag from `main`; the PyPI deployment workflow builds from the release state.
- After publication, verify:
  - PyPI has the new version.
  - `pip install "torchsparsegradutils[all]==<version>"` no longer warns about missing extras.
  - GitHub Actions and docs builds are green.
