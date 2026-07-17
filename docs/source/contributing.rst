Contributing to torchsparsegradutils
=====================================

We welcome contributions to torchsparsegradutils! This guide will help you get started.

Development Setup
-----------------

Option 1: Development Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a consistent development environment with GPU support and all dependencies pre-installed, use VS Code Dev Containers:

**Prerequisites:**

- `Docker <https://docs.docker.com/get-docker/>`_ with NVIDIA Container Toolkit (for GPU support)
- `VS Code <https://code.visualstudio.com/>`_ with the `Dev Containers extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_

**Quick Start:**

1. Clone the repository and open in VS Code:

.. code-block:: bash

   git clone https://github.com/cai4cai/torchsparsegradutils
   cd torchsparsegradutils
   code .

2. When prompted, click **"Reopen in Container"** or use the Command Palette:
   - Press ``Ctrl+Shift+P`` (or ``Cmd+Shift+P`` on macOS)
   - Type "Dev Containers: Reopen in Container"

**Available Configurations:**

- ``.devcontainer/Dockerfile.stable`` (default): stable PyTorch with CUDA support
- ``.devcontainer/Dockerfile.nightly``: nightly PyTorch builds for latest features

To switch configurations, modify the ``dockerfile`` field in ``.devcontainer/devcontainer.json``.

Option 2: Local Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository is managed with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/cai4cai/torchsparsegradutils
   cd torchsparsegradutils

   uv sync --group dev      # environment + dev dependencies from uv.lock
   pre-commit install       # install pre-commit hooks

Working on the CUDA kernels additionally requires an NVIDIA GPU and, for
building the backend package under ``cuda/``, `Hugging Face kernel-builder
<https://huggingface.co/docs/kernels>`_ (Nix-based).

Development Workflow
--------------------

1. Create a new branch for your feature:

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. Make your changes
3. Run tests and ensure they pass
4. Add tests for new functionality
5. Update documentation if needed
6. Submit a pull request

Code Style
----------

Tooling (enforced by pre-commit and CI):

- **ruff** — linting, formatting, and import sorting for Python (line length 120):

.. code-block:: bash

   uv run ruff check .
   uv run ruff format .

- **clang-format / clang-tidy** — formatting and linting for the CUDA/C++
  kernel code under ``cuda/csrc/``.
- **pyrefly** — type checking (baseline-and-burn-down; do not add new errors).

**Type hints** are encouraged:

.. code-block:: python

   def sparse_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
       """Matrix multiplication with type hints."""
       pass

**Docstring format** follows Google style:

.. code-block:: python

   def example_function(param1: int, param2: str) -> bool:
       """Example function with Google-style docstring.

       Args:
           param1: The first parameter.
           param2: The second parameter.

       Returns:
           The return value description.

       Raises:
           ValueError: If param1 is negative.
       """
       pass

Testing
-------

We use pytest, orchestrated through tox (via tox-uv). Tests live in
``tests/`` at the repository root:

.. code-block:: bash

   # Run the test suite directly (CUDA-only tests skip cleanly without a GPU)
   uv run pytest tests/

   # Run specific test file
   uv run pytest tests/test_sparse_matmul.py

   # CPU test matrix (python x torch versions)
   uv run tox -e py312-torch-stable

   # Full six-stage GPU gate: opcheck, parity vs the frozen oracle,
   # gradcheck, hypothesis property tests, statistical suites
   uv run tox -e gpu

**Writing Tests**

Follow these guidelines:

1. The kernel-backed ops are CUDA-only: write op tests on CUDA tensors and
   guard them so they skip cleanly when no GPU/backend is available (see the
   existing ``requires_cuda_backend`` markers in ``tests/``)
2. Test both sparse layouts an op accepts (COO, CSR) and both index dtypes
   (int32, int64)
3. Test unbatched and batched inputs, including ragged batches for COO
4. Test edge cases (empty matrices, single elements) — invalid input must
   raise, never be silently accepted
5. Test gradients: pattern and layout of sparse gradients, plus numerical
   accuracy against the frozen parity oracle in ``tests/oracle/``
6. New ``tsgu::`` ops must pass ``torch.library.opcheck``

Documentation
-------------

See :doc:`naming` for the repository's conventions for sparse layouts, shapes,
batching, dimensions, specified-entry counts, and the rewrite vocabulary
(descriptors, folded rows, kernel short names). These conventions are binding
for new code, tests, docstrings, error messages, and pull requests.

Documentation is built with Sphinx. To build locally:

.. code-block:: bash

   uv sync --group docs
   cd docs/
   uv run make html
   # Open docs/_build/html/index.html

**Documentation Guidelines**

1. All public functions must have docstrings
2. Include mathematical notation where appropriate
3. Provide usage examples — on CUDA tensors, since the ops are CUDA-only
4. Document parameters, return values, and exceptions

Design record
-------------

The CUDA rewrite is specified in ``spec/`` at the repository root (goal,
public-surface map, architecture, kernel designs, testing gates, benchmark
protocol). It is the frozen design record for the migration: consult it to
understand why things are the way they are; changes to current behaviour go
through issues and PRs, not spec edits.

Adding New Features
-------------------

When adding new functionality:

1. **Start with an issue** describing the feature
2. **Design the API** — consider consistency with existing functions and the
   naming conventions
3. **Implement core functionality** with appropriate error handling (error
   messages state the accepted logical shapes and the received shape)
4. **Add comprehensive tests** covering edge cases
5. **Document thoroughly** with examples and mathematical background
6. **Benchmark performance** against alternatives when relevant — the
   benchmark protocol in ``spec/benchmarks.md`` applies (provenance labels,
   memory measured alongside time)

Continuous Integration
----------------------

Our CI pipeline runs:

1. **Lint and format** checks (ruff; clang-format for CUDA/C++)
2. **Type checking** (pyrefly)
3. **Unit tests** across the tox matrix (Python and PyTorch versions)
4. **Documentation** building

The GPU gate (``tox -e gpu``) runs on GPU-equipped machines; hosted GPU CI is
planned. All checks must pass before merging.

Submitting Pull Requests
------------------------

1. **Ensure all tests pass** locally
2. **Write clear commit messages** (conventional commits)
3. **Reference related issues**

PR template checklist:

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All CI checks pass
- [ ] Naming conventions followed (:doc:`naming`)

Code Review Process
-------------------

1. **Maintainer review** for architecture and design
2. **Automated checks** must pass
3. **Community review** welcome for all PRs
4. **Final approval** from project maintainers

Common review feedback:

- **Performance considerations** for sparse operations
- **Memory efficiency** improvements (sparse gradients must stay at the
  input's pattern — never densify)
- **Numerical stability** concerns
- **API consistency** with existing functions

Issue Reporting
---------------

When reporting bugs:

1. **Check existing issues** first
2. **Provide minimal reproducible example**
3. **Include system information** (Python version, PyTorch version, CUDA
   version, GPU model, OS)
4. **Describe expected vs. actual behavior**

Feature Requests
----------------

For feature requests:

1. **Describe the use case** and motivation
2. **Propose API design** if applicable
3. **Consider implementation complexity**
4. **Reference relevant literature** for mathematical operations

Release Process
---------------

Releases follow semantic versioning:

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backwards compatible
- **Patch** (0.0.X): Bug fixes

Release checklist:

1. Update version numbers
2. Run full test suite (CPU matrix and GPU gate)
3. Build documentation
4. Create GitHub release
5. Publish to PyPI (front package); publish the CUDA backend via
   kernel-builder to the Hugging Face Hub

Getting Help
------------

- **Documentation**: Read the docs first
- **GitHub Discussions**: For questions and design discussions
- **GitHub Issues**: For bug reports and feature requests

Thank you for contributing to torchsparsegradutils!
