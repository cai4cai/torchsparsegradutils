Contributing to torchsparsegradutils
=====================================

We welcome contributions to torchsparsegradutils! This guide will help you get started.

Development Setup
-----------------

#### Option 1: Development Containers (Recommended)

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

- ``.devcontainer/Dockerfile.stable`` (default): Uses stable PyTorch with CUDA 12.8 support
- ``.devcontainer/Dockerfile.nightly``: Uses nightly PyTorch builds for latest features

To switch configurations, modify the ``dockerfile`` field in ``.devcontainer/devcontainer.json``:

.. code-block:: json

   "build": {
       "dockerfile": "./Dockerfile.nightly",  // or "./Dockerfile.stable"
       "context": "."
   }

**What's Included:**

- **CUDA 12.8**: Full GPU development support with NVIDIA drivers
- **Pre-installed Dependencies**: PyTorch, CuPy, JAX, SciPy, and all development tools
- **VS Code Extensions**: Python, Pylance, Jupyter, GitHub Copilot, and code formatting tools
- **Development Tools**: pytest, black, flake8, pre-commit hooks
- **Python Environment**: Python 3.10+ with all optional dependencies

**Benefits:**

- ✅ **Consistent Environment**: Same setup across different machines
- ✅ **GPU Support**: Pre-configured CUDA environment
- ✅ **Zero Setup**: All dependencies and tools pre-installed
- ✅ **Isolated**: No conflicts with host system packages
- ✅ **VS Code Integration**: Seamless debugging, IntelliSense, and testing

#### Option 2: Local Development

If you prefer local development:

.. code-block:: bash

   git clone https://github.com/cai4cai/torchsparsegradutils
   cd torchsparsegradutils

   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install in development mode
   pip install -e ".[dev]"  # Install in development mode
   pre-commit install       # Install pre-commit hooks

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

We use several tools to maintain code quality:

**Black** for code formatting:

.. code-block:: bash

   black torchsparsegradutils/ tests/

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

We use pytest for testing. Tests are organized by module:

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_sparse_matmul.py

   # Run with coverage
   pytest --cov=torchsparsegradutils

   # Run CUDA memory/performance experiments manually
   pytest --run-manual-cuda -m manual_cuda -s

**Writing Tests**

Follow these guidelines:

1. Test both CPU and GPU (when available)
2. Test different sparse formats (COO, CSR)
3. Test edge cases (empty matrices, single elements)
4. Test gradients and backpropagation
5. Include numerical accuracy tests

Example test structure:

.. code-block:: python

   import torch
   import pytest
   from torchsparsegradutils import sparse_mm

   class TestSparseMatmul:

       @pytest.mark.parametrize("device", ["cpu", "cuda"])
       def test_basic_multiplication(self, device):
           if device == "cuda" and not torch.cuda.is_available():
               pytest.skip("CUDA not available")

           # Test setup
           A = create_test_sparse_matrix().to(device)
           B = torch.randn(3, 4).to(device)

           # Test
           result = sparse_mm(A, B)
           expected = torch.sparse.mm(A, B)

           # Assertions
           assert torch.allclose(result, expected)

       def test_gradient_sparsity(self):
           """Test that gradients preserve sparsity."""
           A = create_test_sparse_matrix(requires_grad=True)
           B = torch.randn(3, 4, requires_grad=True)

           result = sparse_mm(A, B)
           loss = result.sum()
           loss.backward()

           assert A.grad.is_sparse
           assert A.grad._nnz() == A._nnz()

Documentation
-------------

Documentation is built with Sphinx. To build locally:

.. code-block:: bash

   cd docs/
   make html
   # Open docs/_build/html/index.html

**Documentation Guidelines**

1. All public functions must have docstrings
2. Include mathematical notation where appropriate
3. Provide usage examples
4. Document parameters, return values, and exceptions

**Adding Examples**

Add examples to the docstrings:

.. code-block:: python

   def sparse_triangular_solve(A, B, upper=False):
       """Solve triangular sparse linear system.

       Args:
           A: Sparse triangular matrix
           B: Dense right-hand side
           upper: Whether A is upper triangular

       Returns:
           Solution tensor X such that AX = B

       Example:
           >>> import torch
           >>> from torchsparsegradutils import sparse_triangular_solve
           >>> # Create lower triangular matrix
           >>> indices = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 2]])
           >>> values = torch.tensor([1., 2., 3., 4.])
           >>> A = torch.sparse_coo_tensor(indices, values, (3, 3))
           >>> B = torch.randn(3, 2)
           >>> X = sparse_triangular_solve(A, B, upper=False)
       """

Adding New Features
-------------------

When adding new functionality:

1. **Start with an issue** describing the feature
2. **Design the API** - consider consistency with existing functions
3. **Implement core functionality** with appropriate error handling
4. **Add comprehensive tests** covering edge cases
5. **Document thoroughly** with examples and mathematical background
6. **Benchmark performance** against alternatives when relevant

Example feature addition:

.. code-block:: python

   # 1. Core implementation
   def new_sparse_operation(A, B, option="default"):
       """New sparse operation description."""
       # Input validation
       if not A.is_sparse:
           raise ValueError("A must be sparse")

       # Implementation
       result = _compute_operation(A, B, option)
       return result

   # 2. Tests
   def test_new_sparse_operation():
       # Basic functionality test
       # Edge case tests
       # Gradient tests
       # Performance test
       pass

   # 3. Documentation
   # Add to API docs, tutorials, examples

Benchmarking New Features
-------------------------

Include benchmarks for new operations:

.. code-block:: python

   # benchmarks/benchmark_new_feature.py
   import time
   import torch
   from torchsparsegradutils import new_sparse_operation

   def benchmark_new_operation():
       sizes = [100, 500, 1000, 5000]
       results = []

       for n in sizes:
           A = create_sparse_matrix(n, n)
           B = torch.randn(n, n)

           # Timing
           start = time.time()
           result = new_sparse_operation(A, B)
           end = time.time()

           results.append((n, end - start))

       return results

Continuous Integration
----------------------

Our CI pipeline runs:

1. **Code formatting** checks (Black)
2. **Type checking** (mypy, when available)
3. **Unit tests** on multiple Python versions
4. **Integration tests** with different PyTorch versions
5. **Documentation** building
6. **Benchmark** regression tests

All checks must pass before merging.

Submitting Pull Requests
------------------------

1. **Ensure all tests pass** locally
2. **Write clear commit messages**
3. **Reference related issues**

PR template checklist:

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All CI checks pass
- [ ] Backwards compatibility maintained

Code Review Process
-------------------

1. **Maintainer review** for architecture and design
2. **Automated checks** must pass
3. **Community review** welcome for all PRs
4. **Final approval** from project maintainers

Common review feedback:

- **Performance considerations** for sparse operations
- **Memory efficiency** improvements
- **Numerical stability** concerns
- **API consistency** with existing functions

Issue Reporting
---------------

When reporting bugs:

1. **Check existing issues** first
2. **Provide minimal reproducible example**
3. **Include system information** (Python version, PyTorch version, OS)
4. **Describe expected vs. actual behavior**

Example bug report:

.. code-block:: python

   # System info
   Python 3.10.0
   PyTorch 2.5.0
   CUDA 12.1 (if relevant)

   # Minimal example
   import torch
   from torchsparsegradutils import sparse_mm

   A = torch.sparse_coo_tensor(...)
   B = torch.randn(...)

   # This fails with error message:
   result = sparse_mm(A, B)

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
2. Run full test suite
3. Build documentation
4. Create GitHub release
5. Publish to PyPI

Getting Help
------------

- **Documentation**: Read the docs first
- **GitHub Discussions**: For questions and design discussions
- **GitHub Issues**: For bug reports and feature requests

Thank you for contributing to torchsparsegradutils!
