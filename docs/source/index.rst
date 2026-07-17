torchsparsegradutils Documentation
==================================

.. image:: https://github.com/cai4cai/torchsparsegradutils/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/cai4cai/torchsparsegradutils/actions/workflows/python-package.yml
   :alt: Python tests

.. image:: https://img.shields.io/github/license/cai4cai/torchsparsegradutils
   :target: https://github.com/cai4cai/torchsparsegradutils?tab=Apache-2.0-1-ov-file#readme
   :alt: License

**torchsparsegradutils** provides sparse autograd utilities for PyTorch,
implemented on native CUDA kernels (the ``tsgu::`` op set). Every operation
computes gradients with respect to a sparse input at that input's sparsity
pattern, in that input's layout — never by materialising a dense gradient.
The package fills long-standing gaps in PyTorch's sparse ecosystem: sparse
gradients for matmul and linear solves, and a native sparse ``logsumexp``.

Key Features
============

Native CUDA kernels
-------------------

- All core ops route to custom CUDA kernels registered as ``torch.library``
  custom ops — ``torch.compile``-compatible and ``opcheck``-tested.
- Batched-first: one leading batch axis, with ragged specified-entry counts
  per batch item supported through an internal ``BatchedCSR`` descriptor.
- CUDA-required at runtime: the compiled backend package
  ``torchsparsegradutils_cuda`` registers the kernels at import (see
  :doc:`installation`).

Core Sparse Operations with Sparse Gradient Support
----------------------------------------------------

- :func:`~torchsparsegradutils.sparse_mm`: sparse × dense matrix
  multiplication with batch support and sparse gradients
- :func:`~torchsparsegradutils.sparse_logsumexp`: sparse-aware
  ``log-sum-exp`` reduction mirroring ``torch.logsumexp``
- :func:`~torchsparsegradutils.sparse_bidir_logsumexp`: row and column
  ``log-sum-exp`` in one fused kernel traversal
- :func:`~torchsparsegradutils.sparse_triangular_solve`: sparse triangular
  solver with batch support
- :func:`~torchsparsegradutils.sparse_generic_solve`: generic sparse linear
  solver with pluggable iterative backends
- :func:`~torchsparsegradutils.sparse_generic_lstsq`: generic sparse linear
  least-squares solver
- :func:`~torchsparsegradutils.segment_mm` /
  :func:`~torchsparsegradutils.gather_mm`: grouped dense matmul with
  DGL-compatible semantics

Built-in Iterative Solvers (No External Dependencies)
------------------------------------------------------

Host-side loops whose matrix-vector products run on the native kernels:

- **BICGSTAB**: Biconjugate Gradient Stabilized method
- **CG**: Conjugate Gradient method
- **LSMR**: Least Squares Minimal Residual method
- **MINRES**: Minimal Residual method

Sparse Multivariate Normal Distributions
-----------------------------------------

- **SparseMultivariateNormal**: Structured Gaussian distribution with
  reparameterised sampling
- **SparseMultivariateNormalNative**: Native implementation using
  ``torch.sparse.mm``

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   benchmarks
   contributing
   naming

Installation
============

The front package installs from PyPI; the CUDA backend is required at
runtime (see :doc:`installation` for the full story):

.. code-block:: bash

   pip install torchsparsegradutils

For development installation:

.. code-block:: bash

   git clone https://github.com/cai4cai/torchsparsegradutils.git
   cd torchsparsegradutils
   uv sync --group dev

Quick Start
===========

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   # Create sparse matrix on GPU
   indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
   values = torch.tensor([3., 4., 5.])
   A = torch.sparse_coo_tensor(indices, values, (2, 3)).cuda()
   A.requires_grad_(True)

   # Dense matrix
   B = torch.randn(3, 4, device="cuda")

   # Sparse matrix multiplication
   result = sparse_mm(A, B)

   # The gradient w.r.t. A is sparse, at A's pattern
   loss = result.sum()
   loss.backward()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
