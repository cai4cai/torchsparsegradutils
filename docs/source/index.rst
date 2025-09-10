torchsparsegradutils Documentation
==================================

.. image:: https://github.com/cai4cai/torchsparsegradutils/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/cai4cai/torchsparsegradutils/actions/workflows/python-package.yml
   :alt: Python tests

.. image:: https://img.shields.io/github/license/cai4cai/torchsparsegradutils
   :target: https://github.com/cai4cai/torchsparsegradutils?tab=Apache-2.0-1-ov-file#readme
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code Style: Black

**torchsparsegradutils** is a comprehensive collection of utility functions to work with PyTorch sparse tensors,
ensuring memory efficiency and supporting various sparsity-preserving tensor operations with automatic differentiation.
This package addresses fundamental gaps in PyTorch's sparse tensor ecosystem, providing essential operations that
preserve sparsity in gradients during backpropagation.

🚀 Key Features
===============

Core Sparse Operations with Sparse Gradient Support
----------------------------------------------------

**Memory-Efficient Sparse Matrix Multiplication**

- :func:`~torchsparsegradutils.sparse_mm`: Memory-efficient sparse matrix multiplication with batch support
- Preserves sparsity in gradients during backpropagation
- Workaround for `PyTorch issue #41128 <https://github.com/pytorch/pytorch/issues/41128>`_
- Supports both COO and CSR formats with optional batching

**Sparse Linear System Solvers**

- :func:`~torchsparsegradutils.sparse_triangular_solve`: Sparse triangular solver with batch support
- :func:`~torchsparsegradutils.sparse_generic_solve`: Generic sparse linear solver with pluggable backends
- :func:`~torchsparsegradutils.sparse_generic_lstsq`: Generic sparse linear least-squares solver

Built-in Iterative Solvers (No External Dependencies)
------------------------------------------------------

**Pure PyTorch Implementations**

- **BICGSTAB**: Biconjugate Gradient Stabilized method
- **CG**: Conjugate Gradient method
- **LSMR**: Least Squares Minimal Residual method
- **MINRES**: Minimal Residual method

Sparse Multivariate Normal Distributions
-----------------------------------------

- **SparseMultivariateNormal**: Structured Gaussian Distribution with reparameterised sampling
- **SparseMultivariateNormalNative**: Native implementation using torch.sparse.mm

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   benchmarks
   contributing

Installation
============

Install from PyPI:

.. code-block:: bash

   pip install torchsparsegradutils

For development installation:

.. code-block:: bash

   git clone https://github.com/cai4cai/torchsparsegradutils.git
   cd torchsparsegradutils
   pip install -e .

Quick Start
===========

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm, sparse_triangular_solve

   # Create sparse matrix
   indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
   values = torch.tensor([3., 4., 5.])
   A = torch.sparse_coo_tensor(indices, values, (2, 3))

   # Dense matrix
   B = torch.randn(3, 4)

   # Sparse matrix multiplication
   result = sparse_mm(A, B)

   # The result preserves sparsity in gradients
   loss = result.sum()
   loss.backward()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
