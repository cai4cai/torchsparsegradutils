Quick Start Guide
=================

This guide will help you get started with torchsparsegradutils quickly.

All kernel-backed ops are CUDA-only (see :doc:`installation`): the examples
below run on CUDA tensors and require the ``torchsparsegradutils_cuda``
backend to be installed.

Basic Sparse Matrix Operations
-------------------------------

Sparse Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core function :func:`~torchsparsegradutils.sparse_mm` performs sparse
matrix multiplication with memory-sparse gradients:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   # Create a sparse matrix in COO layout, on GPU
   indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
   values = torch.tensor([3., 4., 5.])
   A = torch.sparse_coo_tensor(indices, values, (2, 3)).cuda()
   A.requires_grad_(True)

   # Create a dense matrix
   B = torch.randn(3, 4, requires_grad=True, device="cuda")

   # Perform sparse matrix multiplication with gradient support
   result = sparse_mm(A, B)
   print(f"Result shape: {result.shape}")  # torch.Size([2, 4])

   # The operation preserves sparsity in gradients
   loss = result.sum()
   loss.backward()
   print(f"A gradient is sparse: {A.grad.is_sparse}")  # True
   print(f"A gradient nse: {A.grad._nnz()}")  # 3 — A's own pattern

Batched Operations
~~~~~~~~~~~~~~~~~~

Operations are batched-first: one leading batch axis, and (for COO) batch
items may have different numbers of specified entries:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   # Create batched sparse matrices
   batch_size = 2

   # Stack individual sparse matrices
   A1 = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1., 2.], (2, 2)).cuda()
   A2 = torch.sparse_coo_tensor([[0, 1], [1, 0]], [3., 4.], (2, 2)).cuda()
   A_batch = torch.stack([A1, A2])  # Shape: (2, 2, 2)

   # Dense batch
   B = torch.randn(batch_size, 2, 3, device="cuda")

   # Batched multiplication
   result = sparse_mm(A_batch, B)
   print(result.shape)  # torch.Size([2, 2, 3])

Sparse Reductions
-----------------

:func:`~torchsparsegradutils.sparse_logsumexp` mirrors
:func:`torch.logsumexp` directly on the specified entries, and
:func:`~torchsparsegradutils.sparse_bidir_logsumexp` computes the reductions
over rows *and* over columns in a single fused kernel traversal:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_logsumexp, sparse_bidir_logsumexp
   from torchsparsegradutils.utils import rand_sparse

   A = rand_sparse((512, 256), nnz=4096, layout=torch.sparse_csr, device="cuda")

   lse_over_rows = sparse_logsumexp(A, dim=0)  # one value per column
   lse_over_cols = sparse_logsumexp(A, dim=1)  # one value per row

   # Both reductions in one traversal
   lse0, lse1 = sparse_bidir_logsumexp(A)

Solving Linear Systems
----------------------

Triangular Systems
~~~~~~~~~~~~~~~~~~

For sparse triangular systems, use :func:`~torchsparsegradutils.sparse_triangular_solve`:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_triangular_solve
   from torchsparsegradutils.utils.random_sparse import rand_sparse_tri

   # Create a sparse lower triangular matrix
   L = rand_sparse_tri((3, 3), nnz=5, upper=False, layout=torch.sparse_csr,
                       device="cuda")

   # Right-hand side
   b = torch.randn(3, 2, device="cuda")

   # Solve Lx = b
   x = sparse_triangular_solve(L, b, upper=False)

   # Verify solution (should be close to zero)
   from torchsparsegradutils import sparse_mm
   residual = sparse_mm(L, x) - b
   print(f"Residual norm: {torch.norm(residual):.6f}")

General Linear Systems
~~~~~~~~~~~~~~~~~~~~~~

For general sparse linear systems, use :func:`~torchsparsegradutils.sparse_generic_solve`
with one of the built-in iterative solvers (or your own callable):

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_generic_solve
   from torchsparsegradutils.utils.random_sparse import make_spd_sparse
   from torchsparsegradutils.utils import minres

   # Create a sparse symmetric positive definite matrix
   A_sparse, A_dense = make_spd_sparse(10, torch.sparse_coo, torch.float32,
                                       torch.int64, "cuda")
   b = torch.randn(10, device="cuda")

   # Solve using MINRES solver
   x_minres = sparse_generic_solve(A_sparse, b, solve=minres)
   print(f"MINRES solution shape: {x_minres.shape}")

   # Verify solution
   residual = A_sparse @ x_minres - b
   print(f"MINRES residual norm: {torch.norm(residual):.6f}")

Sparse Multivariate Normal Distributions
-----------------------------------------

Basic Usage with LDL^T Parameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and sample from sparse multivariate normal distributions:

.. code-block:: python

   import torch
   from torchsparsegradutils.distributions import SparseMultivariateNormal
   from torchsparsegradutils.utils.random_sparse import rand_sparse_tri

   # Create parameters
   dim = 10
   loc = torch.zeros(dim, device="cuda")

   # LDL^T parameterization (recommended for stability)
   diagonal = torch.ones(dim, device="cuda") * 0.5
   scale_tril = rand_sparse_tri((dim, dim), nnz=15, upper=False,
                               layout=torch.sparse_coo, strict=False,
                               device="cuda")

   # Create distribution with LDL^T covariance parameterization
   dist = SparseMultivariateNormal(
       loc=loc,
       diagonal=diagonal,
       scale_tril=scale_tril
   )

   # Sample from the distribution
   samples = dist.rsample((100,))
   print(f"Samples shape: {samples.shape}")  # torch.Size([100, 10])

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

The distributions support reparameterized sampling for gradient computation:

.. code-block:: python

   # Enable gradients for parameters
   loc = torch.zeros(dim, requires_grad=True, device="cuda")
   diagonal = torch.ones(dim, requires_grad=True, device="cuda")
   scale_tril = rand_sparse_tri((dim, dim), nnz=15, upper=False,
                               layout=torch.sparse_coo, strict=True,
                               device="cuda")
   scale_tril.requires_grad_(True)

   # Create distribution
   dist = SparseMultivariateNormal(
       loc=loc,
       diagonal=diagonal,
       scale_tril=scale_tril
   )

   # Sample using rsample for gradient flow
   samples = dist.rsample((10,))

   # Compute some loss
   loss = samples.mean()
   loss.backward()

   print(f"Location gradient norm: {torch.norm(loc.grad):.6f}")
   print(f"Diagonal gradient norm: {torch.norm(diagonal.grad):.6f}")
   print(f"Scale gradient nse: {scale_tril.grad._nnz()}")

Next Steps
----------

- Explore the :doc:`api/index` for complete function references
- View :doc:`benchmarks` for performance comparisons
- See :doc:`contributing` for development setup
