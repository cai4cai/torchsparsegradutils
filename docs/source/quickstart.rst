Quick Start Guide
=================

This guide will help you get started with torchsparsegradutils quickly.

Basic Sparse Matrix Operations
-------------------------------

Sparse Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core function :func:`~torchsparsegradutils.sparse_mm` performs memory-efficient sparse matrix multiplication:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   # Create a sparse matrix in COO format
   indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
   values = torch.tensor([3., 4., 5.])
   A = torch.sparse_coo_tensor(indices, values, (2, 3))

   # Create a dense matrix
   B = torch.randn(3, 4, requires_grad=True)

   # Perform sparse matrix multiplication
   result = sparse_mm(A, B)
   print(result.shape)  # torch.Size([2, 4])

   # The operation preserves sparsity in gradients
   loss = result.sum()
   loss.backward()
   print(B.grad.is_sparse)  # Gradients preserve sparsity structure

Batched Operations
~~~~~~~~~~~~~~~~~~

torchsparsegradutils supports batched operations:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   # Create batched sparse matrices
   batch_size = 2

   # Method 1: Stack individual sparse matrices
   A1 = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1., 2.], (2, 2))
   A2 = torch.sparse_coo_tensor([[0, 1], [1, 0]], [3., 4.], (2, 2))
   A_batch = torch.stack([A1, A2])  # Shape: (2, 2, 2)

   # Dense batch
   B = torch.randn(batch_size, 2, 3)

   # Batched multiplication
   result = sparse_mm(A_batch, B)
   print(result.shape)  # torch.Size([2, 2, 3])

Solving Linear Systems
----------------------

Triangular Systems
~~~~~~~~~~~~~~~~~~

For sparse triangular systems, use :func:`~torchsparsegradutils.sparse_triangular_solve`:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_triangular_solve

   # Create a sparse lower triangular matrix
   indices = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 2]])
   values = torch.tensor([1., 2., 3., 4.])
   L = torch.sparse_coo_tensor(indices, values, (3, 3))

   # Right-hand side
   b = torch.randn(3, 2)

   # Solve Lx = b
   x = sparse_triangular_solve(L, b, upper=False)

   # Verify solution
   residual = sparse_mm(L, x) - b
   print(torch.norm(residual))  # Should be close to zero

General Linear Systems
~~~~~~~~~~~~~~~~~~~~~~

For general sparse linear systems, use :func:`~torchsparsegradutils.sparse_generic_solve`:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_generic_solve

   # Create a sparse symmetric positive definite matrix
   n = 100
   # ... create your sparse matrix A and RHS b ...

   # Solve using Conjugate Gradient
   x_cg = sparse_generic_solve(A, b, method='cg', tol=1e-6)

   # Solve using BICGSTAB
   x_bicgstab = sparse_generic_solve(A, b, method='bicgstab', tol=1e-6)

   # Solve using MINRES
   x_minres = sparse_generic_solve(A, b, method='minres', tol=1e-6)

Sparse Multivariate Normal Distributions
-----------------------------------------

Basic Usage
~~~~~~~~~~~

Create and sample from sparse multivariate normal distributions:

.. code-block:: python

   import torch
   from torchsparsegradutils.distributions import SparseMultivariateNormal

   # Create a sparse precision matrix (inverse covariance)
   dim = 10
   indices = torch.tensor([[i, i] for i in range(dim)] +
                          [[i, i+1] for i in range(dim-1)] +
                          [[i+1, i] for i in range(dim-1)]).T
   values = torch.tensor([2.0] * dim + [-0.5] * (dim-1) + [-0.5] * (dim-1))
   precision = torch.sparse_coo_tensor(indices, values, (dim, dim))

   # Create mean vector
   mean = torch.zeros(dim)

   # Create distribution
   dist = SparseMultivariateNormal(
       loc=mean,
       precision_matrix=precision,
       param='precision_LDL'
   )

   # Sample from the distribution
   samples = dist.sample((1000,))
   print(samples.shape)  # torch.Size([1000, 10])

   # Compute log probability
   log_prob = dist.log_prob(samples)
   print(log_prob.shape)  # torch.Size([1000])

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

The distributions support reparameterized sampling for gradient computation:

.. code-block:: python

   # Enable gradients for parameters
   mean.requires_grad_(True)
   precision.values().requires_grad_(True)

   # Sample using rsample for gradient flow
   samples = dist.rsample((100,))

   # Compute some loss
   loss = samples.mean()
   loss.backward()

   print("Mean gradient:", mean.grad)
   print("Precision gradient:", precision.values().grad)

Working with Different Backends
-------------------------------

CuPy Backend
~~~~~~~~~~~~

For GPU acceleration with CuPy:

.. code-block:: python

   import torch
   from torchsparsegradutils.cupy import sparse_solve_c4t

   # Move tensors to GPU
   A = A.cuda()
   b = b.cuda()

   # Solve using CuPy backend
   x = sparse_solve_c4t(A, b, method='cg')

JAX Backend
~~~~~~~~~~~

For JAX integration:

.. code-block:: python

   import torch
   from torchsparsegradutils.jax import sparse_solve_j4t

   # Solve using JAX backend
   x = sparse_solve_j4t(A, b, method='cg')

Performance Tips
----------------

1. **Use CSR format** for repeated operations on the same sparsity pattern
2. **Batch operations** when possible to amortize overhead
3. **Precompute factorizations** for repeated solves with the same matrix
4. **Use appropriate tolerances** for iterative solvers
5. **Consider mixed precision** for large problems

.. code-block:: python

   # Convert to CSR for repeated operations
   A_csr = A.to_sparse_csr()

   # Batch multiple RHS vectors
   B_batch = torch.stack([b1, b2, b3], dim=-1)
   X_batch = sparse_generic_solve(A_csr, B_batch)

Next Steps
----------

- Read the :doc:`tutorials/index` for more detailed examples
- Explore the :doc:`api/index` for complete function references
- Check out the :doc:`mathematical_background` for theory
- View :doc:`benchmarks` for performance comparisons
