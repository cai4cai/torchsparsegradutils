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
   indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
   values = torch.tensor([3., 4., 5.])
   A = torch.sparse_coo_tensor(indices, values, (2, 3)).coalesce()
   A.requires_grad_(True)

   # Create a dense matrix
   B = torch.randn(3, 4, requires_grad=True)

   # Perform sparse matrix multiplication with gradient support
   result = sparse_mm(A, B)
   print(f"Result shape: {result.shape}")  # torch.Size([2, 4])

   # The operation preserves sparsity in gradients
   loss = result.sum()
   loss.backward()
   print(f"A gradient is sparse: {A.grad.is_sparse}")  # True
   print(f"A gradient nnz: {A.grad._nnz()}")  # Number of non-zeros

COO coalescing policy
~~~~~~~~~~~~~~~~~~~~~

``sparse_mm(A, B, *, assume_unique=False)`` requires a coalesced COO input
when gradient recording is enabled and ``A.requires_grad`` is true. Otherwise,
uncoalesced inputs are accepted, including for inference and B-only gradients
when ``A.requires_grad`` is false.
This check is disabled under ``torch.no_grad()`` and ``torch.inference_mode()``.
Inference mode takes precedence even with ``torch.enable_grad()`` nested inside it.

Compared with earlier releases, an uncoalesced trainable COO input now raises
by default: call ``A.coalesce()`` first, or pass ``assume_unique=True`` only when
coordinates are known to be unique.

Pass ``A.coalesce()`` explicitly when A contains duplicate coordinates. To train
the original values independently, keep them as the leaf parameter:

.. code-block:: python

   values = torch.tensor([2., 3.], requires_grad=True)
   indices = torch.tensor([[0, 0], [0, 0]])
   A = torch.sparse_coo_tensor(indices, values, (1, 1)).coalesce()
   sparse_mm(A, torch.tensor([[4.]])).sum().backward()
   # values.grad is tensor([4., 4.])

Rebuild A from trainable values for each new training step. For a frozen operator,
coalesce once and reuse it. If coordinates are already unique but unsorted,
``assume_unique=True`` skips the check. This is an unchecked promise: duplicates
can give incorrect A gradients. CSR inputs do not need this option.

The returned A gradient preserves the stored indices, entry count and order of
the actual input to ``sparse_mm``, including for batched COO. This does not promise
to preserve duplicates in an earlier tensor that was coalesced before the call.
Neither option modifies the input. Native kernels may still sort internally.
For COO batches with more than one item, block-diagonal assembly coalesces
extracted items that are not flagged as coalesced. Extracted items may lose that
flag, so pre-coalescing a batch may not avoid further per-item coalescing.
Broader batching performance and memory work remains separate.

Batched Operations
~~~~~~~~~~~~~~~~~~

torchsparsegradutils supports batched operations, for sparse_mm and triangular_solve:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   # Create batched sparse matrices
   batch_size = 2

   # Method 1: Stack individual sparse matrices
   A1 = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1., 2.], (2, 2))
   A2 = torch.sparse_coo_tensor([[0, 1], [1, 0]], [3., 4.], (2, 2))
   A_batch = torch.stack([A1, A2]).coalesce()  # Shape: (2, 2, 2)

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
   from torchsparsegradutils.utils.random_sparse import rand_sparse_tri

   # Create a sparse lower triangular matrix
   L = rand_sparse_tri((3, 3), nnz=5, upper=False, layout=torch.sparse_csr)

   # Right-hand side
   b = torch.randn(3, 2)

   # Solve Lx = b
   x = sparse_triangular_solve(L, b, upper=False)

   # Verify solution (should be close to zero)
   from torchsparsegradutils import sparse_mm
   residual = sparse_mm(L, x) - b
   print(f"Residual norm: {torch.linalg.vector_norm(residual):.6f}")

General Linear Systems
~~~~~~~~~~~~~~~~~~~~~~

For general sparse linear systems, use :func:`~torchsparsegradutils.sparse_generic_solve`:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_generic_solve, sparse_mm
   from torchsparsegradutils.utils.random_sparse import make_spd_sparse
   from torchsparsegradutils.utils import minres

   # Create a sparse symmetric positive definite matrix
   A_sparse, A_dense = make_spd_sparse(10, torch.sparse_csr, torch.float32, torch.int64, 'cpu')
   b = torch.randn(10)

   # Solve using MINRES solver
   x_minres = sparse_generic_solve(A_sparse, b, solve=minres, tol=1e-6)
   print(f"MINRES solution shape: {x_minres.shape}")

   # Verify solution
   residual = A_sparse @ x_minres - b
   print(f"MINRES residual norm: {torch.linalg.vector_norm(residual):.6f}")

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
   loc = torch.zeros(dim)

   # LDL^T parameterization (recommended for stability)
   diagonal = torch.ones(dim) * 0.5
   scale_tril = rand_sparse_tri((dim, dim), nnz=15, upper=False,
                               layout=torch.sparse_coo, strict=False)

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
   loc = torch.zeros(dim, requires_grad=True)
   diagonal = torch.ones(dim, requires_grad=True)
   scale_tril = rand_sparse_tri((dim, dim), nnz=15, upper=False,
                               layout=torch.sparse_coo, strict=True)
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

   print(f"Location gradient norm: {torch.linalg.vector_norm(loc.grad):.6f}")
   print(f"Diagonal gradient norm: {torch.linalg.vector_norm(diagonal.grad):.6f}")
   print(f"Scale gradient nnz: {scale_tril.grad._nnz()}")

Working with Different Backends
-------------------------------

CuPy Backend (GPU Acceleration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration with CuPy:

.. code-block:: python

   import torch
   from torchsparsegradutils.cupy import sparse_solve_c4t
   from torchsparsegradutils.utils.random_sparse import make_spd_sparse

   # Create matrices on GPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   A_sparse, A_dense = make_spd_sparse(50, torch.sparse_csr, torch.float32, torch.int64, device)
   b = torch.randn(50, device=device)

   # Solve using CuPy backend (only works on GPU)
   if device.type == 'cuda':
       x = sparse_solve_c4t(A_sparse, b, solve='cg', tol=1e-6)
       print(f"CuPy solution on GPU: {x.shape}")
   else:
       print("CUDA not available, skipping CuPy example")

JAX Backend
~~~~~~~~~~~

For JAX integration:

.. code-block:: python

   import torch
   from torchsparsegradutils.utils.random_sparse import make_spd_sparse

   try:
       from torchsparsegradutils.jax import sparse_solve_j4t
       from jax.scipy.sparse.linalg import cg

       # Create test matrices
       A_sparse, A_dense = make_spd_sparse(20, torch.sparse_csr, torch.float32, torch.int64, 'cpu')
       b = torch.randn(20)

       # Solve using JAX backend
       x = sparse_solve_j4t(A_sparse, b, solve=cg, tol=1e-6)
       print(f"JAX solution shape: {x.shape}")

   except ImportError:
       print("JAX not available, skipping JAX example")

Next Steps
----------

- Explore the :doc:`api/index` for complete function references
- View :doc:`benchmarks` for performance comparisons
- See :doc:`contributing` for development setup with devcontainers
