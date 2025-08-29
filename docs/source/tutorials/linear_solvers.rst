Linear Solvers Tutorial
=======================

This tutorial demonstrates how to use the various linear solvers in torchsparsegradutils.

Triangular Systems
------------------

For sparse triangular systems, we use :func:`~torchsparsegradutils.sparse_triangular_solve`:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_triangular_solve

   # Create a lower triangular matrix
   n = 5
   indices = []
   values = []

   # Fill lower triangle
   for i in range(n):
       for j in range(i + 1):
           indices.append([i, j])
           values.append(float(i + j + 1))  # Example values

   indices = torch.tensor(indices).T
   values = torch.tensor(values)
   L = torch.sparse_coo_tensor(indices, values, (n, n))

   # Right-hand side
   b = torch.randn(n, 3)

   # Solve Lx = b
   x = sparse_triangular_solve(L, b, upper=False)

   # Verify the solution
   residual = torch.sparse.mm(L, x) - b
   print(f"Residual norm: {torch.norm(residual)}")

Upper Triangular Systems
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create upper triangular matrix
   indices = []
   values = []

   for i in range(n):
       for j in range(i, n):
           indices.append([i, j])
           values.append(float(i + j + 1))

   indices = torch.tensor(indices).T
   values = torch.tensor(values)
   U = torch.sparse_coo_tensor(indices, values, (n, n))

   # Solve Ux = b
   x = sparse_triangular_solve(U, b, upper=True)

General Linear Systems
----------------------

For general sparse linear systems, use :func:`~torchsparsegradutils.sparse_generic_solve`:

Symmetric Positive Definite Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_generic_solve
   from torchsparsegradutils.utils.random_sparse import random_sparse_pd

   # Create a symmetric positive definite matrix
   n = 100
   density = 0.05  # 5% non-zeros
   A = random_sparse_pd(n, density)
   b = torch.randn(n, 1)

   # Solve using Conjugate Gradient
   x_cg = sparse_generic_solve(A, b, method='cg', tol=1e-8, max_iter=1000)

   print(f"CG solution shape: {x_cg.shape}")

Non-symmetric Systems
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For non-symmetric matrices, use BICGSTAB or MINRES
   A_nonsym = torch.sparse_coo_tensor(
       torch.randint(0, n, (2, int(n * n * 0.05))),
       torch.randn(int(n * n * 0.05)),
       (n, n)
   )

   # BICGSTAB for general non-symmetric matrices
   x_bicgstab = sparse_generic_solve(
       A_nonsym, b,
       method='bicgstab',
       tol=1e-6,
       max_iter=1000
   )

Least Squares Problems
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For overdetermined systems (more rows than columns)
   m, n = 200, 100
   A_rect = torch.sparse_coo_tensor(
       torch.randint(0, min(m, n), (2, int(m * n * 0.02))),
       torch.randn(int(m * n * 0.02)),
       (m, n)
   )
   b_rect = torch.randn(m, 1)

   # Use LSMR for least squares
   x_lsmr = sparse_generic_solve(
       A_rect, b_rect,
       method='lsmr',
       tol=1e-6
   )

Solver Comparison
-----------------

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Solver Comparison
   :header-rows: 1
   :widths: 20 30 25 25

   * - Method
     - Best For
     - Convergence
     - Memory
   * - CG
     - SPD matrices
     - Fast
     - Low
   * - BICGSTAB
     - General non-symmetric
     - Good
     - Medium
   * - MINRES
     - Symmetric indefinite
     - Reliable
     - Low
   * - LSMR
     - Least squares
     - Stable
     - Medium

Choosing the Right Solver
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def choose_solver(A, b):
       """Recommend solver based on matrix properties."""

       # Check if matrix is square
       if A.shape[0] != A.shape[1]:
           return 'lsmr'  # Rectangular -> least squares

       # For square matrices, check symmetry (approximately)
       A_T = A.transpose(0, 1).coalesce()
       diff = (A - A_T).coalesce()

       if torch.allclose(diff.values(), torch.zeros_like(diff.values()), atol=1e-10):
           # Symmetric matrix
           # Check if positive definite (rough estimate)
           try:
               L = torch.linalg.cholesky(A.to_dense())  # This is expensive!
               return 'cg'  # SPD
           except:
               return 'minres'  # Symmetric but not PD
       else:
           return 'bicgstab'  # Non-symmetric

Advanced Usage
--------------

Custom Linear Operators
~~~~~~~~~~~~~~~~~~~~~~~

You can also use the iterative solvers with custom linear operators:

.. code-block:: python

   from torchsparsegradutils.utils.linear_cg import cg

   def matvec(A, x):
       """Custom matrix-vector product."""
       return torch.sparse.mm(A, x)

   # Use CG with custom operator
   x_custom = cg(
       matvec_func=lambda x: matvec(A, x),
       b=b,
       x0=torch.zeros_like(b),
       tol=1e-8,
       max_iter=1000
   )

Preconditioning
~~~~~~~~~~~~~~~

For better convergence, you can use preconditioning:

.. code-block:: python

   # Simple diagonal preconditioner
   def diag_preconditioner(A):
       """Extract diagonal for preconditioning."""
       diag_indices = A.coalesce().indices()
       mask = diag_indices[0] == diag_indices[1]
       diag_values = A.coalesce().values()[mask]
       return torch.sparse_coo_tensor(
           torch.stack([torch.arange(A.shape[0]), torch.arange(A.shape[0])]),
           1.0 / (diag_values + 1e-12),  # Avoid division by zero
           A.shape
       )

   M_inv = diag_preconditioner(A)

   # Apply preconditioning: solve M^{-1}Ax = M^{-1}b
   b_precond = torch.sparse.mm(M_inv, b)

   def precond_matvec(x):
       return torch.sparse.mm(M_inv, torch.sparse.mm(A, x))

   x_precond = cg(
       matvec_func=precond_matvec,
       b=b_precond,
       x0=torch.zeros_like(b),
       tol=1e-8
   )

Error Handling and Convergence
------------------------------

.. code-block:: python

   import warnings

   try:
       x = sparse_generic_solve(A, b, method='cg', tol=1e-10, max_iter=100)
   except RuntimeError as e:
       if "convergence" in str(e).lower():
           print("Solver did not converge, trying with relaxed tolerance")
           x = sparse_generic_solve(A, b, method='cg', tol=1e-6, max_iter=500)
       else:
           print(f"Unexpected error: {e}")
           # Try a different solver
           x = sparse_generic_solve(A, b, method='bicgstab', tol=1e-6)

Next Steps
----------

- Learn about :doc:`distributions` for probabilistic operations
- Explore :doc:`backends` for GPU acceleration
- Check out :doc:`optimization_examples` for ML applications
