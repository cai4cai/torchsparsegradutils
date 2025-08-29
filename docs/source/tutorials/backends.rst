Backend Integration Tutorial
=============================

This tutorial covers how to use different computational backends (CuPy, JAX) with torchsparsegradutils.

CuPy Backend
------------

The CuPy backend provides GPU acceleration for sparse linear algebra operations using NVIDIA's cuSPARSE library.

Installation and Setup
~~~~~~~~~~~~~~~~~~~~~~~

First, ensure CuPy is installed:

.. code-block:: bash

   pip install cupy-cuda12x  # For CUDA 12.x
   # or
   pip install cupy-cuda11x  # For CUDA 11.x

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from torchsparsegradutils.cupy import sparse_solve_c4t

   # Create sparse matrix on GPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Example: tridiagonal matrix
   n = 1000
   indices = torch.tensor([
       [i for i in range(n)] + [i for i in range(n-1)] + [i for i in range(1, n)],
       [i for i in range(n)] + [i for i in range(1, n)] + [i for i in range(n-1)]
   ]).to(device)

   values = torch.tensor(
       [2.0] * n + [-1.0] * (n-1) + [-1.0] * (n-1)
   ).to(device)

   A = torch.sparse_coo_tensor(indices, values, (n, n)).to(device)
   b = torch.randn(n, 1).to(device)

   # Solve using CuPy backend
   x = sparse_solve_c4t(A, b, method='cg')

   print(f"Solution shape: {x.shape}")
   print(f"Device: {x.device}")

   # Verify solution
   residual = torch.sparse.mm(A, x) - b
   print(f"Residual norm: {torch.norm(residual)}")

Available CuPy Solvers
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Iterative solvers
   solvers = ['cg', 'cgs', 'minres', 'gmres']

   for solver in solvers:
       try:
           x = sparse_solve_c4t(A, b, method=solver, tol=1e-6)
           residual_norm = torch.norm(torch.sparse.mm(A, x) - b)
           print(f"{solver.upper()}: residual norm = {residual_norm:.2e}")
       except Exception as e:
           print(f"{solver.upper()}: failed with {e}")

.. code-block:: python

   # Direct solvers (for smaller matrices)
   if n < 5000:  # Direct solvers are memory intensive
       try:
           x_direct = sparse_solve_c4t(A, b, method='spsolve')
           print(f"Direct solve residual: {torch.norm(torch.sparse.mm(A, x_direct) - b)}")
       except Exception as e:
           print(f"Direct solve failed: {e}")

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time

   def benchmark_solver(A, b, method, backend='pytorch'):
       if backend == 'cupy':
           start = time.time()
           x = sparse_solve_c4t(A, b, method=method)
           torch.cuda.synchronize()
           end = time.time()
       else:
           from torchsparsegradutils import sparse_generic_solve
           start = time.time()
           x = sparse_generic_solve(A, b, method=method)
           torch.cuda.synchronize() if torch.cuda.is_available() else None
           end = time.time()

       return end - start, x

   # Compare PyTorch vs CuPy
   methods = ['cg', 'bicgstab']

   for method in methods:
       if method in ['cg']:  # CuPy CG
           time_cupy, x_cupy = benchmark_solver(A, b, method, 'cupy')
           print(f"CuPy {method}: {time_cupy:.4f}s")

       time_torch, x_torch = benchmark_solver(A, b, method, 'pytorch')
       print(f"PyTorch {method}: {time_torch:.4f}s")
       print()

JAX Backend
-----------

The JAX backend provides integration with Google's JAX ecosystem for high-performance computing.

Installation and Setup
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install jax[cuda12]  # For CUDA 12.x
   # or
   pip install jax[cpu]     # CPU only

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from torchsparsegradutils.jax import sparse_solve_j4t

   # JAX backend works with same PyTorch tensors
   x_jax = sparse_solve_j4t(A, b, method='cg', tol=1e-8)

   print(f"JAX solution device: {x_jax.device}")
   print(f"JAX residual: {torch.norm(torch.sparse.mm(A, x_jax) - b)}")

Available JAX Solvers
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # JAX supports fewer solvers but they're highly optimized
   jax_solvers = ['cg', 'bicgstab']

   for solver in jax_solvers:
       x_jax = sparse_solve_j4t(A, b, method=solver)
       residual = torch.norm(torch.sparse.mm(A, x_jax) - b)
       print(f"JAX {solver}: residual = {residual:.2e}")

Advanced Usage
--------------

Mixed Backend Workflows
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use different backends for different parts of computation

   # 1. Data preparation in PyTorch
   A_pytorch = torch.sparse_coo_tensor(indices, values, (n, n)).cuda()
   b_pytorch = torch.randn(n, 1).cuda()

   # 2. Linear solve with CuPy for speed
   x_cupy = sparse_solve_c4t(A_pytorch, b_pytorch, method='cg')

   # 3. Post-processing back in PyTorch
   result = torch.sparse.mm(A_pytorch, x_cupy)
   loss = torch.nn.functional.mse_loss(result, b_pytorch)

   print(f"Final loss: {loss.item()}")

Backend Selection Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def auto_select_backend(A, b, method='cg'):
       """Automatically select best backend based on problem characteristics."""

       n = A.shape[0]
       nnz = A._nnz() if A.is_sparse else n * n
       density = nnz / (n * n)

       # Decision logic
       if not torch.cuda.is_available():
           return 'pytorch'
       elif method in ['spsolve', 'spsolve_triangular']:
           return 'cupy'  # CuPy has better direct solvers
       elif n > 10000 and density < 0.01:
           return 'cupy'  # CuPy better for large sparse problems
       elif method in ['bicgstab'] and n > 5000:
           return 'jax'   # JAX has optimized BICGSTAB
       else:
           return 'pytorch'  # Default to PyTorch

   # Usage
   backend = auto_select_backend(A, b, 'cg')
   print(f"Recommended backend: {backend}")

   if backend == 'cupy':
       x = sparse_solve_c4t(A, b, method='cg')
   elif backend == 'jax':
       x = sparse_solve_j4t(A, b, method='cg')
   else:
       from torchsparsegradutils import sparse_generic_solve
       x = sparse_generic_solve(A, b, method='cg')

Gradient Computation with Backends
-----------------------------------

CuPy with Gradients
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # CuPy backend preserves gradients
   A.requires_grad_(True)
   b.requires_grad_(True)

   x = sparse_solve_c4t(A, b, method='cg')
   loss = (x ** 2).sum()
   loss.backward()

   print(f"A gradient preserved: {A.grad is not None}")
   print(f"b gradient preserved: {b.grad is not None}")

JAX with Gradients
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # JAX also preserves PyTorch gradients
   A_grad = A.clone().requires_grad_(True)
   b_grad = b.clone().requires_grad_(True)

   x_jax = sparse_solve_j4t(A_grad, b_grad, method='cg')
   loss_jax = (x_jax ** 2).sum()
   loss_jax.backward()

   print(f"JAX preserves gradients: {A_grad.grad is not None}")

Large-Scale Example
-------------------

Solving Large Sparse Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_large_sparse_system(n=50000, density=0.001):
       """Create large sparse system for benchmarking."""
       nnz = int(n * n * density)

       # Random sparse matrix
       indices = torch.randint(0, n, (2, nnz))
       values = torch.randn(nnz)

       # Make it SPD by A^T A + I
       A_rand = torch.sparse_coo_tensor(indices, values, (n, n))
       A_T = A_rand.transpose(0, 1)
       A_spd = torch.sparse.mm(A_T, A_rand)

       # Add identity for conditioning
       eye_indices = torch.stack([torch.arange(n), torch.arange(n)])
       eye_values = torch.ones(n) * 0.1
       eye = torch.sparse_coo_tensor(eye_indices, eye_values, (n, n))

       A = (A_spd + eye).cuda()
       b = torch.randn(n, 1).cuda()

       return A, b

   # Create large system
   print("Creating large sparse system...")
   A_large, b_large = create_large_sparse_system(20000, 0.0005)
   print(f"System size: {A_large.shape}, nnz: {A_large._nnz()}")

.. code-block:: python

   # Benchmark different backends
   backends = [
       ('PyTorch', lambda: sparse_generic_solve(A_large, b_large, method='cg')),
       ('CuPy', lambda: sparse_solve_c4t(A_large, b_large, method='cg')),
       ('JAX', lambda: sparse_solve_j4t(A_large, b_large, method='cg'))
   ]

   results = {}
   for name, solver_func in backends:
       try:
           start = time.time()
           x = solver_func()
           torch.cuda.synchronize()
           elapsed = time.time() - start

           # Verify accuracy
           residual = torch.norm(torch.sparse.mm(A_large, x) - b_large)

           results[name] = {
               'time': elapsed,
               'residual': residual.item(),
               'success': True
           }
           print(f"{name}: {elapsed:.2f}s, residual: {residual:.2e}")

       except Exception as e:
           results[name] = {'success': False, 'error': str(e)}
           print(f"{name}: Failed - {e}")

Error Handling and Fallbacks
-----------------------------

Robust Backend Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def robust_sparse_solve(A, b, preferred_backend='auto'):
       """Robust sparse solver with automatic fallback."""

       methods_by_backend = {
           'cupy': ['cg', 'bicgstab', 'minres'],
           'jax': ['cg', 'bicgstab'],
           'pytorch': ['cg', 'bicgstab', 'minres', 'lsmr']
       }

       if preferred_backend == 'auto':
           preferred_backend = auto_select_backend(A, b)

       backends_to_try = [preferred_backend]
       if preferred_backend != 'pytorch':
           backends_to_try.append('pytorch')  # Always fallback to PyTorch

       for backend in backends_to_try:
           for method in methods_by_backend[backend]:
               try:
                   if backend == 'cupy':
                       x = sparse_solve_c4t(A, b, method=method)
                   elif backend == 'jax':
                       x = sparse_solve_j4t(A, b, method=method)
                   else:
                       x = sparse_generic_solve(A, b, method=method)

                   # Verify solution quality
                   residual = torch.norm(torch.sparse.mm(A, x) - b)
                   if residual < 1e-3:  # Acceptable tolerance
                       return x, backend, method

               except Exception as e:
                   print(f"Failed {backend}/{method}: {e}")
                   continue

       raise RuntimeError("All solver backends failed")

   # Usage
   try:
       solution, used_backend, used_method = robust_sparse_solve(A, b)
       print(f"Success with {used_backend}/{used_method}")
   except RuntimeError as e:
       print(f"All methods failed: {e}")

Best Practices
--------------

1. **Start with PyTorch** for development and debugging
2. **Use CuPy** for production workloads with large sparse matrices
3. **Try JAX** for specialized algorithms or integration with JAX ecosystems
4. **Always verify solutions** regardless of backend
5. **Have fallback strategies** for robustness

Performance Summary
~~~~~~~~~~~~~~~~~~~

.. list-table:: Backend Performance Characteristics
   :header-rows: 1
   :widths: 20 30 25 25

   * - Backend
     - Best For
     - Pros
     - Cons
   * - PyTorch
     - Development, Small-Medium problems
     - Easy debugging, Full gradient support
     - Slower for large problems
   * - CuPy
     - Large sparse systems, Production
     - Fast cuSPARSE, Many algorithms
     - Additional dependency
   * - JAX
     - Research, Specialized algorithms
     - JIT compilation, XLA optimization
     - Limited solver selection

Next Steps
----------

- Explore :doc:`optimization_examples` using these backends
- Learn about :doc:`basic_operations` for core functionality
- Check out :doc:`linear_solvers` for algorithm details
