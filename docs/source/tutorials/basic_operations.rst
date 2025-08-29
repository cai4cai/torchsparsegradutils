Basic Sparse Operations
=======================

This tutorial covers the fundamental sparse operations in torchsparsegradutils.

Creating Sparse Tensors
------------------------

First, let's create some sparse tensors to work with:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   # Create a simple COO sparse matrix
   indices = torch.tensor([[0, 1, 1, 2],
                           [0, 1, 2, 2]])
   values = torch.tensor([1., 2., 3., 4.])
   A_coo = torch.sparse_coo_tensor(indices, values, (3, 3))

   print("COO matrix:")
   print(A_coo.to_dense())

   # Convert to CSR format
   A_csr = A_coo.to_sparse_csr()
   print("\\nCSR format - same matrix:")
   print(A_csr.to_dense())

Memory-Efficient Sparse Matrix Multiplication
----------------------------------------------

The main advantage of :func:`~torchsparsegradutils.sparse_mm` over PyTorch's native sparse operations is gradient sparsity preservation:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   # Create sparse matrix A and dense matrix B
   A = torch.sparse_coo_tensor(
       [[0, 1, 2], [1, 2, 0]],
       [1., 2., 3.],
       (3, 3),
       requires_grad=True
   )

   B = torch.randn(3, 4, requires_grad=True)

   # Using torchsparsegradutils
   C = sparse_mm(A, B)
   loss = C.sum()
   loss.backward()

   print("A gradient sparsity preserved:", A.grad.is_sparse)
   print("A gradient nnz:", A.grad._nnz())
   print("Original A nnz:", A._nnz())

Compare this with PyTorch's native operation:

.. code-block:: python

   # Reset gradients
   A.grad = None
   B.grad = None

   # Using PyTorch native (converts gradient to dense)
   C_native = torch.sparse.mm(A, B)
   loss_native = C_native.sum()
   loss_native.backward()

   print("Native PyTorch - A gradient sparsity:", A.grad.is_sparse)
   # This will be False, showing dense gradients

Batched Operations
------------------

torchsparsegradutils supports efficient batched sparse operations:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm

   batch_size = 4
   n, m, k = 100, 100, 50

   # Create batched sparse matrices
   # Each matrix in the batch has the same sparsity pattern
   indices = torch.randint(0, n, (2, 200))  # 200 non-zeros per matrix
   batch_indices = torch.stack([
       torch.cat([torch.full((1, 200), i), indices])
       for i in range(batch_size)
   ], dim=1).view(3, -1)

   values = torch.randn(batch_size * 200)
   A_batch = torch.sparse_coo_tensor(
       batch_indices, values, (batch_size, n, m)
   )

   B_batch = torch.randn(batch_size, m, k)

   # Batched multiplication
   C_batch = sparse_mm(A_batch, B_batch)
   print(f"Batch result shape: {C_batch.shape}")
   print(f"Expected shape: ({batch_size}, {n}, {k})")

Working with Different Sparse Formats
--------------------------------------

Understanding when to use COO vs CSR formats:

.. code-block:: python

   import torch
   import time

   # Create the same sparse matrix in both formats
   n = 1000
   nnz = 5000

   indices = torch.randint(0, n, (2, nnz))
   values = torch.randn(nnz)

   A_coo = torch.sparse_coo_tensor(indices, values, (n, n))
   A_csr = A_coo.to_sparse_csr()

   B = torch.randn(n, 100)

   # Timing comparison
   def time_operation(A, B, name, iterations=10):
       torch.cuda.synchronize() if torch.cuda.is_available() else None
       start = time.time()
       for _ in range(iterations):
           result = sparse_mm(A, B)
       torch.cuda.synchronize() if torch.cuda.is_available() else None
       end = time.time()
       print(f"{name}: {(end-start)/iterations:.4f}s per operation")

   time_operation(A_coo, B, "COO format")
   time_operation(A_csr, B, "CSR format")

Handling Edge Cases
-------------------

Empty Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Empty sparse matrix
   A_empty = torch.sparse_coo_tensor(
       torch.zeros(2, 0, dtype=torch.long),
       torch.zeros(0),
       (5, 5)
   )

   B = torch.randn(5, 3)
   result = sparse_mm(A_empty, B)
   print("Empty sparse matrix result:")
   print(result)  # Should be all zeros

Very Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Matrix with only one non-zero element
   A_minimal = torch.sparse_coo_tensor(
       [[0], [0]], [5.0], (10, 10)
   )

   B = torch.ones(10, 1)
   result = sparse_mm(A_minimal, B)
   print("Minimal sparse matrix result:")
   print(result.flatten())

Mixed Precision
---------------

For large-scale problems, consider mixed precision:

.. code-block:: python

   # Create matrices in half precision
   A_half = torch.sparse_coo_tensor(
       indices, values.half(), (n, n)
   ).half()
   B_half = B.half()

   # Computation in half precision
   result_half = sparse_mm(A_half, B_half)

   # Convert back to full precision if needed
   result_full = result_half.float()

   print(f"Memory saved: {A_half.values().nbytes / values.nbytes:.2f}x")

Performance Tips
----------------

1. **Use CSR format for repeated operations** with the same sparsity pattern
2. **Batch operations** when possible to amortize overhead
3. **Consider mixed precision** for memory-constrained scenarios
4. **Preallocate output tensors** when shape is known

.. code-block:: python

   # Good practice: convert once, use many times
   A_csr = A_coo.to_sparse_csr()

   results = []
   for i in range(100):
       B_i = generate_random_matrix()  # Your data
       result = sparse_mm(A_csr, B_i)
       results.append(result)

Next Steps
----------

- Learn about :doc:`linear_solvers` for solving sparse systems
- Explore :doc:`distributions` for probabilistic operations
- Check out :doc:`backends` for GPU acceleration
