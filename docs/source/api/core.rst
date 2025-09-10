Core Operations
===============

This module contains the core sparse tensor operations.

Sparse Matrix Multiplication
-----------------------------

.. currentmodule:: torchsparsegradutils.sparse_matmul

.. autofunction:: sparse_mm

.. autoclass:: SparseMatMul
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: forward, backward

Indexed Matrix Multiplication
-----------------------------

.. currentmodule:: torchsparsegradutils.indexed_matmul

.. autofunction:: gather_mm

.. autofunction:: segment_mm

Sparse Linear Solvers
----------------------

.. currentmodule:: torchsparsegradutils.sparse_solve

.. autofunction:: sparse_triangular_solve

.. autoclass:: SparseTriangularSolve
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: forward, backward

.. autofunction:: sparse_generic_solve

.. autoclass:: SparseGenericSolve
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: forward, backward

Sparse Least Squares
---------------------

.. currentmodule:: torchsparsegradutils.sparse_lstsq

.. autofunction:: sparse_generic_lstsq

.. autoclass:: SparseGenericLstsq
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: forward, backward
