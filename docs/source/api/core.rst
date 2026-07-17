Core Operations
===============

The core sparse tensor operations. Each public function wraps a native CUDA
kernel registered as a ``torch.library`` custom op in the ``tsgu::``
namespace; forward and backward both run on the GPU, and gradients with
respect to a sparse input keep that input's sparsity pattern and layout.

Sparse Matrix Multiplication
-----------------------------

.. currentmodule:: torchsparsegradutils.ops.matmul

.. autofunction:: sparse_mm

Indexed Matrix Multiplication
-----------------------------

.. currentmodule:: torchsparsegradutils.ops.indexed_matmul

.. autofunction:: gather_mm

.. autofunction:: segment_mm

Sparse Reductions
-----------------

.. currentmodule:: torchsparsegradutils.ops.logsumexp

.. autofunction:: sparse_logsumexp

.. autofunction:: sparse_bidir_logsumexp

Sparse Linear Solvers
----------------------

.. currentmodule:: torchsparsegradutils.ops.triangular_solve

.. autofunction:: sparse_triangular_solve

.. currentmodule:: torchsparsegradutils.ops.generic_solve

.. autofunction:: sparse_generic_solve

Sparse Least Squares
---------------------

.. currentmodule:: torchsparsegradutils.ops.lstsq

.. autofunction:: sparse_generic_lstsq
