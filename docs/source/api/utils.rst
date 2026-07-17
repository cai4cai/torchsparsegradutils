Utility Functions
=================

Utility functions for sparse tensor operations, and the built-in iterative
solvers. Everything below is re-exported from ``torchsparsegradutils.utils``.

Iterative Solvers
------------------

Host-side Krylov loops whose matrix-vector products run on the native CUDA
kernels. Each solver can be passed to
:func:`~torchsparsegradutils.sparse_generic_solve` /
:func:`~torchsparsegradutils.sparse_generic_lstsq` via their ``solve`` /
``lstsq`` arguments.

.. automodule:: torchsparsegradutils.solvers.bicgstab
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchsparsegradutils.solvers.cg
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchsparsegradutils.solvers.lsmr
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchsparsegradutils.solvers.minres
   :members:
   :undoc-members:
   :show-inheritance:

Sparse Utilities and Layout Conversion
---------------------------------------

.. automodule:: torchsparsegradutils.utils.convert
   :members:
   :undoc-members:
   :show-inheritance:

Random Sparse Tensor Generation
--------------------------------

.. automodule:: torchsparsegradutils.utils.random_sparse
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Distribution Helpers
---------------------------------

.. automodule:: torchsparsegradutils.utils.dist_stats_helpers
   :members:
   :undoc-members:
   :show-inheritance:
