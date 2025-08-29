Changelog
=========

This page tracks all notable changes to torchsparsegradutils.

Version 0.2.0 (Current)
------------------------

**Major Features**

* **Big package revisions** - Complete restructuring and API improvements
* Enhanced sparse matrix multiplication with improved gradient preservation
* Expanded distribution support with multiple parameterizations
* New encoder utilities for pairwise operations
* Comprehensive benchmarking suite

**New Features**

* :func:`~torchsparsegradutils.gather_mm` - Indexed matrix multiplication
* :func:`~torchsparsegradutils.segment_mm` - Segment-based matrix operations
* Enhanced :class:`~torchsparsegradutils.distributions.SparseMultivariateNormal` with LDL parameterization
* CuPy backend integration for GPU acceleration
* JAX backend integration for cross-framework compatibility
* Built-in iterative solvers (CG, BICGSTAB, LSMR, MINRES)

**API Changes**

* Improved error handling and input validation
* Consistent API across all sparse operations
* Better support for batched operations
* Enhanced documentation and examples

**Performance Improvements**

* Optimized sparse matrix multiplication algorithms
* Reduced memory usage for gradient computation
* Improved numerical stability for iterative solvers
* Better GPU utilization

**Bug Fixes**

* Fixed gradient computation edge cases
* Resolved memory leaks in batched operations
* Corrected sparse format conversion issues
* Fixed distribution sampling numerical stability

**Documentation**

* Complete documentation overhaul with Sphinx
* Mathematical background sections
* Comprehensive tutorials and examples
* Benchmarking and performance guides
* API reference with autodoc

Version 0.1.x (Legacy)
-----------------------

Version 0.1.2
~~~~~~~~~~~~~~

**Features**

* Basic sparse matrix multiplication
* Triangular system solver
* Initial distribution support

**Bug Fixes**

* Memory efficiency improvements
* Gradient flow corrections

Version 0.1.1
~~~~~~~~~~~~~~

**Features**

* Enhanced sparse solve functionality
* Better PyTorch integration
* Basic documentation

Version 0.1.0
~~~~~~~~~~~~~~

**Initial Release**

* Core sparse matrix operations
* Basic linear system solvers
* Sparse multivariate normal distribution
* Initial test suite

Upcoming Features (Roadmap)
---------------------------

Version 0.3.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~~

* **Advanced Preconditioning**: ILU, AMG preconditioners
* **Sparse Eigensolvers**: ARPACK integration
* **Automatic Differentiation**: Enhanced AD support
* **Performance**: CUDA kernel optimizations
* **New Distributions**: Sparse Wishart, Inverse-Wishart

Version 0.4.0 (Future)
~~~~~~~~~~~~~~~~~~~~~~~

* **Distributed Computing**: Multi-GPU and cluster support
* **Advanced Algorithms**: Multigrid methods
* **Integration**: Better NumPy/SciPy compatibility
* **Specialized Solvers**: Domain-specific optimizations

Migration Guide
---------------

Upgrading from 0.1.x to 0.2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**API Changes**

The main API remains stable, but some function signatures have been enhanced:

.. code-block:: python

   # Old (0.1.x)
   from torchsparsegradutils import sparse_mm
   result = sparse_mm(A, B)

   # New (0.2.0) - same interface, enhanced functionality
   from torchsparsegradutils import sparse_mm
   result = sparse_mm(A, B)  # Now supports batching, better gradients

**New Import Structure**

.. code-block:: python

   # Access to new submodules
   from torchsparsegradutils.distributions import SparseMultivariateNormal
   from torchsparsegradutils.cupy import sparse_solve_c4t
   from torchsparsegradutils.jax import sparse_solve_j4t

**Distribution API Updates**

.. code-block:: python

   # Enhanced parameterizations available
   dist = SparseMultivariateNormal(
       loc=mu,
       precision_matrix=Lambda,
       param='precision_LDL'  # New parameterization
   )

**Deprecated Features**

* Some internal utilities have been moved or renamed
* Legacy test utilities replaced with improved versions

**Breaking Changes**

* None in public API
* Internal module structure changes may affect advanced users

Compatibility Matrix
--------------------

.. list-table:: Version Compatibility
   :header-rows: 1
   :widths: 25 25 25 25

   * - torchsparsegradutils
     - PyTorch
     - Python
     - Status
   * - 0.2.0
     - ≥2.5.0
     - ≥3.10
     - Current
   * - 0.1.2
     - ≥2.0.0
     - ≥3.8
     - Supported
   * - 0.1.1
     - ≥1.13.0
     - ≥3.8
     - Legacy
   * - 0.1.0
     - ≥1.10.0
     - ≥3.7
     - Deprecated

Development Timeline
--------------------

* **2024 Q4**: Version 0.1.0 initial release
* **2025 Q1**: Version 0.1.x stability improvements
* **2025 Q3**: Version 0.2.0 major revision (current)
* **2025 Q4**: Version 0.3.0 advanced features (planned)
* **2026 Q1**: Version 0.4.0 distributed computing (future)

Contributing to Changes
-----------------------

* **Bug fixes**: Welcome for any version
* **New features**: Target latest development branch
* **Documentation**: Always appreciated
* **Benchmarks**: Help improve performance

See :doc:`contributing` for detailed guidelines.

License Changes
---------------

* **All versions**: Apache License 2.0
* **No license changes** planned
