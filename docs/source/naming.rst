Naming and shape conventions
============================

This guide defines the terminology used for shapes, batching, sparse layouts,
and dimensions in ``torchsparsegradutils``. Use it for code, docstrings,
tests, issues, and pull requests. State the logical object first and then its
shape: sparse tensor terminology otherwise makes it easy to give different
concepts the same name.

Core terminology
----------------

Use **matrix** for a logical object with row and column axes, and use
**tensor** for the PyTorch object or for a genuinely non-matrix object. For
the non-hybrid matrix tensors used by most operations in this repository,
the row and column axes are the final two axes. Hybrid sparse tensors may
have trailing dense value dimensions.

* An unbatched sparse matrix has shape ``(n_rows, n_cols)``.
* A batched sparse matrix (a batch of sparse matrices) has shape
  ``(batch_size, n_rows, n_cols)``.
* An unbatched square sparse matrix has shape ``(n, n)``; its batched form has
  shape ``(batch_size, n, n)``.
* A dense matrix of right-hand sides has shape ``(n_rows, n_rhs)``; its
  batched form has shape ``(batch_size, n_rows, n_rhs)``.
* A vector has shape ``(n,)`` and a batch of vectors has shape
  ``(batch_size, n)``.

``2D``, ``3D``, ``rank 2``, and ``rank 3`` describe only the number of
tensor axes. They do not identify an axis as a batch, matrix, spatial, or
dense-value axis. Prefer, for example, “a batched sparse matrix with shape
``(batch_size, n_rows, n_cols)``” to “a 3D sparse tensor”.

When an API supports at most one batch axis, say so directly. Unless an API
documents an exception, batch sizes must match exactly and batch broadcasting
is not supported.

Rank, batch, sparse, and dense dimensions
------------------------------------------

These terms are not interchangeable:

* **Tensor rank** is ``tensor.ndim``, the total number of logical axes.
* A **batch dimension** indexes independent problem instances. Repository
  convention is one leading batch axis.
* A **sparse dimension** has coordinates represented by sparse indices. For
  COO, ``sparse_dim()`` is the number of leading indexed dimensions. CSR and
  CSC always have two sparse matrix dimensions.
* A **dense dimension of a sparse tensor** is a trailing dimension stored in
  each specified value. It is not the same as a dense, ``torch.strided``
  tensor. Most matrix operations here require ``dense_dim() == 0`` unless
  documented otherwise.

For example, a hybrid sparse matrix with shape
``(n_rows, n_cols, feature_size)`` can have two sparse dimensions and one
dense feature dimension. A true three-dimensional sparse object, such as a
sparse volume ``(depth, height, width)``, is different from a rank-3 tensor
whose leading axis is a batch of matrices.

COO and compressed-layout batching
----------------------------------

PyTorch represents batching differently for COO and compressed layouts, so
documentation must preserve that distinction.

For COO, a tensor of shape ``(batch_size, n_rows, n_cols)`` commonly has three
sparse dimensions. PyTorch does not label the first one as a batch dimension;
an API in this package *logically interprets* it as one. COO batch items may
have different numbers of specified entries, including zero.

For CSR and CSC, PyTorch represents tensors as
``(*batch_shape, n_rows, n_cols, *dense_shape)``. The normal form supported by
this package is ``(batch_size, n_rows, n_cols)``. Batched CSR and CSC require
the same number of specified entries in every batch item; that is a storage
constraint, not a mathematical requirement of batching. Do not generalise
this restriction to COO.

Specified-element counts in batched layouts
--------------------------------------------

``_nnz()`` is layout-specific for a batched sparse tensor. This distinction is
exercised by the sparse-matrix multiplication backward tests in
``torchsparsegradutils/tests/test_sparse_matmul.py``:

* For batched CSR, ``A.grad._nnz()`` is the number of specified entries
  **per batch element**.
* For batched COO, ``A.grad._nnz()`` is the number of specified entries in the
  **whole tensor**.

Thus, if every batch item has ``nse_per_matrix`` specified entries and the
batch size is ``batch_size``, the checked invariants are:

.. code-block:: python

   # Batched CSR: PyTorch reports a per-batch-item count.
   assert A_csr.grad._nnz() == nse_per_matrix

   # Batched COO: PyTorch reports a count over the complete tensor.
   assert A_coo.grad._nnz() == nse_per_matrix * batch_size

This does not make CSR's count more local mathematically; it describes the
storage-count API exposed by PyTorch. For an unbatched COO or CSR matrix,
``_nnz()`` is simply the count for that one matrix.

Specified entries, explicit zeros, and ``nnz``
----------------------------------------------

Prefer **specified entry** and **number of specified elements** (``nse``) when
the distinction matters. Sparse storage can contain an explicit value equal to
zero, so “number of nonzeros” can be misleading.

* An **explicit zero** is a stored value equal to zero.
* A **structural** or **implicit zero** is an absent coordinate interpreted as
  the sparse fill value.
* Existing code and benchmarks may use ``nnz``. When retained, define it as
  the count of stored/specified entries, not necessarily mathematically
  nonzero values.

Avoid calling ``values()`` “nonzero values” unless explicit zeros are known to
be absent; use “stored values” or “explicit values” instead.

Layout and index names
----------------------

Use PyTorch's layout names consistently: sparse COO (``torch.sparse_coo``),
sparse CSR (``torch.sparse_csr``), sparse CSC (``torch.sparse_csc``), sparse
BSR (``torch.sparse_bsr``), sparse BSC (``torch.sparse_bsc``), and dense or
strided (``torch.strided``). Prefer **layout** over mixing “format” and
“layout” in the same API description. “Compressed sparse layout” can refer to
CSR, CSC, BSR, and BSC collectively.

Name index arrays after the represented axis:

* COO: ``batch_indices``, ``row_indices``, and ``col_indices``.
* CSR: ``crow_indices``, ``col_indices``, and ``values``.
* CSC: ``ccol_indices``, ``row_indices``, and ``values``.

``crow_indices`` and ``ccol_indices`` are compressed pointers, not coordinate
lists. When converting them to coordinates, say **uncompress**, **expand**, or
**repeat-interleave the row/column indices**.

Shape symbols and operations
----------------------------

Within a function or document, choose one scheme and keep it consistent:
``batch_size`` (or ``b``), ``nrows``/``n_rows`` (or ``M``),
``ncols``/``n_cols`` (or ``N``), and ``nrhs``/``n_rhs`` (or ``K``). Short local
names such as ``b, nrows, ncols`` are appropriate in compact numerical code.

For multiplication, use:

.. code-block:: text

   A: (n_rows, n_inner) or (batch_size, n_rows, n_inner)
   B: (n_inner, n_cols) or (batch_size, n_inner, n_cols)
   C: (n_rows, n_cols) or (batch_size, n_rows, n_cols)

For solves, use:

.. code-block:: text

   A: (n, n) or (batch_size, n, n)
   B: (n, n_rhs) or (batch_size, n, n_rhs)
   X: same shape as B

Call ``B`` the **right-hand side**; its final axis is the number of
right-hand sides, not a batch dimension. For an unbatched matrix, ``A.T``
transposes the matrix axes. For a batched matrix, use ``A.mT`` or
``A.transpose(-2, -1)`` to transpose the final two matrix axes while preserving
the leading batch axis. Do not use ``A.T`` for batched matrices because it
reverses all tensor axes.

Reduction, distribution, and memory wording
--------------------------------------------

For ``A`` with shape ``(n_rows, n_cols)``, “reduce over rows” means reduce
axis 0 and produce one value per column; “reduce over columns” means reduce
axis 1 and produce one value per row. Avoid “row-wise” and “column-wise”
unless the reduced and retained axes are both stated.

For ``torch.distributions``-compatible code, use
``sample_shape + batch_shape + event_shape``. Sample dimensions are not batch
dimensions. Spatial axes, channels, and batch axes are also distinct; use
``spatial_dims``, ``spatial_shape``, ``num_channels``, and ``batch_size`` as
appropriate and document a concrete volume ordering such as
``(depth, height, width)``.

Use **view** only when storage is shared, **copy** when storage is newly
allocated, and **reshape** when either may occur. “Materialise dense” means
constructing a full dense tensor, for example with ``to_dense()``. Do not claim
that an operation has no allocation merely because an input was expanded:
``stack``, ``cat``, ``clone``, ``contiguous``, and output creation may allocate.

COO coalescing and validation messages
--------------------------------------

Only COO tensors are **coalesced** or **uncoalesced**. Coalesced COO has unique,
sorted coordinates; uncoalesced COO can have duplicate coordinates that sum at
the same logical position. Coalesce before nonlinear operations on values
unless the implementation has proved an equivalent treatment of duplicates.
Do not apply this terminology to CSR or CSC.

Shape-validation errors should describe the accepted logical forms as well as
the received shape:

.. code-block:: python

   raise ValueError(
       "A must be an unbatched sparse matrix with shape (n_rows, n_cols) "
       "or a batched sparse matrix with shape (batch_size, n_rows, n_cols); "
       f"got shape {tuple(A.shape)}."
   )

Before submitting shape-related changes, make sure public inputs and outputs
state explicit shapes; distinguish batch, matrix, spatial, channel, sample,
and event axes; and separate COO-specific from compressed-layout-specific
constraints.

References
----------

* `PyTorch sparse tensor documentation <https://docs.pytorch.org/docs/stable/sparse.html>`_
* `PyTorch distributions documentation <https://docs.pytorch.org/docs/stable/distributions.html>`_
* `PyTorch tensor views documentation <https://docs.pytorch.org/docs/stable/tensor_view.html>`_
