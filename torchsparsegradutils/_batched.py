"""BatchedCSR / BatchedCSC descriptors — the canonical internal sparse format.

Architecture: architecture.md §3. Vocabulary: naming.md §2 (descriptor, folded
row, local column, ``nse_total``/``nse_per_item``, ``B = 1`` encodes
unbatched). This module is private: nothing outside the front package
re-exports it, and no op wires it in yet — that starts in later commits
(``torch.library`` op schemas take plain dense tensors, per architecture §2;
the descriptor is how those tensors get assembled/disassembled from user
layouts).

One frozen descriptor replaces the block-diag batching hack everywhere, with
ragged ``nse`` per batch item as first-class:

- ``values (nse_total,)`` · ``rowptr (B·n+1,)`` absolute over folded rows
  ``row_global = b · n + r`` · ``col (nse_total,)`` **local** columns in
  ``[0, m)`` · ``shape (B, n, m)``; ``B = 1`` encodes unbatched.
- Local (unoffset) columns keep int32 viable and let kernels address
  ``Bdense[b, col, :]`` directly; the batch of an entry recovers as
  ``row_global // n``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import torch

from torchsparsegradutils.utils.convert import _compress_row_indices

# int32 index arrays can address at most 2**31 - 1 positions; the eligibility
# bound (architecture.md §3 / naming.md §2) checks nse_total, rowptr length,
# and n_cols all stay strictly under this.
_INT32_BOUND = 2**31

_ACCEPTED_LOGICAL_FORMS = (
    "a 2D sparse matrix (n_rows, n_cols) in COO or CSR layout, a batched sparse "
    "matrix (batch_size, n_rows, n_cols) as 3D COO (leading sparse dim logically "
    "interpreted as batch, ragged nse per item allowed) or as batched CSR (equal "
    "nse per item), or a list of 2D CSR sparse matrices (ragged nse per item "
    "allowed)"
)


def _invalid_source(received: str) -> ValueError:
    """naming.md §1 error template: accepted logical forms + received shape/layout."""
    return ValueError(f"BatchedCSR.from_torch/to_torch accepts {_ACCEPTED_LOGICAL_FORMS}; got {received}.")


def _resolve_index_dtype(source_dtype: torch.dtype, *, nse_total: int, rowptr_len: int, n_cols: int) -> torch.dtype:
    """Apply the int32 storage rule (architecture.md §3): keep int32 only when
    the source already used int32 *and* max(nse_total, rowptr_len, n_cols) is
    still below 2**31 — int64 sources are never downcast, and eligible int32
    sources are never silently upcast to int64.
    """
    if source_dtype == torch.int64:
        return torch.int64
    if source_dtype != torch.int32:
        raise TypeError(f"Unsupported index dtype {source_dtype}; expected torch.int32 or torch.int64.")
    bound = max(nse_total, rowptr_len, n_cols)
    return torch.int32 if bound < _INT32_BOUND else torch.int64


def _fold_coo_to_csr(
    batch: torch.Tensor,
    row_local: torch.Tensor,
    col_local: torch.Tensor,
    values: torch.Tensor,
    *,
    batch_size: int,
    n_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort scattered (batch, row_local, col_local) coordinates and compress
    them into a folded CSR ``(rowptr, col, values)`` triple (kernel replaces
    these internals in commit 19). Callers must guarantee no duplicate
    coordinates (a valid CSR/CSC pattern has none) — with no duplicates
    present this is exactly a stable lexicographic sort by
    ``(row_global, col_local)``, which is why it's implemented as one below
    rather than reused from ``utils.convert``'s ``_sort_coo_indices``: that
    helper sorts via ``torch.unique(..., return_inverse=True)``, whose output
    is dedup'd and therefore data-dependent in *size* — commit 15
    (spec/commit.md Phase 3; tsgu::spmm's gradB, the first real caller of
    this lazy member, torchsparsegradutils/ops/matmul.py) surfaced that as a
    ``GuardOnDataDependentSymNode`` failure under opcheck's AOTAutograd
    dynamic-shape test, since the folded row/col arrays here are themselves
    already data-dependent-length (derived from another row's ``rowptr``
    diffs). A two-pass **stable** ``argsort`` (secondary key first, then
    primary — the standard multi-key stable-sort composition) produces the
    identical total order without ever changing the array's length, so no
    unbacked symint is created.
    """
    row_global = batch * n_rows + row_local
    permutation_secondary = torch.argsort(col_local, stable=True)
    row_global_by_secondary = row_global[permutation_secondary]
    permutation_primary = torch.argsort(row_global_by_secondary, stable=True)
    permutation = permutation_secondary[permutation_primary]

    row_global_sorted = row_global[permutation]
    col_sorted = col_local[permutation]
    # _validate=False: row_global_sorted is derived from already-validated
    # batch/row_local inputs (never raw user input) -- see
    # _compress_row_indices's "Notes on _validate" for why this also avoids
    # a GuardOnDataDependentSymNode under dynamic-shape tracing.
    rowptr = _compress_row_indices(row_global_sorted, batch_size * n_rows, _validate=False)
    return rowptr, col_sorted, values[permutation]


@dataclass(frozen=True)
class BatchedCSR:
    """Canonical internal CSR descriptor (architecture.md §3).

    A *descriptor* — never call it a tensor; it holds tensors (naming.md §2).

    Attributes
    ----------
    values : torch.Tensor
        Stored values, shape ``(nse_total,)``.
    rowptr : torch.Tensor
        Absolute CSR pointer over **folded rows** ``row_global = b * n_rows + r``,
        shape ``(batch_size * n_rows + 1,)``.
    col : torch.Tensor
        **Local** column indices in ``[0, n_cols)``, shape ``(nse_total,)``.
    shape : tuple of int
        ``(batch_size, n_rows, n_cols)``. ``batch_size == 1`` encodes an
        unbatched matrix — code must never branch on "has batch dim".
    """

    values: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    shape: Tuple[int, int, int]

    def __post_init__(self) -> None:
        batch_size, n_rows, _n_cols = self.shape
        expected_rowptr_len = batch_size * n_rows + 1
        if tuple(self.rowptr.shape) != (expected_rowptr_len,):
            raise ValueError(
                "BatchedCSR.rowptr must be absolute over folded rows with shape "
                f"(batch_size * n_rows + 1,) = ({expected_rowptr_len},) for shape {self.shape}; "
                f"got {tuple(self.rowptr.shape)}."
            )
        if self.col.shape[0] != self.values.shape[0]:
            raise ValueError(
                "BatchedCSR.col and BatchedCSR.values must both have length nse_total; "
                f"got col {tuple(self.col.shape)} and values {tuple(self.values.shape)}."
            )
        if self.rowptr.dtype != self.col.dtype:
            raise ValueError(
                f"BatchedCSR.rowptr and BatchedCSR.col must share one index dtype; "
                f"got rowptr {self.rowptr.dtype} and col {self.col.dtype}."
            )
        if self.rowptr.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"BatchedCSR index dtype must be torch.int32 or torch.int64; got {self.rowptr.dtype}.")

    @property
    def index_dtype(self) -> torch.dtype:
        return self.rowptr.dtype

    @property
    def nse_total(self) -> int:
        return self.values.shape[0]

    @functools.cached_property
    def row_indices(self) -> torch.Tensor:
        """Uncompressed folded row index (``row_global = b * n_rows + r``) per
        specified entry — expanded/repeat-interleaved from ``rowptr``, never
        "decompressed rows" (naming.md §1)."""
        batch_size, n_rows, _n_cols = self.shape
        return torch.repeat_interleave(
            torch.arange(batch_size * n_rows, dtype=self.rowptr.dtype, device=self.rowptr.device),
            self.rowptr[1:] - self.rowptr[:-1],
        )

    @functools.cached_property
    def transposed(self) -> "BatchedCSC":
        """The BatchedCSC of the same logical matrix (gradB, dim=-2 reductions)."""
        batch_size, n_rows, n_cols = self.shape
        row_global = self.row_indices
        batch = row_global // n_rows
        row_local = row_global % n_rows
        colptr, row, values = _fold_coo_to_csr(
            batch, self.col, row_local, self.values, batch_size=batch_size, n_rows=n_cols
        )
        return BatchedCSC(
            values=values,
            colptr=colptr.to(self.index_dtype),
            row=row.to(self.index_dtype),
            shape=self.shape,
        )

    @functools.cached_property
    def plans(self) -> dict:
        # Placeholder: real SpSM analysis plans land in commit 16 (kernels.md
        # open Q3) — the plan cache lives on the descriptor so its lifetime is
        # the descriptor's lifetime; no global cache, no hashing.
        return {}

    @classmethod
    def from_torch(cls, tensor: Union[torch.Tensor, Sequence[torch.Tensor]]) -> "BatchedCSR":
        """Build a BatchedCSR from one of the accepted logical forms: 2D CSR
        (zero-copy), 2D COO, 3D batched COO (ragged nse OK, coalesced),
        torch batched CSR (equal nse per item), or a list of 2D CSR tensors
        (ragged-native)."""
        if isinstance(tensor, (list, tuple)):
            return cls._from_csr_list(tensor)
        if not isinstance(tensor, torch.Tensor):
            raise _invalid_source(f"object of type {type(tensor)!r}")
        if tensor.layout == torch.sparse_csr:
            if tensor.ndim == 2:
                return cls._from_csr_2d(tensor)
            if tensor.ndim == 3:
                return cls._from_csr_batched(tensor)
            raise _invalid_source(f"a CSR tensor of shape {tuple(tensor.shape)}")
        if tensor.layout == torch.sparse_coo:
            if tensor.ndim == 2:
                return cls._from_coo_2d(tensor)
            if tensor.ndim == 3:
                return cls._from_coo_batched(tensor)
            raise _invalid_source(f"a COO tensor of shape {tuple(tensor.shape)}")
        raise _invalid_source(f"a tensor with layout {tensor.layout}")

    @classmethod
    def _from_csr_2d(cls, tensor: torch.Tensor) -> "BatchedCSR":
        n_rows, n_cols = tensor.shape
        return cls(
            values=tensor.values(), rowptr=tensor.crow_indices(), col=tensor.col_indices(), shape=(1, n_rows, n_cols)
        )

    @classmethod
    def _from_coo_2d(cls, tensor: torch.Tensor) -> "BatchedCSR":
        n_rows, n_cols = tensor.shape
        coalesced = tensor if tensor.is_coalesced() else tensor.coalesce()
        indices = coalesced.indices()
        row, col = indices[0], indices[1]
        index_dtype = _resolve_index_dtype(indices.dtype, nse_total=row.shape[0], rowptr_len=n_rows + 1, n_cols=n_cols)
        row = row.to(index_dtype)
        col = col.to(index_dtype)
        rowptr = _compress_row_indices(row, n_rows)
        return cls(values=coalesced.values(), rowptr=rowptr, col=col, shape=(1, n_rows, n_cols))

    @classmethod
    def _from_coo_batched(cls, tensor: torch.Tensor) -> "BatchedCSR":
        batch_size, n_rows, n_cols = tensor.shape
        coalesced = tensor if tensor.is_coalesced() else tensor.coalesce()
        indices = coalesced.indices()
        batch, row, col = indices[0], indices[1], indices[2]
        index_dtype = _resolve_index_dtype(
            indices.dtype, nse_total=row.shape[0], rowptr_len=batch_size * n_rows + 1, n_cols=n_cols
        )
        batch = batch.to(index_dtype)
        row = row.to(index_dtype)
        col = col.to(index_dtype)
        row_global = batch * n_rows + row
        rowptr = _compress_row_indices(row_global, batch_size * n_rows)
        return cls(values=coalesced.values(), rowptr=rowptr, col=col, shape=(batch_size, n_rows, n_cols))

    @classmethod
    def _from_csr_batched(cls, tensor: torch.Tensor) -> "BatchedCSR":
        batch_size, n_rows, n_cols = tensor.shape
        crow = tensor.crow_indices()
        col = tensor.col_indices()
        values = tensor.values()
        nse_per_item = col.shape[-1]
        nse_total = batch_size * nse_per_item
        index_dtype = _resolve_index_dtype(
            crow.dtype, nse_total=nse_total, rowptr_len=batch_size * n_rows + 1, n_cols=n_cols
        )
        crow = crow.to(index_dtype)
        col = col.to(index_dtype)
        offsets = (torch.arange(batch_size, dtype=index_dtype, device=crow.device) * nse_per_item).unsqueeze(1)
        tail = torch.tensor([nse_total], dtype=index_dtype, device=crow.device)
        rowptr = torch.cat([(crow[:, :-1] + offsets).reshape(-1), tail])
        return cls(values=values.reshape(-1), rowptr=rowptr, col=col.reshape(-1), shape=(batch_size, n_rows, n_cols))

    @classmethod
    def _from_csr_list(cls, tensors: Sequence[torch.Tensor]) -> "BatchedCSR":
        if len(tensors) == 0:
            raise _invalid_source("an empty list")
        if not all(isinstance(t, torch.Tensor) and t.layout == torch.sparse_csr and t.ndim == 2 for t in tensors):
            raise _invalid_source("a list containing a non-2D-CSR element")
        shape0 = tuple(tensors[0].shape)
        if not all(tuple(t.shape) == shape0 for t in tensors):
            raise _invalid_source(f"a list of CSR matrices with mismatched shapes {[tuple(t.shape) for t in tensors]}")
        index_dtype0 = tensors[0].crow_indices().dtype
        if not all(t.crow_indices().dtype == index_dtype0 for t in tensors):
            raise _invalid_source("a list of CSR matrices with mismatched index dtypes")

        n_rows, n_cols = shape0
        batch_size = len(tensors)
        nse_per_item = [t.col_indices().shape[0] for t in tensors]
        nse_total = sum(nse_per_item)
        index_dtype = _resolve_index_dtype(
            index_dtype0, nse_total=nse_total, rowptr_len=batch_size * n_rows + 1, n_cols=n_cols
        )

        offsets = [0]
        for nse in nse_per_item[:-1]:
            offsets.append(offsets[-1] + nse)

        device = tensors[0].device
        rowptr_parts = [t.crow_indices()[:-1].to(index_dtype) + offset for t, offset in zip(tensors, offsets)]
        tail = torch.tensor([nse_total], dtype=index_dtype, device=device)
        rowptr = torch.cat(rowptr_parts + [tail])
        col = torch.cat([t.col_indices().to(index_dtype) for t in tensors])
        values = torch.cat([t.values() for t in tensors])
        return cls(values=values, rowptr=rowptr, col=col, shape=(batch_size, n_rows, n_cols))

    def to_torch(self, *, like: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reconstruct the layout/batching form of ``like`` (COO in → COO out,
        batched CSR → batched CSR, list-of-CSR → list) from this descriptor."""
        if isinstance(like, (list, tuple)):
            return self._to_csr_list(expected_len=len(like))
        if not isinstance(like, torch.Tensor):
            raise _invalid_source(f"object of type {type(like)!r}")
        if like.layout == torch.sparse_csr:
            if like.ndim == 2:
                return self._to_csr_2d()
            if like.ndim == 3:
                return self._to_csr_batched()
            raise _invalid_source(f"a CSR tensor of shape {tuple(like.shape)}")
        if like.layout == torch.sparse_coo:
            if like.ndim == 2:
                return self._to_coo_2d()
            if like.ndim == 3:
                return self._to_coo_batched()
            raise _invalid_source(f"a COO tensor of shape {tuple(like.shape)}")
        raise _invalid_source(f"a tensor with layout {like.layout}")

    def _require_unbatched(self, form: str) -> None:
        batch_size = self.shape[0]
        if batch_size != 1:
            raise ValueError(f"BatchedCSR.to_torch(like=<{form}>) requires batch_size == 1; got {batch_size}.")

    def _to_csr_2d(self) -> torch.Tensor:
        self._require_unbatched("2D CSR")
        _batch_size, n_rows, n_cols = self.shape
        return torch.sparse_csr_tensor(self.rowptr, self.col, self.values, (n_rows, n_cols))

    def _to_coo_2d(self) -> torch.Tensor:
        self._require_unbatched("2D COO")
        _batch_size, n_rows, n_cols = self.shape
        indices = torch.stack([self.row_indices, self.col], dim=0)
        return torch.sparse_coo_tensor(indices, self.values, (n_rows, n_cols), is_coalesced=True)

    def _to_coo_batched(self) -> torch.Tensor:
        batch_size, n_rows, n_cols = self.shape
        row_global = self.row_indices
        batch = row_global // n_rows
        row_local = row_global % n_rows
        indices = torch.stack([batch, row_local, self.col], dim=0)
        return torch.sparse_coo_tensor(indices, self.values, (batch_size, n_rows, n_cols), is_coalesced=True)

    def _to_csr_batched(self) -> torch.Tensor:
        batch_size, n_rows, n_cols = self.shape
        positions = torch.arange(0, batch_size + 1, device=self.rowptr.device) * n_rows
        boundary = self.rowptr[positions]
        nse_per_item_arr = boundary[1:] - boundary[:-1]
        if batch_size > 0 and not torch.all(nse_per_item_arr == nse_per_item_arr[0]):
            raise ValueError(
                "BatchedCSR.to_torch(like=<batched CSR>) requires equal nse per batch item "
                f"(torch batched CSR storage constraint, naming.md §1); got ragged nse "
                f"{nse_per_item_arr.tolist()}."
            )
        nse_per_item = int(nse_per_item_arr[0].item()) if batch_size > 0 else 0
        crow = torch.stack(
            [self.rowptr[i * n_rows : (i + 1) * n_rows + 1] - self.rowptr[i * n_rows] for i in range(batch_size)]
        )
        col = self.col.reshape(batch_size, nse_per_item)
        values = self.values.reshape(batch_size, nse_per_item)
        return torch.sparse_csr_tensor(crow, col, values, (batch_size, n_rows, n_cols))

    def _to_csr_list(self, *, expected_len: int) -> List[torch.Tensor]:
        batch_size, n_rows, n_cols = self.shape
        if expected_len != batch_size:
            raise ValueError(
                f"BatchedCSR.to_torch(like=<list of {expected_len} CSR matrices>) does not match "
                f"descriptor batch_size {batch_size}."
            )
        result = []
        for i in range(batch_size):
            start, end = i * n_rows, (i + 1) * n_rows
            crow = self.rowptr[start : end + 1] - self.rowptr[start]
            nse_start = int(self.rowptr[start].item())
            nse_end = int(self.rowptr[end].item())
            col = self.col[nse_start:nse_end]
            values = self.values[nse_start:nse_end]
            result.append(torch.sparse_csr_tensor(crow, col, values, (n_rows, n_cols)))
        return result


@dataclass(frozen=True)
class BatchedCSC:
    """Canonical internal CSC descriptor — the transpose companion of
    BatchedCSR (architecture.md §3). Lazily produced via
    ``BatchedCSR.transposed``; nothing constructs it directly yet.

    Attributes
    ----------
    values : torch.Tensor
        Stored values, shape ``(nse_total,)``.
    colptr : torch.Tensor
        Absolute CSC pointer over **folded columns** ``col_global = b * n_cols + c``,
        shape ``(batch_size * n_cols + 1,)``.
    row : torch.Tensor
        **Local** row indices in ``[0, n_rows)``, shape ``(nse_total,)``.
    shape : tuple of int
        ``(batch_size, n_rows, n_cols)`` of the underlying logical matrix
        (not transposed) — matching PyTorch's own CSC convention.
    """

    values: torch.Tensor
    colptr: torch.Tensor
    row: torch.Tensor
    shape: Tuple[int, int, int]

    def __post_init__(self) -> None:
        batch_size, _n_rows, n_cols = self.shape
        expected_colptr_len = batch_size * n_cols + 1
        if tuple(self.colptr.shape) != (expected_colptr_len,):
            raise ValueError(
                "BatchedCSC.colptr must be absolute over folded columns with shape "
                f"(batch_size * n_cols + 1,) = ({expected_colptr_len},) for shape {self.shape}; "
                f"got {tuple(self.colptr.shape)}."
            )
        if self.row.shape[0] != self.values.shape[0]:
            raise ValueError(
                "BatchedCSC.row and BatchedCSC.values must both have length nse_total; "
                f"got row {tuple(self.row.shape)} and values {tuple(self.values.shape)}."
            )
        if self.colptr.dtype != self.row.dtype:
            raise ValueError(
                f"BatchedCSC.colptr and BatchedCSC.row must share one index dtype; "
                f"got colptr {self.colptr.dtype} and row {self.row.dtype}."
            )
        if self.colptr.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"BatchedCSC index dtype must be torch.int32 or torch.int64; got {self.colptr.dtype}.")

    @property
    def index_dtype(self) -> torch.dtype:
        return self.colptr.dtype

    @property
    def nse_total(self) -> int:
        return self.values.shape[0]

    @functools.cached_property
    def col_indices(self) -> torch.Tensor:
        """Uncompressed folded column index (``col_global = b * n_cols + c``)
        per specified entry — expanded/repeat-interleaved from ``colptr``."""
        batch_size, _n_rows, n_cols = self.shape
        return torch.repeat_interleave(
            torch.arange(batch_size * n_cols, dtype=self.colptr.dtype, device=self.colptr.device),
            self.colptr[1:] - self.colptr[:-1],
        )

    @functools.cached_property
    def transposed(self) -> "BatchedCSR":
        """Back to the BatchedCSR of the same logical matrix."""
        batch_size, n_rows, n_cols = self.shape
        col_global = self.col_indices
        batch = col_global // n_cols
        col_local = col_global % n_cols
        rowptr, col, values = _fold_coo_to_csr(
            batch, self.row, col_local, self.values, batch_size=batch_size, n_rows=n_rows
        )
        return BatchedCSR(
            values=values,
            rowptr=rowptr.to(self.index_dtype),
            col=col.to(self.index_dtype),
            shape=self.shape,
        )

    @functools.cached_property
    def plans(self) -> dict:
        # Placeholder: real SpSM analysis plans land in commit 16 (kernels.md
        # open Q3) — the plan cache lives on the descriptor so its lifetime is
        # the descriptor's lifetime; no global cache, no hashing.
        return {}
