"""Shared matvec adapter for the batched Krylov solvers (spec/commit.md
Phase 3 commit 17; map.md §2: solver hot path is SpMV/SpMM).

Every solver in ``solvers/`` canonicalises its right-hand side to one batched
dense tensor of shape ``(batch_size, n, n_rhs)`` (``B = 1`` encodes unbatched
— naming.md §2) and drives its iteration through a :class:`BatchedOperator`
built from whatever the caller passed as the operator:

- a **sparse matrix** (COO/CSR tensor, 2D or batched 3D) routes its matvec
  through ``tsgu::spmm`` when the CUDA backend can take it (map.md kernel
  routing: composites call ``tsgu::spmm``; ``spmv`` = ``spmm`` with
  ``p = 1``). The transpose matvec uses the descriptor's lazily **cached
  BatchedCSC** (``BatchedCSR.transposed``, architecture.md §3) — built once
  per operator, reused every iteration. Off-CUDA (or for value dtypes the
  kernels don't cover) it falls back to plain torch sparse matmul semantics,
  exactly what the solvers did before this commit.
- a **dense matrix** uses ``.matmul`` (broadcasts over the batch axis).
- a **callable** is honoured as-is (map.md contract: pluggable user solvers
  and closures keep working). The adapter feeds it operands shaped the way
  the caller's own right-hand side was shaped — a user closure written for
  ``(n, n_rhs)`` operands never sees the internal batch axis.
- a **``BatchedCSR`` descriptor** is accepted directly (package-internal
  callers that already hold one skip the re-extraction).

This module is private to ``solvers/``; nothing outside the front package
imports it.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch

from torchsparsegradutils._batched import BatchedCSR
from torchsparsegradutils._dispatch import backend_available

# Value dtypes the CUDA kernels are templated over (architecture.md §3:
# "kernels templated over {f32,f64} x {i32,i64}").
_KERNEL_VALUE_DTYPES = (torch.float32, torch.float64)

_SPARSE_LAYOUTS = (torch.sparse_coo, torch.sparse_csr)

OperatorLike = Union[torch.Tensor, BatchedCSR, Callable[[torch.Tensor], torch.Tensor]]


def as_batched_rhs(rhs: torch.Tensor) -> Tuple[torch.Tensor, bool, bool]:
    """Canonicalise a right-hand side to ``(batch_size, n, n_rhs)``.

    Returns ``(rhs_batched, was_vector, was_batched)`` where ``was_vector``
    records a 1D ``(n,)`` input and ``was_batched`` a 3D
    ``(batch_size, n, n_rhs)`` input; 2D ``(n, n_rhs)`` input has both False.
    Use :func:`restore_rhs_shape` to undo.
    """
    if rhs.ndim == 1:
        return rhs.unsqueeze(0).unsqueeze(-1), True, False
    if rhs.ndim == 2:
        return rhs.unsqueeze(0), False, False
    if rhs.ndim == 3:
        return rhs, False, True
    raise ValueError(
        "rhs must be a vector (n,), a matrix of right-hand sides (n, n_rhs), or a "
        f"batched matrix (batch_size, n, n_rhs); got shape {tuple(rhs.shape)}."
    )


def restore_rhs_shape(result: torch.Tensor, was_vector: bool, was_batched: bool) -> torch.Tensor:
    """Undo :func:`as_batched_rhs` on a ``(batch_size, n, n_rhs)`` result."""
    if was_batched:
        return result
    result = result.squeeze(0)
    if was_vector:
        result = result.squeeze(-1)
    return result


def _spmm_eligible(descriptor_values: torch.Tensor) -> bool:
    return backend_available() and descriptor_values.is_cuda and descriptor_values.dtype in _KERNEL_VALUE_DTYPES


def _descriptor_matvec(descriptor: BatchedCSR) -> Callable[[torch.Tensor], torch.Tensor]:
    batch_size, n_rows, n_cols = descriptor.shape
    if _spmm_eligible(descriptor.values):

        def matvec(operand: torch.Tensor) -> torch.Tensor:
            return torch.ops.tsgu.spmm(
                descriptor.values, descriptor.rowptr, descriptor.col, operand, batch_size, n_rows, n_cols
            )

        return matvec

    # Pure-torch fallback (CPU, disabled backend, or uncovered value dtype):
    # out[row_g, :] += vals * operand[b, col, :], folded rows scattered back.
    def matvec_fallback(operand: torch.Tensor) -> torch.Tensor:
        row_global = descriptor.row_indices.long()
        batch = row_global // n_rows
        gathered = descriptor.values.unsqueeze(-1) * operand[batch, descriptor.col.long()]
        out = operand.new_zeros(batch_size * n_rows, operand.shape[-1])
        out.index_add_(0, row_global, gathered)
        return out.reshape(batch_size, n_rows, operand.shape[-1])

    return matvec_fallback


def _descriptor_rmatvec(descriptor: BatchedCSR) -> Callable[[torch.Tensor], torch.Tensor]:
    batch_size, n_rows, n_cols = descriptor.shape
    csc = descriptor.transposed  # cached BatchedCSC (architecture.md §3)
    if _spmm_eligible(csc.values):

        def rmatvec(operand: torch.Tensor) -> torch.Tensor:
            # The CSC arrays read as the CSR of the transposed matrix with
            # n_rows/n_cols swapped (ops/matmul.py gradB does the same).
            return torch.ops.tsgu.spmm(csc.values, csc.colptr, csc.row, operand, batch_size, n_cols, n_rows)

        return rmatvec

    def rmatvec_fallback(operand: torch.Tensor) -> torch.Tensor:
        col_global = csc.col_indices.long()
        batch = col_global // n_cols
        gathered = csc.values.unsqueeze(-1) * operand[batch, csc.row.long()]
        out = operand.new_zeros(batch_size * n_cols, operand.shape[-1])
        out.index_add_(0, col_global, gathered)
        return out.reshape(batch_size, n_cols, operand.shape[-1])

    return rmatvec_fallback


class BatchedOperator:
    """Uniform batched matvec interface over the accepted operator forms.

    ``matvec`` maps ``(batch_size, n_cols, n_rhs) -> (batch_size, n_rows,
    n_rhs)``; ``rmatvec`` is the transpose map. Both always speak the
    canonical batched shape — the adapter, not the solver, hides the batch
    axis from unbatched user callables.

    Parameters
    ----------
    operator : torch.Tensor, BatchedCSR, or callable
        The operator. A sparse tensor (COO/CSR, 2D or batched 3D) or a
        descriptor routes through ``tsgu::spmm`` when eligible; a dense
        tensor uses ``.matmul``; a callable is wrapped as-is.
    callable_operand_ndim : int
        Only consulted for callables: the operand rank the user's closure
        expects, i.e. the rank of the caller's own right-hand side — the
        pluggable-callable contract (map.md) freezes what a closure sees.
        ``3``: the closure takes the canonical ``(batch_size, n, n_rhs)``
        operand as-is. ``2``: it takes unbatched ``(n, n_rhs)`` slabs
        (``linear_cg``/``minres`` behaviour). ``1``: it takes single
        ``(n,)`` vectors (``bicgstab``/``lsmr`` behaviour) — the adapter
        loops the closure over right-hand-side columns, so batched iterates
        never leak a widened operand into a legacy vector closure.
        Tensors and descriptors carry their own batching and ignore this.
    n_rows, n_cols : int, optional
        Logical matrix shape. Required for callables (cannot be inferred);
        ignored for tensors/descriptors.
    """

    def __init__(
        self,
        operator: OperatorLike,
        *,
        callable_operand_ndim: int = 2,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
    ) -> None:
        if callable_operand_ndim not in (1, 2, 3):
            raise ValueError(f"callable_operand_ndim must be 1, 2, or 3; got {callable_operand_ndim}.")
        self._descriptor: Optional[BatchedCSR] = None
        self._tensor: Optional[torch.Tensor] = None
        self._callable: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self._callable_operand_ndim = callable_operand_ndim

        self._swapped = False  # True: `operator` is the transpose of the held descriptor

        if isinstance(operator, BatchedCSR):
            self._descriptor = operator
            batch_size, self.n_rows, self.n_cols = operator.shape
            self.batch_size: Optional[int] = batch_size
        elif isinstance(operator, torch.Tensor):
            if operator.layout == torch.sparse_csc:
                # A CSC matrix is the O(1) transpose view of a CSR matrix
                # (``torch.adjoint`` of sparse CSR yields CSC — how lstsq's
                # default transpose solve arrives here). Hold the descriptor
                # of its transpose and swap matvec/rmatvec. Real value dtypes
                # only: ``.mT`` equals the adjoint only without conjugation.
                if operator.values().is_complex():
                    raise TypeError("A sparse CSC operator with complex values is not supported; pass a callable.")
                self._descriptor = BatchedCSR.from_torch(operator.mT)
                self._swapped = True
                batch_size, transposed_rows, transposed_cols = self._descriptor.shape
                self.n_rows, self.n_cols = transposed_cols, transposed_rows
                self.batch_size = batch_size
            elif operator.layout in _SPARSE_LAYOUTS:
                self._descriptor = BatchedCSR.from_torch(operator)
                batch_size, self.n_rows, self.n_cols = self._descriptor.shape
                self.batch_size = batch_size
            elif operator.layout == torch.strided:
                if operator.ndim not in (2, 3):
                    raise ValueError(
                        "a dense operator must be a matrix (n_rows, n_cols) or a batched matrix "
                        f"(batch_size, n_rows, n_cols); got shape {tuple(operator.shape)}."
                    )
                self._tensor = operator
                self.n_rows, self.n_cols = operator.shape[-2], operator.shape[-1]
                self.batch_size = operator.shape[0] if operator.ndim == 3 else None
            else:
                raise TypeError(
                    f"Unsupported operator layout {operator.layout}; expected a strided, sparse COO, "
                    "or sparse CSR tensor, a BatchedCSR descriptor, or a callable."
                )
        elif callable(operator):
            if n_rows is None or n_cols is None:
                raise RuntimeError("n_rows and n_cols must be provided when the operator is a callable.")
            self._callable = operator
            self.n_rows, self.n_cols = n_rows, n_cols
            self.batch_size = None
        else:
            raise RuntimeError("matmul_closure must be a tensor, a BatchedCSR descriptor, or a callable object!")

    def matvec(self, operand: torch.Tensor) -> torch.Tensor:
        """``(batch_size, n_cols, n_rhs) -> (batch_size, n_rows, n_rhs)``."""
        if self._descriptor is not None:
            apply = _descriptor_rmatvec if self._swapped else _descriptor_matvec
            return apply(self._descriptor)(operand)
        if self._tensor is not None:
            return self._tensor.matmul(operand)
        return self._apply_callable(self._callable, operand)

    def rmatvec(self, operand: torch.Tensor) -> torch.Tensor:
        """Transpose matvec ``(batch_size, n_rows, n_rhs) -> (batch_size, n_cols, n_rhs)``.

        For a callable operator there is no derivable transpose — callers
        must supply their own transpose closure (lsmr's ``Armat`` contract).
        """
        if self._descriptor is not None:
            apply = _descriptor_matvec if self._swapped else _descriptor_rmatvec
            return apply(self._descriptor)(operand)
        if self._tensor is not None:
            return torch.adjoint(self._tensor).matmul(operand)
        raise RuntimeError("The transpose matvec of a callable operator cannot be derived; provide it explicitly.")

    def _apply_callable(self, closure, operand: torch.Tensor) -> torch.Tensor:
        if self._callable_operand_ndim == 3:
            return closure(operand)
        if self._callable_operand_ndim == 2:
            # Unbatched closure: hide the canonical batch axis (batch_size == 1).
            return closure(operand.squeeze(0)).unsqueeze(0)
        # Vector closure (legacy bicgstab/lsmr operand contract): one call per
        # right-hand-side column, each fed a plain (n,) vector.
        slab = operand.squeeze(0)
        columns = [closure(slab[:, j]) for j in range(slab.shape[-1])]
        return torch.stack(columns, dim=-1).unsqueeze(0)
