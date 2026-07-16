"""Shared pattern-construction and reference-computation helpers for
tsgu::spsm's gate suite (spec/commit.md Phase 3 commit 16). Not a gate
module itself -- no ``gateN`` marker, nothing here is collected by pytest as
a test (no ``test_*`` names), imported by ``tests/test_gate{2,3,4,5,6}_*_spsm.py``.

Naming.md §2: rowptr is absolute over folded rows (``row_global = b * n +
r``); col is **local** in ``[0, n)`` (A is square for a triangular solve).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def make_triangular_mask(
    n: int, *, upper: bool, unitriangular: bool, density: float, generator: torch.Generator
) -> torch.Tensor:
    """A dense ``(n, n)`` 0/1 mask for a triangular pattern: the strict
    upper/lower part is filled at ``density``, and the diagonal is included
    unless ``unitriangular`` (map.md contract: "the stored matrix must be
    strictly triangular... when unitriangular=True")."""
    strict = torch.triu(torch.ones(n, n), diagonal=1) if upper else torch.tril(torch.ones(n, n), diagonal=-1)
    fill = (torch.rand(n, n, generator=generator) < density).to(torch.float32)
    mask = strict * fill
    if not unitriangular:
        mask = mask + torch.eye(n)
    return mask.clamp(max=1.0)


def make_batched_triangular_csr(
    n: int,
    B: int,
    index_dtype: torch.dtype,
    value_dtype: torch.dtype,
    device: torch.device,
    *,
    upper: bool,
    unitriangular: bool,
    density: float = 0.3,
    generator: Optional[torch.Generator] = None,
    diag_boost: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build a folded-CSR ``(rowptr, col, vals)`` triple for a batch of ``B``
    independent ``(n, n)`` triangular patterns (equal nse per item -- so the
    result is also a valid input to ``csr_to_sparse_tensor(..., layout=CSR)``).
    Diagonal entries (when present) are boosted in magnitude (``diag_boost``)
    so the resulting system is reasonably well-conditioned -- an unscaled
    random triangular system is often numerically singular in float32/64,
    which would make every parity/gradcheck comparison meaningless (not a
    kernel bug, just an ill-posed test fixture)."""
    gen = generator if generator is not None else torch.Generator().manual_seed(0)
    rowptr_list = [0]
    cols: List[int] = []
    diag_positions: List[int] = []
    # One mask shared across every batch item -- equal nse per item, the
    # torch batched-CSR storage constraint (naming.md §1); values still
    # differ per item (drawn fresh below). Ragged-nse coverage lives in
    # test_oracle_b_batched_ragged (COO only, per naming.md §1).
    shared_mask = make_triangular_mask(n, upper=upper, unitriangular=unitriangular, density=density, generator=gen)
    for _b in range(B):
        mask = shared_mask
        for r in range(n):
            row_cols = torch.nonzero(mask[r], as_tuple=False).flatten().tolist()
            for c in row_cols:
                cols.append(c)
                diag_positions.append(1 if c == r else 0)
            rowptr_list.append(rowptr_list[-1] + len(row_cols))
    rowptr = torch.tensor(rowptr_list, dtype=index_dtype, device=device)
    col = torch.tensor(cols, dtype=index_dtype, device=device)
    n_specified = len(cols)
    vals = torch.randn(n_specified, generator=gen).to(value_dtype)
    diag_mask = torch.tensor(diag_positions, dtype=torch.bool)
    vals = vals * 0.2
    vals[diag_mask] = vals[diag_mask].abs() + diag_boost
    return rowptr.to(device), col.to(device), vals.to(device), B, n


def dense_from_csr(
    rowptr: torch.Tensor, col: torch.Tensor, vals: torch.Tensor, B: int, n: int, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Materialise the folded-CSR pattern densely (Oracle B territory --
    independent of tsgu::spsm's own implementation)."""
    row_g = torch.repeat_interleave(
        torch.arange(B * n, device=rowptr.device, dtype=torch.int64), (rowptr[1:] - rowptr[:-1]).long()
    )
    batch = row_g // n
    local_row = row_g % n
    dense = torch.zeros(B, n, n, dtype=dtype, device=rowptr.device)
    if vals.numel() > 0:
        dense[batch, local_row, col.long()] = vals.to(dtype)
    return dense


def dense_reference_solve(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    vals: torch.Tensor,
    rhs: torch.Tensor,
    B: int,
    n: int,
    *,
    upper: bool,
    unitriangular: bool,
    transpose: bool,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Oracle B (testing.md Pillar 1): fp64 dense ``torch.linalg.solve_triangular``,
    independent of tsgu::spsm/Oracle A."""
    dense = dense_from_csr(rowptr, col, vals, B, n, dtype=dtype)
    if not unitriangular:
        pass  # diagonal already stored explicitly
    else:
        eye = torch.eye(n, dtype=dtype, device=rowptr.device).unsqueeze(0).expand(B, n, n)
        dense = dense + eye
    rhs64 = rhs.to(dtype)
    if transpose:
        dense = dense.transpose(-2, -1)
        eff_upper = not upper
    else:
        eff_upper = upper
    return torch.linalg.solve_triangular(dense, rhs64, upper=eff_upper, unitriangular=False)


def csr_to_sparse_tensor(
    rowptr: torch.Tensor, col: torch.Tensor, vals: torch.Tensor, B: int, n: int, *, layout
) -> torch.Tensor:
    """Rewrap a folded-CSR ``(rowptr, col, vals)`` triple as a public-facing
    sparse tensor (COO or CSR, batched or not) -- mirrors
    tests/_spmm_helpers.py's helper of the same name."""
    if layout == torch.sparse_csr:
        if B == 1:
            return torch.sparse_csr_tensor(rowptr, col, vals, (n, n))
        nse_per_item = int(rowptr[n].item())
        crow = torch.stack([rowptr[b * n : (b + 1) * n + 1] - rowptr[b * n] for b in range(B)])
        col_b = col.reshape(B, nse_per_item)
        vals_b = vals.reshape(B, nse_per_item)
        return torch.sparse_csr_tensor(crow, col_b, vals_b, (B, n, n))

    row_g = torch.repeat_interleave(
        torch.arange(B * n, device=rowptr.device, dtype=rowptr.dtype), (rowptr[1:] - rowptr[:-1]).long()
    )
    batch = row_g // n
    local_row = row_g % n
    if B == 1:
        indices = torch.stack([local_row, col], dim=0)
        return torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce()
    indices = torch.stack([batch, local_row, col], dim=0)
    return torch.sparse_coo_tensor(indices, vals, (B, n, n)).coalesce()


def bidiagonal_chain_csr(
    n: int, index_dtype: torch.dtype, value_dtype: torch.dtype, device: torch.device, *, upper: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Adversarial: a deep dependency chain -- bidiagonal (each row has the
    diagonal plus exactly one neighbour), the worst case for the level
    schedule (n levels, one row each -- kernels.md/this commit's brief)."""
    rows: List[int] = []
    cols: List[int] = []
    for r in range(n):
        if upper:
            if r + 1 < n:
                rows.append(r)
                cols.append(r + 1)
            rows.append(r)
            cols.append(r)
        else:
            if r - 1 >= 0:
                rows.append(r)
                cols.append(r - 1)
            rows.append(r)
            cols.append(r)
    # sort by row for a valid CSR (entries within a row need not be sorted,
    # but rows themselves must be grouped contiguously and in order).
    order = sorted(range(len(rows)), key=lambda i: rows[i])
    rows = [rows[i] for i in order]
    cols = [cols[i] for i in order]
    rowptr_list = [0]
    ri = 0
    for r in range(n):
        cnt = 0
        while ri < len(rows) and rows[ri] == r:
            cnt += 1
            ri += 1
        rowptr_list.append(rowptr_list[-1] + cnt)
    rowptr = torch.tensor(rowptr_list, dtype=index_dtype, device=device)
    col = torch.tensor(cols, dtype=index_dtype, device=device)
    gen = torch.Generator().manual_seed(7)
    vals = torch.randn(len(cols), generator=gen).to(value_dtype) * 0.2
    diag_mask = torch.tensor([c == r for r, c in zip(rows, cols)], dtype=torch.bool)
    vals[diag_mask] = vals[diag_mask].abs() + 4.0
    return rowptr.to(device), col.to(device), vals.to(device), 1, n
