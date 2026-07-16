"""Shared pattern-construction and reference-computation helpers for
tsgu::spmm's gate suite (spec/commit.md Phase 3 commit 15). Not a gate
module itself -- no ``gateN`` marker, nothing here is collected by pytest as
a test (no ``test_*`` names), imported by ``tests/test_gate{2,3,4,5,6}_*_spmm.py``.

Naming.md §2: rowptr is absolute over folded rows (``row_global = b * n +
r``); col is **local** in ``[0, m)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def random_masks(
    B: int, n: int, m: int, density: float, *, generator: torch.Generator, empty_item: bool = False
) -> List[torch.Tensor]:
    """``B`` independent random 0/1 ``(n, m)`` masks at ``density`` (ragged nse
    per item falls out of independent sampling) -- CPU-generated. If
    ``empty_item``, batch item 0 is forced all-zero (naming.md §1's "ragged
    nse per item... even zero", testing.md's "empty batch items")."""
    masks = []
    for b in range(B):
        if empty_item and b == 0:
            masks.append(torch.zeros(n, m))
            continue
        masks.append((torch.rand(n, m, generator=generator) < density).to(torch.float32))
    return masks


def make_batched_csr(
    masks: Sequence[torch.Tensor],
    index_dtype: torch.dtype,
    value_dtype: torch.dtype,
    device: torch.device,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Build a folded-CSR ``(rowptr, col, vals)`` triple from a list of ``B``
    dense ``(n, m)`` 0/1 masks -- one mask per batch item, so ragged nse per
    item (including an all-zero mask for an "empty batch item") falls out
    naturally. Returns ``(rowptr, col, vals, B, n, m)``."""
    B = len(masks)
    n, m = masks[0].shape
    rowptr_list = [0]
    cols: List[int] = []
    n_specified = 0
    for mask in masks:
        assert tuple(mask.shape) == (n, m), f"all masks must share one (n, m) = {(n, m)}; got {tuple(mask.shape)}"
        for r in range(n):
            row_cols = torch.nonzero(mask[r], as_tuple=False).flatten().tolist()
            cols.extend(row_cols)
            n_specified += len(row_cols)
            rowptr_list.append(rowptr_list[-1] + len(row_cols))
    rowptr = torch.tensor(rowptr_list, dtype=index_dtype, device=device)
    col = torch.tensor(cols, dtype=index_dtype, device=device)
    gen = generator if generator is not None else torch.Generator().manual_seed(0)
    vals = torch.randn(n_specified, generator=gen).to(value_dtype).to(device)
    return rowptr, col, vals, B, n, m


def dense_reference(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    vals: torch.Tensor,
    dense: torch.Tensor,
    B: int,
    n: int,
    m: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Oracle B (testing.md Pillar 1): the folded-CSR pattern materialised
    densely in ``dtype`` and multiplied against ``dense`` via
    ``torch.einsum`` -- independent of tsgu::spmm's own (or Oracle A's)
    implementation."""
    row_g = torch.repeat_interleave(
        torch.arange(B * n, device=rowptr.device, dtype=torch.int64), (rowptr[1:] - rowptr[:-1]).long()
    )
    batch = row_g // n
    local_row = row_g % n
    dense_pattern = torch.zeros(B, n, m, dtype=dtype, device=rowptr.device)
    if vals.numel() > 0:
        dense_pattern[batch, local_row, col.long()] = vals.to(dtype)
    dense_ref = dense.to(dtype)
    return torch.einsum("bnm,bmp->bnp", dense_pattern, dense_ref)


def csr_to_sparse_tensor(
    rowptr: torch.Tensor, col: torch.Tensor, vals: torch.Tensor, B: int, n: int, m: int, *, layout
) -> torch.Tensor:
    """Rewrap a folded-CSR ``(rowptr, col, vals)`` triple as a public-facing
    sparse tensor (COO or CSR, batched or not) -- the layout the public
    ``sparse_mm`` wrapper actually accepts (naming.md §1: never a raw
    rowptr/col/vals triple, that's kernel-only vocabulary)."""
    if layout == torch.sparse_csr:
        if B == 1:
            return torch.sparse_csr_tensor(rowptr, col, vals, (n, m))
        # torch batched CSR requires equal nse per item -- callers of this
        # helper with layout=CSR must already guarantee that.
        nse_per_item = int(rowptr[n].item())
        crow = torch.stack([rowptr[b * n : (b + 1) * n + 1] - rowptr[b * n] for b in range(B)])
        col_b = col.reshape(B, nse_per_item)
        vals_b = vals.reshape(B, nse_per_item)
        return torch.sparse_csr_tensor(crow, col_b, vals_b, (B, n, m))

    # COO: uncompress rowptr -> row indices (naming.md §1: "expanded /
    # repeat-interleaved", never "decompressed").
    row_g = torch.repeat_interleave(
        torch.arange(B * n, device=rowptr.device, dtype=rowptr.dtype), (rowptr[1:] - rowptr[:-1]).long()
    )
    batch = row_g // n
    local_row = row_g % n
    if B == 1:
        indices = torch.stack([local_row, col], dim=0)
        return torch.sparse_coo_tensor(indices, vals, (n, m)).coalesce()
    indices = torch.stack([batch, local_row, col], dim=0)
    return torch.sparse_coo_tensor(indices, vals, (B, n, m)).coalesce()
