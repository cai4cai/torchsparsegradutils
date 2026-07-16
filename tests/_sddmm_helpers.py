"""Shared pattern-construction and reference-computation helpers for
tsgu::sddmm's gate suite (spec/commit.md Phase 3 commit 14). Not a gate
module itself -- no ``gateN`` marker, nothing here is collected by pytest as
a test (no ``test_*`` names), imported by ``tests/test_gate{2,3,4,5,6}_*_sddmm.py``.

Naming.md §2: rowptr is absolute over folded rows (``row_global = b * n +
r``); col is **local** in ``[0, m)``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def make_batched_pattern(
    masks: Sequence[torch.Tensor], index_dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Build a folded-CSR ``(rowptr, col)`` pattern from a list of ``B`` dense
    ``(n, m)`` 0/1 masks -- one mask per batch item, so ragged nse per item
    (including an all-zero mask for an "empty batch item") and zero-row
    segments fall out naturally from each mask's own content, no special
    casing needed. Returns ``(rowptr, col, B, n, m)``.
    """
    B = len(masks)
    n, m = masks[0].shape
    rowptr_list = [0]
    cols: List[int] = []
    for mask in masks:
        assert tuple(mask.shape) == (n, m), f"all masks must share one (n, m) = {(n, m)}; got {tuple(mask.shape)}"
        for r in range(n):
            row_cols = torch.nonzero(mask[r], as_tuple=False).flatten().tolist()
            cols.extend(row_cols)
            rowptr_list.append(rowptr_list[-1] + len(row_cols))
    rowptr = torch.tensor(rowptr_list, dtype=index_dtype, device=device)
    col = torch.tensor(cols, dtype=index_dtype, device=device)
    return rowptr, col, B, n, m


def random_masks(
    B: int, n: int, m: int, density: float, *, generator: torch.Generator, empty_item: bool = False
) -> List[torch.Tensor]:
    """``B`` independent random 0/1 ``(n, m)`` masks at ``density`` (ragged nse
    per item falls out of independent sampling) -- CPU-generated (index
    construction is host-side regardless of the eventual pattern's device).
    If ``empty_item``, batch item 0 is forced all-zero (naming.md §1's "ragged
    nse per item... even zero" case, testing.md's "empty batch items")."""
    masks = []
    for b in range(B):
        if empty_item and b == 0:
            masks.append(torch.zeros(n, m))
            continue
        masks.append((torch.rand(n, m, generator=generator) < density).to(torch.float32))
    return masks


def dense_reference(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    g: torch.Tensor,
    mat: torch.Tensor,
    B: int,
    n: int,
    m: int,
    negate: bool,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Oracle B (testing.md Pillar 1): ``G @ mat^T`` computed densely in
    ``dtype``, gathered at the ``(rowptr, col)`` pattern -- independent of
    tsgu::sddmm's own (or Oracle A's) implementation."""
    g_ref = g.to(dtype)
    mat_ref = mat.to(dtype)
    dense = torch.einsum("bnp,bmp->bnm", g_ref, mat_ref)  # (B, n, m), no sparsity assumed
    row_g = torch.repeat_interleave(
        torch.arange(B * n, device=rowptr.device, dtype=torch.int64), (rowptr[1:] - rowptr[:-1]).long()
    )
    batch = row_g // n
    local_row = row_g % n
    out = dense[batch, local_row, col.long()]
    return -out if negate else out
