"""benchmarks/corpus.py — seeded synthetic corpus generators.

spec/benchmarks.md §2, "Synthetic batched/ragged" tier (the migration
workhorse — no public dataset covers batched/ragged COO): ``rand_sparse*``
sweep, B in {1, 8, 64, 256}, ragged nse (+/-50% across items), n in
{10^3, 10^4, 10^5}, int32/int64; generator seed pinned. ``n_rhs`` (p) swept
over {1, 8, 32, 128, 512}.

Wraps ``torchsparsegradutils.utils.random_sparse``'s generators (already
tested elsewhere) rather than reimplementing sparse-matrix generation; adds
the sweep grid, ragged batching, and this module's one departure from the
spec's raw numbers:

**VRAM auto-scaling** (spec/commit.md #11 clarification): this dev machine
has a 4 GB card. The spec's ``B in {1,8,64,256}`` shape sweep stays exactly
as written; **nse-per-item shrinks** so the largest swept configuration's
estimated peak memory stays under a documented budget
(:func:`peak_budget_gb`). :func:`available_vram_gb` queries
``torch.cuda.get_device_properties`` when a GPU is visible, so a bigger card
later grows the sweep back automatically with no code change; with no GPU
visible at all (down or absent), it falls back to this machine's known 4 GB
so the corpus still scales sanely in degraded mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
)

# Generator seed -- pinned per benchmarks.md §2 ("generator seed pinned").
# Reruns of any sweep config below are byte-identical.
SEED = 20260714

SWEEP_B = (1, 8, 64, 256)
SWEEP_N = (10**3, 10**4, 10**5)
SWEEP_P = (1, 8, 32, 128, 512)

# Reference density for the synthetic workhorse's nse-per-item. Anchored to
# the one concrete density benchmarks.md §5's illustrative (dummy) example
# implies (a 10k^2 matrix with ~500k nse/item -> 0.5%); not itself a number
# benchmarks.md pins exactly, so treated as a documented default rather than
# a spec'd constant.
REFERENCE_DENSITY = 0.005
RAGGED_JITTER = 0.5  # +/-50% per item, benchmarks.md §2

# --- VRAM auto-scaling -------------------------------------------------------

# This dev machine's card (spec/commit.md #11 worker context: "the machine's
# GPU is currently DOWN"). Used only when torch can't see any CUDA device at
# all, so the corpus still scales as if targeting the known card rather than
# defaulting to the full, unscaled spec sweep.
THIS_MACHINE_VRAM_GB_FALLBACK = 4.0
# Absolute cap during migration, regardless of the card actually detected --
# "sized for THIS machine ... so peak stays < 3GB" (spec/commit.md #11).
ABSOLUTE_PEAK_BUDGET_GB = 3.0
# Never target more than this fraction of any detected card's total memory,
# so a bigger GPU later doesn't get a corpus sized to fully occupy it either.
PEAK_BUDGET_FRACTION = 0.75


def available_vram_gb() -> float:
    """Total VRAM of GPU 0 in GiB, or :data:`THIS_MACHINE_VRAM_GB_FALLBACK`
    if no CUDA device is visible to torch (down, absent, or disabled)."""
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
        except Exception:
            pass
    return THIS_MACHINE_VRAM_GB_FALLBACK


def peak_budget_gb(vram_gb: Optional[float] = None) -> float:
    """The peak-memory budget this corpus is sized against: the smaller of
    the absolute migration-period cap and a fraction of the detected (or
    fallback) card's total VRAM."""
    vram_gb = available_vram_gb() if vram_gb is None else vram_gb
    return min(ABSOLUTE_PEAK_BUDGET_GB, vram_gb * PEAK_BUDGET_FRACTION)


_BYTES_PER_VALUE = {torch.float32: 4, torch.float64: 8, torch.float16: 2, torch.bfloat16: 2}
_BYTES_PER_INDEX = {torch.int32: 4, torch.int64: 8}


def estimate_peak_bytes(
    *,
    B: int,
    n: int,
    m: int,
    p: int,
    nse_per_item: int,
    value_dtype: torch.dtype,
    index_dtype: torch.dtype,
    workspace_fudge: float = 1.5,
) -> int:
    """A rough bytes-moved model (benchmarks.md §1's "achieved bandwidth ...
    bytes-moved model") for a batched SpMM-shaped op: sparse operand
    (values + col + rowptr) + dense rhs + dense output, inflated by a
    documented fudge factor for allocator/workspace overhead.

    This is a corpus-sizing heuristic, not a substitute for
    :mod:`benchmarks.memory`'s real measurement -- there's no kernel to
    measure yet, so this is what picks safe corpus sizes today.
    """
    v = _BYTES_PER_VALUE[value_dtype]
    idx = _BYTES_PER_INDEX[index_dtype]
    nse_total = nse_per_item * B
    sparse_bytes = nse_total * (v + idx) + (B * n + 1) * idx
    dense_in_bytes = B * m * p * v
    dense_out_bytes = B * n * p * v
    return int(workspace_fudge * (sparse_bytes + dense_in_bytes + dense_out_bytes))


def max_nse_per_item_for_budget(
    *,
    B: int,
    n: int,
    m: int,
    p: int,
    value_dtype: torch.dtype,
    index_dtype: torch.dtype,
    budget_gb: Optional[float] = None,
    workspace_fudge: float = 1.5,
) -> int:
    """Inverse of :func:`estimate_peak_bytes`: the largest ``nse_per_item``
    that keeps the estimated peak under ``budget_gb`` (default
    :func:`peak_budget_gb`)."""
    budget_bytes = (peak_budget_gb() if budget_gb is None else budget_gb) * (1024**3)
    v = _BYTES_PER_VALUE[value_dtype]
    idx = _BYTES_PER_INDEX[index_dtype]
    dense_in_bytes = B * m * p * v
    dense_out_bytes = B * n * p * v
    rowptr_bytes = (B * n + 1) * idx
    remaining = budget_bytes / workspace_fudge - dense_in_bytes - dense_out_bytes - rowptr_bytes
    if remaining <= 0:
        return 1
    max_nse_total = remaining / (v + idx)
    return max(1, int(max_nse_total // max(B, 1)))


def reference_nse_per_item(n: int, m: int) -> int:
    return max(1, min(n * m, round(REFERENCE_DENSITY * n * m)))


def _scaled_nse_per_item(B: int, n: int, m: int, p: int, value_dtype: torch.dtype, index_dtype: torch.dtype) -> int:
    ref = reference_nse_per_item(n, m)
    cap = max_nse_per_item_for_budget(B=B, n=n, m=m, p=p, value_dtype=value_dtype, index_dtype=index_dtype)
    return max(1, min(ref, cap))


@dataclass(frozen=True)
class SweepConfig:
    name: str
    B: int
    n: int
    m: int
    p: int
    nse_per_item: int
    layout: torch.layout
    ragged: bool
    value_dtype: torch.dtype
    index_dtype: torch.dtype
    seed: int


def sweep_configs(
    *,
    layout: torch.layout = torch.sparse_csr,
    value_dtype: torch.dtype = torch.float32,
    index_dtype: torch.dtype = torch.int32,
    p_values: tuple = SWEEP_P,
) -> list[SweepConfig]:
    """The full B x n x p grid (benchmarks.md §2), nse-per-item auto-scaled
    to this machine's VRAM budget (see module docstring). Ragged batching
    only applies to COO -- batched CSR requires equal nse per item (a torch
    layout constraint, not ours; see
    ``torchsparsegradutils/utils/random_sparse.py``'s module docstring)."""
    ragged = layout == torch.sparse_coo
    configs = []
    for B in SWEEP_B:
        for n in SWEEP_N:
            m = n  # square synthetic matrices, matching benchmarks.md's own dummy examples
            for p in p_values:
                nse = _scaled_nse_per_item(B, n, m, p, value_dtype, index_dtype)
                configs.append(
                    SweepConfig(
                        name=f"synth-{'ragged' if ragged else 'uniform'}-B{B}-n{n}-p{p}",
                        B=B,
                        n=n,
                        m=m,
                        p=p,
                        nse_per_item=nse,
                        layout=layout,
                        ragged=ragged,
                        value_dtype=value_dtype,
                        index_dtype=index_dtype,
                        seed=SEED,
                    )
                )
    return configs


def smoke_config() -> SweepConfig:
    """The single smallest, safest sweep point: fast on CPU in today's
    degraded mode, and still a legitimate point in the real sweep once a
    GPU and a kernel exist. Used by ``benchmarks.run``'s baseline entry
    point."""
    return sweep_configs(layout=torch.sparse_coo, p_values=(8,))[0]


def _jitter_nnz(reference: int, *, cap: int, rng: torch.Generator) -> int:
    lo = max(1, int(reference * (1 - RAGGED_JITTER)))
    hi = min(cap, max(lo, int(reference * (1 + RAGGED_JITTER))))
    if hi <= lo:
        return lo
    return int(torch.randint(lo, hi + 1, (1,), generator=rng).item())


def make_sparse_batch(cfg: SweepConfig, *, device: torch.device) -> torch.Tensor:
    """Build a batched sparse tensor for ``cfg``: ragged nse per item
    (+/-50%, seeded) for COO; uniform nse per item for CSR (torch
    constraint)."""
    if cfg.layout == torch.sparse_csr:
        return generate_random_sparse_csr_matrix(
            (cfg.B, cfg.n, cfg.m),
            cfg.nse_per_item,
            indices_dtype=cfg.index_dtype,
            values_dtype=cfg.value_dtype,
            device=device,
            well_conditioned=True,
        )

    rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
    cap = cfg.n * cfg.m
    idx_parts, val_parts = [], []
    for b in range(cfg.B):
        nnz_b = _jitter_nnz(cfg.nse_per_item, cap=cap, rng=rng) if cfg.ragged else cfg.nse_per_item
        item = generate_random_sparse_coo_matrix(
            (cfg.n, cfg.m),
            nnz_b,
            indices_dtype=cfg.index_dtype,
            values_dtype=cfg.value_dtype,
            device=device,
            well_conditioned=True,
        )
        item_idx = item.indices()
        batch_row = torch.full((1, item_idx.shape[1]), b, dtype=cfg.index_dtype, device=device)
        idx_parts.append(torch.cat([batch_row, item_idx], dim=0))
        val_parts.append(item.values())

    indices = torch.cat(idx_parts, dim=1)
    values = torch.cat(val_parts)
    return torch.sparse_coo_tensor(indices, values, (cfg.B, cfg.n, cfg.m), device=device).coalesce()


_DTYPE_SHORT = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int32: "i32",
    torch.int64: "i64",
}
_LAYOUT_NAME = {torch.sparse_coo: "COO", torch.sparse_csr: "CSR"}


def dtype_short(dtype: torch.dtype) -> str:
    return _DTYPE_SHORT.get(dtype, str(dtype))


def bytes_per_index(dtype: torch.dtype) -> int:
    return _BYTES_PER_INDEX[dtype]


def bytes_per_value(dtype: torch.dtype) -> int:
    return _BYTES_PER_VALUE[dtype]


def layout_name(layout: torch.layout) -> str:
    return _LAYOUT_NAME.get(layout, str(layout))
