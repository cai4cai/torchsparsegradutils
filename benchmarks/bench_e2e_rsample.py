"""End-to-end composite benchmark: PairwiseEncoder -> SparseMultivariateNormal
rsample forward + backward (spec/commit.md Phase 4 commit 20).

    uv run python -m benchmarks.bench_e2e_rsample

spec/benchmarks.md §3: "End-to-end composites (rsample of
`SparseMultivariateNormal`, CG solve loop) get one benchmark each — the
user-visible number, catching dispatch overhead that per-op benchmarks hide."

THE PIN (benchmarks.md §3 memory bars, verbatim): "e2e rsample: the
encoder-CSR backward blow-up documented in the old README is a pinned
regression case — peak_bwd <= 1.2x the COO path's, or the suite fails."
This script builds the same encoder volume in BOTH layouts, measures
``peak_bwd`` per benchmarks.md §1's reset -> forward -> record -> backward ->
record protocol (benchmarks/memory.py), records the ratio as a Result row
with bar "<=1.2x COO", and **exits nonzero** if any config busts the bar.

Speed baseline: none exists on this branch — the pre-rewrite pure-torch path
survives only as the frozen parity oracle (tests/oracle/), not an importable
end-to-end composite — so the speed rows record absolute ms per layout plus
the CSR-vs-COO ratio (backend="custom", benchmarks.md §1 provenance).

VRAM guard (4 GB card, ~3.68 usable): configs step strictly upward from
2 x 32^3. The smallest config is bounded by an analytic bytes model before it
runs; each larger config's peak is extrapolated linearly in voxel count
(~cubically in spatial side) from the previous *measured* peak, inflated by a
safety factor, and any config whose estimate exceeds the budget
(min(corpus.peak_budget_gb(), 2.5 GB)) is skipped with a printed note —
never attempted, so no config can OOM.
"""

from __future__ import annotations

import sys

import torch

from benchmarks import corpus, harness, memory
from benchmarks.env import Fingerprint, fingerprint
from benchmarks.results import Result, write_result

# Encoder volume: (num_channels, spatial side) -> volume_shape (C, s, s, s).
# Steps upward; later configs are guarded by extrapolation from earlier ones.
CONFIGS: tuple[tuple[int, int], ...] = ((2, 32), (2, 48), (2, 64))
RADIUS = 1.5
NUM_SAMPLES = 32  # sample_shape (S,): rsample's dense operand is (S, n)
VALUE_DTYPE = torch.float32
INDEX_DTYPE = torch.int64  # COO coalesce coerces int32 -> int64; keep layouts symmetric

MEM_BAR = "<=1.2x COO"
MEM_BAR_RATIO = 1.2
SAFETY_FACTOR = 1.5  # inflation on every peak estimate before attempting a config
HARD_PEAK_CAP_GB = 2.5  # task-level hard guard, on top of corpus.peak_budget_gb()


def _budget_gb() -> float:
    return min(corpus.peak_budget_gb(), HARD_PEAK_CAP_GB)


def _build(layout: torch.layout, channels: int, side: int, device: torch.device):
    """Encoder + LL^T parameters (positive diagonal) + zero mean, one layout."""
    from torchsparsegradutils.encoders import PairwiseEncoder

    volume_shape = (channels, side, side, side)
    encoder = PairwiseEncoder(
        radius=RADIUS,
        volume_shape=volume_shape,
        diag=True,  # LL^T: diagonal lives in the sparse factor
        upper=False,  # lower triangular
        channel_voxel_relation="indep",
        layout=layout,
        indices_dtype=INDEX_DTYPE,
        device=device,
    )
    torch.manual_seed(corpus.SEED)
    params = torch.randn(len(encoder.offsets), *volume_shape, dtype=VALUE_DTYPE, device=device)
    with torch.no_grad():
        params.mul_(0.1)
        for i, offset in enumerate(encoder.offsets):
            if all(o == 0 for o in offset):
                params[i] = params[i].abs() + 0.5  # positive diagonal for a valid Cholesky factor
                break
    params.requires_grad_(True)
    loc = torch.zeros(encoder.volume_numel, dtype=VALUE_DTYPE, device=device)
    return encoder, params, loc


def _make_fns(encoder, params, loc):
    from torchsparsegradutils.distributions import SparseMultivariateNormal

    def _fwd():
        scale_tril = encoder(params)
        dist = SparseMultivariateNormal(loc=loc, scale_tril=scale_tril)
        return dist.rsample((NUM_SAMPLES,))

    def _bwd(samples):
        samples.sum().backward()

    def _step():
        params.grad = None
        _bwd(_fwd())

    return _fwd, _bwd, _step


def _nse(encoder) -> int:
    if encoder.layout == torch.sparse_coo:
        return encoder.indices.shape[1]
    return encoder.col_indices.numel()


def _index_bytes(encoder) -> int:
    if encoder.layout == torch.sparse_coo:
        return encoder.indices.numel() * encoder.indices.element_size()
    return sum(
        t.numel() * t.element_size() for t in (encoder.crow_indices, encoder.col_indices, encoder.csr_permutation)
    )


def _analytic_first_peak_mb(channels: int, side: int) -> float:
    """Crude bytes model for the smallest config only (larger configs are
    guarded by extrapolation from measurement instead): ~10 offsets at
    radius 1.5 / upper=False / diag, forward values copies + index arrays +
    (eps, samples) dense pair, backward ~2.5x forward."""
    voxels = channels * side**3
    nse = 10 * voxels
    fwd = nse * (4 * 4 + 8 * 3) + voxels * NUM_SAMPLES * 4 * 4
    return 2.5 * fwd / 2**20


def _layout_tag(layout: torch.layout) -> str:
    return corpus.layout_name(layout).lower()


def _run_layout(layout: torch.layout, channels: int, side: int, device: torch.device) -> dict:
    encoder, params, loc = _build(layout, channels, side, device)
    fwd, bwd, step = _make_fns(encoder, params, loc)
    nse = _nse(encoder)

    io_bytes = (
        params.numel() * params.element_size()
        + loc.numel() * loc.element_size()
        + 2 * encoder.volume_numel * NUM_SAMPLES * params.element_size()  # eps + samples
        + _index_bytes(encoder)
    )
    mem = memory.measure(fwd, bwd, io_bytes=io_bytes, device=device)
    params.grad = None
    timing = harness.do_bench(step, device=device)
    params.grad = None

    tag = _layout_tag(layout)
    print(
        f"  [{tag}] n={encoder.volume_numel} nse={nse} S={NUM_SAMPLES}: "
        f"fwd+bwd median={timing.median_ms:.3f}ms (p10={timing.p10_ms:.3f}, p90={timing.p90_ms:.3f}) "
        f"peak_fwd={mem.peak_fwd_mb:.1f}MB peak_bwd={mem.peak_bwd_mb:.1f}MB workspace={mem.workspace_mb:.1f}MB"
    )
    result = {
        "layout": tag,
        "n": encoder.volume_numel,
        "nse": nse,
        "timing": timing,
        "mem": mem,
    }
    del encoder, params, loc, fwd, bwd, step
    torch.cuda.empty_cache()
    return result


def _write_speed_row(res: dict, *, matrix: str, fp: Fingerprint, baseline: dict | None) -> None:
    baseline_ms = baseline["timing"].median_ms if baseline is not None else None
    speedup = baseline_ms / res["timing"].median_ms if baseline_ms is not None else None
    row = Result.build(
        op="e2e_rsample",
        family="e2e",
        backend="custom",
        variant=f"{res['layout']}·B=1·f32/i64·S={NUM_SAMPLES}",
        matrix=matrix,
        n=res["n"],
        m=res["n"],
        nse=res["nse"],
        p=NUM_SAMPLES,
        fp=fp,
        baseline_name="e2e rsample COO layout (same config)" if baseline is not None else None,
        baseline_ms=baseline_ms,
        ours_ms=res["timing"].median_ms,
        speedup=speedup,
        peak_fwd_mb=res["mem"].peak_fwd_mb,
        peak_bwd_mb=res["mem"].peak_bwd_mb,
        workspace_mb=res["mem"].workspace_mb,
        meta={"p10_ms": res["timing"].p10_ms, "p90_ms": res["timing"].p90_ms, "n_iters": res["timing"].n_iters},
    )
    path = write_result(row, filename=f"e2e_rsample_{res['layout']}_{matrix}.json")
    ratio = f" csr_vs_coo_speed={speedup:.2f}x" if speedup is not None else ""
    print(f"  row [{res['layout']}] ours={res['timing'].median_ms:.3f}ms{ratio} -> {path}")


def _write_pin_row(csr: dict, coo: dict, *, matrix: str, fp: Fingerprint) -> bool:
    ratio = csr["mem"].peak_bwd_mb / coo["mem"].peak_bwd_mb
    met = ratio <= MEM_BAR_RATIO
    row = Result.build(
        op="e2e_rsample",
        family="e2e",
        backend="custom",
        variant=f"csr_vs_coo·B=1·f32/i64·S={NUM_SAMPLES}",
        matrix=matrix,
        n=csr["n"],
        m=csr["n"],
        nse=csr["nse"],
        p=NUM_SAMPLES,
        fp=fp,
        baseline_name="COO layout peak_bwd (same config)",
        peak_fwd_mb=csr["mem"].peak_fwd_mb,
        peak_bwd_mb=csr["mem"].peak_bwd_mb,
        workspace_mb=csr["mem"].workspace_mb,
        mem_bar_met=met,
        bar=MEM_BAR,
        bar_met=met,
        meta={
            "coo_peak_bwd_mb": coo["mem"].peak_bwd_mb,
            "csr_peak_bwd_mb": csr["mem"].peak_bwd_mb,
            "csr_over_coo_peak_bwd": ratio,
        },
    )
    path = write_result(row, filename=f"e2e_rsample_mem_pin_{matrix}.json")
    print(
        f"  PIN [{matrix}] peak_bwd CSR={csr['mem'].peak_bwd_mb:.1f}MB "
        f"COO={coo['mem'].peak_bwd_mb:.1f}MB ratio={ratio:.3f}x bar={MEM_BAR} met={met} -> {path}"
    )
    return met


def main() -> int:
    if not torch.cuda.is_available():
        print("bench_e2e_rsample: no CUDA device visible -- the rsample memory pin cannot be evaluated.")
        return 1
    import torchsparsegradutils  # noqa: F401  (registers tsgu:: ops)
    from torchsparsegradutils._dispatch import backend_available

    if not backend_available():
        print("bench_e2e_rsample: torchsparsegradutils_cuda backend not available -- pin cannot be evaluated.")
        return 1

    device = torch.device("cuda")
    fp = fingerprint()
    budget_mb = _budget_gb() * 1024
    print(f"peak budget: {budget_mb:.0f}MB (min of corpus.peak_budget_gb and {HARD_PEAK_CAP_GB}GB hard cap)")

    failures: list[str] = []
    evaluated = 0
    prev: tuple[int, float] | None = None  # (voxels, measured max peak_bwd_mb across layouts)

    for channels, side in CONFIGS:
        matrix = f"encoder-{channels}x{side}^3"
        voxels = channels * side**3
        if prev is None:
            est_mb = _analytic_first_peak_mb(channels, side)
            est_src = "analytic bytes model"
        else:
            prev_voxels, prev_peak = prev
            est_mb = prev_peak * (voxels / prev_voxels) * SAFETY_FACTOR
            est_src = f"extrapolated ~cubically from {prev_voxels} voxels"
        if est_mb > budget_mb:
            print(
                f"SKIP {matrix}: estimated peak_bwd ~{est_mb:.0f}MB ({est_src}) exceeds "
                f"budget {budget_mb:.0f}MB -- not attempted (VRAM guard)."
            )
            continue

        print(f"config {matrix}: n={voxels}, estimated peak ~{est_mb:.0f}MB ({est_src})")
        coo = _run_layout(torch.sparse_coo, channels, side, device)
        csr = _run_layout(torch.sparse_csr, channels, side, device)

        _write_speed_row(coo, matrix=matrix, fp=fp, baseline=None)
        _write_speed_row(csr, matrix=matrix, fp=fp, baseline=coo)
        if not _write_pin_row(csr, coo, matrix=matrix, fp=fp):
            failures.append(
                f"{matrix}: peak_bwd(CSR)={csr['mem'].peak_bwd_mb:.1f}MB > "
                f"{MEM_BAR_RATIO}x peak_bwd(COO)={coo['mem'].peak_bwd_mb:.1f}MB"
            )
        evaluated += 1
        prev = (voxels, max(coo["mem"].peak_bwd_mb, csr["mem"].peak_bwd_mb))

    if evaluated == 0:
        print("bench_e2e_rsample: no config fit the VRAM budget -- the pin was never evaluated. Failing.")
        return 1
    if failures:
        print("bench_e2e_rsample: MEMORY PIN FAILED (benchmarks.md §3: 'or the suite fails'):")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"bench_e2e_rsample: pin held on all {evaluated} evaluated config(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
