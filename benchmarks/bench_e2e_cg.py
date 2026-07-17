"""End-to-end composite benchmark: the CG solve loop through
``sparse_generic_solve`` (spec/commit.md Phase 4 commit 20).

    uv run python -m benchmarks.bench_e2e_cg

spec/benchmarks.md §3: "End-to-end composites (rsample of
`SparseMultivariateNormal`, CG solve loop) get one benchmark each — the
user-visible number, catching dispatch overhead that per-op benchmarks hide."

The composite: ``sparse_generic_solve(A, B, solve=linear_cg)`` on an SPD
matrix from ``utils.random_sparse.make_spd_sparse``, forward + backward
(``x.sum().backward()`` — backward runs the recursive transposed CG solve
plus one ``tsgu::sddmm`` for gradA), timed per benchmarks.md §1 and with
memory measured per benchmarks/memory.py. Sweep: n in {4096, 16384, 65536}
x n_rhs in {1, 8}, f32/i64 CSR. All rows backend="custom".

Baseline: ``torch.linalg.solve`` on the dense matrix, forward + backward —
an honesty reference (bar "n/a", bar_met null), recorded ONLY where the
dense system fits the budget (n=4096); larger n skip it with a printed note.

VRAM/host-RAM guard (4 GB card, spec task hard guards): ``make_spd_sparse``
materialises a dense n x n during construction, so each n is budget-checked
against a documented bytes model *before* any allocation: build on-device if
the model fits the device budget, fall back to a host (CPU) build + transfer
when it fits host RAM, and skip the config with a printed note otherwise
(n=65536's dense build wants ~100 GB — it is expected to skip on this
machine and is kept in the sweep so a bigger card grows into it).
"""

from __future__ import annotations

import sys

import torch

from benchmarks import corpus, harness, memory
from benchmarks.env import Fingerprint, fingerprint
from benchmarks.results import Result, write_result

SWEEP_N = (4096, 16384, 65536)
SWEEP_P = (1, 8)
VALUE_DTYPE = torch.float32
INDEX_DTYPE = torch.int64
LAYOUT = torch.sparse_csr

# ~0.5% off-diagonal density (corpus.REFERENCE_DENSITY regime): fraction of
# off-diagonal pairs make_spd_sparse zeroes out.
SPARSITY_RATIO = 0.995

BASELINE_N = 4096  # the only n whose dense torch.linalg.solve fits the budget
HARD_PEAK_CAP_GB = 2.5
# Dense-build workspace model for make_spd_sparse (bytes): M, M@M^T temp,
# n*eye temp, A_dense (4 x f32 dense) + the triu nonzero-index/randperm
# workspace (~12 bytes/element int64) => ~28 bytes per matrix element.
_SPD_BUILD_BYTES_PER_ELEM = 28
# Dense baseline model: A, LU factor, gradA, backward temps (~6 f32 denses).
_DENSE_SOLVE_BYTES_PER_ELEM = 24


def _budget_bytes() -> float:
    return min(corpus.peak_budget_gb(), HARD_PEAK_CAP_GB) * 2**30


def _host_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _build_spd(n: int, device: torch.device):
    """make_spd_sparse with the pre-allocation budget guard. Returns
    (A_sparse_on_device, A_dense_or_None) or None when n must be skipped."""
    from torchsparsegradutils.utils.random_sparse import make_spd_sparse

    build_bytes = _SPD_BUILD_BYTES_PER_ELEM * n * n
    if build_bytes <= _budget_bytes():
        build_device = device
    elif build_bytes <= 0.6 * _host_available_bytes():
        print(
            f"  note: n={n} dense SPD construction (~{build_bytes / 2**30:.1f}GB workspace) exceeds the "
            "device budget -- building on host CPU and transferring the sparse matrix."
        )
        build_device = torch.device("cpu")
    else:
        print(
            f"SKIP n={n}: make_spd_sparse materialises a dense {n}x{n} (~{build_bytes / 2**30:.1f}GB build "
            "workspace) -- exceeds both device budget and host RAM. Not attempted (memory guard)."
        )
        return None

    torch.manual_seed(corpus.SEED + n)
    A, A_dense = make_spd_sparse(n, LAYOUT, VALUE_DTYPE, INDEX_DTYPE, build_device, sparsity_ratio=SPARSITY_RATIO)
    if build_device.type != "cuda":
        A = A.to(device)
        A_dense = None  # host-built configs never run the dense baseline
    elif n != BASELINE_N:
        A_dense = None
    return A, A_dense


def _rel_residual(A: torch.Tensor, x: torch.Tensor, B: torch.Tensor) -> float:
    with torch.no_grad():
        r = torch.sparse.mm(A.detach(), x.detach()) - B.detach()
        return (r.norm() / B.detach().norm()).item()


def _bench_config(n: int, p: int, A: torch.Tensor, A_dense, device: torch.device, fp: Fingerprint) -> None:
    from torchsparsegradutils import sparse_generic_solve
    from torchsparsegradutils.solvers.cg import LinearCGSettings
    from torchsparsegradutils.utils import linear_cg

    settings = LinearCGSettings(max_cg_iterations=400, cg_tolerance=1e-5)
    nse = A.values().numel()

    A.requires_grad_(True)
    B_rhs = torch.randn(n, p, dtype=VALUE_DTYPE, device=device, requires_grad=True)

    def _fwd():
        A.grad = None
        B_rhs.grad = None
        return sparse_generic_solve(A, B_rhs, solve=linear_cg, settings=settings)

    def _bwd(x):
        x.sum().backward()

    def _step():
        _bwd(_fwd())

    with torch.no_grad():
        x_check = sparse_generic_solve(A.detach(), B_rhs.detach(), solve=linear_cg, settings=settings)
    rel_res = _rel_residual(A, x_check, B_rhs)
    del x_check

    io_bytes = (
        A.values().numel() * A.values().element_size()
        + A.col_indices().numel() * A.col_indices().element_size()
        + A.crow_indices().numel() * A.crow_indices().element_size()
        + 2 * B_rhs.numel() * B_rhs.element_size()  # B and x
    )
    mem = memory.measure(_fwd, _bwd, io_bytes=io_bytes, device=device)
    timing = harness.do_bench(_step, device=device)
    A.grad = None
    B_rhs.grad = None

    print(
        f"  ours (CG fwd+bwd) n={n} p={p} nse={nse}: median={timing.median_ms:.3f}ms "
        f"(p10={timing.p10_ms:.3f}, p90={timing.p90_ms:.3f}) rel_residual={rel_res:.2e} "
        f"peak_fwd={mem.peak_fwd_mb:.1f}MB peak_bwd={mem.peak_bwd_mb:.1f}MB workspace={mem.workspace_mb:.1f}MB"
    )

    baseline_name = None
    baseline_ms = None
    speedup = None
    bar = None
    if A_dense is not None:
        dense_bytes = _DENSE_SOLVE_BYTES_PER_ELEM * n * n
        if dense_bytes <= _budget_bytes():
            Ad = A_dense.detach().clone().requires_grad_(True)
            Bd = B_rhs.detach().clone().requires_grad_(True)

            def _dense_step():
                Ad.grad = None
                Bd.grad = None
                torch.linalg.solve(Ad, Bd).sum().backward()

            dense_timing = harness.do_bench(_dense_step, device=device)
            baseline_name = "dense torch.linalg.solve"
            baseline_ms = dense_timing.median_ms
            speedup = baseline_ms / timing.median_ms
            bar = "n/a"  # honesty reference, not an acceptance bar (benchmarks.md §3 has no CG-vs-dense bar)
            print(f"  baseline (dense torch.linalg.solve fwd+bwd): median={baseline_ms:.3f}ms speedup={speedup:.2f}x")
            del Ad, Bd
            torch.cuda.empty_cache()
        else:
            print(f"  note: dense baseline skipped at n={n} -- dense solve workspace exceeds budget.")
    elif n != BASELINE_N:
        print(f"  note: dense baseline only recorded at n={BASELINE_N} (larger dense systems bust the budget).")

    row = Result.build(
        op="e2e_cg",
        family="e2e",
        backend="custom",
        variant=f"csr·B=1·f32/i64·p={p}",
        matrix=f"synth-spd-n{n}",
        n=n,
        m=n,
        nse=nse,
        p=p,
        fp=fp,
        baseline_name=baseline_name,
        baseline_ms=baseline_ms,
        ours_ms=timing.median_ms,
        speedup=speedup,
        peak_fwd_mb=mem.peak_fwd_mb,
        peak_bwd_mb=mem.peak_bwd_mb,
        workspace_mb=mem.workspace_mb,
        bar=bar,
        bar_met=None,
        meta={
            "p10_ms": timing.p10_ms,
            "p90_ms": timing.p90_ms,
            "n_iters": timing.n_iters,
            "rel_residual": rel_res,
            "solver": "linear_cg",
            "cg_tolerance": settings.cg_tolerance,
            "max_cg_iterations": settings.max_cg_iterations,
        },
    )
    path = write_result(row, filename=f"e2e_cg_n{n}_p{p}.json")
    print(f"  row -> {path}")

    A.requires_grad_(False)
    torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("bench_e2e_cg: no CUDA device visible -- the tsgu:: solve path cannot run.")
        return 1
    import torchsparsegradutils  # noqa: F401  (registers tsgu:: ops)
    from torchsparsegradutils._dispatch import backend_available

    if not backend_available():
        print("bench_e2e_cg: torchsparsegradutils_cuda backend not available.")
        return 1

    device = torch.device("cuda")
    fp = fingerprint()
    print(f"peak budget: {_budget_bytes() / 2**30:.2f}GB (min of corpus.peak_budget_gb and {HARD_PEAK_CAP_GB}GB cap)")

    ran_any = False
    for n in SWEEP_N:
        built = _build_spd(n, device)
        if built is None:
            continue
        A, A_dense = built
        print(f"config n={n}: nse={A.values().numel()} ({A.values().numel() / (n * n):.2%} density)")
        for p in SWEEP_P:
            _bench_config(n, p, A, A_dense, device, fp)
            ran_any = True
        del A, A_dense
        torch.cuda.empty_cache()

    if not ran_any:
        print("bench_e2e_cg: no config fit the memory budget -- nothing measured. Failing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
