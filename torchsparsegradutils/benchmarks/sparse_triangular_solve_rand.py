#!/usr/bin/env python3
import sys
import os

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import torch
import pandas as pd

from torchsparsegradutils import sparse_triangular_solve
from torchsparsegradutils.utils import rand_sparse, rand_sparse_tri

from torchsparsegradutils.cupy.cupy_sparse_solve import sparse_solve_c4t
from cupyx.scipy.sparse.linalg._solve import spsolve_triangular

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

# problem sizes: (label, N, M, nnz)
SIZES = [
    ("small", 2000, 128, 4000),
    ("medium", 5000, 256, 10000),
    ("large", 10000, 512, 20000),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

ALGORITHMS = [
    (
        "dense.triangular_solve",
        lambda A, B: torch.triangular_solve(
            B.detach(), A.to_dense().detach(), upper=True, unitriangular=False, transpose=False
        ).solution,
    ),
    ("sparse_triangular_solve", sparse_triangular_solve),
    ("cupy.spsolve_triangular", lambda A, B: sparse_solve_c4t(A, B, solve=spsolve_triangular)),
]

# ("cupy.spsolve_tri", lambda A, B: sparse_solve_c4t(A, B, solve=spsolve_triangular)),

REPEATS = 100


def measure_op(op, A, B, repeats=REPEATS):
    """
    Measure forward/backward times and peak mem.
    Returns (t_fwd, mem_fwd, t_bwd, mem_bwd)
    """
    # -- forward timing over repeats --
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        A1 = A.detach().clone().requires_grad_(True)
        B1 = B.detach().clone().requires_grad_(True)
        op(A1, B1)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    avg_fwd = (t1 - t0) / repeats
    mem_fwd = torch.cuda.max_memory_allocated(device) / 1e6

    # -- backward timing over repeats --
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    for _ in range(repeats):
        A1 = A.detach().clone().requires_grad_(True)
        B1 = B.detach().clone().requires_grad_(True)
        out = op(A1, B1)
        out.sum().backward()
    torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    avg_bwd = (t3 - t2) / repeats
    mem_bwd = torch.cuda.max_memory_allocated(device) / 1e6

    return avg_fwd, mem_fwd, avg_bwd, mem_bwd


def main():
    records = []
    for size_label, N, M, nnz in SIZES:
        A_shape = (N, N)
        B_shape = (N, M)

        for idx_dt in INDEX_DTYPES:
            for val_dt in VALUE_DTYPES:
                B = torch.randn(B_shape, dtype=val_dt, device=device)

                for alg_name, alg_fn in ALGORITHMS:
                    # rebuild A per-algorithm
                    if "triangular_solve" in alg_name:
                        A_coo = rand_sparse_tri(
                            A_shape,
                            nnz,
                            torch.sparse_coo,
                            upper=True,
                            strict=True,
                            indices_dtype=idx_dt,
                            values_dtype=val_dt,
                            device=device,
                        ).coalesce()
                        A_csr = rand_sparse_tri(
                            A_shape,
                            nnz,
                            torch.sparse_csr,
                            upper=True,
                            strict=True,
                            indices_dtype=idx_dt,
                            values_dtype=val_dt,
                            device=device,
                        )
                    else:
                        A_coo = rand_sparse(
                            A_shape, nnz, torch.sparse_coo, indices_dtype=idx_dt, values_dtype=val_dt, device=device
                        ).coalesce()
                        A_csr = A_coo.to_sparse_csr()

                    for layout_name, A in [("coo", A_coo), ("csr", A_csr)]:
                        # run
                        t_fwd, mem_fwd, t_bwd, mem_bwd = measure_op(alg_fn, A, B)
                        records.append(
                            {
                                "size": size_label,
                                "layout": layout_name,
                                "algo": alg_name,
                                "index_dt": str(idx_dt).split(".")[-1],
                                "value_dt": str(val_dt).split(".")[-1],
                                "N": N,
                                "M": M,
                                "nnz": nnz,
                                "fwd_time_s": f"{t_fwd:.3f}",
                                "fwd_mem_MB": f"{mem_fwd:.1f}",
                                "bwd_time_s": f"{t_bwd:.3f}",
                                "bwd_mem_MB": f"{mem_bwd:.1f}",
                            }
                        )

    df = pd.DataFrame.from_records(records)
    df = df[
        [
            "size",
            "layout",
            "algo",
            "index_dt",
            "value_dt",
            "N",
            "M",
            "nnz",
            "fwd_time_s",
            "fwd_mem_MB",
            "bwd_time_s",
            "bwd_mem_MB",
        ]
    ]

    # write all results into a single markdown file
    out_path = "torchsparsegradutils/benchmarks/sparse_triangular_solve_rand_results.md"
    md = df.to_markdown(index=False)
    with open(out_path, "w") as f:
        f.write("# Sparse triangular and generic solve benchmark\n\n")
        f.write(md)
        f.write("\n")
    print(f"Written results to {out_path}")


if __name__ == "__main__":
    main()
