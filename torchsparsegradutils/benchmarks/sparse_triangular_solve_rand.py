#!/usr/bin/env python3
"""
Sparse Triangular Solve Benchmark - Random Matrices

This benchmark tests sparse triangular solve operations using randomly
generated sparse triangular matrices of various sizes.
"""

import sys
import os

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pandas as pd
from tqdm import tqdm

from torchsparsegradutils import sparse_triangular_solve
from torchsparsegradutils.utils import rand_sparse, rand_sparse_tri

from torchsparsegradutils.cupy.cupy_sparse_solve import sparse_solve_c4t
from cupyx.scipy.sparse.linalg._solve import spsolve_triangular

from benchmark_utils import (
    measure_op,
    print_benchmark_header,
    print_results_table_header,
    print_result_row,
    format_time,
    save_benchmark_results,
    REPEATS,
)

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

# problem sizes: (label, N, M, nnz)
SIZES = [
    ("small", 2**10, 2**6, 2**12),
    ("medium", 2**14, 2**8, 2**14),
    ("large", 2**18, 2**9, 2**16),
]

UPPER = False
UNITRIANGULAR = False
TRANSPOSE = False

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

ALGORITHMS = [
    (
        "dense.triangular_solve",
        lambda A, B: torch.triangular_solve(
            B, A.to_dense(), upper=UPPER, unitriangular=UNITRIANGULAR, transpose=TRANSPOSE
        ).solution,
    ),
    (
        "torch_triangular_solve",
        lambda A, B: torch.triangular_solve(
            B, A, upper=UPPER, unitriangular=UNITRIANGULAR, transpose=TRANSPOSE
        ).solution,
    ),
    (
        "sparse_triangular_solve",
        lambda A, B: sparse_triangular_solve(A, B, upper=UPPER, unitriangular=UNITRIANGULAR, transpose=TRANSPOSE),
    ),
    # ("cupy.spsolve_triangular", lambda A, B: sparse_solve_c4t(A, B, solve=spsolve_triangular)),  # TODO: add lower and unit_diagonal
    # NOTE: This is very very slow
]


def run_sparse_triangular_solve_benchmark():
    """Run the sparse triangular solve benchmark suite."""

    print_benchmark_header("Sparse Triangular Solve Benchmark - Random Matrices")

    records = []

    for size_label, N, M, nnz in tqdm(SIZES, desc="Problem sizes"):
        print(f"\n🔍 Testing size: {size_label} (N={N}, M={M}, nnz={nnz})")

        A_shape = (N, N)
        B_shape = (N, M)

        for idx_dt in tqdm(INDEX_DTYPES, desc="Index dtypes", leave=False):
            for val_dt in tqdm(VALUE_DTYPES, desc="Value dtypes", leave=False):
                for layout in tqdm(LAYOUTS, desc="Layouts", leave=False):
                    layout_name = "coo" if layout == torch.sparse_coo else "csr"
                    print(f"\n  📊 Configuration: idx_dtype={idx_dt}, val_dtype={val_dt}, layout={layout_name}")

                    B = torch.randn(B_shape, dtype=val_dt, device=device)

                    print_results_table_header()

                    for alg_name, alg_fn in ALGORITHMS:
                        try:
                            print(f"    🧮 Testing {alg_name}...")

                            # Generate random triangular matrix for triangular solve
                            A_sparse = rand_sparse_tri(
                                A_shape,
                                nnz,
                                layout,
                                upper=UPPER,
                                strict=not UNITRIANGULAR,
                                indices_dtype=idx_dt,
                                values_dtype=val_dt,
                                device=device,
                            )

                            # Measure performance
                            t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd = measure_op(
                                alg_fn,
                                A_sparse,
                                B,
                                repeats=REPEATS,
                                device=device,
                                desc=f"{alg_name} ({layout_name})",
                                warmup_runs=10,
                                remove_outliers=True,
                            )

                            # Print result
                            print_result_row(
                                f"{alg_name} ({layout_name})",
                                (N, M),
                                t_fwd,
                                std_fwd,
                                mem_fwd,
                                std_mem_fwd,
                                t_bwd,
                                std_bwd,
                                mem_bwd,
                                std_mem_bwd,
                            )

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
                                    "fwd_time_us": t_fwd,
                                    "fwd_time_std_us": std_fwd,
                                    "fwd_mem_MB": mem_fwd,
                                    "fwd_mem_std_MB": std_mem_fwd,
                                    "bwd_time_us": t_bwd,
                                    "bwd_time_std_us": std_bwd,
                                    "bwd_mem_MB": mem_bwd,
                                    "bwd_mem_std_MB": std_mem_bwd,
                                }
                            )

                        except Exception as e:
                            print(f"    ❌ {alg_name} failed: {e}")

                            records.append(
                                {
                                    "size": size_label,
                                    "layout": "unknown",
                                    "algo": alg_name,
                                    "index_dt": str(idx_dt).split(".")[-1],
                                    "value_dt": str(val_dt).split(".")[-1],
                                    "N": N,
                                    "M": M,
                                    "nnz": nnz,
                                    "fwd_time_s": 0.0,
                                    "fwd_mem_MB": 0.0,
                                    "bwd_time_s": 0.0,
                                    "bwd_mem_MB": 0.0,
                                    "error": str(e)[:100],
                                }
                            )

    # Save results
    if records:
        save_benchmark_results(records, "sparse_triangular_solve_rand")

    print("\n✅ Sparse triangular solve benchmark completed!")


if __name__ == "__main__":
    run_sparse_triangular_solve_benchmark()
