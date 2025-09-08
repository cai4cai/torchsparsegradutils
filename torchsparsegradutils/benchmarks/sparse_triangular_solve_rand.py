#!/usr/bin/env python3
"""
Sparse Triangular Solve Benchmark - Random Matrices

This benchmark tests sparse triangular solve operations using randomly
generated sparse triangular matrices of various sizes.
"""

import os
import sys

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch
from benchmark_utils import (
    format_time,
    measure_op,
    print_benchmark_header,
    print_result_row,
    print_results_table_header,
    save_benchmark_results,
)
from cupyx.scipy.sparse.linalg._solve import spsolve_triangular
from tqdm import tqdm

from torchsparsegradutils import sparse_triangular_solve
from torchsparsegradutils.cupy.cupy_sparse_solve import sparse_solve_c4t
from torchsparsegradutils.utils import rand_sparse, rand_sparse_tri

# from jax.lax.linalg import triangular_solve  # NOTE: jax doesn't have a sparse triangular solve
# import torchsparsegradutils.jax as tsgujax


REPEATS = 10
WARMUP_RUNS = 1

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

# problem sizes: (label, N, M, nnz)
SIZES = [
    ("small", 2**10, 2**1, 2**11),
    ("medium", 2**14, 2**2, 2**15),
    ("large", 2**18, 2**3, 2**19),
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
    (
        "cupy.spsolve_triangular",
        lambda A, B: sparse_solve_c4t(
            A,
            B,
            solve=lambda A_inner, b_inner: spsolve_triangular(
                A_inner, b_inner, lower=not UPPER, unit_diagonal=UNITRIANGULAR
            ),
        ),
    ),
    # (
    #     "jax.triangular_solve",
    #     lambda A, B: tsgujax.sparse_solve_j4t(
    #         A,
    #         B,
    #         solve=lambda A_jax, B_jax: triangular_solve(
    #             A_jax, B_jax, left_side=True, lower=not UPPER, transpose_a=TRANSPOSE, unit_diagonal=UNITRIANGULAR
    #         ),
    #     ),
    # ),
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
                                strict=UNITRIANGULAR,
                                indices_dtype=idx_dt,
                                values_dtype=val_dt,
                                device=device,
                                well_conditioned=True,
                                min_diag_value=1.0,
                            )

                            # Measure performance
                            t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd = measure_op(
                                alg_fn,
                                A_sparse,
                                B,
                                repeats=REPEATS,
                                device=device,
                                desc=f"{alg_name} ({layout_name})",
                                warmup_runs=WARMUP_RUNS,
                                remove_outliers=True,
                            )

                            # Calculate residual norm for solution accuracy
                            with torch.no_grad():
                                x = alg_fn(A_sparse, B)
                                residual = A_sparse @ x - B
                                resnorm = torch.norm(residual).cpu().item()
                                relative_resnorm = resnorm / torch.norm(B).cpu().item()

                            # # Calculate residual norm for solution accuracy
                            # with torch.no_grad():
                            #     x = alg_fn(A_sparse, B)
                            #     # Calculate Ax - B for triangular solve verification
                            #     if A_sparse.layout == torch.sparse_coo or A_sparse.layout == torch.sparse_csr:
                            #         Ax = torch.sparse.mm(A_sparse, x)
                            #     else:
                            #         Ax = A_sparse @ x
                            #     residual = Ax - B
                            #     resnorm = torch.norm(residual).cpu().item()
                            #     relative_resnorm = resnorm / torch.norm(B).cpu().item()

                            # Print result with residual norm
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
                            print(f"      📐 Residual norm: {resnorm:.2e}, Relative: {relative_resnorm:.2e}")

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
                                    "resnorm": resnorm,
                                    "relative_resnorm": relative_resnorm,
                                }
                            )

                        except Exception as e:
                            print(f"    ❌ {alg_name} failed: {e}")

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
                                    "fwd_time_us": np.nan,
                                    "fwd_time_std_us": np.nan,
                                    "fwd_mem_MB": np.nan,
                                    "fwd_mem_std_MB": np.nan,
                                    "bwd_time_us": np.nan,
                                    "bwd_time_std_us": np.nan,
                                    "bwd_mem_MB": np.nan,
                                    "bwd_mem_std_MB": np.nan,
                                    "resnorm": np.nan,
                                    "relative_resnorm": np.nan,
                                    "error": str(e)[:100],
                                }
                            )

    # Save results
    if records:
        save_benchmark_results(records, "sparse_triangular_solve_rand")

    print("\n✅ Sparse triangular solve benchmark completed!")


if __name__ == "__main__":
    run_sparse_triangular_solve_benchmark()
