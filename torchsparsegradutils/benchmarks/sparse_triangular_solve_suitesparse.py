#!/usr/bin/env python3
"""
Sparse Triangular Solve Benchmark - SuiteSparse Collection

This benchmark tests sparse triangular solve operations using matrices from
the SuiteSparse Matrix Collection. It compares different algorithms for
solving triangular systems Ax = B where A is sparse and triangular.

Key Features:
- Extracts the appropriate triangular part from general sparse matrices
- Calculates residuals using the extracted triangular matrix for accuracy verification
- This ensures meaningful residual computation that reflects the actual system solved
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
    load_mat_from_suitesparse_collection,
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

# from jax.lax.linalg import triangular_solve  # NOTE: jax doesn't have a sparse triangular solve
# import torchsparsegradutils.jax as tsgujax


REPEATS = 10
WARMUP_RUNS = 1

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

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
            A, B, solve=lambda A_c, B_c: spsolve_triangular(A_c, B_c, lower=not UPPER, unit_diagonal=UNITRIANGULAR)
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


def run_triangular_solve_benchmark():
    """Run the sparse triangular solve benchmark suite with SuiteSparse matrices."""

    print_benchmark_header("Sparse Triangular Solve Benchmark - SuiteSparse Collection")

    # Load the same sparse matrix as mm benchmark
    A_np_coo = load_mat_from_suitesparse_collection("Rothberg", "cfd2")
    N = A_np_coo.shape[0]
    M = 2  # number of columns in B

    triangle_type = "upper" if UPPER else "lower"
    unit_str = " (unit diagonal)" if UNITRIANGULAR else ""
    print(f"📊 Matrix: Rothberg/cfd2, shape={A_np_coo.shape}, original_nnz={A_np_coo.nnz}")
    print(f"🔺 Extracting {triangle_type} triangular part{unit_str} for proper benchmarking")

    records = []

    # Build and test for each dtype and layout
    for idx_dt in tqdm(INDEX_DTYPES, desc="Index dtypes"):
        for val_dt in tqdm(VALUE_DTYPES, desc="Value dtypes", leave=False):
            for layout in tqdm(LAYOUTS, desc="Layouts", leave=False):
                layout_name = "coo" if layout == torch.sparse_coo else "csr"
                print(f"\n🔍 Testing configuration: idx_dtype={idx_dt}, val_dtype={val_dt}, layout={layout_name}")

                # Convert matrix to torch sparse tensor - optimize tensor creation
                # Use numpy.stack to avoid the slow list-to-tensor conversion
                indices_np = np.stack([A_np_coo.row, A_np_coo.col], axis=0)
                indices = torch.from_numpy(indices_np).to(dtype=idx_dt, device=device)
                values = torch.from_numpy(A_np_coo.data).to(dtype=val_dt, device=device)
                A_full = torch.sparse_coo_tensor(
                    indices, values, A_np_coo.shape, dtype=val_dt, device=device
                ).coalesce()

                # Extract triangular part of the matrix for proper triangular solve benchmarking
                # This ensures residual calculation is meaningful and reflects realistic usage
                if UPPER:
                    # Keep only upper triangular part (i <= j)
                    if UNITRIANGULAR:
                        # Exclude diagonal for unit triangular (implicit diagonal of 1s)
                        mask = A_full.indices()[0] < A_full.indices()[1]
                    else:
                        # Include diagonal for non-unit upper triangular
                        mask = A_full.indices()[0] <= A_full.indices()[1]
                else:
                    # Keep only lower triangular part (i >= j)
                    if UNITRIANGULAR:
                        # Exclude diagonal for unit triangular (implicit diagonal of 1s)
                        mask = A_full.indices()[0] > A_full.indices()[1]
                    else:
                        # Include diagonal for non-unit lower triangular
                        mask = A_full.indices()[0] >= A_full.indices()[1]

                # Filter indices and values to keep only triangular part
                tri_indices = A_full.indices()[:, mask]
                tri_values = A_full.values()[mask]

                # Note: For unit triangular matrices, the diagonal is implicitly 1 and handled by the solver
                A_sparse = torch.sparse_coo_tensor(
                    tri_indices, tri_values, A_full.shape, dtype=val_dt, device=device
                ).coalesce()

                # Convert to requested layout
                if layout == torch.sparse_csr:
                    A_sparse = A_sparse.to_sparse_csr()

                print(f"    Original nnz: {A_full._nnz()}, Triangular nnz: {A_sparse._nnz()}")

                print_results_table_header()

                for alg_name, alg_fn in ALGORITHMS:
                    try:
                        print(f"  🧮 Testing {alg_name}...")

                        # Right-hand side matrix B
                        B = torch.randn(N, M, dtype=val_dt, device=device, requires_grad=True)

                        # Measure performance
                        t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd = measure_op(
                            alg_fn,
                            A_sparse,
                            B,
                            repeats=REPEATS,
                            device=device,
                            desc=alg_name,
                            warmup_runs=WARMUP_RUNS,
                            remove_outliers=True,
                        )

                        # Calculate residual norm for solution accuracy using the triangular matrix
                        with torch.no_grad():
                            x = alg_fn(A_sparse, B)
                            # Use the triangular matrix A_sparse (not A_full) for residual calculation
                            # This ensures the residual is computed correctly for the actual triangular system solved
                            residual = A_sparse @ x - B
                            resnorm = torch.norm(residual).cpu().item()
                            relative_resnorm = (
                                resnorm / torch.norm(B).cpu().item() if torch.norm(B).cpu().item() > 0 else 0.0
                            )

                        # Print result
                        print_result_row(
                            alg_name, (N, M), t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd
                        )
                        print(f"      📐 Residual norm: {resnorm:.2e}, Relative: {relative_resnorm:.2e}")

                        records.append(
                            {
                                "matrix": "Rothberg/cfd2",
                                "N": N,
                                "M": M,
                                "original_nnz": A_np_coo.nnz,
                                "triangular_nnz": A_sparse._nnz(),
                                "triangle_type": "upper" if UPPER else "lower",
                                "unitriangular": UNITRIANGULAR,
                                "index_dt": str(idx_dt).split(".")[-1],
                                "value_dt": str(val_dt).split(".")[-1],
                                "layout": layout_name,
                                "algorithm": alg_name,
                                "fwd_time_us": t_fwd,
                                "fwd_time_std_us": std_fwd,
                                "fwd_mem_MB": mem_fwd,
                                "fwd_mem_std_MB": std_mem_fwd,
                                "bwd_time_us": t_bwd,
                                "bwd_time_std_us": std_bwd,
                                "bwd_mem_MB": mem_bwd,
                                "bwd_mem_std_MB": std_mem_bwd,
                                "residual_norm": resnorm,
                                "relative_residual_norm": relative_resnorm,
                            }
                        )

                    except Exception as e:
                        print(f"  ❌ {alg_name} failed: {e}")

                        records.append(
                            {
                                "matrix": "Rothberg/cfd2",
                                "N": N,
                                "M": M,
                                "original_nnz": A_np_coo.nnz,
                                "triangular_nnz": A_sparse._nnz() if "A_sparse" in locals() else np.nan,
                                "triangle_type": "upper" if UPPER else "lower",
                                "unitriangular": UNITRIANGULAR,
                                "index_dt": str(idx_dt).split(".")[-1],
                                "value_dt": str(val_dt).split(".")[-1],
                                "layout": layout_name,
                                "algorithm": alg_name,
                                "fwd_time_us": np.nan,
                                "fwd_time_std_us": np.nan,
                                "fwd_mem_MB": np.nan,
                                "fwd_mem_std_MB": np.nan,
                                "bwd_time_us": np.nan,
                                "bwd_time_std_us": np.nan,
                                "bwd_mem_MB": np.nan,
                                "bwd_mem_std_MB": np.nan,
                                "residual_norm": np.nan,
                                "relative_residual_norm": np.nan,
                                "error": str(e)[:100],
                            }
                        )

    # Save results
    if records:
        save_benchmark_results(records, "sparse_triangular_solve_suitesparse")

    print("\n✅ Sparse triangular solve suite benchmark completed!")


if __name__ == "__main__":
    run_triangular_solve_benchmark()
    run_triangular_solve_benchmark()
