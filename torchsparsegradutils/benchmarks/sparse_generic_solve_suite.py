#!/usr/bin/env python3
"""
Sparse Generic Solve Benchmark - SuiteSparse Collection

This benchmark tests sparse linear solve operations using matrices from
the SuiteSparse Matrix Collection. It compares different iterative solvers
for solving linear systems Ax = B where A is sparse.
"""

import sys
import os

# add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torchsparsegradutils.sparse_solve import sparse_generic_solve
from torchsparsegradutils.utils import convert_coo_to_csr, linear_cg, bicgstab, minres
from jax.scipy.sparse.linalg import cg as cg_jax, bicgstab as bicgstab_jax
from torchsparsegradutils.cupy.cupy_sparse_solve import sparse_solve_c4t
from cupyx.scipy.sparse.linalg._solve import lsqr as lsqr_cu, minres as minres_cu, spsolve as spsolve_cu
import torchsparsegradutils.jax as tsgujax

from benchmark_utils import (
    load_mat_from_suitesparse_collection,
    measure_op,
    print_benchmark_header,
    print_results_table_header,
    print_result_row,
    format_time,
    format_memory,
    save_benchmark_results,
)

M = 1  # number of columns in B
REPEATS = 1
WARMUP_RUNS = 0
# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

# BUG: Sometimes I am getting: Backward OOM: attempted 36373575.7 MiB. Not sure what is causing this.

ALGORITHMS = [
    ("dense.solve", lambda A, B: torch.linalg.solve(A.to_dense(), B)),
    ("sparse_generic_cg", lambda A, B: sparse_generic_solve(A, B, solve=linear_cg, transpose_solve=linear_cg)),
    ("sparse_generic_bicgstab", lambda A, B: sparse_generic_solve(A, B, solve=bicgstab, transpose_solve=bicgstab)),
    ("sparse_generic_minres", lambda A, B: sparse_generic_solve(A, B, solve=minres, transpose_solve=minres)),
    ("cupy_spsolve", lambda A, B: sparse_solve_c4t(A, B, solve=spsolve_cu, transpose_solve=spsolve_cu)),
    ("cupy_lsqr", lambda A, B: sparse_solve_c4t(A, B, solve=lsqr_cu, transpose_solve=lsqr_cu)),
    ("cupy_minres", lambda A, B: sparse_solve_c4t(A, B, solve=minres_cu, transpose_solve=minres_cu)),
    ("jax_default", lambda A, B: tsgujax.sparse_solve_j4t(A, B)),
    ("jax_cg", lambda A, B: tsgujax.sparse_solve_j4t(A, B, solve=cg_jax, transpose_solve=cg_jax)),
    ("jax_bicgstab", lambda A, B: tsgujax.sparse_solve_j4t(A, B, solve=bicgstab_jax, transpose_solve=bicgstab_jax)),
]


def run_sparse_solve_benchmark():
    """Run the sparse solve benchmark suite with SuiteSparse matrices."""

    print_benchmark_header("Sparse Generic Solve Benchmark - SuiteSparse Collection")

    # Load the same sparse matrix as other benchmarks
    A_np_coo = load_mat_from_suitesparse_collection("Rothberg", "cfd2")
    N = A_np_coo.shape[0]

    print(f"📊 Matrix: Rothberg/cfd2, shape={A_np_coo.shape}, nnz={A_np_coo.nnz}")

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
                A_sparse = torch.sparse_coo_tensor(
                    indices, values, A_np_coo.shape, dtype=val_dt, device=device
                ).coalesce()

                # Convert to requested layout
                if layout == torch.sparse_csr:
                    A_sparse = A_sparse.to_sparse_csr()

                print_results_table_header()

                for alg_name, alg_fn in ALGORITHMS:
                    try:
                        print(f"  🧮 Testing {alg_name}...")

                        # Right-hand side matrix B
                        B = torch.randn(N, M, dtype=val_dt, device=device, requires_grad=True)
                        B = B.squeeze(-1)  # Ensure B is 1D for M = 1

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

                        # Calculate residual norm for solution accuracy
                        with torch.no_grad():
                            x = alg_fn(A_sparse, B)
                            residual = A_sparse @ x - B
                            resnorm = torch.norm(residual).cpu().item()

                        # Print result with residual norm
                        print_result_row(
                            alg_name, (N, M), t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd
                        )
                        print(f"    📐 Residual norm: {resnorm:.2e}")

                        records.append(
                            {
                                "matrix": "Rothberg/cfd2",
                                "N": N,
                                "M": M,
                                "nnz": A_np_coo.nnz,
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
                                "resnorm": resnorm,
                            }
                        )

                    except Exception as e:
                        print(f"  ❌ {alg_name} failed: {e}")

                        records.append(
                            {
                                "matrix": "Rothberg/cfd2",
                                "N": N,
                                "M": M,
                                "nnz": A_np_coo.nnz,
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
                                "error": str(e)[:100],
                                "resnorm": np.nan,
                            }
                        )

    # Save results
    if records:
        save_benchmark_results(records, "sparse_generic_solve_suite")

    print("\n✅ Sparse generic solve benchmark completed!")


if __name__ == "__main__":
    run_sparse_solve_benchmark()
