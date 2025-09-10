#!/usr/bin/env python3
"""
Sparse Matrix Multiplication Benchmark - SuiteSparse Collection

This benchmark tests sparse matrix multiplication operations using matrices from
the SuiteSparse Matrix Collection.
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
from tqdm import tqdm

from torchsparsegradutils import sparse_mm

REPEATS = 100
WARMUP_RUNS = 10

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

ALGORITHMS = [
    ("sparse.mm", lambda A, B: torch.sparse.mm(A, B)),  # NOTE: OOM
    ("sparse_mm", sparse_mm),
    ("dense.mm", lambda A, B: torch.matmul(A.to_dense(), B)),  # NOTE: OOM
]


def run_sparse_mm_suite_benchmark():
    """Run the sparse matrix multiplication benchmark suite with SuiteSparse matrices."""

    print_benchmark_header("Sparse Matrix Multiplication Benchmark - SuiteSparse Collection")

    # Load the same sparse matrix as solve benchmark
    A_np_coo = load_mat_from_suitesparse_collection("Rothberg", "cfd2")
    N = A_np_coo.shape[0]
    M = 128  # number of columns in B
    print(f"📊 Matrix: Rothberg/cfd2, shape={A_np_coo.shape}, nnz={A_np_coo.nnz}")

    records = []

    # Build and test for each dtype and layout
    for idx_dt in tqdm(INDEX_DTYPES, desc="Index dtypes"):
        for val_dt in tqdm(VALUE_DTYPES, desc="Value dtypes", leave=False):
            for layout in tqdm(LAYOUTS, desc="Layouts", leave=False):
                layout_name = "coo" if layout == torch.sparse_coo else "csr"
                print(f"\n🔍 Testing configuration: idx_dtype={idx_dt}, val_dtype={val_dt}, layout={layout_name}")

                # Convert matrix to torch sparse tensor
                indices = torch.tensor([A_np_coo.row, A_np_coo.col], dtype=idx_dt, device=device)
                values = torch.tensor(A_np_coo.data, dtype=val_dt, device=device)
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
                        B = torch.randn(N, M, dtype=val_dt, device=device)

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

                        # Print result
                        print_result_row(
                            alg_name, (N, M), t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd
                        )

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
                                "error": str(e),
                            }
                        )

    # Save results
    if records:
        save_benchmark_results(records, "sparse_mm_suite")

    print("\n✅ Sparse matrix multiplication suite benchmark completed!")


if __name__ == "__main__":
    run_sparse_mm_suite_benchmark()
