#!/usr/bin/env python3
"""
Sparse Matrix Multiplication Benchmark - Random Matrices

This benchmark tests sparse matrix multiplication operations using randomly
generated sparse matrices of various sizes and sparsity patterns.
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
from tqdm import tqdm

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.utils import rand_sparse

REPEATS = 100
WARMUP_RUNS = 10

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

# problem sizes: (label, N, M, nnz)
SIZES = [
    ("small", 2**10, 2**6, 2**12),
    ("medium", 2**14, 2**8, 2**14),
    ("large", 2**18, 2**9, 2**16),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

ALGORITHMS = [
    ("sparse.mm", lambda A, B: torch.sparse.mm(A, B)),
    ("sparse_mm", sparse_mm),
    ("dense.mm", lambda A, B: torch.matmul(A.to_dense(), B)),
]


def run_sparse_mm_benchmark():
    """Run the sparse matrix multiplication benchmark suite."""

    print_benchmark_header("Sparse Matrix Multiplication Benchmark - Random Matrices")

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
                            print(f"    🧮 Testing {alg_name} ({layout_name})...")

                            # build random A for matmul
                            A_sparse = rand_sparse(
                                A_shape, nnz, layout, indices_dtype=idx_dt, values_dtype=val_dt, device=device
                            )

                            # run benchmark
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
                            print(f"    ❌ {alg_name} ({layout_name}) failed: {e}")

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
                                    "error": str(e),
                                }
                            )

    # Save results
    if records:
        save_benchmark_results(records, "sparse_mm_rand")

    print("\n✅ Sparse matrix multiplication benchmark completed!")


if __name__ == "__main__":
    run_sparse_mm_benchmark()
