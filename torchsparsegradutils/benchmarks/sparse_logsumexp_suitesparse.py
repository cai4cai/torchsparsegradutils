#!/usr/bin/env python3
"""
Sparse Log-Sum-Exp Benchmark - SuiteSparse Collection

Benchmarks ``sparse_logsumexp`` against a dense ``torch.logsumexp`` baseline on a
real matrix from the SuiteSparse Matrix Collection (Rothberg/cfd2, ~123k x 123k).
The reduction is over ``dim=1``; forward and backward time and peak memory are
measured. The dense baseline materialises ``A.to_dense()`` and is the memory
ceiling the sparse path exists to avoid -- at this size it OOMs (recorded as NaN).
"""

import os
import sys

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
from benchmark_utils import (
    load_mat_from_suitesparse_collection,
    measure_op,
    print_benchmark_header,
    print_result_row,
    print_results_table_header,
    save_benchmark_results,
)
from tqdm import tqdm

from torchsparsegradutils import sparse_logsumexp

REPEATS = 100
WARMUP_RUNS = 10

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

# reduce over dim=1; measure_op takes a binary op, so B is an unused placeholder.
ALGORITHMS = [
    ("sparse_logsumexp", lambda A, B: sparse_logsumexp(A, dim=1)),
    ("dense.logsumexp", lambda A, B: torch.logsumexp(A.to_dense(), dim=1)),
]

MATRICES = [
    ("Rothberg", "cfd2"),  # 123k x 123k
    ("Williams", "webbase-1M"),  # 1M x 1M
]


def run_sparse_logsumexp_suitesparse_benchmark():
    """Run the sparse log-sum-exp benchmark suite with SuiteSparse matrices."""

    print_benchmark_header("Sparse Log-Sum-Exp Benchmark - SuiteSparse Collection")

    records = []
    B = torch.zeros(1, device=device)  # unused placeholder for measure_op's binary signature

    for dirname, matname in tqdm(MATRICES, desc="Matrices"):
        A_np_coo = load_mat_from_suitesparse_collection(dirname, matname)
        matrix_name = f"{dirname}/{matname}"
        N, M = A_np_coo.shape
        nnz = A_np_coo.nnz
        print(f"📊 Matrix: {matrix_name}, shape={A_np_coo.shape}, nnz={nnz}")

        for idx_dt in tqdm(INDEX_DTYPES, desc="Index dtypes", leave=False):
            for val_dt in tqdm(VALUE_DTYPES, desc="Value dtypes", leave=False):
                for layout in tqdm(LAYOUTS, desc="Layouts", leave=False):
                    layout_name = "coo" if layout == torch.sparse_coo else "csr"
                    print(f"\n🔍 Testing configuration: idx_dtype={idx_dt}, val_dtype={val_dt}, layout={layout_name}")

                    indices = torch.tensor([A_np_coo.row, A_np_coo.col], dtype=idx_dt, device=device)
                    values = torch.tensor(A_np_coo.data, dtype=val_dt, device=device)
                    A_sparse = torch.sparse_coo_tensor(
                        indices, values, A_np_coo.shape, dtype=val_dt, device=device
                    ).coalesce()
                    if layout == torch.sparse_csr:
                        A_sparse = A_sparse.to_sparse_csr()

                    print_results_table_header()

                    for alg_name, alg_fn in ALGORITHMS:
                        try:
                            print(f"  🧮 Testing {alg_name} ({layout_name})...")

                            (
                                t_fwd,
                                std_fwd,
                                mem_fwd,
                                std_mem_fwd,
                                t_bwd,
                                std_bwd,
                                mem_bwd,
                                std_mem_bwd,
                            ) = measure_op(
                                alg_fn,
                                A_sparse,
                                B,
                                repeats=REPEATS,
                                device=device,
                                desc=f"{alg_name} ({layout_name})",
                                warmup_runs=WARMUP_RUNS,
                                remove_outliers=True,
                            )

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
                                    "matrix": matrix_name,
                                    "N": N,
                                    "M": M,
                                    "nnz": nnz,
                                    "index_dt": str(idx_dt).split(".")[-1],
                                    "value_dt": str(val_dt).split(".")[-1],
                                    "layout": layout_name,
                                    "algo": alg_name,
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
                            print(f"  ❌ {alg_name} ({layout_name}) failed: {e}")

                            records.append(
                                {
                                    "matrix": matrix_name,
                                    "N": N,
                                    "M": M,
                                    "nnz": nnz,
                                    "index_dt": str(idx_dt).split(".")[-1],
                                    "value_dt": str(val_dt).split(".")[-1],
                                    "layout": layout_name,
                                    "algo": alg_name,
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

    if records:
        save_benchmark_results(records, "sparse_logsumexp_suitesparse")

    print("\n✅ Sparse log-sum-exp SuiteSparse benchmark completed!")


if __name__ == "__main__":
    run_sparse_logsumexp_suitesparse_benchmark()
