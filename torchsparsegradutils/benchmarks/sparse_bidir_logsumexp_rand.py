#!/usr/bin/env python3
"""
Sparse Bidirectional Log-Sum-Exp Benchmark - Random Matrices

Benchmarks ``sparse_bidir_logsumexp`` (both row- and column-wise reductions in a
single traversal) against the two-call baseline
``(sparse_logsumexp(A, dim=0), sparse_logsumexp(A, dim=1))`` on randomly generated
sparse matrices of various sizes, sparsity patterns, layouts and dtypes. Forward
and backward time and peak memory are measured; the baseline is what the single-
pass primitive exists to beat. Both algorithms are driven through a single tensor
(``output_layout="padded"`` / ``torch.cat``) so ``measure_op``'s ``.sum().backward()``
exercises both reductions.
"""

import os
import sys

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
from benchmark_utils import (
    measure_op,
    print_benchmark_header,
    print_result_row,
    print_results_table_header,
    save_benchmark_results,
)
from tqdm import tqdm

from torchsparsegradutils import sparse_bidir_logsumexp, sparse_logsumexp
from torchsparsegradutils.utils import rand_sparse

REPEATS = 100
WARMUP_RUNS = 10

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

# problem sizes: (label, N, M, nnz)
SIZES = [
    ("small", 2**10, 2**10, 2**12),
    ("medium", 2**12, 2**12, 2**14),
    ("large", 2**14, 2**14, 2**16),
    ("xlarge", 2**16, 2**16, 2**18),
    ("million", 2**20, 2**20, 2**22),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

# Both ops return a single tensor so measure_op's out.sum().backward() drives both
# reductions. B is an unused placeholder for measure_op's binary signature.
ALGORITHMS = [
    ("sparse_bidir_logsumexp", lambda A, B: sparse_bidir_logsumexp(A, output_layout="padded")),
    ("two_call_baseline", lambda A, B: torch.cat([sparse_logsumexp(A, dim=0), sparse_logsumexp(A, dim=1)])),
]


def run_sparse_bidir_logsumexp_benchmark():
    """Run the sparse bidirectional log-sum-exp benchmark suite."""

    print_benchmark_header("Sparse Bidirectional Log-Sum-Exp Benchmark - Random Matrices")

    records = []

    for size_label, N, M, nnz in tqdm(SIZES, desc="Problem sizes"):
        print(f"\n🔍 Testing size: {size_label} (N={N}, M={M}, nnz={nnz})")

        A_shape = (N, M)
        B = torch.zeros(1, device=device)  # unused placeholder for measure_op's binary signature

        for idx_dt in tqdm(INDEX_DTYPES, desc="Index dtypes", leave=False):
            for val_dt in tqdm(VALUE_DTYPES, desc="Value dtypes", leave=False):
                for layout in tqdm(LAYOUTS, desc="Layouts", leave=False):
                    layout_name = "coo" if layout == torch.sparse_coo else "csr"
                    print(f"\n  📊 Configuration: idx_dtype={idx_dt}, val_dtype={val_dt}, layout={layout_name}")

                    # COO indices are always int64 -> skip int32 rather than mislabel a duplicate.
                    if layout == torch.sparse_coo and idx_dt != torch.int64:
                        print(f"    ⏭  skipping {layout_name} + {idx_dt} (COO indices are always int64)")
                        continue

                    # Generate once per config, outside the try so generation errors surface here.
                    A_sparse = rand_sparse(
                        A_shape, nnz, layout, indices_dtype=idx_dt, values_dtype=val_dt, device=device
                    )
                    actual_idx_dt = (
                        A_sparse.col_indices().dtype
                        if layout == torch.sparse_csr
                        else A_sparse.coalesce().indices().dtype
                    )
                    assert actual_idx_dt == idx_dt, f"index dtype mislabel: requested {idx_dt}, built {actual_idx_dt}"

                    print_results_table_header()

                    for alg_name, alg_fn in ALGORITHMS:
                        try:
                            print(f"    🧮 Testing {alg_name} ({layout_name})...")

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
        save_benchmark_results(records, "sparse_bidir_logsumexp_rand")

    print("\n✅ Sparse bidirectional log-sum-exp benchmark completed!")


if __name__ == "__main__":
    run_sparse_bidir_logsumexp_benchmark()
