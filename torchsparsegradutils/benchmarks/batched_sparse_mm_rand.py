#!/usr/bin/env python3
"""
Batched Sparse Matrix Multiplication Benchmark

This benchmark tests batched sparse matrix multiplication operations using randomly
generated sparse matrices, comparing different batching strategies.
"""

import sys
import os

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.utils import rand_sparse

from benchmark_utils import (
    _parse_oom,
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
    # ("medium", 2**14, 2**8, 2**14),
    # ("large", 2**18, 2**9, 2**16),
]

BATCH_SIZES = [4, 128]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

ALGORITHMS = [
    (
        "torch_sparse_mm_list",
        # NOTE: Cannot do .backward() with indexing batched sparse matrix, so we have to keep everything in a list
        lambda A, B: torch.stack([torch.sparse.mm(A[i], B[i]) for i in range(len(A))]),
    ),
    (
        "sparse_mm_list",
        lambda A, B: torch.stack([sparse_mm(A[i], B[i]) for i in range(len(A))]),
    ),
    ("batched_sparse_mm", sparse_mm),
]


def measure_batched_op(op, A, B, repeats=REPEATS, device=None, desc="operation"):
    """
    Measure performance for batched sparse operations.
    A, B may be either Tensors (batched) or lists of Tensors (unbatched).

    Returns:
        tuple: (avg_fwd_time_us, std_fwd_time_us, max_fwd_mem_mb, std_fwd_mem_mb,
                avg_bwd_time_us, std_bwd_time_us, max_bwd_mem_mb, std_bwd_mem_mb)
    """
    if device is None:
        device = torch.device("cuda")

    is_list = isinstance(A, list)

    try:
        # -- Forward timing & memory --
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        import time
        from tqdm import trange

        fwd_times = []
        fwd_mems = []

        for _ in trange(repeats, desc=f"{desc} (forward)", leave=False):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            if is_list:
                As = [a.detach().clone().requires_grad_(True) for a in A]
                Bs = [b.detach().clone().requires_grad_(True) for b in B]
                _ = op(As, Bs)
            else:
                A1 = A.detach().clone().requires_grad_(True)
                B1 = B.detach().clone().requires_grad_(True)
                _ = op(A1, B1)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            fwd_times.append((t1 - t0) * 1e6)  # Convert to microseconds
            fwd_mems.append(torch.cuda.max_memory_allocated(device) / 1e6)  # Convert to MB

        avg_fwd = np.mean(fwd_times)
        std_fwd = np.std(fwd_times)
        avg_fwd_mem = np.mean(fwd_mems)
        std_fwd_mem = np.std(fwd_mems)

    except torch.cuda.OutOfMemoryError as e:
        # Forward OOM: return NaN values
        mem_attempt = _parse_oom(e)
        print(f"⚠️  Forward OOM: attempted {mem_attempt:.1f} MiB")
        return np.nan, np.nan, mem_attempt, np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        # -- Backward timing & memory --
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        bwd_times = []
        bwd_mems = []

        for _ in trange(repeats, desc=f"{desc} (backward)", leave=False):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            if is_list:
                As = [a.detach().clone().requires_grad_(True) for a in A]
                Bs = [b.detach().clone().requires_grad_(True) for b in B]
                out = op(As, Bs)
            else:
                A1 = A.detach().clone().requires_grad_(True)
                B1 = B.detach().clone().requires_grad_(True)
                out = op(A1, B1)
            out.sum().backward()
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            bwd_times.append((t1 - t0) * 1e6)  # Convert to microseconds
            bwd_mems.append(torch.cuda.max_memory_allocated(device) / 1e6)  # Convert to MB

        avg_bwd = np.mean(bwd_times)
        std_bwd = np.std(bwd_times)
        avg_bwd_mem = np.mean(bwd_mems)
        std_bwd_mem = np.std(bwd_mems)

    except torch.cuda.OutOfMemoryError as e:
        # Backward OOM: record attempted alloc
        mem_attempt = _parse_oom(e)
        print(f"⚠️  Backward OOM: attempted {mem_attempt:.1f} MiB")
        return avg_fwd, std_fwd, avg_fwd_mem, std_fwd_mem, np.nan, np.nan, mem_attempt, np.nan

    return avg_fwd, std_fwd, avg_fwd_mem, std_fwd_mem, avg_bwd, std_bwd, avg_bwd_mem, std_bwd_mem


def run_batched_sparse_mm_benchmark():
    """Run the batched sparse matrix multiplication benchmark suite."""

    print_benchmark_header("Batched Sparse Matrix Multiplication Benchmark")

    records = []

    for size_label, N, M, nnz in tqdm(SIZES, desc="Problem sizes"):
        for batch in tqdm(BATCH_SIZES, desc="Batch sizes", leave=False):
            print(f"\n🔍 Testing: {size_label} (N={N}, M={M}, nnz={nnz}, batch={batch})")

            for idx_dt in tqdm(INDEX_DTYPES, desc="Index dtypes", leave=False):
                for val_dt in tqdm(VALUE_DTYPES, desc="Value dtypes", leave=False):
                    for layout in tqdm(LAYOUTS, desc="Layouts", leave=False):
                        layout_name = "coo" if layout == torch.sparse_coo else "csr"
                        print(f"\n  📊 Configuration: idx_dtype={idx_dt}, val_dtype={val_dt}, layout={layout_name}")
                        print_results_table_header()

                        # Generate batched inputs
                        A_batch = rand_sparse(
                            (batch, N, N), nnz, layout, indices_dtype=idx_dt, values_dtype=val_dt, device=device
                        )
                        if layout == torch.sparse_coo:
                            A_batch = A_batch.coalesce()
                        B_batch = torch.randn((batch, N, M), dtype=val_dt, device=device)

                        # list-of-unbatched inputs
                        list_A = [
                            rand_sparse((N, N), nnz, layout, indices_dtype=idx_dt, values_dtype=val_dt, device=device)
                            for _ in range(batch)
                        ]
                        if layout == torch.sparse_coo:
                            list_A = [a.coalesce() for a in list_A]
                        list_B = [torch.randn((N, M), dtype=val_dt, device=device) for _ in range(batch)]

                        for alg_name, alg_fn in ALGORITHMS:
                            try:
                                if alg_name == "batched_sparse_mm":
                                    # use the batched tensors
                                    print(f"    🧮 Testing {alg_name} ({layout_name})...")

                                    t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd = (
                                        measure_batched_op(
                                            alg_fn,
                                            A_batch,
                                            B_batch,
                                            repeats=REPEATS,
                                            device=device,
                                            desc=f"{alg_name} ({layout_name})",
                                        )
                                    )

                                    print_result_row(
                                        f"{alg_name} ({layout_name})",
                                        f"batch={batch}, {N}x{M}",
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
                                            "batch": batch,
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
                                else:
                                    # list‐of‐unbatched variants
                                    print(f"    🧮 Testing {alg_name} ({layout_name})...")

                                    t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd = (
                                        measure_batched_op(
                                            alg_fn,
                                            list_A,
                                            list_B,
                                            repeats=REPEATS,
                                            device=device,
                                            desc=f"{alg_name} ({layout_name})",
                                        )
                                    )

                                    print_result_row(
                                        f"{alg_name} ({layout_name})",
                                        f"batch={batch}, {N}x{M}",
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
                                            "batch": batch,
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

    # Save results
    if records:
        df = pd.DataFrame.from_records(records)
        df = df[
            [
                "size",
                "batch",
                "layout",
                "algo",
                "index_dt",
                "value_dt",
                "N",
                "M",
                "nnz",
                "fwd_time_us",
                "fwd_time_std_us",
                "fwd_mem_MB",
                "fwd_mem_std_MB",
                "bwd_time_us",
                "bwd_time_std_us",
                "bwd_mem_MB",
                "bwd_mem_std_MB",
            ]
        ]

        # Save CSV
        save_benchmark_results(records, "batched_sparse_mm_rand")

    print("\n✅ Batched sparse matrix multiplication benchmark completed!")


if __name__ == "__main__":
    run_batched_sparse_mm_benchmark()
