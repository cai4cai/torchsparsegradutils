#!/usr/bin/env python3
import sys
import os
# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import torch
import pandas as pd

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.utils import rand_sparse

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

# TODO: also be good to test batched matmul, using torch.bmm or torch.stack([torch.sparsse.mm(A, B) for ...])

# problem sizes: (label, N, M, nnz)
SIZES = [
    ("small",  2_000,  128, 4_000),
    ("medium", 5_000,  256, 10_000),
    ("large",  10_000, 512, 20_000),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

ALGORITHMS = [
    ("sparse.mm", lambda A, B: torch.sparse.mm(A, B)),
    ("sparse_mm", sparse_mm),
    ("dense.mm", lambda A, B: torch.matmul(A.to_dense(), B)),
]

def measure_op(op, A, B):
    """
    Measure forward/backward times and peak mem.
    Returns (t_fwd, mem_fwd, t_bwd, mem_bwd)
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # ensure new tensors for timing
    A1 = A.detach().clone().requires_grad_(True)
    B1 = B.detach().clone().requires_grad_(True)

    # -- forward --
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = op(A1, B1)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    mem_fwd = torch.cuda.max_memory_allocated(device) / 1e6

    # -- backward only --
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    out.sum().backward()
    torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    mem_bwd = torch.cuda.max_memory_allocated(device) / 1e6

    return (t1 - t0), mem_fwd, (t3 - t2), mem_bwd

def main():
    records = []
    for size_label, N, M, nnz in SIZES:
        A_shape = (N, N)
        B_shape = (N, M)

        for idx_dt in INDEX_DTYPES:
            for val_dt in VALUE_DTYPES:
                # build one sparse COO for all algos
                A_coo = rand_sparse(
                    A_shape, nnz, torch.sparse_coo,
                    indices_dtype=idx_dt,
                    values_dtype=val_dt,
                    device=device
                ).coalesce()
                B = torch.randn(B_shape, dtype=val_dt, device=device)

                # also build a CSR copy
                A_csr = A_coo.to_sparse_csr()

                for alg_name, alg_fn in ALGORITHMS:
                    for layout_name, A in [("coo", A_coo), ("csr", A_csr)]:
                        # run
                        t_fwd, mem_fwd, t_bwd, mem_bwd = measure_op(alg_fn, A, B)

                        records.append({
                            "size":      size_label,
                            "layout":    layout_name,
                            "algo":      alg_name,
                            "index_dt":  str(idx_dt).split(".")[-1],
                            "value_dt":  str(val_dt).split(".")[-1],
                            "N":         N,
                            "M":         M,
                            "nnz":       nnz,
                            "fwd_time_s":   f"{t_fwd:.3f}",
                            "fwd_mem_MB":   f"{mem_fwd:.1f}",
                            "bwd_time_s":   f"{t_bwd:.3f}",
                            "bwd_mem_MB":   f"{mem_bwd:.1f}",
                        })

    df = pd.DataFrame.from_records(records)
    # reorder columns for clarity
    df = df[[
        "size", "layout", "algo", "index_dt", "value_dt",
        "N", "M", "nnz",
        "fwd_time_s", "fwd_mem_MB", "bwd_time_s", "bwd_mem_MB"
    ]]

    md = df.to_markdown(index=False)
    with open("torchsparsegradutils/tests/benchmark_results_sparse_mm.md", "w") as f:
        f.write("# sparse_mm vs torch.sparse.mm vs dense.mm benchmark\n\n")
        f.write(md)
        f.write("\n")

    print("Written results to benchmark_results_sparse_mm.md")

if __name__ == "__main__":
    main()
