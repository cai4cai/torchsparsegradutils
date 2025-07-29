#!/usr/bin/env python3
import sys
import os
import time
import statistics
from tqdm import trange

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pandas as pd

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.utils import rand_sparse

# Only run on CUDA
device = torch.device("cuda:1")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

# problem sizes: (label, N, M, nnz)
SIZES = [
    # ("small", 2000, 128, 4000),
    ("medium", 5000, 256, 10000),
    # ("large", 10000, 512, 20000),
]
BATCH_SIZES = [4, 128]
REPEATS = 100

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

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


def measure_op(op, A, B, repeats=REPEATS):
    """
    A, B may be either Tensors (batched) or lists of Tensors (unbatched).
    """
    is_list = isinstance(A, list)

    # -- forward timing & memory --
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in trange(repeats, desc="forward repeats", leave=False):
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
    mem_fwd = torch.cuda.max_memory_allocated(device) / 1e6

    # -- backward timing & memory --
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    for _ in trange(repeats, desc="backward repeats", leave=False):
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
    t3 = time.perf_counter()
    mem_bwd = torch.cuda.max_memory_allocated(device) / 1e6

    avg_fwd = (t1 - t0) / repeats
    avg_bwd = (t3 - t2) / repeats
    return avg_fwd, mem_fwd, avg_bwd, mem_bwd


def main():
    records = []
    for size_label, N, M, nnz in SIZES:
        for batch in BATCH_SIZES:
            for idx_dt in INDEX_DTYPES:
                for val_dt in VALUE_DTYPES:
                    # batched inputs
                    A_coo_batch = rand_sparse(
                        (batch, N, N), nnz, torch.sparse_coo, indices_dtype=idx_dt, values_dtype=val_dt, device=device
                    ).coalesce()
                    A_csr_batch = rand_sparse(
                        (batch, N, N), nnz, torch.sparse_csr, indices_dtype=idx_dt, values_dtype=val_dt, device=device
                    )
                    B_batch = torch.randn((batch, N, M), dtype=val_dt, device=device)

                    # list-of-unbatched inputs
                    list_A_coo = [
                        rand_sparse(
                            (N, N), nnz, torch.sparse_coo, indices_dtype=idx_dt, values_dtype=val_dt, device=device
                        ).coalesce()
                        for _ in range(batch)
                    ]
                    list_A_csr = [
                        rand_sparse(
                            (N, N), nnz, torch.sparse_csr, indices_dtype=idx_dt, values_dtype=val_dt, device=device
                        )
                        for _ in range(batch)
                    ]
                    list_B = [torch.randn((N, M), dtype=val_dt, device=device) for _ in range(batch)]

                    for alg_name, alg_fn in ALGORITHMS:
                        if alg_name == "batched_sparse_mm":
                            # use the batched tensors
                            for layout_name, A_mat in [("coo", A_coo_batch), ("csr", A_csr_batch)]:
                                t_fwd, mem_fwd, t_bwd, mem_bwd = measure_op(alg_fn, A_mat, B_batch)
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
                                        "fwd_time_s": f"{t_fwd:.3f}",
                                        "fwd_mem_MB": f"{mem_fwd:.1f}",
                                        "bwd_time_s": f"{t_bwd:.3f}",
                                        "bwd_mem_MB": f"{mem_bwd:.1f}",
                                    }
                                )
                        else:
                            # list‐of‐unbatched variants
                            for layout_name, A_list in [("coo", list_A_coo), ("csr", list_A_csr)]:
                                t_fwd, mem_fwd, t_bwd, mem_bwd = measure_op(alg_fn, A_list, list_B)
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
                                        "fwd_time_s": f"{t_fwd:.3f}",
                                        "fwd_mem_MB": f"{mem_fwd:.1f}",
                                        "bwd_time_s": f"{t_bwd:.3f}",
                                        "bwd_mem_MB": f"{mem_bwd:.1f}",
                                    }
                                )

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
            "fwd_time_s",
            "fwd_mem_MB",
            "bwd_time_s",
            "bwd_mem_MB",
        ]
    ]

    md = df.to_markdown(index=False)
    out_path = "torchsparsegradutils/tests/benchmark_results_batched_mm.md"
    with open(out_path, "w") as f:
        f.write("# Batched sparse_mm vs list-of-unbatched benchmarks\n\n")
        f.write(md)
        f.write("\n")
    print(f"Written results to {out_path}")


if __name__ == "__main__":
    main()
