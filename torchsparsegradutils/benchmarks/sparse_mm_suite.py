#!/usr/bin/env python3
import sys
import os
import re

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import torch
import pandas as pd
from tqdm import trange

from torchsparsegradutils import sparse_mm

# reuse loader from solve benchmark
from torchsparsegradutils.tests.benchmark_sparse_solve import load_mat_from_suitesparse_collection

# Only run on CUDA
device = torch.device("cuda:1")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

REPEATS = 100

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

ALGORITHMS = [
    ("sparse.mm", lambda A, B: torch.sparse.mm(A, B)),  # NOTE: OOM
    ("sparse_mm", sparse_mm),
    ("dense.mm", lambda A, B: torch.matmul(A.to_dense(), B)),  # NOTE: OOM
]


# helper to extract attempted allocation (in MiB) from OOM message
def _parse_oom(e):
    msg = str(e)
    m = re.search(r"Tried to allocate ([\d\.]+) GiB", msg)
    if m:
        return float(m.group(1)) * 1024.0
    m = re.search(r"Tried to allocate ([\d\.]+) MiB", msg)
    if m:
        return float(m.group(1))
    return 0.0


def measure_op(op, A, B, repeats=REPEATS):
    """
    Measure average forward/backward times and peak mem over multiple runs.
    If an OOM occurs, capture attempted allocation and skip further steps.
    Returns (avg_fwd_time, max_fwd_mem, avg_bwd_time, max_bwd_mem).
    """
    # -- forward timing & memory --
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in trange(repeats, desc="forward repeats", leave=False):
            A1 = A.detach().clone().requires_grad_(True)
            B1 = B.detach().clone().requires_grad_(True)
            op(A1, B1)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        avg_fwd = (t1 - t0) / repeats
        mem_fwd = torch.cuda.max_memory_allocated(device) / 1e6
    except torch.cuda.OutOfMemoryError as e:
        # forward OOM: record attempted alloc as mem_fwd, skip backward
        mem_attempt = _parse_oom(e)
        return 0.0, mem_attempt, 0.0, 0.0

    # -- backward timing & memory --
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        for _ in trange(repeats, desc="backward repeats", leave=False):
            A1 = A.detach().clone().requires_grad_(True)
            B1 = B.detach().clone().requires_grad_(True)
            out = op(A1, B1)
            out.sum().backward()
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        avg_bwd = (t3 - t2) / repeats
        mem_bwd = torch.cuda.max_memory_allocated(device) / 1e6
    except torch.cuda.OutOfMemoryError as e:
        # backward OOM: record attempted alloc as mem_bwd
        mem_attempt = _parse_oom(e)
        return avg_fwd, mem_fwd, 0.0, mem_attempt

    return avg_fwd, mem_fwd, avg_bwd, mem_bwd


def main():
    # load the same sparse matrix as solve benchmark
    A_np_coo = load_mat_from_suitesparse_collection("Rothberg", "cfd2")
    N = A_np_coo.shape[0]
    records = []
    # build and test for each dtype
    for idx_dt in INDEX_DTYPES:
        for val_dt in VALUE_DTYPES:
            # construct sparse COO/CSR from numpy COO
            row, col, data = A_np_coo.row, A_np_coo.col, A_np_coo.data
            idx = torch.stack(
                [
                    torch.as_tensor(row, dtype=idx_dt, device=device),
                    torch.as_tensor(col, dtype=idx_dt, device=device),
                ],
                dim=0,
            )
            vals = torch.as_tensor(data, dtype=val_dt, device=device)
            A_coo = torch.sparse_coo_tensor(idx, vals, (N, N), dtype=val_dt, device=device).coalesce()
            A_csr = A_coo.to_sparse_csr()

            # random dense vector
            B = torch.randn((N, 1), dtype=val_dt, device=device)

            for layout_name, A in [("coo", A_coo), ("csr", A_csr)]:
                for alg_name, alg_fn in ALGORITHMS:
                    t_fwd, mem_fwd, t_bwd, mem_bwd = measure_op(alg_fn, A, B)
                    records.append(
                        {
                            "layout": layout_name,
                            "algo": alg_name,
                            "index_dt": str(idx_dt).split(".")[-1],
                            "value_dt": str(val_dt).split(".")[-1],
                            "N": N,
                            "fwd_time_s": f"{t_fwd:.3f}",
                            "fwd_mem_MB": f"{mem_fwd:.1f}",
                            "bwd_time_s": f"{t_bwd:.3f}",
                            "bwd_mem_MB": f"{mem_bwd:.1f}",
                        }
                    )

    df = pd.DataFrame.from_records(records)
    # only select columns we recorded
    df = df[
        [
            "layout",
            "algo",
            "index_dt",
            "value_dt",
            "N",
            "fwd_time_s",
            "fwd_mem_MB",
            "bwd_time_s",
            "bwd_mem_MB",
        ]
    ]

    # write only the mat-mul results
    mm_group = ["sparse.mm", "sparse_mm", "dense.mm"]
    sub = df[df["algo"].isin(mm_group)]
    md = sub.to_markdown(index=False)
    out_path = "torchsparsegradutils/benchmarks/sparse_mm_suite_results.md"
    with open(out_path, "w") as f:
        f.write("# sparse_mm vs torch.sparse.mm vs dense.mm benchmark\n\n")
        f.write(md)
        f.write("\n")
    print(f"Written results to {out_path}")


if __name__ == "__main__":
    main()
