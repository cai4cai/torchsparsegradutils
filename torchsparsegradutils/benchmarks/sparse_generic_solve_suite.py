#!/usr/bin/env python3
import sys
import os

# add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import torch
import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import urllib
import tarfile
import re

from torchsparsegradutils.sparse_solve import sparse_generic_solve
from torchsparsegradutils.utils import convert_coo_to_csr, linear_cg, bicgstab, minres
from jax.scipy.sparse.linalg import cg as cg_jax, bicgstab as bicgstab_jax
from torchsparsegradutils.cupy.cupy_sparse_solve import sparse_solve_c4t
from cupyx.scipy.sparse.linalg._solve import spsolve_triangular
import torchsparsegradutils.jax as tsgujax

# Only run on CUDA
device = torch.device("cuda:1")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

REPEATS = 1
BACKWARD = True

INDEX_DTYPES = [torch.int32]  # , torch.int64
VALUE_DTYPES = [torch.float32]  # , torch.float64
ALGORITHMS = [
    # ("dense.solve", lambda A, B: torch.linalg.solve(A.to_dense(), B)),  # requires ~64GB VRAM for forward
    ("sparse generic cg", lambda A, B: sparse_generic_solve(A, B, solve=linear_cg)),  # requires ~35TB for backward
    ("sparse generic bicgstab", lambda A, B: sparse_generic_solve(A, B, solve=bicgstab)),  # requires ~35TB for backward
    ("sparse generic minres", lambda A, B: sparse_generic_solve(A, B, solve=minres)),  # requires ~35TB for backward
    # ("cupy default", lambda A, B: sparse_solve_c4t(A, B)),  # requires ~35TB for backward  # BUG: cupy_backends.cuda.libs.cusolver.CUSOLVERError: CUSOLVER_STATUS_ALLOC_FAILED
    ("jax default", lambda A, B: tsgujax.sparse_solve_j4t(A, B)),  # requires ~35TB for backward
    ("jax cg", lambda A, B: tsgujax.sparse_solve_j4t(A, B, solve=cg_jax)),  # requires ~35TB for backward
    ("jax bicgstab", lambda A, B: tsgujax.sparse_solve_j4t(A, B, solve=bicgstab_jax)),  # requires ~35TB for backward
]


# -- load SuiteSparse matrix once --
def load_mat_from_suitesparse_collection(dirname, matname):
    # eg: https://suitesparse-collection-website.herokuapp.com/Rothberg/cfd2
    base_url = "https://suitesparse-collection-website.herokuapp.com/MM/"
    url = base_url + dirname + "/" + matname + ".tar.gz"
    compressed = matname + ".tar.gz"
    if not os.path.exists(compressed):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, filename=compressed)
    folder = "./" + matname
    localfile = f"{folder}/{matname}.mtx"
    if not os.path.exists(localfile):
        print(f"Extracting {compressed}")
        with tarfile.open(compressed) as tf:
            tf.extractall("./")
    A_np_coo = scipy.io.mmread(localfile)
    print(f"Loaded suitesparse matrix {dirname}/{matname}: shape={A_np_coo.shape}")
    return A_np_coo


def make_spd_sparse(n, layout, value_dtype, index_dtype, device):
    # ...existing code from tests...
    M = torch.randn(n, n, dtype=value_dtype, device=device)
    A_dense = M @ M.t() + n * torch.eye(n, dtype=value_dtype, device=device)
    idx = A_dense.nonzero(as_tuple=False).t().to(index_dtype)
    vals = A_dense[idx[0], idx[1]].clone()
    if layout == torch.sparse_coo:
        A = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
    else:
        A_coo = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
        A = convert_coo_to_csr(A_coo)
    return A, A_dense


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
    # -- forward timing & memory --
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            A1 = A.detach().clone().requires_grad_(True)
            B1 = B.detach().clone().requires_grad_(True)
            op(A1, B1)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        avg_fwd = (t1 - t0) / repeats
        mem_fwd = torch.cuda.max_memory_allocated(device) / 1e6
    except torch.cuda.OutOfMemoryError as e:
        # forward OOM: record attempted alloc and skip backward
        mem_attempt = _parse_oom(e)
        return 0.0, mem_attempt, 0.0, 0.0

    if not BACKWARD:
        return avg_fwd, mem_fwd, 0.0, 0.0

    # -- backward timing & memory --
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        for _ in range(repeats):
            A1 = A.detach().clone().requires_grad_(True)
            B1 = B.detach().clone().requires_grad_(True)
            out = op(A1, B1)
            out.sum().backward()
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        avg_bwd = (t3 - t2) / repeats
        mem_bwd = torch.cuda.max_memory_allocated(device) / 1e6
    except torch.cuda.OutOfMemoryError as e:
        # backward OOM: record attempted alloc
        mem_attempt = _parse_oom(e)
        return avg_fwd, mem_fwd, 0.0, mem_attempt

    return avg_fwd, mem_fwd, avg_bwd, mem_bwd


def main():

    A_np_coo = load_mat_from_suitesparse_collection("Rothberg", "cfd2")
    N = A_np_coo.shape[0]

    records = []
    # reuse the loaded matrix, build sparse tensors per dtype/layout
    b_np = np.random.randn(N)
    for idx_dt in INDEX_DTYPES:
        for val_dt in VALUE_DTYPES:
            # build torch COO and CSR from A_np_coo using raw row/col to match data
            row = A_np_coo.row
            col = A_np_coo.col
            idx = torch.stack(
                [
                    torch.as_tensor(row, dtype=idx_dt, device=device),
                    torch.as_tensor(col, dtype=idx_dt, device=device),
                ],
                dim=0,
            )
            vals = torch.as_tensor(A_np_coo.data, dtype=val_dt, device=device)

            A_coo = torch.sparse_coo_tensor(idx, vals, (N, N), dtype=val_dt, device=device).coalesce()
            A_csr = A_coo.to_sparse_csr()

            B = torch.from_numpy(b_np).to(dtype=val_dt, device=device)
            for layout_name, A in [("coo", A_coo), ("csr", A_csr)]:
                for alg_name, alg_fn in ALGORITHMS:
                    print(f" Testing index_dt={idx_dt}, value_dt={val_dt} layout={layout_name}, algo={alg_name}")
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
    # changed: only select columns that were actually recorded
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
    out_path = "torchsparsegradutils/tests/benchmark_results_sparse_solve.md"
    with open(out_path, "w") as f:
        f.write("# Sparse solve methods benchmark\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Written results to {out_path}")


if __name__ == "__main__":
    main()
