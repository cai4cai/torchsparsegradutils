#!/usr/bin/env python3
"""
Sparse Generic Solve Benchmark - SuiteSparse Collection

This benchmark tests sparse linear solve operations using matrices from
the SuiteSparse Matrix Collection. It compares different iterative solvers
for solving linear systems Ax = B where A is sparse.

Fair Benchmarking Parameters:
- All iterative solvers use consistent convergence criteria:
  * Relative tolerance: 1e-5
  * Absolute tolerance: 1e-8
  * Maximum iterations: 1000
  * No preconditioners (fair comparison)
  * Verbose output disabled
- Direct solvers (spsolve) ignore tolerance parameters as expected
- BiCGSTAB uses 2x maxiter for matvec_max (2 matrix-vector products per iteration)
"""

import os
import sys

# add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch
from jax.scipy.sparse.linalg import bicgstab as bicgstab_jax, cg as cg_jax
from tqdm import tqdm

import torchsparsegradutils.jax as tsgujax
from torchsparsegradutils.benchmarks.benchmark_utils import (
    format_memory,
    format_time,
    load_mat_from_suitesparse_collection,
    measure_op,
    print_benchmark_header,
    print_result_row,
    print_results_table_header,
    save_benchmark_results,
)
from torchsparsegradutils.cupy.cupy_sparse_solve import sparse_solve_c4t
from torchsparsegradutils.sparse_solve import sparse_generic_solve
from torchsparsegradutils.utils import bicgstab, convert_coo_to_csr, linear_cg, minres

M = 1  # test 1D vector RHS only
REPEATS = 10
WARMUP_RUNS = 1
# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

# Common parameters for fair benchmarking across all iterative solvers
SOLVER_TOL = 1e-5  # Relative tolerance for convergence
SOLVER_ATOL = 1e-8  # Absolute tolerance for convergence
SOLVER_MAXITER = 1000  # Maximum number of iterations
SOLVER_VERBOSE = False  # Disable verbose output for benchmarking


def make_generic_solver_with_tol(solver_func):
    """Create a generic solver wrapper with consistent parameters."""
    if solver_func == linear_cg:
        # linear_cg uses LinearCGSettings with cg_tolerance parameter
        from torchsparsegradutils.utils.linear_cg import LinearCGSettings

        settings = LinearCGSettings(
            cg_tolerance=SOLVER_TOL,
            max_cg_iterations=SOLVER_MAXITER,
            terminate_cg_by_size=False,  # Use fixed maxiter for fair comparison
            verbose_linalg=SOLVER_VERBOSE,
        )
        return lambda A, B: sparse_generic_solve(
            A, B, solve=solver_func, transpose_solve=solver_func, settings=settings
        )
    elif solver_func == bicgstab:
        # bicgstab uses settings with reltol
        from torchsparsegradutils.utils.bicgstab import BICGSTABSettings

        settings = BICGSTABSettings(
            reltol=SOLVER_TOL,
            abstol=SOLVER_ATOL,
            matvec_max=SOLVER_MAXITER * 2,  # bicgstab uses 2 matvecs per iteration
            precon=None,  # No preconditioner for fair comparison
            # logger uses default _null_log (disabled logger) for benchmarking
        )
        return lambda A, B: sparse_generic_solve(
            A, B, solve=solver_func, transpose_solve=solver_func, settings=settings
        )
    elif solver_func == minres:
        # minres uses settings with minres_tolerance
        from torchsparsegradutils.utils.minres import MINRESSettings

        settings = MINRESSettings(
            minres_tolerance=SOLVER_TOL, max_cg_iterations=SOLVER_MAXITER, verbose_linalg=SOLVER_VERBOSE
        )
        return lambda A, B: sparse_generic_solve(
            A, B, solve=solver_func, transpose_solve=solver_func, settings=settings
        )
    else:
        # Default fallback
        return lambda A, B: sparse_generic_solve(A, B, solve=solver_func, transpose_solve=solver_func)


def make_cupy_solver_with_tol(solver_name):
    """Create a CuPy solver wrapper with consistent parameters."""
    return lambda A, B: sparse_solve_c4t(
        A, B, solve=solver_name, transpose_solve=solver_name, tol=SOLVER_TOL, atol=SOLVER_ATOL, maxiter=SOLVER_MAXITER
    )


def make_jax_solver_with_tol(solver_func):
    """Create a JAX solver wrapper with consistent parameters."""
    return lambda A, B: tsgujax.sparse_solve_j4t(
        A, B, solve=solver_func, transpose_solve=solver_func, tol=SOLVER_TOL, atol=SOLVER_ATOL, maxiter=SOLVER_MAXITER
    )


ALGORITHMS = [
    # Dense solve for reference
    ("dense.solve", lambda A, B: torch.linalg.solve(A.to_dense(), B)),
    # Sparse generic solve algorithms (from test_sparse_solve.py) with consistent tolerance
    ("sparse_generic_cg", make_generic_solver_with_tol(linear_cg)),
    ("sparse_generic_bicgstab", make_generic_solver_with_tol(bicgstab)),
    ("sparse_generic_minres", make_generic_solver_with_tol(minres)),
    # CuPy/SciPy solve algorithms (from test_cupy_sparse_solve.py) with consistent tolerance
    ("cupy_cg", make_cupy_solver_with_tol("cg")),
    ("cupy_cgs", make_cupy_solver_with_tol("cgs")),
    ("cupy_minres", make_cupy_solver_with_tol("minres")),
    ("cupy_gmres", make_cupy_solver_with_tol("gmres")),
    (
        "cupy_spsolve",
        lambda A, B: sparse_solve_c4t(A, B, solve="spsolve", transpose_solve="spsolve"),
    ),  # Direct solver - no tolerance
    # JAX solve algorithms (from test_jax_sparse_solve.py) with consistent tolerance
    ("jax_cg", make_jax_solver_with_tol(cg_jax)),
    ("jax_bicgstab", make_jax_solver_with_tol(bicgstab_jax)),
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
                            relative_resnorm = (
                                resnorm / torch.norm(B).cpu().item() if torch.norm(B).cpu().item() > 0 else 0.0
                            )

                        # Print result with residual norm
                        print_result_row(
                            alg_name, (N, M), t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd
                        )
                        print(f"    📐 Residual norm: {resnorm:.2e}, Relative: {relative_resnorm:.2e}")

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
                                "relative_resnorm": relative_resnorm,
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
                                "relative_resnorm": np.nan,
                            }
                        )

    # Save results
    if records:
        save_benchmark_results(records, "sparse_generic_solve_suite")

    print("\n✅ Sparse generic solve benchmark completed!")


if __name__ == "__main__":
    run_sparse_solve_benchmark()
