#!/usr/bin/env python3
"""
Sparse Generic Solve Benchmark - Random SPD Matrices

This benchmark tests sparse generic solve operations using randomly
generated sparse symmetric positive definite (SPD) matrices of various sizes.
SPD matrices ensure that all iterative solvers should converge reliably.

Key Features:
- Generates random SPD matrices using M @ M.T + n*I construction
- Tests various sparsity patterns by zeroing out random off-diagonal elements
- Calculates both absolute and relative residuals for accuracy verification
- Compares multiple iterative and direct solvers
"""

import os
import sys

# Add the parent directory to sys.path to allow importing torchsparsegradutils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch
from jax.scipy.sparse.linalg import bicgstab as bicgstab_jax, cg as cg_jax
from tqdm import tqdm

import torchsparsegradutils.jax as tsgujax
from torchsparsegradutils.benchmarks.benchmark_utils import (
    measure_op,
    print_benchmark_header,
    print_result_row,
    print_results_table_header,
    save_benchmark_results,
)
from torchsparsegradutils.cupy.cupy_sparse_solve import sparse_solve_c4t
from torchsparsegradutils.sparse_solve import sparse_generic_solve
from torchsparsegradutils.utils import bicgstab, convert_coo_to_csr, linear_cg, minres
from torchsparsegradutils.utils.random_sparse import make_spd_sparse

REPEATS = 50
WARMUP_RUNS = 5

# Only run on CUDA
device = torch.device("cuda")
assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"

# Problem sizes: (label, N, M, sparsity_target)
# sparsity_target: approximate percentage of off-diagonal elements to zero out
SIZES = [
    ("tiny", 2**8, 1, 0.3),  # 256x256, ~30% sparse
    ("small", 2**10, 1, 0.5),  # 1024x1024, ~50% sparse
    ("medium", 2**12, 1, 0.7),  # 4096x4096, ~70% sparse
    # ("large", 2**14, 1, 0.8),  # 16384x16384, ~80% sparse
]

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


def make_jax_solver_with_tol(jax_solver):
    """Create a JAX solver wrapper with consistent parameters."""
    return lambda A, B: tsgujax.sparse_solve_j4t(
        A, B, solve=jax_solver, transpose_solve=jax_solver, tol=SOLVER_TOL, atol=SOLVER_ATOL, maxiter=SOLVER_MAXITER
    )


ALGORITHMS = [
    # Dense solve for reference
    ("dense.solve", lambda A, B: torch.linalg.solve(A.to_dense(), B)),
    # Sparse generic solve algorithms with consistent tolerance
    ("sparse_generic_cg", make_generic_solver_with_tol(linear_cg)),
    ("sparse_generic_bicgstab", make_generic_solver_with_tol(bicgstab)),
    ("sparse_generic_minres", make_generic_solver_with_tol(minres)),
    # CuPy/SciPy solve algorithms with consistent tolerance
    ("cupy_cg", make_cupy_solver_with_tol("cg")),
    ("cupy_cgs", make_cupy_solver_with_tol("cgs")),
    ("cupy_minres", make_cupy_solver_with_tol("minres")),
    ("cupy_gmres", make_cupy_solver_with_tol("gmres")),
    (
        "cupy_spsolve",
        lambda A, B: sparse_solve_c4t(A, B, solve="spsolve", transpose_solve="spsolve"),
    ),  # Direct solver - no tolerance
    # JAX solve algorithms with consistent tolerance
    ("jax_cg", make_jax_solver_with_tol(cg_jax)),
    ("jax_bicgstab", make_jax_solver_with_tol(bicgstab_jax)),
]


def run_sparse_generic_solve_benchmark():
    """Run the sparse generic solve benchmark suite with random SPD matrices."""

    print_benchmark_header("Sparse Generic Solve Benchmark - Random SPD Matrices")

    records = []

    for size_label, N, M, sparsity_target in tqdm(SIZES, desc="Problem sizes"):
        print(f"\\n🔍 Testing size: {size_label} (N={N}, M={M}, sparsity_target={sparsity_target:.1%})")

        A_shape = (N, N)
        B_shape = (N, M)

        for idx_dt in tqdm(INDEX_DTYPES, desc="Index dtypes", leave=False):
            for val_dt in tqdm(VALUE_DTYPES, desc="Value dtypes", leave=False):
                for layout in tqdm(LAYOUTS, desc="Layouts", leave=False):
                    layout_name = "coo" if layout == torch.sparse_coo else "csr"
                    print(f"\\n🔧 Configuration: idx_dtype={idx_dt}, val_dtype={val_dt}, layout={layout_name}")

                    # Generate random SPD matrix
                    try:
                        A_sparse, A_dense = make_spd_sparse(
                            N, layout, val_dt, idx_dt, device, sparsity_ratio=sparsity_target
                        )
                        actual_nnz = A_sparse._nnz()
                        total_elements = N * N
                        actual_sparsity = 1.0 - (actual_nnz / total_elements)

                        print(f"    Generated matrix: nnz={actual_nnz}, sparsity={actual_sparsity:.1%}")

                        # Check positive definiteness of dense version
                        try:
                            torch.linalg.cholesky(A_dense)
                            print("    ✅ Matrix is positive definite")
                        except torch.linalg.LinAlgError:
                            print("    ⚠️  Matrix may not be positive definite, results may vary")

                    except Exception as e:
                        print(f"    ❌ Failed to generate matrix: {e}")
                        continue

                    print_results_table_header()

                    for alg_name, alg_fn in ALGORITHMS:
                        try:
                            print(f"  🧮 Testing {alg_name}...")

                            # Right-hand side matrix B
                            B = torch.randn(N, M, dtype=val_dt, device=device, requires_grad=True)
                            if M == 1:
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

                                # Handle different B shapes for residual calculation
                                if M == 1 and B.dim() == 1:
                                    # For 1D case, ensure matrix multiplication works correctly
                                    if x.dim() == 1:
                                        residual = A_sparse @ x.unsqueeze(-1) - B.unsqueeze(-1)
                                        residual = residual.squeeze(-1)
                                    else:
                                        residual = A_sparse @ x - B.unsqueeze(-1)
                                        residual = residual.squeeze(-1)
                                else:
                                    residual = A_sparse @ x - B

                                resnorm = torch.norm(residual).cpu().item()
                                B_norm = torch.norm(B).cpu().item()
                                relative_resnorm = resnorm / B_norm if B_norm > 0 else 0.0

                            # Print result with residual norm
                            print_result_row(
                                alg_name,
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
                            print(f"      📐 Residual norm: {resnorm:.2e}, Relative: {relative_resnorm:.2e}")

                            records.append(
                                {
                                    "size": size_label,
                                    "N": N,
                                    "M": M,
                                    "target_sparsity": sparsity_target,
                                    "actual_nnz": actual_nnz,
                                    "actual_sparsity": actual_sparsity,
                                    "layout": layout_name,
                                    "algorithm": alg_name,
                                    "index_dt": str(idx_dt).split(".")[-1],
                                    "value_dt": str(val_dt).split(".")[-1],
                                    "fwd_time_us": t_fwd,
                                    "fwd_time_std_us": std_fwd,
                                    "fwd_mem_MB": mem_fwd,
                                    "fwd_mem_std_MB": std_mem_fwd,
                                    "bwd_time_us": t_bwd,
                                    "bwd_time_std_us": std_bwd,
                                    "bwd_mem_MB": mem_bwd,
                                    "bwd_mem_std_MB": std_mem_bwd,
                                    "residual_norm": resnorm,
                                    "relative_residual_norm": relative_resnorm,
                                }
                            )

                        except Exception as e:
                            print(f"  ❌ {alg_name} failed: {e}")

                            records.append(
                                {
                                    "size": size_label,
                                    "N": N,
                                    "M": M,
                                    "target_sparsity": sparsity_target,
                                    "actual_nnz": actual_nnz if "actual_nnz" in locals() else np.nan,
                                    "actual_sparsity": actual_sparsity if "actual_sparsity" in locals() else np.nan,
                                    "layout": layout_name,
                                    "algorithm": alg_name,
                                    "index_dt": str(idx_dt).split(".")[-1],
                                    "value_dt": str(val_dt).split(".")[-1],
                                    "fwd_time_us": np.nan,
                                    "fwd_time_std_us": np.nan,
                                    "fwd_mem_MB": np.nan,
                                    "fwd_mem_std_MB": np.nan,
                                    "bwd_time_us": np.nan,
                                    "bwd_time_std_us": np.nan,
                                    "bwd_mem_MB": np.nan,
                                    "bwd_mem_std_MB": np.nan,
                                    "residual_norm": np.nan,
                                    "relative_residual_norm": np.nan,
                                    "error": str(e)[:100],
                                }
                            )

    # Save results
    if records:
        save_benchmark_results(records, "sparse_generic_solve_rand")

    print("\\n✅ Sparse generic solve random benchmark completed!")


if __name__ == "__main__":
    run_sparse_generic_solve_benchmark()
