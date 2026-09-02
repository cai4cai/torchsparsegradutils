#!/usr/bin/env python3
"""Compare linear-CG convergence before and after the diagnostics fix.

This benchmark focuses on numerical behavior rather than throughput. It loads
the historical implementation from Git so both solvers run in one Python
environment on identical deterministic inputs.

Example
-------
python -m torchsparsegradutils.benchmarks.linear_cg_convergence \
    --baseline-ref ea7b8f0 --repeats 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from torchsparsegradutils.utils import linear_cg

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LINEAR_CG_PATH = "torchsparsegradutils/utils/linear_cg.py"


@dataclass(frozen=True)
class Problem:
    name: str
    matrix: torch.Tensor
    rhs: torch.Tensor
    tolerance: float
    max_iter: int
    initial_guess: torch.Tensor | None = None


class CountedMatmul:
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        self.calls = 0

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.matrix @ value


def load_historical_linear_cg(revision: str) -> Callable[..., torch.Tensor]:
    """Load ``linear_cg`` from a Git revision without changing the worktree."""
    command = ["git", "-C", str(REPOSITORY_ROOT), "show", f"{revision}:{LINEAR_CG_PATH}"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("Cannot load the historical solver because the Git executable was not found") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "Git could not resolve the requested revision and file"
        raise RuntimeError(
            f"Cannot load the historical solver from revision {revision!r}. "
            f"Run this benchmark from a Git checkout containing that revision. Git reported: {detail}"
        ) from error
    module = types.ModuleType("historical_linear_cg")
    exec(compile(completed.stdout, f"{revision}:{LINEAR_CG_PATH}", "exec"), module.__dict__)
    return module.linear_cg


def make_problems(device: torch.device) -> list[Problem]:
    dtype = torch.float64

    size = 36
    diagonal = torch.full((size,), 2.0, dtype=dtype, device=device)
    off_diagonal = torch.full((size - 1,), -0.25, dtype=dtype, device=device)
    tight_matrix = torch.diag(diagonal) + torch.diag(off_diagonal, diagonal=1) + torch.diag(off_diagonal, diagonal=-1)
    tight_rhs = torch.zeros(size, dtype=dtype, device=device)
    tight_rhs[size // 2] = 1

    size = 40
    multiple_matrix = torch.diag(torch.linspace(1.0, 100.0, size, dtype=dtype, device=device))
    multiple_rhs = torch.zeros((size, 101), dtype=dtype, device=device)
    multiple_rhs[:, -1] = 1

    zero_matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device))
    zero_rhs = torch.zeros(3, dtype=dtype, device=device)

    return [
        Problem("tight_tolerance", tight_matrix, tight_rhs, tolerance=1e-12, max_iter=200),
        Problem("multiple_rhs", multiple_matrix, multiple_rhs, tolerance=1e-4, max_iter=size),
        Problem(
            "zero_rhs_nonzero_guess",
            zero_matrix,
            zero_rhs,
            tolerance=1e-5,
            max_iter=20,
            initial_guess=torch.ones_like(zero_rhs),
        ),
    ]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def relative_residual_per_rhs(matrix: torch.Tensor, solution: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if rhs.ndim == 1:
        rhs = rhs.unsqueeze(-1)
        solution = solution.unsqueeze(-1)
    residual_norm = torch.linalg.vector_norm(rhs - matrix @ solution, dim=-2)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-2)
    return residual_norm / rhs_norm.masked_fill(rhs_norm.eq(0), 1)


def run_once(
    solver: Callable[..., torch.Tensor],
    problem: Problem,
    *,
    fixed: bool,
) -> tuple[torch.Tensor, int, int, str]:
    matmul = CountedMatmul(problem.matrix)
    arguments = {
        "tolerance": problem.tolerance,
        "max_iter": problem.max_iter,
        "initial_guess": problem.initial_guess,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if fixed:
            solution, info = solver(matmul, problem.rhs, return_info=True, **arguments)
            return solution, info.iterations, matmul.calls, info.reason
        solution = solver(matmul, problem.rhs, **arguments)
    # The historical solver performs one initial matvec followed by one per
    # iteration and does not recompute the true residual before returning.
    return solution, max(0, matmul.calls - 1), matmul.calls, "not_reported"


def measure(
    solver: Callable[..., torch.Tensor],
    problem: Problem,
    *,
    fixed: bool,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    for _ in range(warmup):
        run_once(solver, problem, fixed=fixed)
    synchronize(problem.rhs.device)

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_once(solver, problem, fixed=fixed)
        synchronize(problem.rhs.device)
        durations.append(time.perf_counter() - start)

    solution, iterations, matvecs, reason = run_once(solver, problem, fixed=fixed)
    true_residual = relative_residual_per_rhs(problem.matrix, solution, problem.rhs)
    reference = torch.linalg.solve(problem.matrix, problem.rhs)
    error = torch.linalg.vector_norm(solution - reference) / torch.linalg.vector_norm(reference).clamp_min(1)

    return {
        "iterations": iterations,
        "matvecs": matvecs,
        "reason": reason,
        "true_relative_residual_max": float(true_residual.max()),
        "unconverged_rhs": int((true_residual > problem.tolerance).sum()),
        "relative_solution_error": float(error),
        "median_time_ms": statistics.median(durations) * 1e3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="ea7b8f0", help="Git revision containing the historical solver")
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    device = torch.device(arguments.device)
    historical_linear_cg = load_historical_linear_cg(arguments.baseline_ref)
    results = {}
    for problem in make_problems(device):
        results[problem.name] = {
            "baseline": measure(
                historical_linear_cg,
                problem,
                fixed=False,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
            ),
            "fixed": measure(
                linear_cg,
                problem,
                fixed=True,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
            ),
        }

    payload = {
        "baseline_ref": arguments.baseline_ref,
        "device": str(device),
        "torch_version": torch.__version__,
        "warmup": arguments.warmup,
        "repeats": arguments.repeats,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
