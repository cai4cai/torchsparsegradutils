#!/usr/bin/env python3
"""
Unified Benchmark Suite Runner

This script runs all available benchmarks in the torchsparsegradutils benchmarking suite.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# List of all benchmark scripts
BENCHMARK_SCRIPTS = [
    "sparse_mm_rand.py",
    "sparse_mm_suite.py",
    "sparse_generic_solve_rand.py",
    "sparse_generic_solve_suite.py",
    "sparse_triangular_solve_rand.py",
    "sparse_triangular_solve_suitesparse.py",
    "batched_sparse_mm_rand.py",
]


def run_benchmark(script_name, cwd):
    """Run a single benchmark script."""
    script_path = Path(cwd) / script_name
    if not script_path.exists():
        print(f"Warning: {script_name} not found, skipping.")
        return False

    print(f"Running {script_name}...")
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=cwd, check=True)
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run torchsparsegradutils benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--scripts", nargs="*", help="Specific scripts to run")
    args = parser.parse_args()

    # Get the benchmark directory
    benchmark_dir = Path(__file__).parent

    if args.all:
        scripts_to_run = BENCHMARK_SCRIPTS
    elif args.scripts:
        scripts_to_run = args.scripts
    else:
        print("Use --all to run all benchmarks or --scripts to specify which ones.")
        return

    print("Starting benchmark suite...")
    print(f"Benchmark directory: {benchmark_dir}")

    success_count = 0
    for script in scripts_to_run:
        if run_benchmark(script, benchmark_dir):
            success_count += 1

    print(f"\nBenchmark suite completed: {success_count}/{len(scripts_to_run)} scripts succeeded")

    if success_count == len(scripts_to_run):
        print("All benchmarks completed successfully!")
    else:
        print("Some benchmarks failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
