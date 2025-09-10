"""
Benchmark utilities for torchsparsegradutils.

This module provides common utilities used across all benchmark scripts,
including matrix loading, performance measurement, and data storage functions.
"""

import os
import re
import sys
import tarfile
import time
import urllib.request

import numpy as np
import scipy.io
import torch
from tqdm import trange

# Default configuration
REPEATS = 100
WARMUP_RUNS = 10
BENCHMARK_DATA_DIR = os.path.join(os.path.dirname(__file__), ".benchmark_data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def ensure_benchmark_data_dir():
    """Ensure the benchmark data directory exists."""
    os.makedirs(BENCHMARK_DATA_DIR, exist_ok=True)


def ensure_results_dir():
    """Ensure the results directory exists."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_environment_info():
    """
    Get information about the current environment.

    Returns:
        dict: Dictionary containing Python, PyTorch, and CUDA version information
    """
    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
    else:
        info["cuda_version"] = "N/A"
        info["cuda_device_name"] = "N/A"

    return info


def load_mat_from_suitesparse_collection(dirname, matname):
    """
    Load a matrix from the SuiteSparse Matrix Collection.

    Downloads and extracts the matrix if not already present in the local
    benchmark data directory.

    Args:
        dirname (str): Directory name in the SuiteSparse collection
        matname (str): Matrix name

    Returns:
        scipy.sparse matrix: The loaded matrix in COO format

    Example:
        >>> A = load_mat_from_suitesparse_collection("Rothberg", "cfd2")
    """
    ensure_benchmark_data_dir()

    # eg: https://suitesparse-collection-website.herokuapp.com/Rothberg/cfd2
    base_url = "https://suitesparse-collection-website.herokuapp.com/MM/"
    url = base_url + dirname + "/" + matname + ".tar.gz"

    # Store files in benchmark data directory
    compressed = os.path.join(BENCHMARK_DATA_DIR, matname + ".tar.gz")
    folder = os.path.join(BENCHMARK_DATA_DIR, matname)
    localfile = os.path.join(folder, matname + ".mtx")

    if not os.path.exists(compressed):
        print(f"📥 Downloading {dirname}/{matname} from SuiteSparse Collection...")
        urllib.request.urlretrieve(url, filename=compressed)
        print(f"✓ Downloaded to {compressed}")

    if not os.path.exists(localfile):
        print(f"📦 Extracting {compressed}...")
        with tarfile.open(compressed) as tf:
            tf.extractall(BENCHMARK_DATA_DIR)
        print(f"✓ Extracted to {folder}")

    A_np_coo = scipy.io.mmread(localfile)
    print(f"📊 Loaded SuiteSparse matrix {dirname}/{matname}: shape={A_np_coo.shape}, nnz={A_np_coo.nnz}")
    return A_np_coo


def _parse_oom(e):
    """
    Helper to extract attempted allocation (in MiB) from OOM message.

    Args:
        e (Exception): CUDA OutOfMemoryError exception

    Returns:
        float: Attempted allocation in MiB
    """
    msg = str(e)
    m = re.search(r"Tried to allocate ([\d\.]+) GiB", msg)
    if m:
        return float(m.group(1)) * 1024.0
    m = re.search(r"Tried to allocate ([\d\.]+) MiB", msg)
    if m:
        return float(m.group(1))
    return 0.0


def measure_op(
    op,
    A,
    B,
    repeats=REPEATS,
    device=None,
    backward=True,
    desc="operation",
    warmup_runs=WARMUP_RUNS,
    remove_outliers=True,
):
    """
    Measure average forward/backward times and peak memory over multiple runs with improved accuracy.

    Args:
        op (callable): Operation to measure, should take (A, B) as arguments
        A (torch.Tensor): First input tensor
        B (torch.Tensor): Second input tensor
        repeats (int): Number of repetitions for timing
        device (torch.device): CUDA device to use for memory measurements
        backward (bool): Whether to measure backward pass
        desc (str): Description for progress bar
        warmup_runs (int): Number of warmup runs to discard
        remove_outliers (bool): Whether to remove statistical outliers

    Returns:
        tuple: (avg_fwd_time_us, std_fwd_time_us, max_fwd_mem_mb, std_fwd_mem_mb,
                avg_bwd_time_us, std_bwd_time_us, max_bwd_mem_mb, std_bwd_mem_mb)
               Times in microseconds, memory in MB
    """
    if device is None:
        device = A.device if hasattr(A, "device") else torch.device("cuda")

    def remove_outliers_iqr(data):
        """Remove outliers using IQR method"""
        if len(data) < 4:
            return data
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [x for x in data if lower_bound <= x <= upper_bound]

    # -- Forward timing & memory --
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # Warmup runs
        for _ in range(warmup_runs):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            A1 = A.detach().clone()
            B1 = B.detach().clone()
            result = op(A1, B1)

            torch.cuda.synchronize(device)
            del result, A1, B1  # Explicit cleanup

        fwd_times = []
        fwd_mems = []

        for _ in trange(repeats, desc=f"{desc} (forward)", leave=False):
            # Clear cache and reset memory stats before each run
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            A1 = A.detach().clone().requires_grad_(True)
            B1 = B.detach().clone().requires_grad_(True)
            result = op(A1, B1)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            fwd_times.append((t1 - t0) * 1e6)  # Convert to microseconds
            fwd_mems.append(torch.cuda.max_memory_allocated(device) / 1e6)  # Convert to MB

            # Clean up
            del result, A1, B1

        # Remove outliers if requested
        if remove_outliers and len(fwd_times) >= 4:
            fwd_times_clean = remove_outliers_iqr(fwd_times)
            fwd_mems_clean = remove_outliers_iqr(fwd_mems)

            # Only use cleaned data if we have enough points
            if len(fwd_times_clean) >= max(3, len(fwd_times) // 2):
                fwd_times = fwd_times_clean
                fwd_mems = fwd_mems_clean

        avg_fwd = np.mean(fwd_times)
        std_fwd = np.std(fwd_times)
        avg_fwd_mem = np.mean(fwd_mems)
        std_fwd_mem = np.std(fwd_mems)

    except torch.cuda.OutOfMemoryError as e:
        # Forward OOM: return NaN values
        mem_attempt = _parse_oom(e)
        print(f"⚠️  Forward OOM: attempted {mem_attempt:.1f} MiB")
        return np.nan, np.nan, mem_attempt, np.nan, np.nan, np.nan, np.nan, np.nan

    if not backward:
        return avg_fwd, std_fwd, avg_fwd_mem, std_fwd_mem, np.nan, np.nan, np.nan, np.nan

    # -- Backward timing & memory --
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # Warmup runs for backward pass
        for _ in range(warmup_runs):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            A1 = A.detach().clone().requires_grad_(True)
            B1 = B.detach().clone().requires_grad_(True)
            out = op(A1, B1)
            out.sum().backward()

            torch.cuda.synchronize(device)
            del out, A1, B1

        bwd_times = []
        bwd_mems = []

        for _ in trange(repeats, desc=f"{desc} (backward)", leave=False):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            A1 = A.detach().clone().requires_grad_(True)
            B1 = B.detach().clone().requires_grad_(True)
            out = op(A1, B1)
            out.sum().backward()
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            bwd_times.append((t1 - t0) * 1e6)  # Convert to microseconds
            bwd_mems.append(torch.cuda.max_memory_allocated(device) / 1e6)  # Convert to MB

            # Clean up
            del out, A1, B1

        # Remove outliers if requested
        if remove_outliers and len(bwd_times) >= 4:
            bwd_times_clean = remove_outliers_iqr(bwd_times)
            bwd_mems_clean = remove_outliers_iqr(bwd_mems)

            if len(bwd_times_clean) >= max(3, len(bwd_times) // 2):
                bwd_times = bwd_times_clean
                bwd_mems = bwd_mems_clean

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


def format_time(t_us):
    """Format time in appropriate units from microseconds."""
    if np.isnan(t_us):
        return "N/A"
    if t_us >= 1e6:  # >= 1 second
        return f"{t_us / 1e6:.3f}s"
    elif t_us >= 1e3:  # >= 1 millisecond
        return f"{t_us / 1e3:.1f}ms"
    else:
        return f"{t_us:.1f}μs"


def format_memory(mem_mb):
    """Format memory in appropriate units."""
    if np.isnan(mem_mb):
        return "N/A"
    if mem_mb >= 1024:
        return f"{mem_mb / 1024:.1f}GB"
    else:
        return f"{mem_mb:.1f}MB"


def print_benchmark_header(title, matrix_info=None):
    """Print a formatted benchmark header."""
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)
    if matrix_info:
        print(f"📊 Matrix: {matrix_info}")
    print()


def print_result_row(algorithm, shape, t_fwd, std_fwd, mem_fwd, std_mem_fwd, t_bwd, std_bwd, mem_bwd, std_mem_bwd):
    """Print a formatted result row."""
    shape_str = f"{shape[0]}x{shape[1]}" if isinstance(shape, (tuple, list)) else str(shape)

    if np.isnan(t_fwd):  # Failed operation
        print(f"  {algorithm:<25} {shape_str:<15} {'FAILED':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    else:
        fwd_time_str = f"{format_time(t_fwd)}±{format_time(std_fwd)}"
        fwd_mem_str = f"{format_memory(mem_fwd)}±{format_memory(std_mem_fwd)}"

        if np.isnan(t_bwd):
            bwd_time_str = "N/A"
            bwd_mem_str = "N/A"
        else:
            bwd_time_str = f"{format_time(t_bwd)}±{format_time(std_bwd)}"
            bwd_mem_str = f"{format_memory(mem_bwd)}±{format_memory(std_mem_bwd)}"

        print(
            f"  {algorithm:<25} {shape_str:<15} {fwd_time_str:<15} {fwd_mem_str:<15} {bwd_time_str:<15} {bwd_mem_str:<15}"
        )


def print_results_table_header():
    """Print the header for results table."""
    print(
        f"  {'Algorithm':<25} {'Shape':<15} {'Fwd Time±Std':<15} {'Fwd Mem±Std':<15} {'Bwd Time±Std':<15} {'Bwd Mem±Std':<15}"
    )
    print(f"  {'-' * 25} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15}")


def save_benchmark_results(records, benchmark_name):
    """
    Save benchmark results to CSV file with environment information.

    Args:
        records (list): List of benchmark result dictionaries
        benchmark_name (str): Name of the benchmark (used for filename)

    Returns:
        str: Path to the saved CSV file
    """
    import pandas as pd

    if not records:
        print("⚠️  No results to save")
        return None

    ensure_results_dir()

    # Add environment information to each record
    env_info = get_environment_info()
    for record in records:
        record.update(env_info)

    df = pd.DataFrame.from_records(records)

    # Save CSV to results directory
    csv_filename = f"{benchmark_name}_results.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    df.to_csv(csv_path, index=False)

    print(f"\n💾 Results saved to {csv_path}")

    # Print summary statistics
    df_valid = df[~pd.isna(df.get("fwd_time_us", np.nan))]  # Exclude failed runs
    if not df_valid.empty and "algorithm" in df_valid.columns:
        print("\n📈 Performance Summary:")
        print("=" * 50)

        if "fwd_time_us" in df_valid.columns:
            best_fwd = df_valid.loc[df_valid["fwd_time_us"].idxmin()]
            print("\n🏆 Fastest configuration:")
            print(f"   Algorithm: {best_fwd['algorithm']}")
            print(f"   Time: {format_time(best_fwd['fwd_time_us'])}")
            if "fwd_mem_MB" in best_fwd and not np.isnan(best_fwd["fwd_mem_MB"]):
                print(f"   Memory: {format_memory(best_fwd['fwd_mem_MB'])}")

    return csv_path
