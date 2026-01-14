"""
Centralized test configuration and tolerance helpers.

This module provides:
- Common test constants (DEVICES, DTYPES, etc.)
- Dtype-aware tolerance helpers for different solver types
"""

import torch

# =============================================================================
# Test Constants
# =============================================================================

# Common test devices
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

# Common dtypes
VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32, torch.int64]
SPARSE_LAYOUTS = [torch.sparse_coo, torch.sparse_csr]


# =============================================================================
# Tolerance Helpers
# =============================================================================
# Different solver types have fundamentally different accuracy characteristics.
# Direct solvers are exact up to floating point precision, iterative solvers
# converge to a tolerance, and least squares is inherently approximate.
#
# float32 has ~7 decimal digits precision, float64 has ~16.
# Tolerances should reflect this difference.


class Tolerances:
    """
    Centralized tolerance configuration for different solver types.

    Usage:
        from test_config import Tolerances
        atol, rtol = Tolerances.direct(dtype)
        atol, rtol = Tolerances.iterative(dtype)
    """

    # Base tolerances for float64 (adjusted for float32 automatically)
    DIRECT_ATOL_F64 = 1e-6
    DIRECT_RTOL_F64 = 1e-6

    ITERATIVE_ATOL_F64 = 1e-3
    ITERATIVE_RTOL_F64 = 1e-4

    LSTSQ_RTOL_F64 = 1e-2

    # float32 multiplier (tolerances are this much looser for float32)
    F32_FACTOR = 100  # 1e-6 -> 1e-4, 1e-4 -> 1e-2

    @classmethod
    def direct(cls, dtype: torch.dtype) -> tuple:
        """
        Tolerances for direct solvers (LU, Cholesky, triangular solve).
        These are exact up to floating point precision.

        Returns:
            (atol, rtol) tuple
        """
        if dtype == torch.float32:
            return (cls.DIRECT_ATOL_F64 * cls.F32_FACTOR, cls.DIRECT_RTOL_F64 * cls.F32_FACTOR)
        return (cls.DIRECT_ATOL_F64, cls.DIRECT_RTOL_F64)

    @classmethod
    def iterative(cls, dtype: torch.dtype) -> tuple:
        """
        Tolerances for iterative solvers (CG, BiCGSTAB, MINRES, LSMR).
        These converge to a tolerance, not exact solutions.

        Returns:
            (atol, rtol) tuple
        """
        if dtype == torch.float32:
            return (cls.ITERATIVE_ATOL_F64 * cls.F32_FACTOR, cls.ITERATIVE_RTOL_F64 * cls.F32_FACTOR)
        return (cls.ITERATIVE_ATOL_F64, cls.ITERATIVE_RTOL_F64)

    @classmethod
    def lstsq(cls, dtype: torch.dtype) -> float:
        """
        Tolerance for least squares solvers.
        These are inherently approximate.

        Returns:
            rtol (relative tolerance)
        """
        if dtype == torch.float32:
            return cls.LSTSQ_RTOL_F64 * 10  # 1e-2 -> 1e-1
        return cls.LSTSQ_RTOL_F64


def get_confidence_level(device, value_dtype, is_batched=False, is_covariance_test=False):
    """
    Determine appropriate confidence level for statistical tests.

    CUDA float32 needs more lenient thresholds due to numerical precision differences
    in sparse matrix operations. See commit message for detailed analysis of T_N statistic
    differences between CPU/CUDA float32.

    Args:
        device: torch.device (cpu or cuda)
        value_dtype: torch.float32 or torch.float64
        is_batched: Whether the test is for batched data
        is_covariance_test: Whether this is for covariance (Nagao) vs mean (Hotelling) test

    Returns:
        confidence_level (float): Higher = more lenient threshold
    """
    if device.type == "cuda" and value_dtype == torch.float32:
        # CUDA float32 has higher numerical error in sparse operations
        return 0.999 if is_covariance_test else 0.99
    elif value_dtype == torch.float32:
        return 0.99
    else:
        # float64 can use standard confidence levels
        return 0.95 if is_batched else 0.90
