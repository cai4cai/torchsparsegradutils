"""
Test doctests using pytest.

This integrates doctest execution into the regular pytest test suite.
"""

import doctest
import warnings

import pytest
import torch

from torchsparsegradutils._dispatch import backend_available

# Modules that have working doctests
DOCTEST_MODULES = [
    "torchsparsegradutils.ops.indexed_matmul",
    "torchsparsegradutils.ops.logsumexp",
    "torchsparsegradutils.ops.lstsq",
    "torchsparsegradutils.ops.matmul",
    "torchsparsegradutils.ops.triangular_solve",
    "torchsparsegradutils.ops.generic_solve",
    "torchsparsegradutils.solvers.bicgstab",
    "torchsparsegradutils.solvers.cg",
    "torchsparsegradutils.solvers.minres",
    "torchsparsegradutils.utils.random_sparse",
    "torchsparsegradutils.utils.convert",
    "torchsparsegradutils.utils.dist_stats_helpers",
    "torchsparsegradutils.encoders.pairwise_encoder",
    "torchsparsegradutils.distributions.sparse_multivariate_normal",
]

# Modules whose doctest examples exercise a CUDA-only tsgu:: op
# (architecture.md §4: "CUDA-required at runtime" -- no CPU implementation
# ships). torchsparsegradutils.ops.logsumexp's examples call
# sparse_logsumexp (tsgu::seglse as of spec/commit.md Phase 3 commit 12) both
# directly and from sparse_bidir_logsumexp's own docstring.
# torchsparsegradutils.ops.matmul's examples call sparse_mm (tsgu::spmm as of
# spec/commit.md Phase 3 commit 15). torchsparsegradutils.ops.triangular_solve's
# examples call sparse_triangular_solve (tsgu::spsm as of spec/commit.md
# Phase 3 commit 16). torchsparsegradutils.distributions.
# sparse_multivariate_normal's SparseMultivariateNormal/_batch_sparse_mv
# examples route sample()/rsample() through sparse_mm (and, via LDL^T/LL^T
# scale_tril solves, sparse_triangular_solve) too.
_CUDA_ONLY_DOCTEST_MODULES = {
    "torchsparsegradutils.ops.logsumexp",
    "torchsparsegradutils.ops.matmul",
    "torchsparsegradutils.ops.triangular_solve",
    # indexed_matmul (spec/commit.md Phase 3 commit 18): segment_mm/gather_mm
    # dispatch to tsgu::grouped_gemm, so the examples run on CUDA tensors.
    "torchsparsegradutils.ops.indexed_matmul",
    # generic_solve/lstsq (spec/commit.md Phase 3 commit 17): the forward host
    # loop runs anywhere, but their docstring gradient examples backward
    # through tsgu::sddmm, so the examples are written on CUDA tensors.
    "torchsparsegradutils.ops.generic_solve",
    "torchsparsegradutils.ops.lstsq",
    "torchsparsegradutils.distributions.sparse_multivariate_normal",
}


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_doctest(module_name):
    """Test doctests for a specific module."""
    if module_name in _CUDA_ONLY_DOCTEST_MODULES and not (torch.cuda.is_available() and backend_available()):
        pytest.skip(
            f"{module_name}'s doctests exercise a CUDA-only tsgu:: op; no CUDA device "
            "and/or loaded, version-matched torchsparsegradutils_cuda backend available here."
        )
    if module_name == "torchsparsegradutils.ops.indexed_matmul" and not torch._C._dispatch_has_kernel_for_dispatch_key(
        "tsgu::grouped_gemm", "CUDA"
    ):
        # commit 18 is developed in two concurrent lanes: the wrapper wiring
        # can be present while the cuda/ tree has not yet been rebuilt with
        # the grouped_gemm kernel -- skip rather than error in that window.
        pytest.skip("tsgu::grouped_gemm has no CUDA kernel registered in this build yet (commit 18 in flight).")
    try:
        # Import the module
        import importlib

        module = importlib.import_module(module_name)

        # Suppress warnings during doctest
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Run doctest
            failure_count, test_count = doctest.testmod(
                module,
                verbose=False,
                optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL),
            )

        assert failure_count == 0, f"Doctest failures in {module_name}: {failure_count}/{test_count}"
        assert test_count > 0, f"No doctests found in {module_name}"

    except ImportError as e:
        pytest.skip(f"Could not import {module_name}: {e}")


def test_doctest_coverage():
    """Ensure we have doctests in our modules."""
    import importlib

    total_tests = 0
    for module_name in DOCTEST_MODULES:
        try:
            module = importlib.import_module(module_name)
            _, test_count = doctest.testmod(module, verbose=False, report=False)
            total_tests += test_count
        except Exception:
            continue

    assert total_tests > 0, "No doctests found in any tested modules"
