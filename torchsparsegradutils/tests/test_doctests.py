"""
Test doctests using pytest.

This integrates doctest execution into the regular pytest test suite.
"""

import doctest
import warnings

import pytest

# Modules that have working doctests
DOCTEST_MODULES = [
    "torchsparsegradutils.indexed_matmul",
    "torchsparsegradutils.sparse_lstsq",
    "torchsparsegradutils.sparse_matmul",
    "torchsparsegradutils.sparse_solve",
    "torchsparsegradutils.utils.bicgstab",
    "torchsparsegradutils.utils.linear_cg",
    "torchsparsegradutils.utils.minres",
    "torchsparsegradutils.utils.random_sparse",
    "torchsparsegradutils.utils.utils",
    "torchsparsegradutils.utils.dist_stats_helpers",
    "torchsparsegradutils.encoders.pairwise_encoder",
    "torchsparsegradutils.distributions.sparse_multivariate_normal",
    "torchsparsegradutils.jax.jax_bindings",
    "torchsparsegradutils.jax.jax_sparse_solve",
    "torchsparsegradutils.cupy.cupy_bindings",
    "torchsparsegradutils.cupy.cupy_sparse_solve",
]


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_doctest(module_name):
    """Test doctests for a specific module."""
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
