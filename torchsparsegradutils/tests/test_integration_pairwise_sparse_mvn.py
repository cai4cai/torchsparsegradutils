"""
Integration tests for PairwiseEncoder and SparseMultivariateNormal components.

This module tests the integration of PairwiseVoxelEncoder/PairwiseEncoder with
SparseMultivariateNormal distribution, focusing on:
- Forward sampling and backward passes through multiple iterations
- Memory stability with large tensors
- Gradient flow through both components
- Different parameterizations (LDL^T and LL^T) and matrix types (covariance/precision)
- Both 2D and 3D spatial configurations on CUDA

# NOTE:
# Current failure cases:
1. CSR integration tests use WAY TOO MUCH memory after .backward() compared to COO methods
    TODO: This needs to be fixed. It may be related to the use of the CSR premutation in PairwiseEncoder
2. 2d and 3d Gradient flow consistency "Gradients too large" for:
    - LLt prec  - gradients calculated with LLt precision parameterization are too large, maybe avoid this
"""

import gc
import math
import warnings
from typing import Any, Dict, Tuple

import pytest
import torch

from torchsparsegradutils.distributions import SparseMultivariateNormal
from torchsparsegradutils.encoders.pairwise_voxel_encoder import PairwiseEncoder, PairwiseVoxelEncoder

# Skip entire test module if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Integration tests require CUDA")

# Testing Parameters - Focus on CUDA with large tensors
DEVICES = [torch.device("cuda")]  # Only CUDA devices

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
SPARSE_LAYOUTS = [torch.sparse_coo, torch.sparse_csr]
PARAMETERIZATIONS = ["ldlt", "llt"]  # LDL^T vs LL^T
MATRIX_TYPES = ["cov", "prec"]  # covariance vs precision

# Large tensor configurations for memory/performance testing
TEST_CONFIGS_2D = [
    # name, radius, channels, height, width, sparsity_factor
    ("small_2d", 1.5, 2, 16, 16, 0.8),
    ("medium_2d", 2.0, 4, 32, 32, 0.6),
    ("large_2d", 2.5, 8, 64, 64, 0.4),
]

TEST_CONFIGS_3D = [
    # name, radius, channels, height, depth, width, sparsity_factor
    ("small_3d", 1.5, 2, 12, 12, 12, 0.8),
    ("medium_3d", 2.0, 3, 16, 16, 16, 0.6),
    ("large_3d", 2.2, 4, 24, 24, 24, 0.4),
]

NUM_ITERATIONS = 5  # Number of forward/backward iterations to test
NUM_SAMPLES = 100  # Number of samples per iteration


# Test ID functions
def config_id(config):
    return config[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


def layout_id(layout):
    return str(layout).split(".")[-1].split("_")[-1].upper()


def param_id(param):
    return param.upper()


def matrix_type_id(matrix_type):
    return matrix_type.upper()


# Fixtures
@pytest.fixture(params=TEST_CONFIGS_2D, ids=[config_id(c) for c in TEST_CONFIGS_2D])
def config_2d(request):
    return request.param


@pytest.fixture(params=TEST_CONFIGS_3D, ids=[config_id(c) for c in TEST_CONFIGS_3D])
def config_3d(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=[dtype_id(d) for d in VALUE_DTYPES])
def value_dtype(request):
    return request.param


@pytest.fixture(params=INDEX_DTYPES, ids=[dtype_id(d) for d in INDEX_DTYPES])
def index_dtype(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture(params=SPARSE_LAYOUTS, ids=[layout_id(lay) for lay in SPARSE_LAYOUTS])
def layout(request):
    return request.param


@pytest.fixture(params=PARAMETERIZATIONS, ids=[param_id(p) for p in PARAMETERIZATIONS])
def parameterization(request):
    return request.param


@pytest.fixture(params=MATRIX_TYPES, ids=[matrix_type_id(m) for m in MATRIX_TYPES])
def matrix_type(request):
    return request.param


@pytest.fixture(autouse=True)
def set_seed_and_cleanup():
    """Set random seed and cleanup memory before/after each test."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()

    yield

    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Helper functions
def create_pairwise_encoder_2d(
    config: Tuple[str, float, int, int, int, float],
    layout: torch.layout,
    index_dtype: torch.dtype,
    device: torch.device,
    parameterization: str = None,
) -> PairwiseEncoder:
    """Create a 2D PairwiseEncoder for testing."""
    name, radius, channels, height, width, _ = config
    volume_shape = (channels, height, width)

    # Include diagonal elements when using LLT parameterization
    include_diag = parameterization == "llt"

    encoder = PairwiseEncoder(
        radius=radius,
        volume_shape=volume_shape,
        diag=include_diag,
        upper=False,  # Lower triangular for Cholesky-like decompositions
        channel_voxel_relation="indep",
        layout=layout,
        indices_dtype=index_dtype,
        device=device,
    )

    return encoder


def create_pairwise_encoder_3d(
    config: Tuple[str, float, int, int, int, int, float],
    layout: torch.layout,
    index_dtype: torch.dtype,
    device: torch.device,
    parameterization: str = None,
) -> PairwiseEncoder:
    """Create a 3D PairwiseEncoder for testing."""
    name, radius, channels, height, depth, width, _ = config
    volume_shape = (channels, height, depth, width)

    # Include diagonal elements when using LLT parameterization
    include_diag = parameterization == "llt"

    encoder = PairwiseEncoder(
        radius=radius,
        volume_shape=volume_shape,
        diag=include_diag,
        upper=False,  # Lower triangular for Cholesky-like decompositions
        channel_voxel_relation="indep",
        layout=layout,
        indices_dtype=index_dtype,
        device=device,
    )

    return encoder


def create_parameter_tensor_2d(
    config: Tuple[str, float, int, int, int, float],
    encoder: PairwiseEncoder,
    value_dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = True,
    parameterization: str = None,
) -> torch.Tensor:
    """Create parameter tensor for 2D configuration."""
    name, radius, channels, height, width, sparsity_factor = config
    num_offsets = len(encoder.offsets)

    # Scale initialization based on sparsity to ensure reasonable sparse matrix values
    init_scale = sparsity_factor * 0.1

    # Create tensor and scale in-place to maintain leaf status
    params = torch.randn(
        num_offsets, channels, height, width, dtype=value_dtype, device=device, requires_grad=requires_grad
    )
    with torch.no_grad():
        params.mul_(init_scale)

        # For LLT parameterization, ensure diagonal values are positive
        if parameterization == "llt" and encoder.diag:
            # Find the diagonal offset (should be all zeros)
            for i, offset in enumerate(encoder.offsets):
                if all(o == 0 for o in offset):
                    # Make diagonal values positive
                    params[i] = torch.abs(params[i]) + 0.1
                    break

    return params


def create_parameter_tensor_3d(
    config: Tuple[str, float, int, int, int, int, float],
    encoder: PairwiseEncoder,
    value_dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = True,
    parameterization: str = None,
) -> torch.Tensor:
    """Create parameter tensor for 3D configuration."""
    name, radius, channels, height, depth, width, sparsity_factor = config
    num_offsets = len(encoder.offsets)

    # Scale initialization based on sparsity to ensure reasonable sparse matrix values
    init_scale = sparsity_factor * 0.1

    # Create tensor and scale in-place to maintain leaf status
    params = torch.randn(
        num_offsets, channels, height, depth, width, dtype=value_dtype, device=device, requires_grad=requires_grad
    )
    with torch.no_grad():
        params.mul_(init_scale)

        # For LLT parameterization, ensure diagonal values are positive
        if parameterization == "llt" and encoder.diag:
            # Find the diagonal offset (should be all zeros)
            for i, offset in enumerate(encoder.offsets):
                if all(o == 0 for o in offset):
                    # Make diagonal values positive
                    params[i] = torch.abs(params[i]) + 0.1
                    break

    return params


def create_sparse_mvn_distribution(
    sparse_matrix: torch.Tensor, parameterization: str, matrix_type: str, value_dtype: torch.dtype, device: torch.device
) -> SparseMultivariateNormal:
    """Create SparseMultivariateNormal distribution from sparse matrix."""
    matrix_size = sparse_matrix.shape[-1]

    # Create location parameter
    loc = torch.zeros(matrix_size, dtype=value_dtype, device=device)

    # Handle parameterization
    if parameterization == "ldlt":
        # LDL^T: need diagonal and strictly lower triangular matrix
        diagonal = torch.ones(matrix_size, dtype=value_dtype, device=device) * 0.5

        if matrix_type == "cov":
            return SparseMultivariateNormal(loc=loc, diagonal=diagonal, scale_tril=sparse_matrix)
        else:  # precision
            return SparseMultivariateNormal(loc=loc, diagonal=diagonal, precision_tril=sparse_matrix)
    else:  # llt
        # LL^T: lower triangular matrix with positive diagonal
        # The encoder should have already included diagonal elements with positive values
        # No need for any dense operations - the sparse matrix should be ready to use

        if matrix_type == "cov":
            return SparseMultivariateNormal(loc=loc, scale_tril=sparse_matrix)
        else:  # precision
            return SparseMultivariateNormal(loc=loc, precision_tril=sparse_matrix)


def check_memory_usage(device: torch.device) -> Dict[str, float]:
    """Check current memory usage on device."""
    if device.type == "cuda":
        memory_stats = {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
    else:
        memory_stats = {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}

    return memory_stats


def run_forward_backward_iterations(
    params: torch.Tensor,
    encoder: PairwiseEncoder,
    parameterization: str,
    matrix_type: str,
    device: torch.device,
    num_iterations: int = NUM_ITERATIONS,
    num_samples: int = NUM_SAMPLES,
) -> Dict[str, Any]:
    """Run multiple forward/backward iterations and return statistics."""

    initial_memory = check_memory_usage(device)
    iteration_memories = []
    gradients_history = []

    for iteration in range(num_iterations):
        # Forward pass: encode parameters to sparse matrix
        sparse_matrix = encoder(params)

        # Create distribution
        dist = create_sparse_mvn_distribution(sparse_matrix, parameterization, matrix_type, params.dtype, device)

        # Sample from distribution
        samples = dist.rsample((num_samples,))

        # Compute loss (simple sum to trigger backprop)
        loss = samples.sum()

        # Backward pass
        loss.backward()

        # Record gradient statistics
        if params.grad is not None:
            grad_norm = params.grad.norm().item()
            grad_max = params.grad.abs().max().item()
            grad_mean = params.grad.abs().mean().item()

            gradients_history.append(
                {"iteration": iteration, "grad_norm": grad_norm, "grad_max": grad_max, "grad_mean": grad_mean}
            )

        # Record memory usage
        iter_memory = check_memory_usage(device)
        iteration_memories.append(iter_memory)

        # Zero gradients for next iteration
        if params.grad is not None:
            params.grad.zero_()

        # Optional: small parameter update to simulate optimization
        if params.grad is not None:
            with torch.no_grad():
                params.data -= 0.001 * params.grad.data

    final_memory = check_memory_usage(device)

    return {
        "initial_memory": initial_memory,
        "iteration_memories": iteration_memories,
        "final_memory": final_memory,
        "gradients_history": gradients_history,
        "max_memory_increase": (
            max(mem["allocated_mb"] - initial_memory["allocated_mb"] for mem in iteration_memories)
            if iteration_memories
            else 0
        ),
    }


# Integration Tests


def test_integration_2d_forward_backward_stability(
    config_2d, device, layout, parameterization, matrix_type, value_dtype, index_dtype
):
    """Test 2D integration with multiple forward/backward passes for memory stability."""

    # Create encoder and parameters
    encoder = create_pairwise_encoder_2d(config_2d, layout, index_dtype, device, parameterization)
    params = create_parameter_tensor_2d(config_2d, encoder, value_dtype, device, parameterization=parameterization)

    # Run iterations and check stability
    stats = run_forward_backward_iterations(params, encoder, parameterization, matrix_type, device)

    # Check that gradients are computed
    assert len(stats["gradients_history"]) == NUM_ITERATIONS
    assert all(grad["grad_norm"] > 0 for grad in stats["gradients_history"])

    # Check memory stability (should not continuously increase)
    memory_increases = [
        mem["allocated_mb"] - stats["initial_memory"]["allocated_mb"] for mem in stats["iteration_memories"]
    ]

    # Memory should stabilize - last few iterations shouldn't keep increasing dramatically
    if len(memory_increases) >= 3:
        recent_increases = memory_increases[-3:]
        max_recent_increase = max(recent_increases)
        min_recent_increase = min(recent_increases)
        memory_variance = max_recent_increase - min_recent_increase

        # Memory variance in recent iterations should be reasonable (< 100MB)
        assert memory_variance < 100, f"Memory not stable: variance={memory_variance:.2f}MB"

    print(f"Config: {config_2d[0]}, Max memory increase: {stats['max_memory_increase']:.2f}MB")


def test_integration_3d_forward_backward_stability(
    config_3d, device, layout, parameterization, matrix_type, value_dtype, index_dtype
):
    """Test 3D integration with multiple forward/backward passes for memory stability."""

    # Create encoder and parameters
    encoder = create_pairwise_encoder_3d(config_3d, layout, index_dtype, device, parameterization)
    params = create_parameter_tensor_3d(config_3d, encoder, value_dtype, device, parameterization=parameterization)

    # Run iterations and check stability
    stats = run_forward_backward_iterations(params, encoder, parameterization, matrix_type, device)

    # Check that gradients are computed
    assert len(stats["gradients_history"]) == NUM_ITERATIONS
    assert all(grad["grad_norm"] > 0 for grad in stats["gradients_history"])

    # Check memory stability
    memory_increases = [
        mem["allocated_mb"] - stats["initial_memory"]["allocated_mb"] for mem in stats["iteration_memories"]
    ]

    if len(memory_increases) >= 3:
        recent_increases = memory_increases[-3:]
        max_recent_increase = max(recent_increases)
        min_recent_increase = min(recent_increases)
        memory_variance = max_recent_increase - min_recent_increase

        # 3D configurations may use more memory, be more lenient
        assert memory_variance < 200, f"Memory not stable: variance={memory_variance:.2f}MB"

    print(f"Config: {config_3d[0]}, Max memory increase: {stats['max_memory_increase']:.2f}MB")


def test_integration_gradient_flow_consistency_2d(
    config_2d, device, layout, parameterization, matrix_type, value_dtype, index_dtype
):
    """Test that gradients flow consistently through the 2D integration."""

    encoder = create_pairwise_encoder_2d(config_2d, layout, index_dtype, device, parameterization)
    params = create_parameter_tensor_2d(config_2d, encoder, value_dtype, device, parameterization=parameterization)

    # Track gradient norms across iterations
    gradient_norms = []

    for _ in range(3):  # Fewer iterations for this test
        # Forward pass
        sparse_matrix = encoder(params)
        dist = create_sparse_mvn_distribution(sparse_matrix, parameterization, matrix_type, params.dtype, device)

        samples = dist.rsample((50,))  # Fewer samples
        loss = samples.var(dim=0).sum()  # Use variance for more interesting gradients

        # Backward pass
        loss.backward()

        gradient_norms.append(params.grad.norm().item())

        # Zero gradients
        params.grad.zero_()

    # Check that gradients exist and are reasonable
    assert all(norm > 0 for norm in gradient_norms), "Some gradients are zero"
    assert all(torch.isfinite(torch.tensor(norm)) for norm in gradient_norms), "Some gradients are not finite"

    # Gradients should be in reasonable range (not too small or too large)
    max_grad = max(gradient_norms)
    min_grad = min(gradient_norms)

    assert max_grad < 1e6, f"Gradients too large: {max_grad}"
    assert min_grad > 1e-8, f"Gradients too small: {min_grad}"


def test_integration_gradient_flow_consistency_3d(
    config_3d, device, layout, parameterization, matrix_type, value_dtype, index_dtype
):
    """Test that gradients flow consistently through the 3D integration."""

    encoder = create_pairwise_encoder_3d(config_3d, layout, index_dtype, device, parameterization)
    params = create_parameter_tensor_3d(config_3d, encoder, value_dtype, device, parameterization=parameterization)

    # Track gradient norms across iterations
    gradient_norms = []

    for _ in range(3):  # Fewer iterations for this test
        # Forward pass
        sparse_matrix = encoder(params)
        dist = create_sparse_mvn_distribution(sparse_matrix, parameterization, matrix_type, params.dtype, device)

        samples = dist.rsample((50,))  # Fewer samples
        loss = samples.var(dim=0).sum()  # Use variance for more interesting gradients

        # Backward pass
        loss.backward()

        gradient_norms.append(params.grad.norm().item())

        # Zero gradients
        params.grad.zero_()

    # Check that gradients exist and are reasonable
    assert all(norm > 0 for norm in gradient_norms), "Some gradients are zero"
    assert all(torch.isfinite(torch.tensor(norm)) for norm in gradient_norms), "Some gradients are not finite"

    # Gradients should be in reasonable range
    max_grad = max(gradient_norms)
    min_grad = min(gradient_norms)

    assert max_grad < 1e6, f"Gradients too large: {max_grad}"
    assert min_grad > 1e-8, f"Gradients too small: {min_grad}"


def test_integration_parameter_optimization_2d(
    config_2d, device, layout, parameterization, matrix_type, value_dtype, index_dtype
):
    """Test that parameters can be optimized through the integrated components in 2D."""

    encoder = create_pairwise_encoder_2d(config_2d, layout, index_dtype, device, parameterization)
    params = create_parameter_tensor_2d(config_2d, encoder, value_dtype, device, parameterization=parameterization)

    # Target distribution (simple case)
    sparse_matrix = encoder(params)
    target_dist = create_sparse_mvn_distribution(
        sparse_matrix.detach(), parameterization, matrix_type, params.dtype, device
    )
    target_samples = target_dist.rsample((200,))
    target_mean = target_samples.mean(dim=0)

    # Optimization loop
    optimizer = torch.optim.Adam([params], lr=0.01)
    initial_param_norm = params.norm().item()

    for _ in range(10):
        optimizer.zero_grad()

        # Forward pass
        sparse_matrix = encoder(params)
        dist = create_sparse_mvn_distribution(sparse_matrix, parameterization, matrix_type, params.dtype, device)

        samples = dist.rsample((100,))
        sample_mean = samples.mean(dim=0)

        # Loss: difference from target mean
        loss = torch.nn.functional.mse_loss(sample_mean, target_mean.detach())

        # Backward pass
        loss.backward()
        optimizer.step()

    # Check that parameters changed
    final_param_norm = params.norm().item()
    param_change = abs(final_param_norm - initial_param_norm) / initial_param_norm

    assert param_change > 0.01, f"Parameters barely changed: {param_change:.6f}"

    print(f"Parameter change: {param_change:.4f}")


def test_integration_parameter_optimization_3d(
    config_3d, device, layout, parameterization, matrix_type, value_dtype, index_dtype
):
    """Test that parameters can be optimized through the integrated components in 3D."""

    encoder = create_pairwise_encoder_3d(config_3d, layout, index_dtype, device, parameterization)
    params = create_parameter_tensor_3d(config_3d, encoder, value_dtype, device, parameterization=parameterization)

    # Target distribution (simple case)
    sparse_matrix = encoder(params)
    target_dist = create_sparse_mvn_distribution(
        sparse_matrix.detach(), parameterization, matrix_type, params.dtype, device
    )
    target_samples = target_dist.rsample((200,))
    target_mean = target_samples.mean(dim=0)

    # Optimization loop
    optimizer = torch.optim.Adam([params], lr=0.01)
    initial_param_norm = params.norm().item()

    for _ in range(10):
        optimizer.zero_grad()

        # Forward pass
        sparse_matrix = encoder(params)
        dist = create_sparse_mvn_distribution(sparse_matrix, parameterization, matrix_type, params.dtype, device)

        samples = dist.rsample((100,))
        sample_mean = samples.mean(dim=0)

        # Loss: difference from target mean
        loss = torch.nn.functional.mse_loss(sample_mean, target_mean.detach())

        # Backward pass
        loss.backward()
        optimizer.step()

    # Check that parameters changed
    final_param_norm = params.norm().item()
    param_change = abs(final_param_norm - initial_param_norm) / initial_param_norm

    assert param_change > 0.01, f"Parameters barely changed: {param_change:.6f}"

    print(f"Parameter change: {param_change:.4f}")


# Test sparse matrix properties
def test_integration_sparse_matrix_properties_2d(config_2d, device, layout, value_dtype, index_dtype):
    """Test that the sparse matrices generated have expected properties in 2D."""

    # Use LLT to ensure diagonals are included for better testing
    encoder = create_pairwise_encoder_2d(config_2d, layout, index_dtype, device, "llt")
    params = create_parameter_tensor_2d(
        config_2d, encoder, value_dtype, device, requires_grad=False, parameterization="llt"
    )

    sparse_matrix = encoder(params)

    # Check basic properties
    assert sparse_matrix.is_sparse or sparse_matrix.layout in [
        torch.sparse_coo,
        torch.sparse_csr,
    ], "Matrix should be sparse"
    assert sparse_matrix.layout == layout, f"Matrix layout should be {layout}"
    assert sparse_matrix.dtype == value_dtype, f"Matrix dtype should be {value_dtype}"
    assert sparse_matrix.device.type == device.type, f"Matrix device should be on {device.type}"

    # Check dimensions
    expected_size = encoder.volume_numel
    assert sparse_matrix.shape == (
        expected_size,
        expected_size,
    ), f"Matrix shape should be ({expected_size}, {expected_size})"

    # Check sparsity
    if layout == torch.sparse_coo:
        nnz = sparse_matrix._nnz()
    else:  # CSR
        nnz = sparse_matrix.values().numel()

    total_elements = expected_size * expected_size
    sparsity = 1.0 - (nnz / total_elements)

    # Should be quite sparse for reasonable radius values
    assert sparsity > 0.5, f"Matrix should be sparse, got sparsity={sparsity:.3f}"

    print(f"Matrix size: {expected_size}x{expected_size}, NNZ: {nnz}, Sparsity: {sparsity:.3f}")


def test_integration_sparse_matrix_properties_3d(config_3d, device, layout, value_dtype, index_dtype):
    """Test that the sparse matrices generated have expected properties in 3D."""

    # Use LLT to ensure diagonals are included for better testing
    encoder = create_pairwise_encoder_3d(config_3d, layout, index_dtype, device, "llt")
    params = create_parameter_tensor_3d(
        config_3d, encoder, value_dtype, device, requires_grad=False, parameterization="llt"
    )

    sparse_matrix = encoder(params)

    # Check basic properties
    assert sparse_matrix.is_sparse or sparse_matrix.layout in [
        torch.sparse_coo,
        torch.sparse_csr,
    ], "Matrix should be sparse"
    assert sparse_matrix.layout == layout, f"Matrix layout should be {layout}"
    assert sparse_matrix.dtype == value_dtype, f"Matrix dtype should be {value_dtype}"
    assert sparse_matrix.device.type == device.type, f"Matrix device should be on {device.type}"

    # Check dimensions
    expected_size = encoder.volume_numel
    assert sparse_matrix.shape == (
        expected_size,
        expected_size,
    ), f"Matrix shape should be ({expected_size}, {expected_size})"

    # Check sparsity
    if layout == torch.sparse_coo:
        nnz = sparse_matrix._nnz()
    else:  # CSR
        nnz = sparse_matrix.values().numel()

    total_elements = expected_size * expected_size
    sparsity = 1.0 - (nnz / total_elements)

    # Should be quite sparse for reasonable radius values
    assert sparsity > 0.5, f"Matrix should be sparse, got sparsity={sparsity:.3f}"

    print(f"Matrix size: {expected_size}x{expected_size}, NNZ: {nnz}, Sparsity: {sparsity:.3f}")


@pytest.mark.parametrize("config", TEST_CONFIGS_2D[-1:])  # Only largest config
def test_integration_large_2d_memory_efficiency(config, device, layout, value_dtype, index_dtype):
    """Test memory efficiency with large 2D configurations."""
    # CUDA check is no longer needed since the entire module is skipped without CUDA

    parameterization = "llt"  # LLT parameterization

    initial_memory = check_memory_usage(device)

    encoder = create_pairwise_encoder_2d(config, layout, index_dtype, device, parameterization)
    params = create_parameter_tensor_2d(config, encoder, value_dtype, device, parameterization=parameterization)

    # Single forward/backward pass
    sparse_matrix = encoder(params)
    dist = create_sparse_mvn_distribution(sparse_matrix, parameterization, "cov", value_dtype, device)

    samples = dist.rsample((100,))
    loss = samples.sum()
    loss.backward()

    final_memory = check_memory_usage(device)
    memory_used = final_memory["max_allocated_mb"] - initial_memory["max_allocated_mb"]

    # Should use reasonable amount of memory (adjust threshold as needed)
    assert memory_used < 1000, f"Memory usage too high: {memory_used:.2f}MB"

    print(f"Large 2D config memory usage: {memory_used:.2f}MB")


@pytest.mark.parametrize("config", TEST_CONFIGS_3D[-1:])  # Only largest config
def test_integration_large_3d_memory_efficiency(config, device, layout, value_dtype, index_dtype):
    """Test memory efficiency with large 3D configurations."""
    # CUDA check is no longer needed since the entire module is skipped without CUDA

    parameterization = "llt"  # LLT parameterization

    initial_memory = check_memory_usage(device)

    encoder = create_pairwise_encoder_3d(config, layout, index_dtype, device, parameterization)
    params = create_parameter_tensor_3d(config, encoder, value_dtype, device, parameterization=parameterization)

    # Single forward/backward pass
    sparse_matrix = encoder(params)
    dist = create_sparse_mvn_distribution(sparse_matrix, parameterization, "cov", value_dtype, device)

    samples = dist.rsample((100,))
    loss = samples.sum()
    loss.backward()

    final_memory = check_memory_usage(device)
    memory_used = final_memory["max_allocated_mb"] - initial_memory["max_allocated_mb"]

    # 3D configs will use more memory, be more lenient
    assert memory_used < 2500, f"Memory usage too high: {memory_used:.2f}MB"

    print(f"Large 3D config memory usage: {memory_used:.2f}MB")
