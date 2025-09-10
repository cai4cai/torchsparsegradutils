"""
Unit tests for statistical distribution validation helpers.

This module provides comprehensive tests for the Hotelling T² test and Nagao
covariance test functions, validating their behavior under various conditions
including:
- Correct parameters (should pass at reasonable confidence levels)
- Incorrect parameters (should fail at reasonable confidence levels)
- Edge cases (small samples, different dimensions, etc.)
- Confidence level sensitivity
"""

import numpy as np
import pytest
import torch

from torchsparsegradutils.utils.dist_stats_helpers import cov_nagao_test, mean_hotelling_t2_test

# Test configurations
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

# Test parameters
TEST_DIMENSIONS = [2, 4, 8]  # Different dimensionalities
TEST_BATCH_SIZES = [1, 3]  # Different batch sizes


# Define Test Names:
def device_id(device):
    return str(device)


def dimension_id(p):
    return f"p{p}"


def batch_id(batch_size):
    return f"b{batch_size}"


# Define Fixtures
@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture(params=TEST_DIMENSIONS, ids=[dimension_id(d) for d in TEST_DIMENSIONS])
def dimension(request):
    return request.param


@pytest.fixture(params=TEST_BATCH_SIZES, ids=[batch_id(d) for d in TEST_BATCH_SIZES])
def batch_size_fixture(request):
    return request.param


class TestMeanHotellingT2Test:
    """Test suite for the Hotelling T² test for multivariate means."""

    def test_correct_mean_should_pass(self, device, dimension, batch_size_fixture):
        """Test that correct means pass the test at reasonable confidence levels."""
        torch.manual_seed(42)
        n = 100_000  # Large sample for stability

        # True parameters
        true_mean = torch.zeros(batch_size_fixture, dimension, device=device)
        true_cov = torch.eye(dimension, device=device).unsqueeze(0).repeat(batch_size_fixture, 1, 1)

        # Generate samples from the true distribution
        from torch.distributions import MultivariateNormal

        dist = MultivariateNormal(true_mean, true_cov)
        samples = dist.sample((n,))  # [n, batch_size, p]

        # Compute sample statistics
        sample_mean = samples.mean(0)  # [batch_size, p]
        sample_cov = torch.stack([torch.cov(samples[:, i, :].T) for i in range(batch_size_fixture)])

        # Test with high confidence level (should pass most of the time)
        result, t2_stat, t2_thresh = mean_hotelling_t2_test(sample_mean, true_mean, sample_cov, n, confidence_level=0.9)

        assert result.any().item(), f"T² stats: {t2_stat}, threshold: {t2_thresh}"

    def test_wrong_mean_should_fail(self, device, dimension):
        """Test that significantly wrong means fail the test."""
        torch.manual_seed(42)
        n = 100_000
        batch_size = 2

        # True parameters
        true_mean = torch.zeros(batch_size, dimension, device=device)
        true_cov = torch.eye(dimension, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Generate samples from the true distribution
        from torch.distributions import MultivariateNormal

        dist = MultivariateNormal(true_mean, true_cov)
        samples = dist.sample((n,))

        # Compute sample statistics
        sample_mean = samples.mean(0)
        sample_cov = torch.stack([torch.cov(samples[:, i, :].T) for i in range(batch_size)])

        # Test against significantly wrong mean
        wrong_mean = true_mean + 1.0  # Shift by 1

        # NOTE: value of confidence_level doesn't seem to make a difference here
        # With a mean that wrong it always fails the test
        result, t2_stat, t2_thresh = mean_hotelling_t2_test(
            sample_mean, wrong_mean, sample_cov, n, confidence_level=0.75
        )

        # Should fail for all batches
        assert not result.any().item(), (
            f"Test should fail with wrong parameters. " f"T² stats: {t2_stat}, threshold: {t2_thresh}"
        )

    def test_input_validation(self):
        """Test input validation and edge cases."""
        p, batch_size, n = 2, 1, 100

        sample_mean = torch.zeros(batch_size, p)
        true_mean = torch.zeros(batch_size, p)
        sample_cov = torch.eye(p).unsqueeze(0)

        # Test valid inputs
        result, _, _ = mean_hotelling_t2_test(sample_mean, true_mean, sample_cov, n)
        assert isinstance(result, torch.Tensor)

        # Test small sample size
        small_n = 10
        result, _, _ = mean_hotelling_t2_test(sample_mean, true_mean, sample_cov, small_n)
        assert isinstance(result, torch.Tensor)

    def test_return_types_and_shapes(self):
        """Test that return types and shapes are correct."""
        p, batch_size, n = 3, 2, 1000

        sample_mean = torch.randn(batch_size, p)
        true_mean = torch.zeros(batch_size, p)
        sample_cov = torch.eye(p).unsqueeze(0).repeat(batch_size, 1, 1)

        result, t2_stat, t2_thresh = mean_hotelling_t2_test(
            sample_mean, true_mean, sample_cov, n, confidence_level=0.95
        )

        # Check types
        assert isinstance(result, torch.Tensor)
        assert isinstance(t2_stat, torch.Tensor)
        assert isinstance(t2_thresh, float)

        # Check shapes
        assert result.shape == (batch_size,)
        assert t2_stat.shape == (batch_size,)
        assert result.dtype == torch.bool


class TestCovNagaoTest:
    """Test suite for the Nagao covariance test."""

    def test_correct_covariance_should_pass(self, device, dimension, batch_size_fixture):
        """Test that correct covariances pass the test at reasonable confidence levels."""
        torch.manual_seed(42)
        n = 100_000  # Large sample for stability

        # True parameters
        true_mean = torch.zeros(batch_size_fixture, dimension, device=device)
        true_cov = torch.eye(dimension, device=device).unsqueeze(0).repeat(batch_size_fixture, 1, 1)
        # Add some correlation for more interesting test
        if dimension >= 2:
            true_cov[:, 0, 1] = true_cov[:, 1, 0] = 0.3

        # Generate samples from the true distribution
        from torch.distributions import MultivariateNormal

        dist = MultivariateNormal(true_mean, true_cov)
        samples = dist.sample((n,))

        # Compute sample covariance
        sample_cov = torch.stack([torch.cov(samples[:, i, :].T) for i in range(batch_size_fixture)])

        # Test with high confidence level (should pass most of the time)
        result, t_n_stat, chi2_thresh = cov_nagao_test(sample_cov, true_cov, n, confidence_level=0.95)

        assert result.any().item(), f"T_N stats: {t_n_stat}, threshold: {chi2_thresh}"

    def test_wrong_covariance_should_fail(self, device, dimension):
        """Test that significantly wrong covariances fail the test."""
        torch.manual_seed(42)
        n = 10000
        batch_size = 2

        # True parameters
        true_mean = torch.zeros(batch_size, dimension, device=device)
        true_cov = torch.eye(dimension, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Generate samples from the true distribution
        from torch.distributions import MultivariateNormal

        dist = MultivariateNormal(true_mean, true_cov)
        samples = dist.sample((n,))

        # Compute sample covariance
        sample_cov = torch.stack([torch.cov(samples[:, i, :].T) for i in range(batch_size)])

        # Test against significantly wrong covariance (scaled by 2)
        wrong_cov = true_cov * 2.0

        # NOTE: value of confidence_level doesn't seem to make a difference here
        # With covariance that wrong it always fails the test
        result, t_n_stat, chi2_thresh = cov_nagao_test(sample_cov, wrong_cov, n, confidence_level=0.75)

        # Should fail for all batches
        assert not result.any().item(), (
            f"Test should fail with wrong parameters. " f"T_N stats: {t_n_stat}, threshold: {chi2_thresh}"
        )

    def test_return_types_and_shapes_cov(self):
        """Test that return types and shapes are correct for covariance test."""
        p, batch_size, n = 3, 2, 1000

        emp_cov = torch.eye(p).unsqueeze(0).repeat(batch_size, 1, 1) * 1.1
        ref_cov = torch.eye(p).unsqueeze(0).repeat(batch_size, 1, 1)

        result, t_n_stat, chi2_thresh = cov_nagao_test(emp_cov, ref_cov, n, confidence_level=0.95)

        # Check types
        assert isinstance(result, torch.Tensor)
        assert isinstance(t_n_stat, torch.Tensor)
        assert isinstance(chi2_thresh, float)

        # Check shapes
        assert result.shape == (batch_size,)
        assert t_n_stat.shape == (batch_size,)
        assert result.dtype == torch.bool


class TestIntegrationBehavior:
    """Integration tests for expected statistical behavior."""

    def test_confidence_level_ordering(self):
        """Test that higher confidence levels are more permissive."""
        torch.manual_seed(42)
        p, batch_size, n = 3, 1, 1000

        # Create borderline case
        sample_mean = torch.tensor([[0.2, 0.2, 0.2]])
        true_mean = torch.zeros(batch_size, p)
        sample_cov = torch.eye(p).unsqueeze(0)

        results = {}
        thresholds = {}

        for conf_level in [0.50, 0.80, 0.90, 0.95, 0.99]:
            result, _, threshold = mean_hotelling_t2_test(
                sample_mean, true_mean, sample_cov, n, confidence_level=conf_level
            )
            results[conf_level] = result
            thresholds[conf_level] = threshold

        # Thresholds should be increasing with confidence level
        conf_levels = sorted(thresholds.keys())
        for i in range(len(conf_levels) - 1):
            assert (
                thresholds[conf_levels[i]] <= thresholds[conf_levels[i + 1]]
            ), f"Threshold for {conf_levels[i]} should be <= threshold for {conf_levels[i + 1]}"

    def test_known_statistical_example(self):
        """Test with a known statistical example for validation."""
        # Create a simple 2D case where we know the expected behavior
        torch.manual_seed(12345)  # Fixed seed for reproducibility

        n = 100000  # Very large sample for stable statistics
        p = 2
        batch_size = 1

        # True parameters: zero mean, identity covariance
        true_mean = torch.zeros(batch_size, p)
        true_cov = torch.eye(p).unsqueeze(0)

        # Generate samples
        from torch.distributions import MultivariateNormal

        dist = MultivariateNormal(true_mean.squeeze(0), true_cov.squeeze(0))
        samples = dist.sample((n,)).unsqueeze(1)  # Add batch dimension

        # Compute sample statistics
        sample_mean = samples.mean(0)
        sample_cov = torch.cov(samples.squeeze(1).T).unsqueeze(0)

        # Both tests should pass with very high probability for correct parameters
        mean_result, _, _ = mean_hotelling_t2_test(sample_mean, true_mean, sample_cov, n, confidence_level=0.99)
        cov_result, _, _ = cov_nagao_test(sample_cov, true_cov, n, confidence_level=0.99)

        assert mean_result.item(), "Mean test should pass with correct parameters and large sample"
        assert cov_result.item(), "Covariance test should pass with correct parameters and large sample"


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running smoke tests for distribution statistics helpers...")

    # Quick test of both functions
    torch.manual_seed(42)
    p, batch_size, n = 2, 1, 1000

    sample_mean = torch.tensor([[0.1, 0.1]])
    true_mean = torch.zeros(batch_size, p)
    sample_cov = torch.eye(p).unsqueeze(0)

    # Test mean function
    result, t2_stat, t2_thresh = mean_hotelling_t2_test(sample_mean, true_mean, sample_cov, n, confidence_level=0.95)
    print(f"Mean test - Result: {result.item()}, T²: {t2_stat.item():.3f}, Threshold: {t2_thresh:.3f}")

    # Test covariance function
    emp_cov = sample_cov * 1.05
    result, t_n_stat, chi2_thresh = cov_nagao_test(emp_cov, sample_cov, n, confidence_level=0.95)
    print(f"Cov test - Result: {result.item()}, T_N: {t_n_stat.item():.3f}, Threshold: {chi2_thresh:.3f}")

    print("Smoke tests completed successfully!")
