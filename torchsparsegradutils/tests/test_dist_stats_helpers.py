"""
Unit tests for statistical distribution validation helpers.

This module provides comprehensive tests for the Hotelling T² test and Nagao
covariance test functions, validating their behavior under various conditions
including:
- True-model samples (statistics and decisions match independent references)
- Strongly shifted or scaled hypotheses (rejected for the fixed samples)
- Edge cases (small samples, different dimensions, etc.)
- Confidence level sensitivity
"""

import numpy as np
import pytest
import torch
from scipy.linalg import eigvalsh
from scipy.stats import chi2, f
from test_config import DEVICES

from torchsparsegradutils.utils.dist_stats_helpers import cov_nagao_test, mean_hotelling_t2_test

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


@pytest.fixture(params=[32, 256], ids=["n32", "n256"])
def sampled_gaussian_data(request, dimension, batch_size_fixture):
    """Use the same correlated CPU samples on every device, without global RNG state."""
    rng_state = torch.random.get_rng_state()
    generator = torch.Generator(device="cpu").manual_seed(12345)
    n, p, batches = request.param, dimension, batch_size_fixture
    mean = torch.arange(p, dtype=torch.float64).expand(batches, -1) / p
    mean = mean + torch.arange(batches, dtype=torch.float64).unsqueeze(-1)
    factor = torch.diag(torch.linspace(0.8, 1.5, p, dtype=torch.float64))
    factor = factor + torch.tril(torch.full((p, p), 0.2, dtype=torch.float64), diagonal=-1)
    factor = factor.unsqueeze(0) * (1 + torch.arange(batches, dtype=torch.float64)[:, None, None] / 4)
    covariance = factor @ factor.transpose(-1, -2)
    noise = torch.randn(n, batches, p, generator=generator, dtype=torch.float64)
    samples = torch.einsum("nbj,bij->nbi", noise, factor) + mean
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    return samples, mean, covariance


class TestMeanHotellingT2Test:
    """Test suite for the Hotelling T² test for multivariate means."""

    def test_matching_mean_should_pass(self, device, dimension, batch_size_fixture):
        """Matching supplied means have zero Hotelling distance for every batch."""
        n = 100_000
        true_mean = torch.arange(dimension, device=device, dtype=torch.float64).expand(batch_size_fixture, -1)
        covariance = torch.eye(dimension, device=device, dtype=torch.float64).expand(batch_size_fixture, -1, -1)
        result, statistic, threshold = mean_hotelling_t2_test(
            true_mean.clone(), true_mean, covariance, n, confidence_level=0.9
        )
        assert result.all().item()
        torch.testing.assert_close(statistic, torch.zeros_like(statistic), atol=0, rtol=0)
        assert threshold == pytest.approx(dimension * (n - 1) / (n - dimension) * f.ppf(0.9, dimension, n - dimension))

    @pytest.mark.parametrize("mean_shift", [0.0, 2.0], ids=["true-mean", "shifted-mean"])
    @pytest.mark.parametrize("confidence", [0.5, 0.95])
    def test_sampled_mean_matches_reference(self, device, sampled_gaussian_data, mean_shift, confidence):
        """Sample moments give the independently calculated statistic and decision."""
        samples, true_mean, _ = sampled_gaussian_data
        n, batches, p = samples.shape
        hypothesized_mean = true_mean + mean_shift
        # NumPy computes its own sample moments; solve avoids the helper's explicit inverse.
        sample_array = samples.numpy()
        differences = sample_array.mean(axis=0) - hypothesized_mean.numpy()
        reference_covs = [np.cov(sample_array[:, i, :], rowvar=False, ddof=1) for i in range(batches)]
        expected = np.array([n * d @ np.linalg.solve(cov, d) for d, cov in zip(differences, reference_covs)])
        expected_threshold = p * (n - 1) / (n - p) * f.ppf(confidence, p, n - p)

        device_samples = samples.to(device)
        sample_mean = device_samples.mean(dim=0)
        sample_cov = torch.stack([torch.cov(device_samples[:, i, :].T) for i in range(batches)])
        result, statistic, threshold = mean_hotelling_t2_test(
            sample_mean, hypothesized_mean.to(device), sample_cov, n, confidence_level=confidence
        )
        np.testing.assert_allclose(statistic.cpu().numpy(), expected, rtol=1e-10, atol=1e-10)
        assert threshold == pytest.approx(expected_threshold, rel=1e-12)
        # A true model can legitimately be rejected; compare with the fixed sample's reference decision.
        np.testing.assert_array_equal(result.cpu().numpy(), expected <= expected_threshold)
        if mean_shift:
            assert not result.any().item()

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

    def test_matching_covariance_should_pass(self, device, dimension, batch_size_fixture):
        """Matching supplied covariances have zero Nagao distance for every batch.

        An empirical covariance from the true model can legitimately fail a 95%
        test. Increasing its sample size does not eliminate that rejection rate;
        this acceptance case therefore uses exactly matched nontrivial moments.
        """
        n = 100_000
        true_cov = torch.eye(dimension, device=device, dtype=torch.float64).repeat(batch_size_fixture, 1, 1)
        true_cov[:, 0, 1] = true_cov[:, 1, 0] = 0.3
        result, statistic, threshold = cov_nagao_test(true_cov.clone(), true_cov, n, confidence_level=0.95)
        assert result.all().item()
        torch.testing.assert_close(statistic, torch.zeros_like(statistic), atol=1e-20, rtol=0)
        assert threshold == pytest.approx(chi2.ppf(0.95, dimension * (dimension + 1) / 2))

    @pytest.mark.parametrize("covariance_scale", [1.0, 4.0], ids=["true-covariance", "scaled-covariance"])
    @pytest.mark.parametrize("confidence", [0.5, 0.95])
    def test_sampled_covariance_matches_reference(self, device, sampled_gaussian_data, covariance_scale, confidence):
        """Use generalized eigenvalues as an independent reference for whitening."""
        samples, _, true_cov = sampled_gaussian_data
        n, batches, p = samples.shape
        reference_cov = true_cov * covariance_scale
        sample_array = samples.numpy()
        expected = []
        for i in range(batches):
            empirical_cov = np.cov(sample_array[:, i, :], rowvar=False, ddof=1)
            # Eigenvalues of (empirical, reference) equal those of the whitened covariance.
            eigenvalues = eigvalsh(empirical_cov, reference_cov[i].numpy())
            # Preserve the helper's sample-size multiplier, not the residual n - 1
            # degrees of freedom of this unbiased covariance estimate.
            expected.append(n / 2 * np.sum((eigenvalues - 1) ** 2))
        expected = np.array(expected)
        expected_threshold = chi2.ppf(confidence, p * (p + 1) / 2)

        device_samples = samples.to(device)
        empirical_cov = torch.stack([torch.cov(device_samples[:, i, :].T) for i in range(batches)])
        result, statistic, threshold = cov_nagao_test(
            empirical_cov, reference_cov.to(device), n, confidence_level=confidence
        )
        np.testing.assert_allclose(statistic.cpu().numpy(), expected, rtol=1e-10, atol=1e-10)
        assert threshold == pytest.approx(expected_threshold, rel=1e-12)
        np.testing.assert_array_equal(result.cpu().numpy(), expected <= expected_threshold)
        if covariance_scale != 1:
            assert not result.any().item()

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
            assert thresholds[conf_levels[i]] <= thresholds[conf_levels[i + 1]], (
                f"Threshold for {conf_levels[i]} should be <= threshold for {conf_levels[i + 1]}"
            )

    @pytest.mark.parametrize("confidence", [0.9, 0.99])
    def test_known_statistical_example(self, device, confidence):
        """Nonzero, hand-calculated statistics exercise acceptance and rejection."""
        n = 100
        sample_mean = torch.tensor([[0.1, 0.2], [1.0, 2.0]], device=device, dtype=torch.float64)
        true_mean = torch.zeros_like(sample_mean)
        sample_cov = torch.diag_embed(torch.tensor([[1.2, 0.8], [2.0, 3.0]], device=device, dtype=torch.float64))
        true_cov = torch.eye(2, device=device, dtype=torch.float64).expand(2, -1, -1)
        mean_result, mean_stat, mean_threshold = mean_hotelling_t2_test(
            sample_mean, true_mean, sample_cov, n, confidence_level=confidence
        )
        cov_result, cov_stat, cov_threshold = cov_nagao_test(sample_cov, true_cov, n, confidence_level=confidence)
        # T² = n * sum(mean_i² / variance_i); Nagao = n/2 * sum((variance_i - 1)²).
        torch.testing.assert_close(mean_stat.cpu(), torch.tensor([35 / 6, 550 / 3], dtype=torch.float64))
        torch.testing.assert_close(cov_stat.cpu(), torch.tensor([4.0, 250.0], dtype=torch.float64))
        assert mean_threshold == pytest.approx(99 / 49 * f.ppf(confidence, 2, 98))
        assert cov_threshold == pytest.approx(chi2.ppf(confidence, 3))
        # The first mean falls outside the 90% region but inside the 99% region.
        assert mean_result.tolist() == [confidence == 0.99, False]
        assert cov_result.tolist() == [True, False]


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
