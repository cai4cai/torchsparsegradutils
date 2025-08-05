import torch
from scipy.stats import f as _scipy_f
from scipy.stats import chi2

__all__ = [
    "mean_hotelling_t2_test",
    "cov_nagao_test",
]

# -------------------------------------------------------------------------
# NOTES:
# Regarding Confidence Regions
# These tests use confidence regions to determine whether observed statistics
# are consistent with hypothesized parameters.
#
# A confidence region with level C (e.g., 0.95) means:
# - If we repeated the experiment many times, C% of the confidence regions
#   would contain the true parameter values
# - HIGHER confidence levels (closer to 1.0) create LARGER regions, making
#   tests EASIER to pass
# - LOWER confidence levels create SMALLER regions, making tests HARDER to pass
#
# confidence_level = 0.95 → 95% confidence region (moderately strict)
# confidence_level = 0.99 → 99% confidence region (more lenient)
# confidence_level = 0.50 → 50% confidence region (very strict)
# -------------------------------------------------------------------------


def mean_hotelling_t2_test(
    sample_mean: torch.Tensor,  # [B, p]
    true_mean: torch.Tensor,  # [B, p]
    sample_cov: torch.Tensor,  # [B, p, p]
    n: int,
    confidence_level: float = 0.95,
):
    r"""
    One-sample Hotelling T² test for multivariate mean equality using confidence regions.

    This test determines whether the hypothesized mean vector μ₀ lies within the
    confidence region around the sample mean. The confidence region is an ellipsoid
    in p-dimensional space, accounting for the covariance structure among variables.

    The test statistic is:
        T² = n·(x̄ - μ₀)ᵀΣ̂⁻¹(x̄ - μ₀)

    Which follows the relation:
        T² ∼ (p(n-1)/(n-p)) F_{p,n-p}

    The confidence region is defined by all points μ satisfying:
        (x̄ - μ)ᵀΣ̂⁻¹(x̄ - μ) ≤ (p/n) F_{p,n-p;confidence_level}

    Args:
        sample_mean (torch.Tensor): Sample mean vector of shape [B, p] where B is
            the batch size and p is the dimension.
        true_mean (torch.Tensor): Hypothesized true mean vector of shape [B, p].
        sample_cov (torch.Tensor): Sample covariance matrix of shape [B, p, p].
        n (int): Sample size used to compute the sample mean and covariance.
        confidence_level (float, optional): Confidence level for the region.
            Higher values create larger regions (easier to pass). Defaults to 0.95.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Boolean tensor of shape [B] indicating whether the hypothesized
              mean lies within the confidence region (True) or outside it (False).
            - torch.Tensor: T² test statistic values of shape [B].
            - float: T² threshold value for the confidence region.

    References:
        Hotelling, H. (1947). Multivariate Quality Control. In C. Eisenhart,
        M. W. Hastay, and W. A. Wallis, eds. Techniques of Statistical Analysis.
        New York: McGraw-Hill.

        Wikipedia: Confidence region - The case of independent, identically
        normally-distributed errors.

    Note:
        The hypothesized mean μ₀ is accepted if it lies within the confidence region:
        T² ≤ (p(n-1)/(n-p)) F_{p,n-p;confidence_level}
    """
    B, p = sample_mean.shape

    # diff = x̄ - μ₀
    diff = sample_mean - true_mean  # [B,p]

    # Σ̂⁻¹
    invS = torch.linalg.inv(sample_cov)  # [B,p,p]

    # T² = n · diffᵀ Σ̂⁻¹ diff
    t2 = n * torch.einsum("bi,bij,bj->b", diff, invS, diff)  # [B]

    # Confidence region threshold: T² ≤ (p(n-1)/(n-p)) F_{p,n-p;confidence_level}
    df1, df2 = p, n - p
    f_critical = float(_scipy_f.ppf(confidence_level, dfn=df1, dfd=df2))
    t2_threshold = (p * (n - 1) / (n - p)) * f_critical

    # Accept if T² lies within the confidence region
    result = t2.cpu() <= t2_threshold
    return result, t2.cpu(), t2_threshold


def cov_nagao_test(
    emp_cov: torch.Tensor,  # [B, p, p]
    ref_cov: torch.Tensor,  # [B, p, p]
    n: int,
    confidence_level: float = 0.95,
):
    r"""
    Nagao's (1973) one-sample test for covariance matrix equality using confidence regions.

    This test determines whether the hypothesized covariance matrix Σ₀ is consistent
    with the empirical covariance matrix, by checking if the test statistic lies
    within the appropriate confidence region under the χ² distribution.

    The test standardizes the empirical covariance matrix:
        W = Σ₀^{-1/2} Σ̂ Σ₀^{-1/2}

    Where Σ₀^{-1/2} is computed from the Cholesky decomposition of Σ₀.
    Under H₀: Σ = Σ₀, the matrix W should be close to the identity matrix I.

    The test statistic is:
        T_N = (n/2)·‖W - I‖_F²

    Where ‖·‖_F is the Frobenius norm. Under H₀, T_N follows a χ² distribution
    with ν = p(p+1)/2 degrees of freedom, where p is the dimension.

    Args:
        emp_cov (torch.Tensor): Empirical (sample) covariance matrix of shape [B, p, p]
            where B is the batch size and p is the dimension.
        ref_cov (torch.Tensor): Reference (hypothesized) covariance matrix of shape [B, p, p].
        n (int): Sample size used to compute the empirical covariance matrix.
        confidence_level (float, optional): Confidence level for the region.
            Higher values create larger regions (easier to pass). Defaults to 0.95.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Boolean tensor of shape [B] indicating whether the hypothesized
              covariance lies within the confidence region (True) or outside it (False).
            - torch.Tensor: T_N test statistic values of shape [B].
            - float: χ² threshold value for the confidence region.

    References:
        Nagao, H. (1973). On Some Test Criteria for Covariance Matrix.
        The Annals of Statistics, Vol. 1, No. 4, pp. 700-709.
        Institute of Mathematical Statistics.
        Stable URL: https://www.jstor.org/stable/2958313
        Equation 3.3

    Note:
        The hypothesized covariance Σ₀ is accepted if it lies within the confidence region:
        T_N ≤ χ²_{ν,confidence_level} where ν = p(p+1)/2.
        This test is often more stable than raw Frobenius norm tests when
        the reference covariance Σ₀ is ill-conditioned.
    """
    B, p, _ = emp_cov.shape

    # Compute L such that Σ₀ = L·Lᵀ
    L = torch.linalg.cholesky(ref_cov)  # L = Σ₀^{½}

    # Invert L to get Σ₀^{-½}
    invL = torch.linalg.inv(L)  # invL = Σ₀^{-½}

    # Form W = Σ₀^{-½} · Σ̂ · Σ₀^{-½}
    W = invL @ emp_cov @ invL.transpose(-1, -2)  # W = Σ₀^{-½} Σ̂ Σ₀^{-½}

    # Compute deviation from identity: W − I
    diff = W - torch.eye(p, device=emp_cov.device, dtype=emp_cov.dtype)
    # diff = Σ₀^{-½} Σ̂ Σ₀^{-½} − I

    # T_N = (n/2) * ||diff||_F^2
    T_N = (n / 2.0) * (diff * diff).sum(dim=(-2, -1))  # Frobenius norm squared

    # Degrees of freedom ν = ½ p(p+1)
    nu = 0.5 * p * (p + 1)

    # Critical χ² threshold for confidence region
    chi2_critical = float(chi2.ppf(confidence_level, df=nu))

    # Accept if T_N lies within the confidence region
    result = T_N.cpu() <= chi2_critical
    return result, T_N.cpu(), chi2_critical


if __name__ == "__main__":
    """Test both statistical functions with known distributions."""
    import numpy as np

    print("Testing mean_hotelling_t2_test and cov_nagao_test...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test parameters
    p = 4  # dimension
    n = 1_000_000  # sample size
    batch_size = 2

    # Test 1: Generate samples from known multivariate normal distribution
    print("\n=== Test 1: Known multivariate normal distribution ===")

    # True parameters
    true_mean = torch.zeros(batch_size, p)
    true_cov = torch.eye(p).unsqueeze(0).repeat(batch_size, 1, 1)
    true_cov[0, 0, 1] = true_cov[0, 1, 0] = 0.5  # Add some correlation in first batch

    # Generate samples using torch multivariate normal
    from torch.distributions import MultivariateNormal

    dist = MultivariateNormal(true_mean, true_cov)
    samples = dist.sample((n,))  # [n, batch_size, p]

    # Compute sample statistics
    sample_mean = samples.mean(0)  # [batch_size, p]
    sample_cov = torch.stack([torch.cov(samples[:, i, :].T) for i in range(batch_size)])  # [batch_size, p, p]

    print(f"True mean: {true_mean}")
    print(f"Sample mean: {sample_mean}")
    print(f"Mean difference: {torch.abs(sample_mean - true_mean).max().item():.4f}")

    print(f"\nTrue covariance:\n{true_cov}")
    print(f"Sample covariance:\n{sample_cov}")
    print(f"Covariance difference (max): {torch.abs(true_cov - sample_cov).max().item():.6f}")

    # Test mean with Hotelling T² test (should pass with high probability)
    mean_test_result, t2_stat, t2_thresh = mean_hotelling_t2_test(
        sample_mean, true_mean, sample_cov, n, confidence_level=0.95
    )
    print(f"Mean test passed (95% confidence): {mean_test_result}")
    print(f"T² statistic: {t2_stat}, threshold: {t2_thresh:.3f}")

    # Test covariance with Nagao test (should pass with high probability)
    cov_test_result, T_N_stat, chi2_thresh = cov_nagao_test(sample_cov, true_cov, n, confidence_level=0.95)
    print(f"Covariance test passed (95% confidence): {cov_test_result}")
    print(f"T_N statistic: {T_N_stat}, threshold: {chi2_thresh:.3f}")

    # Test 2: Test with wrong parameters (should fail)
    print("\n=== Test 2: Wrong parameters (should fail) ===")

    # Wrong mean (shifted by significant amount)
    wrong_mean = true_mean + 0.5
    mean_test_wrong, t2_wrong, t2_thresh_wrong = mean_hotelling_t2_test(
        sample_mean, wrong_mean, sample_cov, n, confidence_level=0.95
    )
    print(f"Mean test with wrong mean (should fail): {mean_test_wrong}")
    print(f"T² statistic (wrong): {t2_wrong}, threshold: {t2_thresh_wrong:.3f}")

    # Wrong covariance (scaled by significant amount)
    wrong_cov = true_cov * 2.0
    cov_test_wrong, T_N_wrong, chi2_thresh_wrong = cov_nagao_test(sample_cov, wrong_cov, n, confidence_level=0.95)
    print(f"Covariance test with wrong covariance (should fail): {cov_test_wrong}")
    print(f"T_N statistic (wrong): {T_N_wrong}, threshold: {chi2_thresh_wrong:.3f}")

    # Test 3: Test sensitivity to confidence levels
    print("\n=== Test 3: Confidence level sensitivity ===")

    confidence_levels = [0.50, 0.80, 0.90, 0.95, 0.99, 0.999]
    print("Conf Level\tMean Test\tCov Test\tMean T²\t\tCov T_N")
    for conf_level in confidence_levels:
        mean_result, t2_stat, t2_threshold = mean_hotelling_t2_test(
            sample_mean, true_mean, sample_cov, n, confidence_level=conf_level
        )
        cov_result, T_N_stat, chi2_threshold = cov_nagao_test(sample_cov, true_cov, n, confidence_level=conf_level)

        print(
            f"{conf_level:.3f}\t\t{mean_result.all().item()}\t\t{cov_result.all().item()}\t\t{t2_stat.max().item():.3f} (thresh: {t2_threshold:.3f})\t{T_N_stat.max().item():.3f} (thresh: {chi2_threshold:.3f})"
        )

    print("\n=== Test Summary ===")
    print("✓ Both tests should pass for correct parameters with reasonable confidence levels (≥0.90)")
    print("✓ Both tests should fail for significantly wrong parameters")
    print("✓ Higher confidence levels create larger regions, making tests EASIER to pass")
    print("✓ Lower confidence levels create smaller regions, making tests HARDER to pass")
    print("\nNote: Even with correct parameters, sampling variability means test statistics")
    print("won't be exactly zero, so very strict confidence levels (≤0.50) may still reject.")
    print("This is statistically correct behavior!")
    print("\nFor practical use in testing distributions, use confidence levels like 0.95-0.99.")
