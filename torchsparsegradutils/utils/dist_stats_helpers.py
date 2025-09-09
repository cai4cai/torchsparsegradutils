"""
Statistical tests for validating multivariate distributions using confidence regions.

This module provides statistical tests for validating multivariate normal
distributions by testing whether observed sample statistics are consistent
with hypothesized population parameters using confidence regions rather than
traditional hypothesis testing.

Notes
-----
**Understanding Confidence Regions**

These tests use confidence regions to determine whether observed statistics
are consistent with hypothesized parameters:

- A confidence region with level C (e.g., 0.95) means that if we repeated
  the experiment many times, C% of the confidence regions would contain
  the true parameter values
- **HIGHER** confidence levels (closer to 1.0) create **LARGER** regions,
  making tests **EASIER** to pass
- **LOWER** confidence levels create **SMALLER** regions, making tests
  **HARDER** to pass

**Examples of confidence levels:**

- ``confidence_level = 0.50`` → 50% confidence region (very strict)
- ``confidence_level = 0.95`` → 95% confidence region (moderately strict)
- ``confidence_level = 0.99`` → 99% confidence region (more lenient)

**When to use each test:**

* Use higher confidence levels (0.95-0.99) for practical distribution validation
* Use moderate levels (0.80-0.95) for unit testing with some tolerance
* Use lower levels (0.50-0.80) for strict validation or when you expect
  the parameters to be very close

**Practical interpretation:**

Think of confidence levels as "how generous should I be?":

* 0.99: "I'll accept the hypothesis unless there's very strong evidence against it"
* 0.95: "I'll accept the hypothesis unless there's moderate evidence against it"
* 0.50: "I'll only accept the hypothesis if it's very close to the data"

This is the OPPOSITE of traditional hypothesis testing where smaller p-values
(like 0.05) indicate stronger evidence against the null hypothesis. Here,
larger confidence levels create more permissive acceptance regions.

For implementation details, see the references for Hotelling's test [1b]_ and
Nagao's test [2b]_.

References
----------
.. [1b] Hotelling, H. (1947). Multivariate Quality Control. In C. Eisenhart,
   M. W. Hastay, and W. A. Wallis, eds. Techniques of Statistical Analysis.
   New York: McGraw-Hill.
.. [2b] Nagao, H. (1973). On Some Test Criteria for Covariance Matrix.
   The Annals of Statistics, Vol. 1, No. 4, pp. 700-709.
"""

from typing import Tuple

import torch
from scipy.stats import chi2, f as _scipy_f

__all__ = [
    "mean_hotelling_t2_test",
    "cov_nagao_test",
]


def mean_hotelling_t2_test(
    sample_mean: torch.Tensor,
    true_mean: torch.Tensor,
    sample_cov: torch.Tensor,
    n: int,
    confidence_level: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    r"""
    One-sample Hotelling T² test for multivariate mean equality using confidence regions.

    This test checks whether the hypothesized mean :math:`\boldsymbol{\mu}_0` lies inside the
    joint confidence region around the sample mean :math:`\bar{\boldsymbol{x}}`, accounting for
    covariance among variables. The confidence region is an ellipsoid in :math:`p`-dimensional
    space.

    Test statistic (Hotelling's T²):

    .. math::

        T^2 \;=\; n\, (\bar{\boldsymbol{x}} - \boldsymbol{\mu}_0)^{\top}
        \hat{\boldsymbol{\Sigma}}^{-1} (\bar{\boldsymbol{x}} - \boldsymbol{\mu}_0)

    with the distributional relationship (for unknown covariance):

    .. math::

        T^2 \;\sim\; \frac{p(n-1)}{n-p}\,F_{p,\,n-p}.

    Acceptance criterion (confidence region): :math:`\boldsymbol{\mu}_0` is accepted at the
    specified confidence level if

    .. math::

        T^2 \;\le\; \frac{p(n-1)}{n-p} \; F_{p,\,n-p;\,\text{confidence level}},

    equivalently

    .. math::

        (\bar{\boldsymbol{x}} - \boldsymbol{\mu})^{\top} \hat{\boldsymbol{\Sigma}}^{-1}
        (\bar{\boldsymbol{x}} - \boldsymbol{\mu})
        \;\le\; \frac{p(n-1)}{n(n-p)} \; F_{p,\,n-p;\,\text{confidence level}},

    for all :math:`\boldsymbol{\mu}` in the confidence region.

    Parameters
    ----------
    sample_mean : torch.Tensor, shape (B, p)
        Sample mean vector where B is the batch size and p is the dimension.
    true_mean : torch.Tensor, shape (B, p)
        Hypothesized true mean vector.
    sample_cov : torch.Tensor, shape (B, p, p)
        Sample covariance matrix.
    n : int
        Sample size used to compute the sample mean and covariance.
    confidence_level : float, default=0.95
        Confidence level for the region. Higher values create larger regions
        (easier to pass). Must be in (0, 1).

    Returns
    -------
    test_result : torch.Tensor, shape (B,), dtype=bool
        Boolean tensor indicating whether the hypothesized mean lies within
        the confidence region (True) or outside it (False).
    t2_statistic : torch.Tensor, shape (B,), dtype=float
        T² test statistic values.
    t2_threshold : float
        T² threshold value for the confidence region.

    Notes
    -----

    **Interpretation of confidence levels:**

    - Higher confidence levels (e.g., 0.99) create larger regions, making the
      test more lenient (easier to accept the hypothesis)
    - Lower confidence levels (e.g., 0.50) create smaller regions, making the
      test more strict (harder to accept the hypothesis)

    For practical distribution validation, use confidence levels of 0.95-0.99.
    For strict unit testing, consider 0.80-0.95.

    This implementation follows standard Hotelling's T² methodology (see e.g.,
    Hotelling's T-squared distribution [1c]_ [2c]_). For confidence regions,
    see Confidence region [3c]_.

    References
    ----------
     .. [1c] Wikipedia: Hotelling's T-squared distribution.
         https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution
     .. [2c] Hotelling, H. (1947). Multivariate Quality Control. In C. Eisenhart,
         M. W. Hastay, and W. A. Wallis, eds. Techniques of Statistical Analysis.
         New York: McGraw-Hill.
     .. [3c] Wikipedia: Confidence region.
         https://en.wikipedia.org/wiki/Confidence_region

    Examples
    --------
    Test if sample statistics are consistent with known parameters:

    >>> import torch
    >>> from torch.distributions import MultivariateNormal
    >>> from torchsparsegradutils.utils import mean_hotelling_t2_test
    >>>
    >>> # Set random seed for reproducible results
    >>> _ = torch.manual_seed(42)
    >>>
    >>> # True parameters
    >>> true_mean = torch.tensor([[0.0, 0.0]])    # shape (1, 2)
    >>> true_cov = torch.eye(2).unsqueeze(0)      # shape (1, 2, 2)
    >>> n = 1000
    >>>
    >>> # Generate sample data from the true distribution
    >>> dist = MultivariateNormal(true_mean.squeeze(0), true_cov.squeeze(0))
    >>> samples = dist.sample((n,)).unsqueeze(1)  # Add batch dimension
    >>> sample_mean = samples.mean(0)
    >>> sample_cov = torch.cov(samples.squeeze(1).T).unsqueeze(0)
    >>>
    >>> # Test with 95% confidence
    >>> result, t2_stat, threshold = mean_hotelling_t2_test(
    ...     sample_mean, true_mean, sample_cov, n, confidence_level=0.95
    ... )
    >>> print(f"Test passed: {result.item()}")  # Should be True
    Test passed: True
    >>> print(f"T² statistic: {t2_stat.item():.3f}, threshold: {threshold:.3f}")
    T² statistic: ..., threshold: ...
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
    emp_cov: torch.Tensor,
    ref_cov: torch.Tensor,
    n: int,
    confidence_level: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    r"""
    Nagao's (1973) one-sample test for covariance matrix equality using confidence regions.

    This test determines whether the hypothesized covariance matrix :math:`\boldsymbol{\Sigma}_0` is consistent
    with the empirical covariance matrix, by checking if the test statistic lies
    within the appropriate confidence region under the :math:`\chi^2` distribution.

    The test standardizes the empirical covariance matrix:

    .. math::

        \mathbf{W} = \boldsymbol{\Sigma}_0^{-1/2} \, \hat{\boldsymbol{\Sigma}} \, \boldsymbol{\Sigma}_0^{-1/2}

    Here, :math:`\hat{\boldsymbol{\Sigma}}` denotes the empirical (sample) covariance, and
    :math:`\mathbf{W}` is the covariance whitened with respect to :math:`\boldsymbol{\Sigma}_0` (i.e., a similarity
    transform that equals the identity when :math:`\hat{\boldsymbol{\Sigma}} = \boldsymbol{\Sigma}_0`).
    The factor :math:`\boldsymbol{\Sigma}_0^{-1/2}` is computed from the Cholesky decomposition of
    :math:`\boldsymbol{\Sigma}_0`. Under :math:`H_0: \boldsymbol{\Sigma} = \boldsymbol{\Sigma}_0`, the matrix
    :math:`\mathbf{W}` should be close to the identity matrix :math:`\mathbf{I}`.

    The test statistic is:

    .. math::

        T_N = \frac{n}{2} \cdot \left\|\mathbf{W} - \mathbf{I}\right\|_F^2

    Where :math:`\|\cdot\|_F` is the Frobenius norm.

    Distribution under :math:`H_0` (asymptotically):

    .. math::

        T_N \;\sim\; \chi^2_{\,\nu}, \qquad \nu = \tfrac{p(p+1)}{2},

    where :math:`p` is the dimension.

    Acceptance criterion (confidence region): at the specified confidence level,
    accept :math:`\boldsymbol{\Sigma}_0` if

    .. math::

        T_N \;\leq\; \chi^2_{\nu;\,\text{confidence level}} \quad \text{with} \quad \nu = \tfrac{p(p+1)}{2}.

    Parameters
    ----------
    emp_cov : torch.Tensor, shape (B, p, p)
        Empirical (sample) covariance matrix where B is the batch size and p
        is the dimension.
    ref_cov : torch.Tensor, shape (B, p, p)
        Reference (hypothesized) covariance matrix.
    n : int
        Sample size used to compute the empirical covariance matrix.
    confidence_level : float, default=0.95
        Confidence level for the region. Higher values create larger regions
        (easier to pass). Must be in (0, 1).

    Returns
    -------
    test_result : torch.Tensor, shape (B,), dtype=bool
        Boolean tensor indicating whether the hypothesized covariance lies
        within the confidence region (True) or outside it (False).
    t_n_statistic : torch.Tensor, shape (B,), dtype=float
        T_N test statistic values.
    chi2_threshold : float
        :math:`\chi^2` threshold value for the confidence region.

    Notes
    -----
    This test is often more stable than raw Frobenius norm tests when
    the reference covariance :math:`\boldsymbol{\Sigma}_0` is ill-conditioned.

    **Interpretation of confidence levels:**

    - Higher confidence levels (e.g., 0.99) create larger regions, making the
      test more lenient (easier to accept the hypothesis)
    - Lower confidence levels (e.g., 0.50) create smaller regions, making the
      test more strict (harder to accept the hypothesis)

    For practical distribution validation, use confidence levels of 0.95-0.99.
    For strict unit testing, consider 0.80-0.95.

    This implementation follows Nagao's test methodology [1d]_.

    References
    ----------
    .. [1d] Nagao, H. (1973). On Some Test Criteria for Covariance Matrix.
       The Annals of Statistics, Vol. 1, No. 4, pp. 700-709.
       Institute of Mathematical Statistics.
       Stable URL: https://www.jstor.org/stable/2958313
       Equation 3.3

    Examples
    --------
    Test if sample covariance is consistent with known covariance:

    >>> import torch
    >>> from torch.distributions import MultivariateNormal
    >>> from torchsparsegradutils.utils import cov_nagao_test
    >>>
    >>> # Set random seed for reproducible results
    >>> _ = torch.manual_seed(42)
    >>>
    >>> # True parameters
    >>> true_mean = torch.tensor([[0.0, 0.0]])    # shape (1, 2)
    >>> ref_cov = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # shape (1, 2, 2)
    >>> n = 1000
    >>>
    >>> # Generate sample data from the true distribution
    >>> dist = MultivariateNormal(true_mean.squeeze(0), ref_cov.squeeze(0))
    >>> samples = dist.sample((n,)).unsqueeze(1)  # Add batch dimension
    >>> emp_cov = torch.cov(samples.squeeze(1).T).unsqueeze(0)
    >>>
    >>> # Test with 95% confidence
    >>> result, t_n_stat, threshold = cov_nagao_test(
    ...     emp_cov, ref_cov, n, confidence_level=0.95
    ... )
    >>> print(f"Test passed: {result.item()}")  # Should be True
    Test passed: True
    >>> print(f"T_N statistic: {t_n_stat.item():.3f}, threshold: {threshold:.3f}")
    T_N statistic: ..., threshold: ...
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
