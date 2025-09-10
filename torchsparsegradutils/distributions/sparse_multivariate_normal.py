import math
import warnings

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal

from torchsparsegradutils import sparse_mm as spmm, sparse_triangular_solve as spts

# from .contraints import sparse_strictly_lower_triangular

__all__ = ["SparseMultivariateNormal", "SparseMultivariateNormalNative"]


def _batch_sparse_mv(op, bmat, bvec, **kwargs):
    r"""
    Batched sparse–dense matvec helper (no broadcasting).

    Performs a matrix–vector (or matrix–matrix) operation where the matrix is
    sparse (COO/CSR) and the vector/array is dense. Limited batch combinations
    are supported and **no broadcasting** of batch dimensions is performed.

    Supported input ranks
    ---------------------
    - ``bmat.ndim == 2`` with ``bvec.ndim == 1``  → returns ``(n,)``
    - ``bmat.ndim == 2`` with ``bvec.ndim == 2``  → returns ``(n, k)``
    - ``bmat.ndim == 3`` with ``bvec.ndim == 2``  → returns ``(B, n)``
    - ``bmat.ndim == 3`` with ``bvec.ndim == 3``  → returns ``(B, n, k)``

    Parameters
    ----------
    op : callable
        Sparse operator to apply. Typically :func:`torchsparsegradutils.sparse_mm`
        (SpMM) or :func:`torchsparsegradutils.sparse_triangular_solve`.
        Must accept ``(bmat, bvec, **kwargs)`` and return a tensor shaped
        like a dense matmul result.
    bmat : torch.Tensor
        Sparse matrix of shape ``(n, n)`` or batched sparse matrix of shape
        ``(B, n, n)`` in COO or CSR layout.
    bvec : torch.Tensor
        Dense vector/array. Shape may be ``(n,)``, ``(n, k)``, ``(B, n)``,
        or ``(B, n, k)`` as listed above.
    **kwargs
        Extra keyword arguments forwarded to ``op`` (e.g., ``upper``,
        ``unitriangular``, ``transpose`` for triangular solves).

    Returns
    -------
    torch.Tensor
        Dense result with shape as indicated in *Supported input ranks*.

    Raises
    ------
    ValueError
        If the pair of ranks is not one of the supported combinations.

    Notes
    -----
    - Batch sizes must match exactly when ``bmat`` is batched.
    - This helper exists to centralize the small reshaping/permutation logic
      so that SpMM and triangular solves can share a common pathway.

    See Also
    --------
    torchsparsegradutils.sparse_mm : Sparse matrix–dense matrix multiply.
    torchsparsegradutils.sparse_triangular_solve : Sparse triangular solver.

    Examples
    --------
    Vector RHS (2D × 1D)::

        >>> A = torch.eye(4, dtype=torch.float64).to_sparse_csr()
        >>> v = torch.arange(4., dtype=torch.float64)
        >>> _batch_sparse_mv(spmm, A, v).shape
        torch.Size([4])

    Matrix RHS (2D × 2D)::

        >>> X = torch.randn(5, 4, dtype=torch.float64)  # 5 vectors of size 4
        >>> _batch_sparse_mv(spmm, A, X).shape
        torch.Size([5, 4])

    Batched matrix with vector RHS (3D × 2D)::

        >>> # Create batched sparse tensors using stack_csr utility
        >>> from torchsparsegradutils.utils import stack_csr
        >>> Ab = stack_csr([A, A])  # (B=2, 4, 4)
        >>> vb = torch.randn(2, 4, dtype=torch.float64)
        >>> _batch_sparse_mv(spmm, Ab, vb).shape
        torch.Size([2, 4])
    """
    if bmat.dim() == 2 and bvec.dim() == 1:
        return op(bmat, bvec.unsqueeze(-1), **kwargs).squeeze(-1)
    elif bmat.dim() == 2 and bvec.dim() == 2:
        return op(bmat, bvec.t(), **kwargs).t()
    elif bmat.dim() == 3 and bvec.dim() == 2:
        return op(bmat, bvec.unsqueeze(-1), **kwargs).squeeze(-1)
    elif bmat.dim() == 3 and bvec.dim() == 3:
        return op(bmat, bvec.permute(1, 2, 0), **kwargs).permute(2, 0, 1)
    else:
        raise ValueError("Invalid dimensions for bmat and bvec")


class SparseMultivariateNormal(Distribution):
    r"""
    Multivariate normal with sparse Cholesky / LDL^T parameterizations.

    Supports sparse covariance **or** sparse precision factors using either
    the standard Cholesky (:math:`L L^\top`) or the modified Cholesky (:math:`L D L^\top`)
    parameterization. Sparse triangular factors can be given in COO or CSR
    layout and (optionally) batched with a single leading batch dimension.

    Parameterizations
    -----------------
    **Cholesky (LL^T)**
        - Covariance form: :math:`\Sigma = L L^\top` with lower-triangular :math:`L` (incl. diagonal).
        - Precision form:  :math:`\Omega = L L^\top` with lower-triangular :math:`L` (incl. diagonal).

    **Modified Cholesky (LDL^T)**
        - Covariance form: :math:`\Sigma = L D L^\top` with *unit* lower-triangular :math:`L` and
          diagonal :math:`D = \operatorname{diag}(\text{diagonal}) > 0`.
        - Precision form:  :math:`\Omega = L D L^\top` with *unit* lower-triangular :math:`L` and
          diagonal :math:`D = \operatorname{diag}(\text{diagonal})` (entries may be any real numbers if
          you only care about sampling via :math:`\Omega^{-1/2}`).
        - In both cases, the strictly lower part lives in the sparse ``*_tril``
          factor and the diagonal is provided separately via ``diagonal``.

    Parameters
    ----------
    loc : torch.Tensor
        Mean vector, shape ``(n,)`` or ``(B, n)``.
    diagonal : torch.Tensor, optional
        Diagonal entries for the :math:`D` in :math:`L D L^\top`. Shape ``(n,)`` or ``(B, n)``.
        Must be positive for covariance parameterization; free real for precision
        parameterization. If ``None``, the :math:`L L^\top` parameterization is used.
    scale_tril : torch.Tensor, optional
        Sparse lower-triangular factor for covariance (:math:`L`). COO or CSR with
        shape ``(n, n)`` or ``(B, n, n)``. Mutually exclusive with
        ``precision_tril``.
    precision_tril : torch.Tensor, optional
        Sparse lower-triangular factor for precision (:math:`L`). COO or CSR with
        shape ``(n, n)`` or ``(B, n, n)``. Mutually exclusive with
        ``scale_tril``.
    validate_args : bool, optional
        If ``True``, run argument checks.

    Attributes
    ----------
    loc : torch.Tensor
        Mean.
    diagonal : torch.Tensor or None
        Diagonal entries for :math:`D` in :math:`L D L^\top` form, if provided.
    scale_tril : torch.Tensor or None
        Covariance Cholesky factor (sparse lower-triangular), if provided.
    precision_tril : torch.Tensor or None
        Precision Cholesky factor (sparse lower-triangular), if provided.
    has_rsample : bool
        Reparameterized sampling is supported.

    Returns
    -------
    SparseMultivariateNormal
        A distribution instance compatible with :mod:`torch.distributions`.

    Raises
    ------
    ValueError
        On invalid shapes, unsupported layouts, or incompatible batching.

    Notes
    -----
    - Only a **single** batch dimension is supported for the sparse factors.
    - Sampling uses reparameterization:

      - Covariance :math:`L L^\top`: :math:`x = L \varepsilon`.
      - Covariance :math:`L D L^\top`: :math:`x = L (\sqrt{D} \odot \varepsilon) + (\sqrt{D} \odot \varepsilon)` since
        :math:`L` is unit lower-triangular (strictly lower part in sparse factor).
      - Precision forms use sparse triangular solves with the transpose.

    - ``log_prob`` is not implemented in this class.
    - Sparse operations are delegated to
      :func:`torchsparsegradutils.sparse_mm` and
      :func:`torchsparsegradutils.sparse_triangular_solve`.

    See Also
    --------
    torch.distributions.MultivariateNormal :
        Dense baseline distribution.
    torchsparsegradutils.sparse_mm :
        Sparse matrix–dense matrix multiply used during sampling.
    torchsparsegradutils.sparse_triangular_solve :
        Sparse triangular solver used with precision factors.

    Examples
    --------
    :math:`L L^\top` parameterization with sparse covariance::

        >>> import torch
        >>> from torchsparsegradutils.distributions import SparseMultivariateNormal
        >>> loc = torch.zeros(4)
        >>> indices = torch.tensor([[1, 2, 2, 3], [0, 0, 1, 2]])
        >>> values = torch.tensor([0.5, 0.3, 0.8, 0.2])
        >>> diag_indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        >>> diag_values = torch.tensor([1.0, 1.0, 1.0, 1.0])
        >>> all_indices = torch.cat([diag_indices, indices], dim=1)
        >>> all_values = torch.cat([diag_values, values])
        >>> scale_tril = torch.sparse_coo_tensor(all_indices, all_values, (4, 4))
        >>> mvn = SparseMultivariateNormal(loc=loc, scale_tril=scale_tril)
        >>> samples = mvn.sample((100,))
        >>> samples.shape
        torch.Size([100, 4])

    :math:`L D L^\top` parameterization with diagonal precision::

        >>> diagonal = torch.tensor([2.0, 1.5, 3.0, 1.0])  # Precision diagonal
        >>> precision_tril = torch.sparse_coo_tensor(indices, values, (4, 4))
        >>> mvn_ldlt = SparseMultivariateNormal(loc=loc, diagonal=diagonal,
        ...                                     precision_tril=precision_tril)
        >>> samples = mvn_ldlt.sample((50,))

    Batched distributions::

        >>> loc_batch = torch.randn(3, 4)
        >>> diagonal_batch = torch.abs(torch.randn(3, 4)) + 0.1
        >>> precision_batch = torch.stack([precision_tril] * 3)
        >>> mvn_batch = SparseMultivariateNormal(loc=loc_batch, diagonal=diagonal_batch,
        ...                                      precision_tril=precision_batch)
        >>> samples = mvn_batch.sample()
        >>> samples.shape
        torch.Size([3, 4])
    """

    arg_constraints = {}
    # TODO: add in constraints
    # For LDL^T parameterization:
    # arg_constraints = {'loc': constraints.real_vector,
    #                    'diagonal': constraints.independent(constraints.positive, 1),
    #                    'scale_tril': sparse_strictly_lower_triangular,
    #                    'precision_tril': sparse_strictly_lower_triangular}
    # For LL^T parameterization:
    # arg_constraints = {'loc': constraints.real_vector,
    #                    'scale_tril': constraints.lower_cholesky,
    #                    'precision_tril': constraints.lower_cholesky}

    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, diagonal=None, scale_tril=None, precision_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        elif loc.dim() > 2:
            raise ValueError(
                "loc must be at most two-dimensional as the current implementation only supports 1 batch dimension."
            )

        event_shape = loc.shape[-1:]
        self._loc = loc

        if diagonal is not None:
            if diagonal.dim() < 1:
                raise ValueError("diagonal must be at least one-dimensional.")
            elif diagonal.dim() > 2:
                raise ValueError(
                    "diagonal must be at most two-dimensional as the current implementation only supports 1 batch dimension."
                )

            if diagonal.shape[-1:] != event_shape:
                raise ValueError("diagonal must be a batch of vectors with shape {}".format(event_shape))

        self._diagonal = diagonal

        if (scale_tril is not None) + (precision_tril is not None) != 1:
            raise ValueError("Exactly one of scale_tril or precision_tril may be specified.")

        if scale_tril is not None:
            if scale_tril.layout == torch.sparse_coo:
                scale_tril = scale_tril.coalesce() if not scale_tril.is_coalesced() else scale_tril
            elif scale_tril.layout == torch.sparse_csr:
                pass
            else:
                raise ValueError("scale_tril must be sparse COO or CSR, instead of {}".format(scale_tril.layout))

            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, " "with optional leading batch dimension"
                )
            elif scale_tril.dim() > 3:
                raise ValueError("scale_tril can only have 1 batch dimension, but has {}".format(scale_tril.dim() - 2))

            if diagonal is not None:
                batch_shape = torch.broadcast_shapes(loc.shape[:-1], diagonal.shape[:-1], scale_tril.shape[:-2])
            else:
                batch_shape = torch.broadcast_shapes(loc.shape[:-1], scale_tril.shape[:-2])
            self._scale_tril = scale_tril

        else:  # precision_tril is not None
            if precision_tril.layout == torch.sparse_coo:
                precision_tril = precision_tril.coalesce()
            elif precision_tril.layout == torch.sparse_csr:
                pass
            else:
                raise ValueError(
                    "precision_tril must be sparse COO or CSR, instead of {}".format(precision_tril.layout)
                )

            if precision_tril.dim() < 2:
                raise ValueError(
                    "precision_tril must be at least two-dimensional, " "with optional leading batch dimensions"
                )
            elif precision_tril.dim() > 3:
                raise ValueError(
                    "precision_tril can only have 1 batch dimension, but has {}".format(precision_tril.dim() - 2)
                )

            if diagonal is not None:
                batch_shape = torch.broadcast_shapes(loc.shape[:-1], diagonal.shape[:-1], precision_tril.shape[:-2])
            else:
                batch_shape = torch.broadcast_shapes(loc.shape[:-1], precision_tril.shape[:-2])

            self._precision_tril = precision_tril

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def diagonal(self):
        return self._diagonal

    @property
    def scale_tril(self):
        return self._scale_tril

    @property
    def precision_tril(self):
        return self._precision_tril

    @property
    def loc(self):
        return self._loc

    @property
    def mean(self):
        return self._loc

    @property
    def mode(self):
        return self._loc

    @property
    def is_ldlt_parameterization(self):
        r"""Return ``True`` if using :math:`L D L^\top` parameterization (``diagonal`` provided), else ``False`` (:math:`L L^\top`)."""
        return self._diagonal is not None

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)

        if "_scale_tril" in self.__dict__:
            if self._diagonal is not None:
                # LDL^T parameterization: scale_tril is unit lower triangular
                eta = self._diagonal.sqrt() * eps
                x = _batch_sparse_mv(spmm, self._scale_tril, eta) + eta
            else:
                # LL^T parameterization: scale_tril is lower triangular with diagonal
                x = _batch_sparse_mv(spmm, self._scale_tril, eps)

        else:  # 'precision_tril' in self.__dict__
            if self._diagonal is not None:
                # LDL^T parameterization: precision_tril is unit lower triangular
                x = _batch_sparse_mv(
                    spts,
                    self._precision_tril,
                    eps / (self._diagonal.sqrt()),
                    upper=False,
                    unitriangular=True,
                    transpose=True,
                )
            else:
                # LL^T parameterization: precision_tril is lower triangular with diagonal
                x = _batch_sparse_mv(
                    spts,
                    self._precision_tril,
                    eps,
                    upper=False,
                    unitriangular=False,
                    transpose=True,
                )

        return self.loc + x


class SparseMultivariateNormalNative(Distribution):
    r"""
    Sparse multivariate normal (native ``torch.sparse.mm`` backend).

    This distribution models :math:`x \sim \mathcal N(\mu, \Sigma)` where the
    covariance is parameterized via a sparse Cholesky factor
    :math:`\Sigma = L L^\top`. Sampling uses ``torch.sparse.mm`` directly so
    gradients propagate to the CSR values of ``L``.

    **Scope & limitations**

    - Layout: **CSR only** (requirement of ``torch.sparse.mm``)
    - Batching: **unbatched only** (2D factor)
    - Parameterization: **LLᵀ (covariance) only**
    - No precision/LDLᵀ support in this native variant (see :class:`SparseMultivariateNormal`)

    Parameters
    ----------
    loc : torch.Tensor
        Mean vector of shape ``(n,)``.
    scale_tril : torch.Tensor
        Sparse lower-triangular Cholesky factor ``L`` of shape ``(n, n)`` in
        ``torch.sparse_csr`` layout with positive diagonal.
    validate_args : bool, optional
        If ``True``, validate input shapes/dtypes where feasible.

    Attributes
    ----------
    loc : torch.Tensor
        Mean vector.
    scale_tril : torch.Tensor
        CSR Cholesky factor ``L`` such that ``Σ = L @ L.T``.

    Notes
    -----
    Sampling uses
    :math:`x = \mu + L \varepsilon, \quad \varepsilon \sim \mathcal N(0, I)`,
    implemented as dense–sparse matmul via ``torch.sparse.mm``. Although PyTorch
    docs historically understate gradient support for some sparse ops, in
    practice autograd computes gradients w.r.t. the CSR **values** here.

    ``covariance_matrix`` and ``variance`` are computed by densifying the factor,
    which can be memory-expensive for large problems.

    See Also
    --------
    SparseMultivariateNormal
        Full-featured sparse MVN with COO/CSR, batched inputs, and precision/LDLᵀ forms.

    Examples
    --------
    Basic usage (LLᵀ parameterization):

    >>> import torch
    >>> n = 4
    >>> # Build a small lower-triangular with positive diagonal in CSR
    >>> crow = torch.tensor([0, 1, 3, 4, 4], dtype=torch.int64)
    >>> col  = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    >>> vals = torch.tensor([1.0, 0.2, 1.1, 0.3], dtype=torch.float64)
    >>> L = torch.sparse_csr_tensor(crow, col, vals, size=(n, n))
    >>> loc = torch.zeros(n, dtype=torch.float64)
    >>> mvn = SparseMultivariateNormalNative(loc, L)
    >>> x = mvn.rsample()   # (n,)
    >>> x.shape
    torch.Size([4])

    Multiple samples:

    >>> xs = mvn.rsample((100,))  # (100, n)
    >>> xs.shape
    torch.Size([100, 4])

    Log probability (densifies internally):

    >>> lp = mvn.log_prob(x)
    >>> torch.isfinite(lp).item()  # doctest: +SKIP
    True

    """

    arg_constraints = {
        "loc": constraints.real_vector,
        # TODO: create custom sparse lower triangular constraint
        # 'scale_tril': constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, scale_tril, validate_args=None):
        if loc.dim() != 1:
            raise ValueError("loc must be one-dimensional for SparseMultivariateNormalNative.")

        if scale_tril.layout != torch.sparse_csr:
            raise ValueError("scale_tril must be sparse CSR for SparseMultivariateNormalNative.")

        if scale_tril.dim() != 2:
            raise ValueError("scale_tril must be two-dimensional (unbatched) for SparseMultivariateNormalNative.")

        if scale_tril.shape[0] != scale_tril.shape[1]:
            raise ValueError("scale_tril must be square.")

        if scale_tril.shape[0] != loc.shape[0]:
            raise ValueError("scale_tril must have the same size as loc.")

        event_shape = loc.shape
        self._loc = loc
        self._scale_tril = scale_tril

        # No batch dimensions for this implementation
        batch_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def scale_tril(self):
        return self._scale_tril

    @property
    def loc(self):
        return self._loc

    @property
    def mean(self):
        return self._loc

    @property
    def mode(self):
        return self._loc

    @property
    def covariance_matrix(self):
        r"""Compute covariance matrix :math:`\Sigma = L L^\top` as ``L @ L.T`` using sparse operations."""
        # Convert to dense for covariance computation - this is expensive but needed
        warnings.warn(
            "Computing covariance_matrix requires converting sparse matrix to dense format. "
            "This may cause memory issues for large sparse matrices. "
            "Consider using variance property for diagonal elements only.",
            UserWarning,
            stacklevel=2,
        )
        L_dense = self._scale_tril.to_dense()
        return L_dense @ L_dense.T

    @property
    def variance(self):
        r"""Compute diagonal (variance) of the covariance matrix :math:`\operatorname{diag}(L L^\top)`."""
        # For LL^T parameterization, variance is sum of squares of each row
        warnings.warn(
            "Computing variance requires converting sparse matrix to dense format. "
            "This may cause memory issues for large sparse matrices.",
            UserWarning,
            stacklevel=2,
        )
        L_dense = self._scale_tril.to_dense()
        return (L_dense**2).sum(dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        r"""Sample from the distribution using :func:`torch.sparse.mm` (reparameterized)."""
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)

        # For unbatched case: eps is (num_samples, event_size) or (event_size,)
        # We need to use torch.sparse.mm(scale_tril, eps.T).T
        if eps.dim() == 1:
            # Single sample case
            x = torch.sparse.mm(self._scale_tril, eps.unsqueeze(-1)).squeeze(-1)
        else:
            # Multiple samples case
            x = torch.sparse.mm(self._scale_tril, eps.t()).t()

        return self.loc + x

    def log_prob(self, value):
        r"""Compute log probability density (densifies internally, may be memory intensive)."""
        if self._validate_args:
            self._validate_sample(value)

        # Convert to dense for log_prob computation
        warnings.warn(
            "Computing log_prob requires converting sparse matrix to dense format. "
            "This may cause memory issues for large sparse matrices. "
            "Consider using rsample() only if you don't need log_prob computation.",
            UserWarning,
            stacklevel=2,
        )
        L_dense = self._scale_tril.to_dense()

        # Solve L @ z = (value - loc) for z
        diff = value - self.loc
        if diff.dim() == 1:
            z = torch.linalg.solve_triangular(L_dense, diff.unsqueeze(-1), upper=False).squeeze(-1)
        else:
            z = torch.linalg.solve_triangular(L_dense, diff.t(), upper=False).t()

        # Compute log probability
        M = (z**2).sum(-1)  # Mahalanobis distance squared
        half_log_det = L_dense.diagonal().log().sum()

        return -0.5 * (self.event_shape[0] * math.log(2 * math.pi) + M) - half_log_det
