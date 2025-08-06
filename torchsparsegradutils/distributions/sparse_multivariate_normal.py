import math
import warnings

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal

from torchsparsegradutils import sparse_mm as spmm
from torchsparsegradutils import sparse_triangular_solve as spts

# from .contraints import sparse_strictly_lower_triangular

__all__ = ["SparseMultivariateNormal", "SparseMultivariateNormalNative"]


def _batch_sparse_mv(op, bmat, bvec, **kwargs):
    """Performs batched matrix-vector operation between a sparse matrix and a dense vector.

    bmat can have 0 or 1 batch dimension
    bvec can have 0, 1 or 2 leading batch dimensions

    If bmat has a leading batch dimension, it must be the same as the first batch dimension of bvec

    NOTE: this function does not support broadcasting of batch dimensions
          unlike torch.distributions.multivariate_normal._batch_mv

    Args:
        op (callable): Function that performs matrix-vector operation, either sparse_mm or sparse_triangular_solve
        bmat (torch.Tensor): Sparse matrix in sparse_coo or sparse_csr layout with optional leading batch dimension.
            Can be either strictly lower triangular (unit triangular) or lower triangular with diagonal.
        bvec (torch.Tensor): Dense vector with up to 2 optional leading batch dimensions

    Returns:
        torch.Tensor: Dense matrix vector product
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
    Creates a sparse multivariate normal (MVN) distribution
    parameterized by a mean vector :attr: `loc`,
    optional diagonal covariance or precision matrix represented as a vector :attr: `diagonal`,
    and a sparse lower triangular covariance or precision matrix
    :attr: `scale_tril` or :attr: `precision_tril`.

    The distribution supports two parameterizations:

    1. **LDL^T parameterization** (when :attr: `diagonal` is provided):
       Both :attr: `scale_tril` and :attr: `precision_tril` are stored as strictly lower triangular matrices.
       Sampling is performed assuming they are unit lower triangular matrices.
       The parameterization takes the form of:
           covariance = L @ D @ L.T
           precision  = L @ D @ L.T
       Where L is a sparse unit lower triangular matrix and D is the diagonal matrix.

    2. **LL^T parameterization** (when :attr: `diagonal` is None):
       Both :attr: `scale_tril` and :attr: `precision_tril` are stored as lower triangular matrices
       with positive diagonal elements. Sampling is performed similar to torch.distributions.MultivariateNormal.
       The parameterization takes the form of:
           covariance = L @ L.T
           precision  = L @ L.T
       Where L is a sparse lower triangular matrix with positive diagonal.

    The implementation supports sparse COO or CSR layout for scale_tril or precision_tril

    NOTE: The current implementation only supports a single leading batch dimension
    NOTE: The current implementation only supports sampling from the distribution

    Args:
        loc (Tensor): mean of the distribution with shape `batch_shape + event_shape`
        diagonal (Tensor, optional): diagonal of the covariance or precision matrix with shape `batch_shape + event_shape`.
            If None, assumes LL^T parameterization. If provided, assumes LDL^T parameterization.
        scale_tril (Tensor): sparse lower triangular matrix with shape `batch_shape + event_shape + event_shape`
            in either torch.sparse_coo or torch.sparse_csr layout
        precision_tril (Tensor): sparse lower triangular matrix with shape `batch_shape + event_shape + event_shape`
            in either torch.sparse_coo or torch.sparse_csr layout
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
        """Returns True if using LDL^T parameterization (diagonal is provided), False if using LL^T."""
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
    Creates a sparse multivariate normal (MVN) distribution using native torch.sparse.mm
    parameterized by a mean vector :attr: `loc` and a sparse lower triangular matrix
    :attr: `scale_tril` for covariance parameterization.

    This distribution only supports LL^T parameterization where:
        covariance = L @ L.T
    Where L is a sparse lower triangular matrix with positive diagonal stored as CSR.

    This implementation is restricted to:
    - CSR layout only (torch.sparse.mm requirement)
    - Unbatched matrices only (torch.sparse.mm limitation)
    - Scale (covariance) parameterization only (no precision support)
    - LL^T parameterization only (no LDL^T support)

    The advantage is that this uses torch.sparse.mm which supports both forward and
    backward passes with gradients through the sparse CSR matrix values.

    Args:
        loc (Tensor): mean of the distribution with shape `event_shape`
        scale_tril (Tensor): sparse lower triangular matrix with shape `event_shape x event_shape`
            in torch.sparse_csr layout with positive diagonal elements
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
        """Compute covariance matrix as L @ L.T using sparse operations."""
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
        """Compute diagonal of covariance matrix."""
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
        """Sample from the distribution using torch.sparse.mm."""
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
        """Compute log probability density."""
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
