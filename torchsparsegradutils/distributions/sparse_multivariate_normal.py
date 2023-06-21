import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property

from torchsparsegradutils import sparse_mm as spmm
from torchsparsegradutils import sparse_triangular_solve as spts
from torchsparsegradutils.utils import sparse_eye

# from .contraints import sparse_strictly_lower_triangular

__all__ = ["SparseMultivariateNormal"]


def _batch_sparse_mv(op, bmat, bvec, **kwargs):
    """Performs batched matrix-vector operation between a sparse matrix and a dense vector.

    bmat can have 0 or 1 batch dimension
    bvec can have 0, 1 or 2 leading batch dimensions

    If bmat has a leading batch dimension, it must be the same as the first batch dimension of bvec

    NOTE: this function does not support broadcasting of batch dimensions
          unlike torch.distributions.multivariate_normal._batch_mv

    Args:
        op (callable): Function that performs matrix-vector operation, either sparse_mm or sparse_triangular_solve
        bmat (torch.Tensor): Sparse matrix in sparse_coo or sparse_csr layout with optional leading batch dimension
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
    doc string
    LDLt decomposition of covariance matrix: C = L @ D @ L.T
    LDLt decomposition of precision matrix: P = L @ D @ L.T
    This implementation only supports a single batch dimension.
    """
    # TODO: It is confusing that scale_tril and precision_tril return unit triangular and strictly lower triangular matrices respectively
    # TODO: add in constraints
    # arg_constraints = {'loc': constraints.real_vector,
    #                    'diag': constraints.independent(constraints.positive, 1),
    #                    'scale_tril': sparse_strictly_lower_triangular,
    #                    'precision_tril': sparse_strictly_lower_triangular}

    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, diagonal, scale_tril=None, precision_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        elif loc.dim() > 2:
            raise ValueError(
                "loc must be at most two-dimensional as the current implementation only supports 1 batch dimension."
            )

        event_shape = loc.shape[-1:]
        self._loc = loc

        if diagonal.dim() < 1:
            raise ValueError("diagonal must be at least one-dimensional.")
        elif diagonal.dim() > 2:
            raise ValueError(
                "diagonal must be at most two-dimensional as the current implementation only supports 1 batch dimension."
            )

        if diagonal.shape[-1:] != event_shape:
            raise ValueError("cov_diag must be a batch of vectors with shape {}".format(event_shape))

        self._diagonal = diagonal

        if (scale_tril is not None) + (precision_tril is not None) != 1:
            raise ValueError("Exactly one of scale_tril or precision_tril may be specified.")

        if scale_tril is not None:
            if scale_tril.layout == torch.sparse_coo:
                scale_tril = scale_tril.coalesce() if not scale_tril.is_coalesced() else scale_tril
                indices_dtype = scale_tril.indices().dtype
            elif scale_tril.layout == torch.sparse_csr:
                indices_dtype = scale_tril.crow_indices().dtype
            else:
                raise ValueError("scale_tril must be sparse COO or CSR, instead of {}".format(scale_tril.layout))

            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, " "with optional leading batch dimension"
                )
            elif scale_tril.dim() > 3:
                raise ValueError("scale_tril can only have 1 batch dimension, but has {}".format(scale_tril.dim() - 2))

            batch_shape = torch.broadcast_shapes(loc.shape[:-1], diagonal.shape[:-1], scale_tril.shape[:-2])

            # add unit diagonal to scale_tril, as this is required for LDLt decomposition and sampling
            Id = sparse_eye(
                scale_tril.shape,
                layout=scale_tril.layout,
                values_dtype=scale_tril.dtype,
                indices_dtype=indices_dtype,
                device=scale_tril.device,
            )
            if len(batch_shape) == 0:
                self._scale_tril = scale_tril + Id
            else:  # BUG: sparse tensors do not support batched addition
                pass

        else:  # precision_tril is not None
            if precision_tril.layout == torch.sparse_coo:
                precision_tril = precision_tril.coalesce()
            elif precision_tril.layout == torch.sparse_csr:
                pass
            else:
                raise ValueError("scale_tril must be sparse COO or CSR, instead of {}".format(precision_tril.layout))

            if precision_tril.dim() < 2:
                raise ValueError(
                    "precision_tril must be at least two-dimensional, " "with optional leading batch dimensions"
                )
            elif precision_tril.dim() > 3:
                raise ValueError(
                    "precision_tril can only have 1 batch dimension, but has {}".format(precision_tril.dim() - 2)
                )

            batch_shape = torch.broadcast_shapes(loc.shape[:-1], diagonal.shape[:-1], precision_tril.shape[:-2])

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

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)

        if "_scale_tril" in self.__dict__:
            x = _batch_sparse_mv(spmm, self._scale_tril, self._diagonal.sqrt() * eps)

        else:  # 'precision_tril' in self.__dict__
            x = _batch_sparse_mv(
                spts,
                self._precision_tril,
                eps / (self._diagonal.sqrt()),
                upper=False,
                unitriangular=True,
                transpose=True,
            )

        return self.loc + x
