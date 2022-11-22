import jax
from jax import dlpack as jax_dlpack
import jax.experimental.sparse

import torch
from torch.utils import dlpack as torch_dlpack

import warnings


def j2t(x_jax):
    # Convert a jax array to a torch tensor
    # See https://github.com/lucidrains/jax2torch/blob/main/jax2torch/jax2torch.py
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch


def t2j(x_torch):
    # Convert a torch tensor to a jax array
    # See https://github.com/lucidrains/jax2torch/blob/main/jax2torch/jax2torch.py
    x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax


def spmm_t4j(A):
    # This is a wrapper to make a pytorch sparse matrix behave as
    # a linear operator for jax arrays
    # - A should be a pytorch sparse tensor
    # - x should be is a jax array
    # Note however that this is of limited functionality as discussed here:
    # https://github.com/google/jax/discussions/13250
    if (not jax.config.jax_enable_x64) and (A.dtype == torch.float64):
        raise TypeError(
            f"Requested a warpper for torch tensor with dtype={A.dtype} with is not supported with jax.config.jax_enable_x64={jax.config.jax_enable_x64} - See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision"
        )

    def closure(x):
        return t2j(torch.sparse.mm(A, j2t(x)))

    return closure


def t2j_csr(x_torch):
    # Convert a torch CSR tensor to a jax CSR array
    data_j = t2j(x_torch.values())
    idx_j = t2j(x_torch.col_indices())
    ind_ptr_j = t2j(x_torch.crow_indices())
    x_jax = jax.experimental.sparse.CSR((data_j, idx_j, ind_ptr_j), shape=x_torch.shape)
    return x_jax


def j2t_csr(x_jax):
    # Convert a jax CSR array to a torch CSR tensor
    data_t = j2t(x_jax.data)
    idices_t = j2t(x_jax.indices)
    ind_ptr_t = j2t(x_jax.indptr)
    x_torch = torch.sparse_csr_tensor(ind_ptr_t, idices_t, data_t, x_jax.shape)
    return x_torch


def t2j_coo(x_torch):
    # Convert a torch COO tensor to a jax COO array
    if not x_torch.is_coalesced():
        warnings.warn("Requested a conversion from torch to jax on a non-coalesced tensor -> coalescing implicitly")
        x_torch = x_torch.coalesce()
    data_j = t2j(x_torch.values())
    row_j = t2j(x_torch.indices()[0, :])
    col_j = t2j(x_torch.indices()[1, :])
    x_jax = jax.experimental.sparse.COO((data_j, row_j, col_j), shape=x_torch.shape)
    return x_jax


def j2t_coo(x_jax):
    # Convert a jax COO array to a torch COO tensor
    data_t = j2t(x_jax.data)
    row_t = j2t(x_jax.row)
    col_t = j2t(x_jax.col)
    x_torch = torch.sparse_coo_tensor(torch.stack([row_t, col_t], dim=0), data_t, x_jax.shape)
    return x_torch
