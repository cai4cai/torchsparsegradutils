import jax
from jax import dlpack as jax_dlpack

import torch
from torch.utils import dlpack as torch_dlpack


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
    if (not jax.config.jax_enable_x64) and (A.dtype == torch.float64):
        raise TypeError(
            f"Requested a warpper for torch tensor with dtype={A.dtype} with is not supported with jax.config.jax_enable_x64={jax.config.jax_enable_x64} - See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision"
        )

    def closure(x):
        return t2j(torch.sparse.mm(A, j2t(x)))

    return closure
