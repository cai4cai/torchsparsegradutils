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
    x_torch = x_torch.contiguous() # https://github.com/google/jax/issues/8082
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax
