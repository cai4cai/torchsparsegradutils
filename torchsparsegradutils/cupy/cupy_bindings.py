import torchsparsegradutils.cupy as tsgucupy

if tsgucupy.have_cupy:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg

import numpy as np
import scipy.sparse as nsp
import scipy.sparse.linalg

import torch

import warnings


def _get_array_modules(x):
    # Get cupy or numpy based on context
    if tsgucupy.have_cupy:
        if torch.is_tensor(x):
            if x.device == torch.device("cpu"):
                xp = np
                xsp = nsp
            else:
                xp = cp
                xsp = csp
        else:
            xp = cp.get_array_module(x)
            xsp = cupyx.scipy.get_array_module(x)
    else:
        xp = np
        xsp = nsp
    return xp, xsp


def t2c_csr(x_torch):
    # Convert a torch CSR tensor to a cupy/numpy CSR array
    xp, xsp = _get_array_modules(x_torch)

    data_c = xp.asarray(x_torch.values())
    col_idx_c = xp.asarray(x_torch.col_indices())
    ind_ptr_c = xp.asarray(x_torch.crow_indices())
    x_cupy = xsp.csr_matrix((data_c, col_idx_c, ind_ptr_c))
    return x_cupy


def c2t_csr(x_cupy):
    # Convert a cupy/numpy CSR array to a torch CSR tensor
    data_t = torch.as_tensor(x_cupy.data)
    idices_t = torch.as_tensor(x_cupy.indices)
    ind_ptr_t = torch.as_tensor(x_cupy.indptr)
    x_torch = torch.sparse_csr_tensor(ind_ptr_t, idices_t, data_t, x_cupy.shape)
    return x_torch


def t2c_coo(x_torch):
    # Convert a torch COO tensor to a cupy/numpy COO array
    xp, xsp = _get_array_modules(x_torch)

    if not x_torch.is_coalesced():
        warnings.warn(
            "Requested a conversion from torch to cupy/numpy on a non-coalesced tensor -> coalescing implicitly"
        )
        x_torch = x_torch.coalesce()
    data_c = xp.asarray(x_torch.values())
    idx_cp = xp.asarray(x_torch.indices())
    x_cupy = xsp.coo_matrix((data_c, idx_cp))
    return x_cupy


def c2t_coo(x_cupy):
    # Convert a cupy COO array to a torch COO tensor
    data_t = torch.as_tensor(x_cupy.data)
    row_t = torch.as_tensor(x_cupy.row)
    col_t = torch.as_tensor(x_cupy.col)
    x_torch = torch.sparse_coo_tensor(torch.stack([row_t, col_t], dim=0), data_t, x_cupy.shape)
    return x_torch
