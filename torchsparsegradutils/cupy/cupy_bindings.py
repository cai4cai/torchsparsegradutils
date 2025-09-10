r"""CuPy / NumPy ↔ PyTorch sparse interop.

Utilities to convert between PyTorch sparse tensors (COO / CSR) and
SciPy / CuPy sparse matrices. The NumPy / SciPy vs CuPy / cuSPARSE stack is
selected automatically from the device (CPU vs CUDA) and CuPy availability.

Notes
-----
* COO conversions require **coalesced** PyTorch tensors. If an input COO is
    not coalesced, it is coalesced with a warning.
* Index / indptr arrays use zero‑copy views where possible (no host/device
    round‑trips unless needed for dtype / layout changes).
* CSR shapes and index semantics match PyTorch's ``torch.sparse_csr_tensor``.
* Conversion preserves dtype & device (except when backend routines upcast).

See Also
--------
``torch.sparse_coo_tensor``
``torch.sparse_csr_tensor``
``scipy.sparse``
``cupyx.scipy.sparse``
"""

from typing import Any, Tuple

import torchsparsegradutils.cupy as tsgucupy

if tsgucupy.have_cupy:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg

import warnings

import numpy as np
import scipy.sparse as nsp
import scipy.sparse.linalg
import torch


def _get_array_modules(x: Any) -> Tuple[Any, Any]:
    r"""
    Select dense & sparse array modules (NumPy/SciPy or CuPy/cupyx.scipy.sparse).

    Determines ``(xp, xsp)`` based on CuPy availability and the device of ``x``.
    If ``x`` is a PyTorch tensor we inspect ``x.device``; otherwise we infer
    from the underlying array type. When CuPy is unavailable we fall back to
    ``(numpy, scipy.sparse)`` unconditionally.

    Parameters
    ----------
    x : Any
        PyTorch tensor, NumPy array, or CuPy array used to infer device / backend.

    Returns
    -------
    xp : module
        ``numpy`` or ``cupy``.
    xsp : module
        ``scipy.sparse`` or ``cupyx.scipy.sparse``.

    Examples
    --------
    >>> import torch
    >>> xp, xsp = _get_array_modules(torch.zeros(1))  # CPU → (numpy, scipy.sparse)
    """
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


def t2c_csr(x_torch: torch.Tensor) -> Any:
    r"""
    Convert a PyTorch CSR tensor to a CuPy / NumPy CSR matrix.

    Parameters
    ----------
    x_torch : torch.Tensor
        2D sparse CSR tensor (layout ``torch.sparse_csr``).

    Returns
    -------
    Any
        ``cupyx.scipy.sparse.csr_matrix`` (CUDA with CuPy) else ``scipy.sparse.csr_matrix``.

    Raises
    ------
    ValueError
        If ``x_torch`` is not a 2D CSR tensor.

    See Also
    --------
    c2t_csr : PyTorch conversion in the opposite direction.
    t2c_coo, c2t_coo : COO conversions.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils.cupy import t2c_csr
    >>> x = torch.randn(4, 4).to_sparse_csr()
    >>> X = t2c_csr(x)
    >>> X.shape
    (4, 4)
    """
    xp, xsp = _get_array_modules(x_torch)

    data_c = xp.asarray(x_torch.values())
    col_idx_c = xp.asarray(x_torch.col_indices())
    ind_ptr_c = xp.asarray(x_torch.crow_indices())
    x_cupy = xsp.csr_matrix((data_c, col_idx_c, ind_ptr_c), shape=x_torch.shape)
    return x_cupy


def c2t_csr(x_cupy: Any) -> torch.Tensor:
    r"""
    Convert a CuPy / NumPy CSR matrix to a PyTorch CSR tensor.

    Parameters
    ----------
    x_cupy : Any
        CSR matrix with attributes ``data``, ``indices``, ``indptr``.

    Returns
    -------
    torch.Tensor
        2D sparse CSR tensor (same shape & numeric data) with layout ``torch.sparse_csr``.

    See Also
    --------
    t2c_csr : Reverse conversion.

    Examples
    --------
    >>> import numpy as np, scipy.sparse as nsp
    >>> from torchsparsegradutils.cupy import c2t_csr
    >>> X = nsp.random(5, 3, density=0.2, format='csr')
    >>> x = c2t_csr(X)
    >>> x.layout is torch.sparse_csr
    True
    """
    data_t = torch.as_tensor(x_cupy.data)
    idices_t = torch.as_tensor(x_cupy.indices)
    ind_ptr_t = torch.as_tensor(x_cupy.indptr)
    x_torch = torch.sparse_csr_tensor(ind_ptr_t, idices_t, data_t, x_cupy.shape)
    return x_torch


def t2c_coo(x_torch: torch.Tensor) -> Any:
    r"""
    Convert a PyTorch COO tensor to a CuPy / NumPy COO matrix.

    Parameters
    ----------
    x_torch : torch.Tensor
        2D sparse COO tensor (layout ``torch.sparse_coo``). Coalesced automatically
        (with a warning) if duplicates are present.

    Returns
    -------
    Any
        ``cupyx.scipy.sparse.coo_matrix`` (CUDA+CuPy) else ``scipy.sparse.coo_matrix``.

    Warns
    -----
    UserWarning
        If the input is not coalesced.

    Raises
    ------
    ValueError
        If ``x_torch`` is not a 2D COO tensor.

    See Also
    --------
    c2t_coo : Reverse COO conversion.
    t2c_csr, c2t_csr : CSR conversions.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils.cupy import t2c_coo
    >>> idx = torch.tensor([[0, 1], [1, 0]])
    >>> val = torch.tensor([2.0, 3.0])
    >>> x = torch.sparse_coo_tensor(idx, val, (2, 2))
    >>> X = t2c_coo(x)
    >>> X.shape
    (2, 2)
    """
    xp, xsp = _get_array_modules(x_torch)

    if not x_torch.is_coalesced():
        warnings.warn(
            "Requested a conversion from torch to cupy/numpy on a non-coalesced tensor -> coalescing implicitly"
        )
        x_torch = x_torch.coalesce()
    data_c = xp.asarray(x_torch.values())
    idx_cp = xp.asarray(x_torch.indices())
    x_cupy = xsp.coo_matrix((data_c, idx_cp), shape=x_torch.shape)
    return x_cupy


def c2t_coo(x_cupy: Any) -> torch.Tensor:
    r"""
    Convert a CuPy / NumPy COO matrix to a PyTorch COO tensor.

    Parameters
    ----------
    x_cupy : Any
        COO matrix with ``data``, ``row``, ``col`` attributes.

    Returns
    -------
    torch.Tensor
        2D sparse COO tensor with identical numerical content.

    See Also
    --------
    t2c_coo : Reverse conversion.

    Examples
    --------
    >>> import numpy as np, scipy.sparse as nsp
    >>> from torchsparsegradutils.cupy import c2t_coo
    >>> X = nsp.coo_matrix(np.array([[0, 1], [2, 0]]))
    >>> x = c2t_coo(X)
    >>> x.layout is torch.sparse_coo
    True
    """
    data_t = torch.as_tensor(x_cupy.data)
    row_t = torch.as_tensor(x_cupy.row)
    col_t = torch.as_tensor(x_cupy.col)
    x_torch = torch.sparse_coo_tensor(torch.stack([row_t, col_t], dim=0), data_t, x_cupy.shape)
    return x_torch
