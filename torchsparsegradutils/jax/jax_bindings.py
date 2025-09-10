r"""JAX ↔ PyTorch sparse/dense conversion utilities.

Helpers to convert between JAX arrays (``jax.Array``,
``jax.experimental.sparse.COO``, ``jax.experimental.sparse.CSR``) and PyTorch
tensors (dense, COO, CSR). Conversions rely on **DLPack** for (potentially)
zero-copy interop.

Key Features
------------
* ``t2j`` / ``j2t`` – dense conversions (torch ↔ jax)
* ``t2j_coo`` / ``j2t_coo`` – sparse COO conversions
* ``t2j_csr`` / ``j2t_csr`` – sparse CSR conversions
* ``spmm_t4j`` – wrap a PyTorch sparse matrix as a JAX linear operator

Notes
-----
* Memory may be shared across frameworks; avoid unsafe in-place mutation
    visible to both.
* Enable 64-bit support in JAX for ``torch.float64`` tensors [3]_:
    ``jax.config.update("jax_enable_x64", True)``.
* COO tensors must be coalesced before conversion to JAX.
* Conversions rely on DLPack [1]_ for interoperability.
* For JAX sparse implementation details, see [2]_.

References
----------
.. [1] DLPack interop: https://github.com/dmlc/dlpack
.. [2] JAX sparse discussion: https://github.com/google/jax/discussions/13250
.. [3] JAX 64-bit precision notes:
             https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
"""

import warnings
from typing import Any, Callable

import jax
import jax.experimental.sparse  # COO / CSR
import torch
from jax import Array as JAXArray, dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack  # kept for completeness


def j2t(x_jax: JAXArray) -> torch.Tensor:
    r"""Convert a JAX array to a PyTorch tensor (DLPack zero-copy if possible).

    Parameters
    ----------
    x_jax : jax.Array
        Source JAX array.

    Returns
    -------
    torch.Tensor
        Target PyTorch tensor (may share memory with ``x_jax``).

    Notes
    -----
    Utilizes ``torch.utils.dlpack.from_dlpack``; shared storage depends on
    device/backend. Avoid in-place modifications across frameworks.

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> x_j = jnp.asarray(np.arange(6).reshape(2, 3))
    >>> x_t = j2t(x_j)
    >>> x_t.shape
    torch.Size([2, 3])
    >>> import numpy as np
    >>> np.allclose(np.asarray(x_j), x_t.cpu().numpy())
    True

    See Also
    --------
    t2j : PyTorch → JAX conversion.
    """
    return torch_dlpack.from_dlpack(x_jax)


def t2j(x_torch: torch.Tensor) -> JAXArray:
    r"""Convert a PyTorch tensor to a JAX array (DLPack zero-copy if possible).

    Ensures contiguity first to avoid historical interop quirks.

    Parameters
    ----------
    x_torch : torch.Tensor
        Source tensor.

    Returns
    -------
    jax.Array
        Resulting JAX array (may share memory with ``x_torch``).

    Examples
    --------
    >>> import torch, numpy as np
    >>> x_t = torch.randn(4, 2, dtype=torch.float64)
    >>> x_j = t2j(x_t)
    >>> x_j.shape
    (4, 2)
    >>> import numpy as np
    >>> np.allclose(np.asarray(x_j), x_t.cpu().numpy())
    True

    See Also
    --------
    j2t : JAX → PyTorch conversion.
    """
    x_torch = x_torch.contiguous()
    return jax_dlpack.from_dlpack(x_torch)


def spmm_t4j(A: torch.Tensor) -> Callable[[JAXArray], JAXArray]:
    r"""Wrap a PyTorch sparse matrix as a JAX linear-operator-style closure.

    Returns a callable performing ``A @ x`` with conversions handled internally.

    Parameters
    ----------
    A : torch.Tensor
        Sparse 2D tensor (COO or CSR).

    Returns
    -------
    callable
        ``closure(x: jax.Array) -> jax.Array``.

    Raises
    ------
    TypeError
        If double precision is requested without JAX x64 enabled.

    Notes
    -----
    Enable x64: ``jax.config.update("jax_enable_x64", True)``.

    Examples
    --------
    >>> import torch, numpy as np, jax
    >>> A = torch.randn(4, 4, dtype=torch.float64).relu().to_sparse_csr()
    >>> x_j = t2j(torch.randn(4, 2, dtype=torch.float64))
    >>> A_used = A if jax.config.read('jax_enable_x64') else A.to(torch.float32)
    >>> Ax_j = spmm_t4j(A_used)(x_j)
    >>> Ax_t = A_used @ j2t(x_j)
    >>> import numpy as np
    >>> np.allclose(np.asarray(Ax_j), Ax_t.cpu().numpy())
    True
    """
    x64_enabled = jax.config.read("jax_enable_x64")
    if (not x64_enabled) and (A.dtype == torch.float64):
        raise TypeError(
            f"Requested a wrapper for torch tensor with dtype={A.dtype} which is not supported with jax.config.read('jax_enable_x64')={x64_enabled} - See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision"
        )

    def closure(x: JAXArray) -> JAXArray:
        return t2j(torch.sparse.mm(A, j2t(x)))

    return closure


def t2j_csr(x_torch: torch.Tensor) -> Any:
    r"""Convert a PyTorch CSR tensor to a JAX CSR sparse array.

    Parameters
    ----------
    x_torch : torch.Tensor
        2D CSR tensor.

    Returns
    -------
    jax.experimental.sparse.CSR
        JAX CSR array with identical sparsity pattern.

    Notes
    -----
    Moves ``values``, ``col_indices``, ``crow_indices`` buffers via DLPack.

    Examples
    --------
    >>> import torch, numpy as np
    >>> X = torch.randn(4, 4, dtype=torch.float64)
    >>> X_sparse = X.relu().to_sparse_csr()
    >>> X_csr_j = t2j_csr(X_sparse)
    >>> import numpy as np
    >>> np.allclose(np.asarray(X_csr_j.todense()), X_sparse.to_dense().cpu().numpy())
    True

    See Also
    --------
    j2t_csr : JAX → PyTorch CSR conversion.
    """
    data_j = t2j(x_torch.values())
    idx_j = t2j(x_torch.col_indices())
    ind_ptr_j = t2j(x_torch.crow_indices())
    return jax.experimental.sparse.CSR((data_j, idx_j, ind_ptr_j), shape=x_torch.shape)


def j2t_csr(x_jax: Any) -> torch.Tensor:
    r"""Convert a JAX CSR sparse array to a PyTorch CSR tensor.

    Parameters
    ----------
    x_jax : jax.experimental.sparse.CSR
        2D JAX CSR array.

    Returns
    -------
    torch.Tensor
        PyTorch CSR tensor with the same sparsity.

    Notes
    -----
    Transfers ``data``, ``indices``, ``indptr`` buffers via DLPack.

    Examples
    --------
    >>> import jax, jax.numpy as jnp, numpy as np, torch
    >>> Xj = jnp.tril(jnp.ones((3, 3)))
    >>> Xj_csr = jax.experimental.sparse.CSR.fromdense(Xj)
    >>> Xt_csr = j2t_csr(Xj_csr)
    >>> import numpy as np
    >>> np.allclose(np.asarray(Xj_csr.todense()), Xt_csr.to_dense().cpu().numpy())
    True

    See Also
    --------
    t2j_csr : PyTorch → JAX CSR conversion.
    """
    data_t = j2t(x_jax.data)
    indices_t = j2t(x_jax.indices)
    ind_ptr_t = j2t(x_jax.indptr)
    return torch.sparse_csr_tensor(ind_ptr_t, indices_t, data_t, x_jax.shape)


def t2j_coo(x_torch: torch.Tensor) -> Any:
    r"""Convert a PyTorch COO tensor to a JAX COO sparse array.

    Parameters
    ----------
    x_torch : torch.Tensor
        2D COO tensor (coalesced internally if needed).

    Returns
    -------
    jax.experimental.sparse.COO
        JAX COO array mirroring sparsity.

    Notes
    -----
    Coalesces then invokes private ``_sort_indices`` (avoid CuSparse warnings).

    Examples
    --------
    >>> import torch, numpy as np
    >>> X = torch.randn(3, 3)
    >>> X_coo = X.relu().to_sparse_coo()
    >>> X_coo_j = t2j_coo(X_coo)
    >>> import numpy as np
    >>> np.allclose(np.asarray(X_coo_j.todense()), X_coo.to_dense().cpu().numpy())
    True

    See Also
    --------
    j2t_coo : JAX → PyTorch COO conversion.
    """
    if not x_torch.is_coalesced():
        warnings.warn("Requested a conversion from torch to jax on a non-coalesced tensor -> coalescing implicitly")
        x_torch = x_torch.coalesce()
    data_j = t2j(x_torch.values())
    row_j = t2j(x_torch.indices()[0, :])
    col_j = t2j(x_torch.indices()[1, :])
    x_jax = jax.experimental.sparse.COO((data_j, row_j, col_j), shape=x_torch.shape)
    # ensure indices are sorted to avoid CuSparseEfficiencyWarning on GPU
    x_jax = x_jax._sort_indices()
    return x_jax


def j2t_coo(x_jax: Any) -> torch.Tensor:
    r"""Convert a JAX COO sparse array to a PyTorch COO tensor.

    Parameters
    ----------
    x_jax : jax.experimental.sparse.COO
        2D JAX COO array.

    Returns
    -------
    torch.Tensor
        PyTorch COO tensor.

    Notes
    -----
    Indices stacked to shape ``(2, nnz)`` per PyTorch COO format.

    Examples
    --------
    >>> import jax, jax.numpy as jnp, numpy as np, torch
    >>> Xj = jnp.triu(jnp.ones((3, 3)))
    >>> Xj_coo = jax.experimental.sparse.COO.fromdense(Xj)
    >>> Xt_coo = j2t_coo(Xj_coo)
    >>> import numpy as np
    >>> np.allclose(np.asarray(Xj_coo.todense()), Xt_coo.to_dense().cpu().numpy())
    True

    See Also
    --------
    t2j_coo : PyTorch → JAX COO conversion.
    """
    data_t = j2t(x_jax.data)
    row_t = j2t(x_jax.row)
    col_t = j2t(x_jax.col)
    indices = torch.stack([row_t, col_t], dim=0)
    return torch.sparse_coo_tensor(indices, data_t, x_jax.shape)
