r"""Preconditioners for the iterative solvers.

A preconditioner is any callable ``M_inv(X) -> Y`` applying an approximation :math:`\mathbf{M}^{-1}` of
:math:`\mathbf{A}^{-1}` to ``X`` of shape ``(n,)`` or ``(*batch, n, k)``, returning a tensor of the same shape,
dtype and device. All solvers in :mod:`torchsparsegradutils.utils` accept one through their ``preconditioner``
argument.

The :class:`Preconditioner` base class adds what a bare callable lacks: :meth:`Preconditioner.transpose`, which
:func:`~torchsparsegradutils.sparse_generic_solve` uses to precondition the transposed system
:math:`\mathbf{A}^\top \mathbf{Y} = \mathbf{G}` solved in the backward pass, and structural flags that the
symmetric solvers rely on.

Preconditioners are built from a detached copy of the matrix: the implicit-function gradients of
:func:`~torchsparsegradutils.sparse_generic_solve` do not depend on :math:`\mathbf{M}`, so a preconditioner
never needs to be differentiated.
"""

from __future__ import annotations

import abc

import torch

from .utils import sparse_diagonal

__all__ = ["Preconditioner", "JacobiPreconditioner", "MatrixPreconditioner"]


class Preconditioner(abc.ABC):
    r"""Base class for preconditioners approximating :math:`\mathbf{A}^{-1}`.

    Subclasses implement :meth:`__call__` and, unless they are symmetric, :meth:`transpose`.

    Attributes
    ----------
    shape : tuple of int
        Shape ``(n, n)`` of the operator :math:`\mathbf{M}^{-1}`.
    is_symmetric : bool
        Whether :math:`\mathbf{M}^{-1}` is symmetric, in which case :meth:`transpose` returns ``self``.
    is_positive_definite : bool
        Whether :math:`\mathbf{M}^{-1}` is symmetric positive definite, as required by
        :func:`~torchsparsegradutils.utils.linear_cg` and :func:`~torchsparsegradutils.utils.minres`.
    """

    shape: tuple[int, int]
    is_symmetric: bool = False
    is_positive_definite: bool = False

    @abc.abstractmethod
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`\mathbf{M}^{-1}` to ``X`` of shape ``(n,)`` or ``(*batch, n, k)``."""

    def transpose(self) -> Preconditioner:
        r"""Return the preconditioner :math:`\mathbf{M}^{-\top}` for the transposed system."""
        if self.is_symmetric:
            return self
        raise NotImplementedError(f"{type(self).__name__} does not implement transpose()")

    @property
    def T(self) -> Preconditioner:
        """Alias of :meth:`transpose`."""
        return self.transpose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={tuple(self.shape)})"


class JacobiPreconditioner(Preconditioner):
    r"""Jacobi (diagonal) preconditioner :math:`\mathbf{M} = \operatorname{diag}(\mathbf{A})`.

    Applying it costs one elementwise product per vector and building it a single pass over the nonzeros of
    :math:`\mathbf{A}`, so it adds :math:`O(n)` memory. It is effective when :math:`\mathbf{A}` is badly scaled,
    e.g. when its rows have very different magnitudes, but does not help with conditioning that stems from the
    coupling between unknowns.

    Parameters
    ----------
    A : torch.Tensor, sparse COO or CSR, or dense (strided), shape ``(n, n)``
        Matrix to precondition. It is detached, so the preconditioner never tracks gradients.
    absolute : bool, default=False
        Use :math:`|\operatorname{diag}(\mathbf{A})|` instead of the diagonal itself. For symmetric indefinite
        matrices, this yields the symmetric positive definite preconditioner that
        :func:`~torchsparsegradutils.utils.minres` requires.

    Notes
    -----
    Zero diagonal entries are replaced by one, so :math:`\mathbf{M}^{-1}` acts as the identity on those rows and
    stays invertible. :math:`\mathbf{M}^{-1}` is always symmetric; ``is_positive_definite`` reports whether all
    (used) diagonal entries are positive, which is checked once at construction.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils.utils import JacobiPreconditioner, linear_cg
    >>> A = torch.tensor([[100.0, 1.0], [1.0, 0.01]]).to_sparse_csr()
    >>> M_inv = JacobiPreconditioner(A)
    >>> M_inv(torch.tensor([100.0, 0.01]))
    tensor([1., 1.])
    >>> x = linear_cg(A.matmul, torch.tensor([1.0, 2.0]), preconditioner=M_inv)
    """

    is_symmetric = True

    def __init__(self, A: torch.Tensor, *, absolute: bool = False):
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be a square 2D matrix, got shape {tuple(A.shape)}")
        diag = sparse_diagonal(A.detach())
        if absolute:
            diag = diag.abs()
        diag = torch.where(diag == 0, torch.ones_like(diag), diag)
        self.shape = (A.shape[0], A.shape[1])
        self.inv_diag = diag.reciprocal()
        self.is_positive_definite = bool((diag > 0).all())

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        inv_diag = self.inv_diag.to(dtype=X.dtype, device=X.device)
        return inv_diag * X if X.dim() == 1 else inv_diag.unsqueeze(-1) * X


class MatrixPreconditioner(Preconditioner):
    r"""Preconditioner given by an explicit matrix :math:`\mathbf{M}^{-1}`.

    Wraps a sparse (COO/CSR) or dense matrix so that its transpose is available for the backward pass of
    :func:`~torchsparsegradutils.sparse_generic_solve`. This is how a tensor passed as ``preconditioner`` there
    is interpreted.

    Parameters
    ----------
    M_inv : torch.Tensor, sparse COO or CSR, or dense (strided), shape ``(n, n)``
        Matrix approximating :math:`\mathbf{A}^{-1}`. It is detached.
    symmetric : bool, default=False
        Declare :math:`\mathbf{M}^{-1}` symmetric, so :meth:`transpose` returns ``self`` instead of a transposed
        copy. Not checked.
    positive_definite : bool, default=False
        Declare :math:`\mathbf{M}^{-1}` symmetric positive definite. Not checked; implies ``symmetric``.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils.utils import MatrixPreconditioner
    >>> M_inv = MatrixPreconditioner(torch.tensor([[1.0, 2.0], [0.0, 1.0]]))
    >>> M_inv.T(torch.tensor([1.0, 1.0]))
    tensor([1., 3.])
    """

    def __init__(self, M_inv: torch.Tensor, *, symmetric: bool = False, positive_definite: bool = False):
        if M_inv.dim() != 2 or M_inv.shape[0] != M_inv.shape[1]:
            raise ValueError(f"M_inv must be a square 2D matrix, got shape {tuple(M_inv.shape)}")
        if M_inv.layout not in (torch.sparse_coo, torch.sparse_csr, torch.strided):
            raise TypeError(f"Unsupported layout: {M_inv.layout}. Only COO, CSR and dense (strided) are supported.")
        self.M_inv = M_inv.detach()
        self.shape = (M_inv.shape[0], M_inv.shape[1])
        self.is_positive_definite = positive_definite
        self.is_symmetric = symmetric or positive_definite

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.M_inv.matmul(X)

    def transpose(self) -> Preconditioner:
        if self.is_symmetric:
            return self
        if self.M_inv.layout == torch.sparse_csr:
            # Transposing CSR gives CSC; convert back so that matmul stays on the CSR kernels
            M_inv_t = self.M_inv.transpose(0, 1).to_sparse_csr()
        else:
            M_inv_t = self.M_inv.transpose(0, 1)
        return MatrixPreconditioner(M_inv_t)
