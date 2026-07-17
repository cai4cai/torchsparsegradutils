# Code imported from https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/bicgstab/bicgstab.py
# Modifications to fit torchsparsegradutils
#
# Batched rewrite (spec/commit.md commit 17): every iterate lives in the
# canonical (batch_size, n, n_rhs) shape (batch_size = 1 encodes unbatched —
# naming.md §2) with per-(batch item, right-hand-side column) convergence
# masks; the matvec runs through the shared BatchedOperator adapter, which
# routes sparse operators through tsgu::spmm when the CUDA backend can take
# them.

import logging
from typing import Callable, NamedTuple, Optional, Union

import torch

from ._matvec import BatchedOperator, as_batched_rhs, restore_rhs_shape

# Default (null) logger.
_null_log = logging.getLogger("bicgstab")
_null_log.disabled = True


class BICGSTABSettings(NamedTuple):
    matvec_max: Optional[int] = None  # Max number of matvecs (default 2n)
    abstol: float = 1.0e-8  # Absolute stopping tolerance
    reltol: float = 1.0e-6  # Relative stopping tolerance
    precon: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = None
    logger: logging.Logger = _null_log


def _dot_columns(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dot product reduced over the n axis, batch and n_rhs axes retained:
    two ``(batch_size, n, n_rhs)`` tensors -> ``(batch_size, 1, n_rhs)``."""
    return (a * b).sum(dim=-2, keepdim=True)


def _column_norm(t: torch.Tensor) -> torch.Tensor:
    """2-norm reduced over the n axis, batch and n_rhs axes retained:
    ``(batch_size, n, n_rhs) -> (batch_size, 1, n_rhs)``."""
    return torch.linalg.vector_norm(t, dim=-2, keepdim=True)


def _active_max(norms: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Aggregate a per-column residual norm for logging: max over the still
    active (batch item, column) pairs, 0 when none are active."""
    return torch.where(active, norms, torch.zeros_like(norms)).max()


def bicgstab(
    matmul_closure: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    rhs: torch.Tensor,
    initial_guess: Optional[torch.Tensor] = None,
    settings: BICGSTABSettings = BICGSTABSettings(),
) -> torch.Tensor:
    r"""
    Solve linear systems with the BiConjugate Gradient Stabilized (BiCGSTAB) method.

    Solves nonsymmetric, nonsingular systems :math:`A x = b`. Accepts either a matrix-like
    tensor (dense or sparse COO/CSR, 2D or batched 3D) or a callable for the matrix–vector
    product, and optionally a (left) preconditioner, also as tensor or callable,
    approximating :math:`M^{-1}`.

    Multiple right-hand sides are solved simultaneously: all iterates are batched over
    the right-hand-side columns (and over a leading batch axis, if present) with
    per-(batch item, column) convergence masks. A column that meets its stopping
    threshold (or hits a breakdown) freezes — its solution no longer updates — while
    the remaining columns iterate on. There is no shared Krylov subspace, so this is
    mathematically unchanged from the former per-column loop.

    Parameters
    ----------
    matmul_closure : {torch.Tensor, callable(x) -> Ax}
        Matrix–vector multiplication operator. A dense tensor uses ``.matmul``; a
        sparse (COO/CSR) tensor routes through ``tsgu::spmm`` on an eligible CUDA
        backend and plain torch sparse matmul semantics otherwise. A callable
        receives operands shaped like ``rhs``: plain ``(n,)`` vectors when ``rhs``
        is unbatched (one call per right-hand-side column), or the full
        ``(batch_size, n, n_rhs)`` operand when ``rhs`` is batched.
    rhs : torch.Tensor, shape (n,), (n, n_rhs) or (batch_size, n, n_rhs)
        Right-hand side vector(s). Each column is solved independently
        (no block BiCGSTAB), all columns simultaneously.
    initial_guess : torch.Tensor, optional, shape like ``rhs``
        Initial guess. If ``None``, zero initialization is used.
    settings : BICGSTABSettings, optional
        Convergence tolerances, maximum matvecs, optional preconditioner and logger.
        ``matvec_max`` counts batched matvec calls (one call advances every column),
        defaulting to ``2 * n``; convergence thresholds apply per column as
        ``max(abstol, reltol * ||r0||)``.

    Returns
    -------
    torch.Tensor
        Solution(s) ``x`` with the **same shape as ``rhs``**.

    Raises
    ------
    RuntimeError
        If ``matmul_closure`` is neither tensor nor callable, or if the
        ``precon`` is neither tensor nor callable.

    Notes
    -----
    Per iteration (unpreconditioned) BiCGSTAB [1a]_ uses ~2 matvecs, several dot products,
    and vector updates. The algorithm can experience breakdown when certain inner
    products or denominators vanish (e.g., :math:`\langle r_0, v \rangle = 0` or :math:`\langle t, t \rangle = 0`);
    a column hitting a breakdown freezes at its current iterate. This implementation
    follows a standard variant [2a]_.

    References
    ----------
    .. [1a] Van der Vorst, H. A. (1992). *Bi-CGSTAB: A fast and smoothly converging
           variant of Bi-CG for the solution of nonsymmetric linear systems*.
           SIAM J. Sci. Stat. Comput., 13(2), 631–644.
    .. [2a] Kelley, C. T. (1995). *Iterative Methods for Linear and Nonlinear Equations*.
           SIAM.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils.utils import bicgstab
    >>> A = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
    >>> b = torch.tensor([1.0, 2.0])
    >>> x = bicgstab(A.matmul, b)
    >>> x.shape
    torch.Size([2])

    Multiple right-hand sides:

    >>> B = torch.randn(2, 3)
    >>> X = bicgstab(A.matmul, B)
    >>> X.shape
    torch.Size([2, 3])

    With custom settings:

    >>> from torchsparsegradutils.solvers.bicgstab import BICGSTABSettings
    >>> settings = BICGSTABSettings(abstol=1e-10, reltol=1e-8, matvec_max=1000)
    >>> x = bicgstab(A.matmul, b, settings=settings)

    With preconditioning:

    >>> # Diagonal preconditioner
    >>> # Extract and regularize diagonal
    >>> diagA = torch.diag(A)
    >>> eps = 1e-12
    >>> safe_diag = torch.where(diagA.abs() < eps, torch.full_like(diagA, eps), diagA)
    >>> inv_diag = 1.0 / safe_diag
    >>> # Supply as an operator (apply M^{-1} r = inv_diag * r elementwise)
    >>> settings_precond = BICGSTABSettings(
    ...     precon=lambda r: inv_diag * r  # r has same shape as b
    ... )
    >>> x = bicgstab(A.matmul, b, settings=settings_precond)
    """
    rhs_b, was_vector, was_batched = as_batched_rhs(rhs)
    batch_size, n, n_rhs = rhs_b.shape
    # Legacy vector closures receive plain (n,) operands; batched right-hand
    # sides hand the closure the full canonical operand.
    operand_ndim = 3 if was_batched else 1

    op = BatchedOperator(matmul_closure, callable_operand_ndim=operand_ndim, n_rows=n, n_cols=n)

    if settings.precon is None:
        precon = None
    elif torch.is_tensor(settings.precon) or callable(settings.precon):
        precon = BatchedOperator(settings.precon, callable_operand_ndim=operand_ndim, n_rows=n, n_cols=n).matvec
    else:
        raise RuntimeError("settings.precon must be a tensor, or a callable object!")

    nMatvec = 0

    # Initial guess is zero unless one is supplied
    if initial_guess is None:
        x = torch.zeros_like(rhs_b)
    else:
        guess, _, _ = as_batched_rhs(initial_guess)
        if guess.shape != rhs_b.shape:
            guess = guess.expand(batch_size, n, n_rhs)
        x = guess.clone()

    matvec_max = 2 * n if settings.matvec_max is None else settings.matvec_max

    # Initial residual is the fixed vector
    r0 = rhs_b.clone()
    if initial_guess is None:
        r0 = rhs_b - op.matvec(x)
        nMatvec += 1

    # Every scalar iterate of the single-column algorithm becomes a
    # (batch_size, 1, n_rhs) tensor.
    one = torch.ones((batch_size, 1, n_rhs), dtype=rhs_b.dtype, device=rhs_b.device)
    rho = one.clone()
    alpha = one.clone()
    omega = one.clone()
    rho_next = _dot_columns(r0, r0)
    residNorm0 = rho_next.sqrt().abs()
    residNorm = residNorm0
    # Per-column threshold: max(abstol, reltol * ||r0||), elementwise.
    threshold = torch.clamp_min(settings.reltol * residNorm0, settings.abstol)

    active = residNorm0 > threshold
    if nMatvec >= matvec_max:
        active = torch.zeros_like(active)

    settings.logger.info("Initial residual = %8.2e" % residNorm0.max())
    settings.logger.info("Threshold = %8.2e" % threshold.max())
    hdr = "%6s  %8s" % ("Matvec", "Residual")
    settings.logger.info(hdr)
    settings.logger.info("-" * len(hdr))

    r = r0.clone()
    p = torch.zeros_like(rhs_b)
    v = torch.zeros_like(rhs_b)

    while bool(active.any()) and nMatvec < matvec_max:
        beta = rho_next / rho * alpha / omega
        rho = rho_next

        # Update the search direction (frozen columns evolve harmlessly —
        # their x and r no longer update).
        p = r + beta * (p - omega * v)

        # Compute preconditioned search direction
        q = precon(p) if precon is not None else p

        v = op.matvec(q)
        nMatvec += 1

        # <r0, v> = 0 is a breakdown: those columns freeze at their current
        # iterate; the division is made safe so no NaN leaks into the batch.
        den_rv = _dot_columns(r0, v)
        active = active & (den_rv != 0)
        alpha = rho / torch.where(den_rv == 0, torch.ones_like(den_rv), den_rv)
        s = r - alpha * v

        # Check for CGS termination (per column)
        residNorm = _column_norm(s)

        settings.logger.info("%6d  %8.2e" % (nMatvec, _active_max(residNorm, active)))

        # Columns satisfying the CGS test take the half-step update and freeze.
        cgs_hit = active & (residNorm <= threshold)
        x = torch.where(cgs_hit, x + alpha * q, x)
        active = active & ~cgs_hit

        if not bool(active.any()) or nMatvec >= matvec_max:
            break

        z = precon(s) if precon is not None else s

        t = op.matvec(z)
        nMatvec += 1

        # <t, t> = 0 is a breakdown: freeze those columns, keep divisions safe.
        den_tt = _dot_columns(t, t)
        active = active & (den_tt != 0)
        omega = _dot_columns(t, s) / torch.where(den_tt == 0, torch.ones_like(den_tt), den_tt)
        rho_next = -omega * _dot_columns(r0, t)

        # Update residual and solution — masked so frozen columns keep theirs.
        r = torch.where(active, s - omega * t, r)
        x = torch.where(active, x + omega * z + alpha * q, x)

        residNorm = _column_norm(r)

        settings.logger.info("%6d  %8.2e" % (nMatvec, _active_max(residNorm, active)))

        # Columns that met their threshold freeze.
        active = active & (residNorm > threshold)

    return restore_rhs_shape(x, was_vector, was_batched)
