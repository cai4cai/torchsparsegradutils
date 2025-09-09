# Code imported from https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/bicgstab/bicgstab.py
# Modifications to fit torchsparsegradutils

import logging
from typing import Callable, NamedTuple, Optional, Union

import torch

# Default (null) logger.
_null_log = logging.getLogger("bicgstab")
_null_log.disabled = True


class BICGSTABSettings(NamedTuple):
    matvec_max: Optional[int] = None  # Max number of matvecs (default 2n)
    abstol: float = 1.0e-8  # Absolute stopping tolerance
    reltol: float = 1.0e-6  # Relative stopping tolerance
    precon: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = None
    logger: logging.Logger = _null_log


def bicgstab(
    matmul_closure: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    rhs: torch.Tensor,
    initial_guess: Optional[torch.Tensor] = None,
    settings: BICGSTABSettings = BICGSTABSettings(),
) -> torch.Tensor:
    r"""
    Solve linear systems with the BiConjugate Gradient Stabilized (BiCGSTAB) method.

    Solves nonsymmetric, nonsingular systems :math:`A x = b`. Accepts either a matrix-like
    tensor (using ``.matmul``) or a callable for the matrix–vector product, and
    optionally a (left) preconditioner, also as tensor or callable, approximating :math:`M^{-1}`.

    Parameters
    ----------
    matmul_closure : {torch.Tensor, callable(x) -> Ax}
        Matrix–vector multiplication operator. If a tensor is provided, its
        ``.matmul`` is used.
    rhs : torch.Tensor, shape (n,) or (n, k)
        Right-hand side vector(s). For multiple RHS, **each column is solved
        independently** (no block BiCGSTAB).
    initial_guess : torch.Tensor, optional, shape like ``rhs``
        Initial guess. If ``None``, zero initialization is used.
    settings : BICGSTABSettings, optional
        Convergence tolerances, maximum matvecs, optional preconditioner and logger.

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
    products or denominators vanish (e.g., :math:`\langle r_0, v \rangle = 0` or :math:`\langle t, t \rangle = 0`).
    This implementation follows a standard variant [2a]_ and solves multiple RHS by
    looping over columns (no shared Krylov subspace).

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

    >>> from torchsparsegradutils.utils.bicgstab import BICGSTABSettings
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
    # support multiple right‐hand sides by solving each column separately
    if rhs.dim() > 1:
        cols = rhs.shape[1]
        sols = [
            bicgstab(
                matmul_closure,
                rhs[:, i],
                None if initial_guess is None else initial_guess[:, i],
                settings,
            )
            for i in range(cols)
        ]
        return torch.stack(sols, dim=1)

    n = rhs.shape[0]
    nMatvec = 0

    if torch.is_tensor(matmul_closure):
        op = matmul_closure.matmul
    elif callable(matmul_closure):
        op = matmul_closure
    else:
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    if settings.precon is None:
        precon = None
    elif torch.is_tensor(settings.precon):
        precon = settings.precon.matmul
    elif callable(settings.precon):
        precon = settings.precon
    else:
        raise RuntimeError("settings.precon must be a tensor, or a callable object!")

    # Initial guess is zero unless one is supplied
    res_device = rhs.device
    res_dtype = rhs.dtype

    if initial_guess is None:
        x = torch.zeros(n, dtype=res_dtype, device=res_device)
    else:
        x = initial_guess.clone()

    # matvec_max = kwargs.get('matvec_max', 2*n)
    matvec_max = 2 * n if settings.matvec_max is None else settings.matvec_max

    # Initial residual is the fixed vector
    r0 = rhs.clone()
    if initial_guess is None:
        r0 = rhs - op(x)
        nMatvec += 1

    rho = alpha = omega = 1.0
    rho_next = torch.dot(r0, r0)
    residNorm = residNorm0 = torch.abs(torch.sqrt(rho_next))
    threshold = max(settings.abstol, settings.reltol * residNorm0)

    finished = residNorm <= threshold or nMatvec >= matvec_max

    settings.logger.info("Initial residual = %8.2e" % residNorm0)
    settings.logger.info("Threshold = %8.2e" % threshold)
    hdr = "%6s  %8s" % ("Matvec", "Residual")
    settings.logger.info(hdr)
    settings.logger.info("-" * len(hdr))

    if not finished:
        r = r0.clone()
        p = torch.zeros(n, dtype=res_dtype, device=res_device)
        v = torch.zeros(n, dtype=res_dtype, device=res_device)

    while not finished:
        beta = rho_next / rho * alpha / omega
        rho = rho_next

        # Update p in-place
        p *= beta
        p -= beta * omega * v
        p += r

        # Compute preconditioned search direction
        if precon is not None:
            q = precon(p)
        else:
            q = p

        v = op(q)
        nMatvec += 1

        alpha = rho / torch.dot(r0, v)
        s = r - alpha * v

        # Check for CGS termination
        residNorm = torch.linalg.norm(s)

        settings.logger.info("%6d  %8.2e" % (nMatvec, residNorm))

        if residNorm <= threshold:
            x += alpha * q
            finished = True
            continue

        if nMatvec >= matvec_max:
            finished = True
            continue

        if precon is not None:
            z = precon(s)
        else:
            z = s

        t = op(z)
        nMatvec += 1
        omega = torch.dot(t, s) / torch.dot(t, t)
        rho_next = -omega * torch.dot(r0, t)

        # Update residual
        r = s - omega * t

        # Update solution in-place-ish. Note that 'z *= omega' alters s if
        # precon = None. That's ok since s is no longer needed in this iter.
        # 'q *= alpha' would alter p.
        z *= omega
        x += z
        x += alpha * q

        residNorm = torch.linalg.norm(r)

        settings.logger.info("%6d  %8.2e" % (nMatvec, residNorm))

        if residNorm <= threshold or nMatvec >= matvec_max:
            finished = True
            continue

    # converged = residNorm <= threshold  # variable unused
    bestSolution = x

    return bestSolution
