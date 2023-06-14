# Code imported from https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/bicgstab/bicgstab.py
# Modifications to fit torchsparsegradutils

import torch
import warnings

from typing import NamedTuple
import types

import logging

# Default (null) logger.
_null_log = logging.getLogger("bicgstab")
_null_log.disabled = True


class BICGSTABSettings(NamedTuple):
    matvec_max: int = None  # Max. number of matrix-vector produts (2n)
    abstol: float = 1.0e-8  # absolute stopping tolerance
    reltol: float = 1.0e-6  # absolute stopping tolerance
    precon: any = None  # optional preconditioner
    logger: any = _null_log  # a `logging.logger` instance.


def bicgstab(
    matmul_closure,
    rhs,
    initial_guess=None,
    settings=BICGSTABSettings(),
):
    """
    A pytorch implementation of the bi-conjugate gradient stabilized
    (Bi-CGSTAB) algorithm. Bi-CGSTAB may be used to solve unsymmetric systems
    of linear equations, i.e., systems of the form
        A x = b
    where the operator A is unsymmetric and nonsingular.

    Bi-CGSTAB requires 2 operator-vector products, 6 dot products and 6 daxpys
    per iteration.

    In addition, if a preconditioner is supplied, it needs to solve 2
    preconditioning systems per iteration.

    The original description appears in [VdVorst92]_. This implementation is a
    preconditioned version of that given in [Kelley]_.

    Reference:
    .. [VdVorst92] H. Van der Vorst, *Bi-CGSTAB: A Fast and Smoothly Convergent
                   Variant of Bi-CG for the Solution of Nonsymmetric Linear
                   Systems*, SIAM Journal on Scientific and Statistical
                   Computing **13** (2), pp. 631--644, 1992.


    .. [Kelley] C. T. Kelley, *Iterative Methods for Linear and Nonlinear
                Equations*, number 16 in *Frontiers in Applied Mathematics*,
                SIAM, Philadelphia, 1995.

    Solve a linear system with `rhs` as right-hand side by the Bi-CGSTAB
    method. The vector `rhs` should be a Numpy array.

    :keywords:
        :guess:      Initial guess (Numpy array, default: 0)
        :matvec_max: Max. number of matrix-vector produts (2n)
    """
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
    residNorm = residNorm

    return bestSolution
