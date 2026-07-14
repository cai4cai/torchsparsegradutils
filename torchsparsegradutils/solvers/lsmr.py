"""
Code adapted from https://github.com/rfeinman/pytorch-minimize/blob/master/torchmin/lstsq/lsmr.py
Code modified from scipy.sparse.linalg.lsmr

Copyright (C) 2010 David Fong and Michael Saunders
"""

from typing import Callable, Optional, Tuple, Union

import torch


def _sym_ortho(a, b, out):
    torch.hypot(a, b, out=out[2])
    torch.div(a, out[2], out=out[0])
    torch.div(b, out[2], out=out[1])
    return out


@torch.no_grad()
def lsmr(
    A: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    b: torch.Tensor,
    Armat: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = None,
    n: Optional[int] = None,
    damp: float = 0.0,
    atol: float = 1e-6,
    btol: float = 1e-6,
    conlim: float = 1e8,
    maxiter: Optional[int] = None,
    x0: Optional[torch.Tensor] = None,
    check_nonzero: bool = True,
) -> Tuple[torch.Tensor, int]:
    r"""
    Least Squares Minimal Residual (LSMR) solver.

    Iterative solver for :math:`A x = b` and least-squares problems
    :math:`\min_x \|A x - b\|_2`. Works with large, sparse, or rectangular :math:`A` and
    is often more stable than LSQR on ill-conditioned problems.

    Parameters
    ----------
    A : {torch.Tensor, callable(x) -> A @ x}
        System matrix or matvec closure. If a tensor is given, it may be
        dense or sparse and ``.matmul`` is used.
    b : torch.Tensor, shape (m,)
        Right-hand side vector. Must be on the same device/dtype as ``A``.
    Armat : {torch.Tensor, callable(x) -> A^T @ x}, optional
        Transpose matvec or matrix. If ``A`` is a tensor and ``Armat`` is
        ``None``, uses ``A.adjoint().matmul``. If ``A`` is callable,
        ``Armat`` is **required**.
    n : int, optional
        Number of columns of ``A``. Required if ``A`` is callable; inferred
        from ``A.shape[1]`` if ``A`` is a tensor.
    damp : float, optional
        Tikhonov damping parameter (ridge). Solves
        :math:`\min_x \|(A; \text{damp} I) x - (b; 0)\|_2`. Default: 0.0.
    atol : float, optional
        Absolute convergence tolerance. Default: 1e-6.
    btol : float, optional
        Relative residual tolerance. Default: 1e-6.
    conlim : float, optional
        Condition estimate limit; stops if estimate exceeds this value.
        Default: 1e8.
    maxiter : int, optional
        Maximum iterations. If ``None``, uses ``min(m, n)``.
    x0 : torch.Tensor, optional, shape (n,)
        Initial guess. If ``None``, zeros are used.
    check_nonzero : bool, optional
        Skip the rare ``beta == 0`` synchronization check for performance when
        set to ``False`` (use with caution). Default: True.

    Returns
    -------
    x : torch.Tensor, shape (n,)
        Approximate solution that minimizes :math:`\|A x - b\|_2` (with damped
        variant when ``damp > 0``).
    iterations : int
        Number of iterations executed.

    Raises
    ------
    RuntimeError
        If ``A`` is neither a tensor nor a callable.
    RuntimeError
        If ``A`` is callable and ``n`` is not provided.
    RuntimeError
        If ``Armat`` is missing or is neither a tensor nor a callable.

    Notes
    -----
    Uses Golub–Kahan bidiagonalization [1f]_ with specialized QR steps. For
    overdetermined systems (``m > n``), returns the least-squares solution.
    For underdetermined systems (``m < n``) with ``damp = 0``, returns the
    minimum-norm least-squares solution.

    Convergence checks (roughly):

      - Consistent: :math:`\|r\|_2 \le \text{atol} \, \|A\| \, \|x\| + \text{btol} \, \|b\|`
      - Inconsistent: :math:`\|A^\top r\|_2 \le \text{atol} \, \|A\| \, \|r\|`

    References
    ----------
    .. [1f] Fong, D. C., & Saunders, M. (2011). LSMR: An iterative algorithm for
           sparse least-squares problems. SIAM Journal on Scientific Computing,
           33(5), 2950-2971.

    Examples
    --------
    Basic least squares problem:

    >>> import torch
    >>> from torchsparsegradutils.utils import lsmr
    >>> # Over-determined system (3x2)
    >>> A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> b = torch.tensor([1.0, 2.0, 3.0])
    >>> x, iterations = lsmr(A, b)
    >>> x.shape
    torch.Size([2])

    Sparse matrix least squares:

    >>> # Create sparse matrix
    >>> indices = torch.tensor([[0, 1, 2, 1, 2], [0, 0, 0, 1, 1]])
    >>> values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> A_sparse = torch.sparse_coo_tensor(indices, values, (3, 2))
    >>> x, it = lsmr(A_sparse, b)

    With damping for regularization (Tikhonov / ridge):

    >>> # Regularized least squares
    >>> x_reg, it = lsmr(A, b, damp=0.1)

    Callable matrix interface:

    >>> x, it = lsmr(lambda v: A @ v, b, Armat=lambda v: A.T @ v, n=2)

    Under-determined system (minimum norm solution):

    >>> A_under = torch.randn(2, 4)  # 2x4 system
    >>> b_under = torch.randn(2)
    >>> x_min_norm, it = lsmr(A_under, b_under)
    >>> x_min_norm.shape
    torch.Size([4])

    Custom tolerances and limits:

    >>> x, it = lsmr(A, b, atol=1e-10, btol=1e-10, conlim=1e12, maxiter=1000)
    """
    if torch.is_tensor(A):
        if n is None:
            n = A.shape[1]
        if Armat is None:
            Armat = (torch.adjoint(A)).matmul
        A = A.matmul
    elif not callable(A):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    if n is None:
        raise RuntimeError("n needs to be provided or computed from A given as a tensor")

    if torch.is_tensor(Armat):
        Armat = Armat.matmul
    elif not callable(Armat):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    sdtype = b.dtype
    if b.dtype == torch.complex64:
        sdtype = torch.float32
    elif b.dtype == torch.complex128:
        sdtype = torch.float64

    b = torch.atleast_1d(b)
    if b.dim() > 1:
        b = b.squeeze()
    eps = torch.finfo(sdtype).eps
    damp = torch.as_tensor(damp, dtype=sdtype, device=b.device)
    ctol = 1 / conlim if conlim > 0 else 0.0
    m = b.shape[0]
    if maxiter is None:
        maxiter = min(m, n)

    u = b.clone()
    normb = b.norm()
    if x0 is None:
        x = b.new_zeros(n)
        beta = normb.clone()
    else:
        x = torch.atleast_1d(x0).clone()
        u.sub_(A(x))
        beta = u.norm()

    if beta > 0:
        u.div_(beta)
        v = Armat(u)
        alpha = v.norm()
    else:
        v = b.new_zeros(n)
        alpha = b.new_tensor(0, dtype=sdtype)

    v = torch.where(alpha > 0, v / alpha, v)

    # Initialize variables for 1st iteration.

    zetabar = alpha * beta
    alphabar = alpha.clone()
    rho = b.new_tensor(1, dtype=sdtype)
    rhobar = b.new_tensor(1, dtype=sdtype)
    cbar = b.new_tensor(1, dtype=sdtype)
    sbar = b.new_tensor(0, dtype=sdtype)

    h = v.clone()
    hbar = b.new_zeros(n)

    # Initialize variables for estimation of ||r||.

    betadd = beta.clone()
    betad = b.new_tensor(0, dtype=sdtype)
    rhodold = b.new_tensor(1, dtype=sdtype)
    tautildeold = b.new_tensor(0, dtype=sdtype)
    thetatilde = b.new_tensor(0, dtype=sdtype)
    zeta = b.new_tensor(0, dtype=sdtype)
    d = b.new_tensor(0, dtype=sdtype)

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha.square()
    maxrbar = b.new_tensor(0, dtype=sdtype)
    minrbar = b.new_tensor(0.99 * torch.finfo(sdtype).max, dtype=sdtype)
    normA = normA2.sqrt()
    condA = b.new_tensor(1, dtype=sdtype)
    normx = b.new_tensor(0, dtype=sdtype)
    # normar = b.new_tensor(0,dtype=sdtype)
    # normr = b.new_tensor(0,dtype=sdtype)

    normr = beta.clone()
    normar = alpha * beta
    if normar == 0:
        return x, 0

    if normb == 0:
        x[:] = 0
        return x, 0

    # extra buffers (added by Reuben)
    c = b.new_tensor(0, dtype=sdtype)
    s = b.new_tensor(0, dtype=sdtype)
    chat = b.new_tensor(0, dtype=sdtype)
    shat = b.new_tensor(0, dtype=sdtype)
    alphahat = b.new_tensor(0, dtype=sdtype)
    ctildeold = b.new_tensor(0, dtype=sdtype)
    stildeold = b.new_tensor(0, dtype=sdtype)
    rhotildeold = b.new_tensor(0, dtype=sdtype)
    rhoold = b.new_tensor(0, dtype=sdtype)
    rhobarold = b.new_tensor(0, dtype=sdtype)
    zetaold = b.new_tensor(0, dtype=sdtype)
    thetatildeold = b.new_tensor(0, dtype=sdtype)
    betaacute = b.new_tensor(0, dtype=sdtype)
    betahat = b.new_tensor(0, dtype=sdtype)
    betacheck = b.new_tensor(0, dtype=sdtype)
    taud = b.new_tensor(0, dtype=sdtype)

    # Main iteration loop.
    for itn in range(1, maxiter + 1):
        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.

        u.mul_(-alpha).add_(A(v))
        torch.norm(u, out=beta)

        if (not check_nonzero) or beta > 0:
            # check_nonzero option provides a means to avoid the GPU-CPU
            # synchronization of a `beta > 0` check. For most cases
            # beta == 0 is unlikely, but use this option with caution.
            u.div_(beta)
            v.mul_(-beta).add_(Armat(u))
            torch.norm(v, out=alpha)
            v = torch.where(alpha > 0, v / alpha, v)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        _sym_ortho(alphabar, damp, out=(chat, shat, alphahat))

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold.copy_(rho, non_blocking=True)
        _sym_ortho(alphahat, beta, out=(c, s, rho))
        thetanew = torch.mul(s, alpha)
        torch.mul(c, alpha, out=alphabar)

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold.copy_(rhobar, non_blocking=True)
        zetaold.copy_(zeta, non_blocking=True)
        thetabar = sbar * rho
        rhotemp = cbar * rho
        _sym_ortho(cbar * rho, thetanew, out=(cbar, sbar, rhobar))
        torch.mul(cbar, zetabar, out=zeta)
        zetabar.mul_(-sbar)

        # Update h, h_hat, x.

        hbar.mul_(-thetabar * rho).div_(rhoold * rhobarold)
        hbar.add_(h)
        x.addcdiv_(zeta * hbar, rho * rhobar)
        h.mul_(-thetanew).div_(rho)
        h.add_(v)

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        torch.mul(chat, betadd, out=betaacute)
        torch.mul(-shat, betadd, out=betacheck)

        # Apply rotation Q_{k,k+1}.
        torch.mul(c, betaacute, out=betahat)
        torch.mul(-s, betaacute, out=betadd)

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold.copy_(thetatilde, non_blocking=True)
        _sym_ortho(rhodold, thetabar, out=(ctildeold, stildeold, rhotildeold))
        torch.mul(stildeold, rhobar, out=thetatilde)
        torch.mul(ctildeold, rhobar, out=rhodold)
        betad.mul_(-stildeold).addcmul_(ctildeold, betahat)

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold.mul_(-thetatildeold).add_(zetaold).div_(rhotildeold)
        torch.div(zeta - thetatilde * tautildeold, rhodold, out=taud)
        d.addcmul_(betacheck, betacheck)
        torch.sqrt(d + (betad - taud).square() + betadd.square(), out=normr)

        # Estimate ||A||.
        normA2.addcmul_(beta, beta)
        torch.sqrt(normA2, out=normA)
        normA2.addcmul_(alpha, alpha)

        # Estimate cond(A).
        torch.max(maxrbar, rhobarold, out=maxrbar)
        if itn > 1:
            torch.min(minrbar, rhobarold, out=minrbar)

        # ------- Test for convergence --------

        # if itn % 10 == 0:
        if True:
            # Compute norms for convergence testing.
            torch.abs(zetabar, out=normar)
            torch.norm(x, out=normx)
            torch.div(torch.max(maxrbar, rhotemp), torch.min(minrbar, rhotemp), out=condA)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.
            test1 = normr / normb
            test2 = normar / (normA * normr + eps)
            test3 = 1 / (condA + eps)
            t1 = test1 / (1 + normA * normx / normb)
            rtol = btol + atol * normA * normx / normb

            # The first 3 tests guard against extremely small values of
            # atol, btol or ctol.  (The user may have set any or all of
            # the parameters atol, btol, conlim  to 0.)
            # The effect is equivalent to the normAl tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.

            # The second 3 tests allow for tolerances set by the user.

            stop = (
                (1 + test3 <= 1)
                | (1 + test2 <= 1)
                | (1 + t1 <= 1)
                | (test3 <= ctol)
                | (test2 <= atol)
                | (test1 <= rtol)
            )

            if stop:
                break

    return x, itn
