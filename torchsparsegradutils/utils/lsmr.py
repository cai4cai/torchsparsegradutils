"""
Code adapted from https://github.com/rfeinman/pytorch-minimize/blob/master/torchmin/lstsq/lsmr.py
Code modified from scipy.sparse.linalg.lsmr

Copyright (C) 2010 David Fong and Michael Saunders
"""
import torch


def _sym_ortho(a, b, out):
    torch.hypot(a, b, out=out[2])
    torch.div(a, out[2], out=out[0])
    torch.div(b, out[2], out=out[1])
    return out


@torch.no_grad()
def lsmr(
    A, b, Armat=None, n=None, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=None, x0=None, check_nonzero=True
):
    """Iterative solver for least-squares problems.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    ``A`` is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

    Parameters
    ----------
    A : {matrix, sparse matrix, ndarray, LinearOperator}
        Matrix A in the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^H x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : array_like, shape (m,)
        Vector ``b`` in the linear system.
    damp : float
        Damping factor for regularized least-squares. `lsmr` solves
        the regularized least-squares problem::
         min ||(b) - (  A   )x||
             ||(0)   (damp*I) ||_2
        where damp is a scalar.  If damp is None or 0, the system
        is solved without regularization.
    atol, btol : float, optional
        Stopping tolerances. `lsmr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, ``lsmr`` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, lsmr terminates when ``norm(A^H r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (say),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final ``x`` will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of ``A`` and ``b`` respectively.  For example, if the entries
        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        `lsmr` terminates if an estimate of ``cond(A)`` exceeds
        `conlim`.  For compatible systems ``Ax = b``, conlim could be
        as large as 1.0e+12 (say).  For least-squares problems,
        `conlim` should be less than 1.0e+8. If `conlim` is None, the
        default value is 1e+8.  Maximum precision can be obtained by
        setting ``atol = btol = conlim = 0``, but the number of
        iterations may then be excessive.
    maxiter : int, optional
        `lsmr` terminates if the number of iterations reaches
        `maxiter`.  The default is ``maxiter = min(m, n)``.  For
        ill-conditioned systems, a larger value of `maxiter` may be
        needed.
    x0 : array_like, shape (n,), optional
        Initial guess of ``x``, if None zeros are used.

    Returns
    -------
    x : ndarray of float
        Least-square solution returned.
    itn : int
        Number of iterations used.

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
        raise RuntimeError("n needs to be provided of computed from A given as a tensor")

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
