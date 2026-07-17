"""
Code adapted from https://github.com/rfeinman/pytorch-minimize/blob/master/torchmin/lstsq/lsmr.py
Code modified from scipy.sparse.linalg.lsmr

Copyright (C) 2010 David Fong and Michael Saunders

Batched rewrite (spec/commit.md commit 17): every iterate lives in the
canonical ``(batch_size, n, n_rhs)`` shape (``batch_size = 1`` encodes
unbatched — naming.md §2) with per-(batch item, right-hand-side column)
convergence masks; the matvec and transpose matvec run through the shared
:class:`~torchsparsegradutils.solvers._matvec.BatchedOperator` adapter, which
routes sparse operators through ``tsgu::spmm`` (transpose via the
descriptor's cached BatchedCSC) when the CUDA backend can take them.
"""

from typing import Callable, Optional, Tuple, Union

import torch

from ._matvec import BatchedOperator, as_batched_rhs, restore_rhs_shape


def _sym_ortho(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Elementwise stable Givens rotation: returns ``(c, s, r)`` with
    ``r = hypot(a, b)``, applied per (batch item, right-hand-side column)."""
    r = torch.hypot(a, b)
    return a / r, b / r, r


def _column_norm(t: torch.Tensor) -> torch.Tensor:
    """2-norm reduced over the n axis, batch and n_rhs axes retained:
    ``(batch_size, n, n_rhs) -> (batch_size, 1, n_rhs)`` (real-valued)."""
    return torch.linalg.vector_norm(t, dim=-2, keepdim=True)


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

    Multiple right-hand sides are solved simultaneously: all iterates are batched
    over the right-hand-side columns (and over a leading batch axis, if present)
    with per-(batch item, column) convergence masks. A column that satisfies its
    stopping test freezes — its solution no longer updates — while the remaining
    columns iterate on. There is no shared Krylov subspace, so each column's
    iteration is mathematically unchanged from a one-column solve.

    Parameters
    ----------
    A : {torch.Tensor, callable(x) -> A @ x}
        System matrix or matvec closure. A tensor may be dense (strided) or
        sparse (COO/CSR, 2D or batched 3D); sparse operators route through
        ``tsgu::spmm`` on an eligible CUDA backend and plain torch sparse
        matmul semantics otherwise. A callable receives operands shaped like
        ``b``: plain ``(n,)`` vectors when ``b`` is unbatched (one call per
        right-hand-side column), or the full ``(batch_size, n, n_rhs)``
        operand when ``b`` is batched.
    b : torch.Tensor, shape (m,), (m, n_rhs) or (batch_size, m, n_rhs)
        Right-hand side. Must be on the same device/dtype as ``A``. A
        ``(m, 1)`` (or otherwise all-but-one-axes-singleton) input is squeezed
        to a vector, preserving the legacy contract.
    Armat : {torch.Tensor, callable(x) -> A^T @ x}, optional
        Transpose matvec or matrix. If ``A`` is a tensor and ``Armat`` is
        ``None``, the transpose matvec is derived from ``A`` (dense: adjoint
        matmul; sparse: the descriptor's cached BatchedCSC). If ``A`` is
        callable, ``Armat`` is **required**.
    n : int, optional
        Number of columns of ``A``. Required if ``A`` is callable; inferred
        from ``A`` if it is a tensor.
    damp : float, optional
        Tikhonov damping parameter (ridge). Solves
        :math:`\min_x \|(A; \text{damp} I) x - (b; 0)\|_2`. Default: 0.0.
    atol : float, optional
        Absolute convergence tolerance. Default: 1e-6.
    btol : float, optional
        Relative residual tolerance. Default: 1e-6.
    conlim : float, optional
        Condition estimate limit; a column stops if its estimate exceeds this
        value. Default: 1e8.
    maxiter : int, optional
        Maximum iterations. If ``None``, uses ``min(m, n)``.
    x0 : torch.Tensor, optional, shape (n,), (n, n_rhs) or (batch_size, n, n_rhs)
        Initial guess. If ``None``, zeros are used.
    check_nonzero : bool, optional
        When ``False``, skip the guarded (masked) handling of the rare
        ``beta == 0`` bidiagonalization breakdown and divide unconditionally,
        matching the legacy fast path (use with caution). Default: True.

    Returns
    -------
    x : torch.Tensor, shape (n,), (n, n_rhs) or (batch_size, n, n_rhs) matching ``b``
        Approximate solution that minimizes :math:`\|A x - b\|_2` per column
        (with damped variant when ``damp > 0``).
    iterations : int
        Number of iterations executed (until every column stopped or
        ``maxiter`` was reached).

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

    Convergence checks (roughly, applied per (batch item, column)):

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

    Multiple right-hand sides, solved simultaneously:

    >>> B = torch.randn(3, 4)
    >>> X, it = lsmr(A, B)
    >>> X.shape
    torch.Size([2, 4])

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
    # torch.adjoint of a CSR (CSC) matrix is a CSC (CSR) view; re-express CSC
    # as CSR so the operator adapter's descriptor path accepts it.
    if torch.is_tensor(A) and A.layout == torch.sparse_csc:
        A = A.to_sparse_csr()
    if torch.is_tensor(Armat) and Armat.layout == torch.sparse_csc:
        Armat = Armat.to_sparse_csr()

    b = torch.atleast_1d(b)
    if b.ndim > 1 and b.squeeze().ndim <= 1:
        # Legacy contract: an (m, 1)-shaped right-hand side collapses to (m,).
        b = b.squeeze()
    rhs, was_vector, was_batched = as_batched_rhs(b)
    batch_size, m, n_rhs = rhs.shape
    # Legacy vector closures receive plain (n,) operands; batched right-hand
    # sides hand the closure the full canonical operand.
    operand_ndim = 3 if was_batched else 1

    matvec: Callable[[torch.Tensor], torch.Tensor]
    rmatvec: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    if torch.is_tensor(A):
        A_op = BatchedOperator(A)
        if n is None:
            n = A_op.n_cols
        matvec = A_op.matvec
        if Armat is None:
            # Dense: adjoint matmul (identical to the legacy torch.adjoint
            # path); sparse: transpose matvec via the descriptor's cached
            # BatchedCSC (architecture.md §3).
            rmatvec = A_op.rmatvec
    elif not callable(A):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    if n is None:
        raise RuntimeError("n needs to be provided or computed from A given as a tensor")

    if not torch.is_tensor(A):
        matvec = BatchedOperator(A, callable_operand_ndim=operand_ndim, n_rows=m, n_cols=n).matvec

    if rmatvec is None:
        if torch.is_tensor(Armat):
            rmatvec = BatchedOperator(Armat).matvec
        elif callable(Armat):
            rmatvec = BatchedOperator(Armat, callable_operand_ndim=operand_ndim, n_rows=n, n_cols=m).matvec
        else:
            raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    sdtype = rhs.dtype
    if sdtype == torch.complex64:
        sdtype = torch.float32
    elif sdtype == torch.complex128:
        sdtype = torch.float64

    eps = torch.finfo(sdtype).eps
    damp = torch.as_tensor(damp, dtype=sdtype, device=rhs.device)
    ctol = 1 / conlim if conlim > 0 else 0.0
    if maxiter is None:
        maxiter = min(m, n)

    u = rhs.clone()
    normb = _column_norm(rhs)
    if x0 is None:
        x = rhs.new_zeros(batch_size, n, n_rhs)
        beta = normb.clone()
    else:
        x_init, _, _ = as_batched_rhs(torch.atleast_1d(x0))
        if x_init.shape != (batch_size, n, n_rhs):
            x_init = x_init.expand(batch_size, n, n_rhs)
        x = x_init.clone()
        u = u - matvec(x)
        beta = _column_norm(u)

    # Guarded division per column: a zero beta means that column of u is the
    # zero vector, so rmatvec already yields zero for it.
    u = torch.where(beta > 0, u / beta, u)
    v = rmatvec(u)
    alpha = _column_norm(v)
    v = torch.where(alpha > 0, v / alpha, v)

    # Initialize variables for 1st iteration — every scalar iterate of the
    # single-column algorithm becomes a (batch_size, 1, n_rhs) tensor.

    one = torch.ones((batch_size, 1, n_rhs), dtype=sdtype, device=rhs.device)
    zero = torch.zeros_like(one)

    zetabar = alpha * beta
    alphabar = alpha.clone()
    rho = one.clone()
    rhobar = one.clone()
    cbar = one.clone()
    sbar = zero.clone()

    h = v.clone()
    hbar = torch.zeros_like(x)

    # Initialize variables for estimation of ||r||.

    betadd = beta.clone()
    betad = zero.clone()
    rhodold = one.clone()
    tautildeold = zero.clone()
    thetatilde = zero.clone()
    zeta = zero.clone()
    d = zero.clone()

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha.square()
    maxrbar = zero.clone()
    minrbar = torch.full_like(one, 0.99 * torch.finfo(sdtype).max)

    normr = beta.clone()
    normar = alpha * beta

    # Per-column early exits (legacy scalar checks, applied per column and in
    # the legacy order: a zero normar freezes x as-is; a zero normb with
    # nonzero normar zeroes that column of x).
    stopped = (normar == 0) | (normb == 0)
    x = torch.where((normb == 0) & (normar != 0), torch.zeros_like(x), x)
    if bool(stopped.all()):
        return restore_rhs_shape(x, was_vector, was_batched), 0

    # Main iteration loop.
    itn = 0
    for itn in range(1, maxiter + 1):
        # Columns whose stopping test held freeze: their x/h/hbar no longer
        # update, while their (cheap, elementwise) scalar iterates keep
        # evolving harmlessly.
        active = ~stopped

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.

        u = matvec(v) - alpha * u
        beta = _column_norm(u)

        if check_nonzero:
            # Per-column masked equivalent of the legacy `beta > 0` guard —
            # elementwise, so no GPU-CPU synchronization is incurred.
            mask = beta > 0
            u = torch.where(mask, u / beta, u)
            v_next = rmatvec(u) - beta * v
            alpha_next = _column_norm(v_next)
            v_next = torch.where(alpha_next > 0, v_next / alpha_next, v_next)
            v = torch.where(mask, v_next, v)
            alpha = torch.where(mask, alpha_next, alpha)
        else:
            # check_nonzero=False keeps the legacy unguarded divisions.
            u = u / beta
            v = rmatvec(u) - beta * v
            alpha = _column_norm(v)
            v = torch.where(alpha > 0, v / alpha, v)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, h_hat, x — masked so stopped columns freeze.

        hbar = torch.where(active, h - (thetabar * rho / (rhoold * rhobarold)) * hbar, hbar)
        x = torch.where(active, x + (zeta / (rho * rhobar)) * hbar, x)
        h = torch.where(active, v - (thetanew / rho) * h, h)

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = ctildeold * betahat - stildeold * betad

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck.square()
        normr = torch.sqrt(d + (betad - taud).square() + betadd.square())

        # Estimate ||A||.
        normA2 = normA2 + beta.square()
        normA = normA2.sqrt()
        normA2 = normA2 + alpha.square()

        # Estimate cond(A).
        maxrbar = torch.maximum(maxrbar, rhobarold)
        if itn > 1:
            minrbar = torch.minimum(minrbar, rhobarold)

        # ------- Test for convergence --------

        # Compute norms for convergence testing.
        normar = zetabar.abs()
        normx = _column_norm(x)
        condA = torch.maximum(maxrbar, rhotemp) / torch.minimum(minrbar, rhotemp)

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

        stop = (1 + test3 <= 1) | (1 + test2 <= 1) | (1 + t1 <= 1) | (test3 <= ctol) | (test2 <= atol) | (test1 <= rtol)

        stopped = stopped | stop
        if bool(stopped.all()):
            break

    return restore_rhs_shape(x, was_vector, was_batched), itn
