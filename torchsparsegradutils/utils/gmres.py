# Restarted GMRES ported from SciPy's scipy.sparse.linalg.gmres, see
# https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_isolve/iterative.py
# Modifications for torchsparsegradutils: several right-hand sides share each operator application but have their own
# Krylov space, convergence test and inner tolerance; classical Gram-Schmidt with reorthogonalisation is available
# (and the default); past Givens rotations are accumulated in a small orthogonal matrix so that each Arnoldi step only
# launches a fixed number of kernels.
#
# SciPy is distributed under the BSD-3-Clause license, reproduced below as its terms require:
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, overload

import torch

from .preconditioners import _matrix_operator


class GMRESSettings(NamedTuple):
    rtol: float = 1e-5  # Relative tolerance: stop once ||b - A x|| <= max(rtol * ||b||, atol) for every column
    atol: float = 0.0  # Absolute tolerance
    restart: int = 20  # Krylov subspace dimension between restarts, capped at n
    max_iter: int | None = None  # Maximum total number of Arnoldi iterations over all restart cycles (default 10 n).
    # It does not count the residual update ending each cycle nor the initial residual of an initial guess.
    orthogonalization: Literal["cgs2", "mgs"] = "cgs2"  # Classical Gram-Schmidt applied twice, or modified Gram-Schmidt
    check_every: int = 1  # Arnoldi iterations between host synchronisations testing for an early end of the cycle


@dataclass(frozen=True)
class GMRESInfo:
    """Convergence information returned by gmres.

    Tensors have shape (*batch_shape, num_rhs). Vector right-hand sides
    therefore produce length-one tensors.

    true_relative_residual is ``||b - A x||_2 / ||b||_2``, computed by the
    residual update that ends every restart cycle (0 for a zero right-hand
    side). recursive_relative_residual is the GMRES residual estimate at the
    end of the last cycle that updated each column (or the initial residual
    for a column that never iterated), relative to the norm of ``M^{-1} b``
    (``b`` without a preconditioner); it is measured on the
    left-preconditioned residual ``M^{-1} (b - A x)``, so it can differ from
    the true relative residual even in exact arithmetic.

    The reason is "converged" when every right-hand side meets the tolerance,
    "breakdown" when every unconverged right-hand side stopped on an Arnoldi
    breakdown (an invariant Krylov subspace that did not deliver a solution
    within tolerance, e.g. for a singular system) or a non-finite value, and
    "max_iter" when the iteration budget ran out first.
    """

    iterations: int
    restarts: int
    matvecs: int
    converged: torch.Tensor
    recursive_relative_residual: torch.Tensor
    true_relative_residual: torch.Tensor
    rtol: float
    atol: float
    reason: Literal["converged", "breakdown", "max_iter"]


def _as_operator(
    operator: torch.Tensor | Callable[[torch.Tensor], torch.Tensor], name: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(operator, torch.Tensor):
        return operator.matmul
    if callable(operator):
        return operator
    raise TypeError(f"{name} must be a tensor or a callable")


def _givens(f: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real Givens rotation (c, s, r) with ``c f + s g = r`` and ``-s f + c g = 0``, as LAPACK's lartg."""
    r = torch.hypot(f, g)
    nonzero = r.ne(0)
    safe_r = torch.where(nonzero, r, 1)
    c = torch.where(nonzero, f / safe_r, 1)
    s = torch.where(nonzero, g / safe_r, 0)
    return c, s, r


def _scaled_norm(z: torch.Tensor) -> torch.Tensor:
    """Euclidean norm over the last dimension, scaled by the largest entry so that squares do not under/overflow.

    The norm of a finite vector is accurate whenever it is representable (e.g. ``1e-25`` or ``1e20`` entries in
    float32), exact-zero vectors have a zero norm, and vectors with non-finite entries have a non-finite norm.
    """
    scale = z.abs().amax(dim=-1, keepdim=True)
    scale = scale.masked_fill(~(scale.gt(0) & torch.isfinite(scale)), 1)
    return torch.linalg.vector_norm(z / scale, dim=-1).mul_(scale.squeeze(-1))


def _krylov_workspace(
    batch: torch.Size, restart: int, n: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Krylov basis ``V`` and, for the Givens rotations, ``QT`` (transpose of their product) and ``R`` (rotated
    Hessenberg matrix) of one restart cycle, for every right-hand side."""
    V = torch.zeros(*batch, restart + 1, n, dtype=dtype, device=device)
    QT = torch.empty(*batch, restart + 1, restart + 1, dtype=dtype, device=device)
    R = torch.empty(*batch, restart, restart, dtype=dtype, device=device)
    return V, QT, R


def _project_on_full_basis(device: torch.device) -> bool:
    """Whether Gram-Schmidt projects on the full, zero-padded Krylov basis rather than on its filled rows.

    Both give the same coefficients since the rows that are not filled yet are zero. MPS runs batched products whose
    shape changes every iteration an order of magnitude slower, so it uses the fixed-size full basis.
    """
    return device.type == "mps"


@overload
def gmres(
    matmul_closure: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    initial_guess: torch.Tensor | None = None,
    preconditioner: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
    settings: GMRESSettings = GMRESSettings(),
    return_info: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def gmres(
    matmul_closure: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    initial_guess: torch.Tensor | None = None,
    preconditioner: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
    settings: GMRESSettings = GMRESSettings(),
    *,
    return_info: Literal[True],
) -> tuple[torch.Tensor, GMRESInfo]: ...


@overload
def gmres(
    matmul_closure: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    initial_guess: torch.Tensor | None = None,
    preconditioner: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
    settings: GMRESSettings = GMRESSettings(),
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, GMRESInfo]: ...


def gmres(  # noqa: C901 - the restarted Arnoldi recurrence is intentionally kept as one function
    matmul_closure: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    initial_guess: torch.Tensor | None = None,
    preconditioner: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
    settings: GMRESSettings = GMRESSettings(),
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, GMRESInfo]:
    r"""
    Solve general (non-symmetric) linear systems with the restarted GMRES method.

    Solves :math:`A x = b` for a real square, possibly non-symmetric and indefinite, matrix
    :math:`A` by minimising the residual norm over a Krylov subspace that is rebuilt every
    ``settings.restart`` iterations (GMRES(m) [1h]_). The algorithm is a port of SciPy's
    ``scipy.sparse.linalg.gmres``: Arnoldi process, least-squares problem solved with Givens
    rotations, left preconditioning, and SciPy's adaptive inner tolerance.

    Parameters
    ----------
    matmul_closure : {torch.Tensor, callable(X) -> A @ X}
        Operator. If a tensor is provided, its ``.matmul`` is used. A callable receives a
        tensor of shape ``(..., n, k)``, one Krylov vector per right-hand side.
    rhs : torch.Tensor, shape (n,), (n, k) or (..., n, k)
        Right-hand side(s). Each column is solved independently, with its own Krylov space
        and convergence test (no block GMRES), but all of them share each application of
        the operator.
    initial_guess : torch.Tensor, optional, shape like ``rhs``
        Initial guess. If ``None``, zero is used. Columns of a zero right-hand side are
        returned as zero, whatever their initial guess.
    preconditioner : {torch.Tensor, callable(X) -> M^{-1} X}, optional
        Left preconditioner approximating :math:`A^{-1}`, with the same calling convention
        as ``matmul_closure``. GMRES then minimises the preconditioned residual
        :math:`\lVert M^{-1}(b - A x) \rVert_2` within each cycle, but convergence is always
        tested on the true residual :math:`\lVert b - A x \rVert_2`.
    settings : GMRESSettings, optional
        Tolerances (``rtol``, ``atol``), restart length, iteration budget (``max_iter``
        counts Arnoldi iterations over all cycles, not the residual updates between
        cycles; ``None`` uses ``10 n``), Gram-Schmidt
        variant and host synchronisation frequency (``check_every``).
    return_info : bool, optional
        Also return a :class:`GMRESInfo`. The true residuals it reports come for free from
        the residual update that ends every restart cycle.

    Returns
    -------
    torch.Tensor or tuple
        Solution with the same shape as ``rhs``, or ``(solution, info)`` if ``return_info``.

    Raises
    ------
    ValueError
        If a setting is invalid, if ``rhs`` is neither ``float32`` nor ``float64``, or if ``initial_guess`` does not
        have the shape of ``rhs``.
    TypeError
        If ``matmul_closure`` or ``preconditioner`` is neither a tensor nor a callable.

    Warns
    -----
    UserWarning
        If ``return_info`` is false and some right-hand sides did not reach the tolerance.

    Notes
    -----
    **Convergence.** A column converges once
    :math:`\lVert b - A x \rVert_2 \le \max(\mathrm{rtol} \lVert b \rVert_2, \mathrm{atol})`,
    matching SciPy. Within a cycle, a column stops once its (preconditioned) residual
    estimate meets an inner tolerance that SciPy adapts after each cycle so that the
    estimate and the true residual agree (scipy/scipy#8400), or once its Arnoldi process
    breaks down. Columns that stopped no longer change, and the solve ends once all of them
    have converged or broken down, or after ``max_iter`` Arnoldi iterations. A cycle that
    breaks down has found an invariant subspace. If the true residual still misses the
    tolerance, SciPy stops. Here, the column restarts if the cycle at least halved its true
    residual, since a breakdown caused by rounding (typically once the Krylov space spans
    the whole space in single precision) leaves a residual that a restart, acting as
    iterative refinement, reduces further. Otherwise, as for an inconsistent singular
    system, the column stops.

    **Iteration budget.** Unlike SciPy, whose ``maxiter`` counts restart cycles (by default
    ``10 n`` cycles of ``restart`` iterations), ``max_iter`` bounds the total number of
    Arnoldi iterations, whatever the restart length. Each Arnoldi iteration applies the
    operator once (to all right-hand sides at once). The residual update that ends every
    restart cycle, and the initial residual of an ``initial_guess``, also apply it once but
    are not counted in ``max_iter``, so ``info.matvecs`` is
    ``iterations + restarts (+ 1 with an initial guess)``.

    **Orthogonalisation.** SciPy uses modified Gram-Schmidt (MGS, ``"mgs"``), which needs
    ``j + 1`` dependent dot products at Arnoldi step ``j``. The default, ``"cgs2"``,
    applies classical Gram-Schmidt twice: each pass is a pair of matrix-vector products
    with the basis, which maps to BLAS-2 kernels and is much faster, especially on GPUs.
    By the "twice is enough" result [2h]_, the second, unconditional pass keeps the basis
    orthogonal to working precision, which is more than MGS guarantees; it is not made
    conditional (Kahan's :math:`1/\sqrt{2}` test) to avoid a host synchronisation and
    per-column branching. JAX's ``gmres`` intends the conditional variant but, as written,
    stops after one pass, and CuPy's uses a single pass.

    **Least-squares problem.** As in SciPy, the small Hessenberg least-squares problem is
    solved with Givens rotations followed by a triangular solve, not with the normal
    equations (which JAX's default ``"batched"`` method uses and which square its condition
    number). Instead of applying the past rotations one after the other to each new
    Hessenberg column, their product is accumulated in a small orthogonal matrix applied
    with a single batched product, which also yields the residual estimate of every
    iteration for free.

    **GPU friendliness.** All bookkeeping is batched over the right-hand sides and stays
    on the device; converged columns are masked rather than removed. The host
    synchronises once per restart cycle and every ``check_every`` Arnoldi iterations, to
    end a cycle early once all columns have stopped. Raising ``check_every`` trades
    synchronisations for possibly wasted operator applications.

    Only ``float32`` and ``float64`` are supported. Memory use is :math:`O((\mathrm{restart} + 1)\, n\, k)`
    for the Krylov bases.

    See Also
    --------
    bicgstab : BiCGSTAB, a short-recurrence alternative for non-symmetric systems.
    minres : MINRES for symmetric (possibly indefinite) systems.
    linear_cg : Conjugate Gradient for symmetric positive definite systems.

    References
    ----------
    .. [1h] Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual
           algorithm for solving nonsymmetric linear systems. *SIAM J. Sci. Stat.
           Comput.*, 7(3), 856–869.
    .. [2h] Giraud, L., Langou, J., & Rozložník, M. (2005). The loss of orthogonality in
           the Gram-Schmidt orthogonalization process. *Computers & Mathematics with
           Applications*, 50(7), 1069–1075.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils.utils import gmres
    >>> A = torch.tensor([[3.0, 2.0, 0.0], [1.0, -1.0, 0.0], [0.0, 5.0, 1.0]], dtype=torch.float64)
    >>> b = torch.tensor([2.0, 4.0, -1.0], dtype=torch.float64)
    >>> x = gmres(A, b)
    >>> torch.allclose(A @ x, b)
    True

    Several right-hand sides, each with its own convergence test, and convergence information:

    >>> from torchsparsegradutils.utils import GMRESSettings
    >>> B = torch.randn(3, 4, dtype=torch.float64)
    >>> X, info = gmres(A, B, settings=GMRESSettings(rtol=1e-10), return_info=True)
    >>> X.shape, info.converged.shape, info.reason
    (torch.Size([3, 4]), torch.Size([4]), 'converged')

    Sparse operator with a Jacobi preconditioner:

    >>> A_sp = A.to_sparse_csr()
    >>> x = gmres(A_sp, b, preconditioner=lambda r: r / torch.diagonal(A).unsqueeze(-1))
    """
    rtol, atol, restart = settings.rtol, settings.atol, settings.restart
    if not (math.isfinite(rtol) and rtol >= 0 and math.isfinite(atol) and atol >= 0):
        raise ValueError("settings.rtol and settings.atol must be finite and nonnegative")
    if restart < 1:
        raise ValueError("settings.restart must be at least 1")
    if settings.max_iter is not None and settings.max_iter < 0:
        raise ValueError("settings.max_iter must be nonnegative or None")
    if settings.orthogonalization not in ("cgs2", "mgs"):
        raise ValueError("settings.orthogonalization must be 'cgs2' or 'mgs'")
    if settings.check_every < 1:
        raise ValueError("settings.check_every must be at least 1")
    if rhs.dtype not in (torch.float32, torch.float64):
        # Half precision is rejected too: the triangular solve has no float16/bfloat16 kernels on CPU or MPS, and
        # an epsilon around 1e-3 would make the breakdown test and the orthogonalisation unreliable anyway
        raise ValueError(f"gmres only supports float32 and float64, got {rhs.dtype}")
    if initial_guess is not None and initial_guess.shape != rhs.shape:
        raise ValueError(f"initial_guess has shape {tuple(initial_guess.shape)}, expected {tuple(rhs.shape)}")
    op = _as_operator(matmul_closure, "matmul_closure")
    if isinstance(preconditioner, torch.Tensor):
        preconditioner = _matrix_operator(preconditioner)
    precon = None if preconditioner is None else _as_operator(preconditioner, "preconditioner")

    squeeze = rhs.dim() == 1
    if squeeze:
        rhs = rhs.unsqueeze(-1)
        if initial_guess is not None:
            initial_guess = initial_guess.unsqueeze(-1)

    # Work with one row per right-hand side, shape (*batch, k, n): Krylov bases are then (*batch, k, m + 1, n) and
    # Gram-Schmidt uses batched matrix-vector products. Operators see the usual (*batch, n, k) layout.
    def apply_op(z: torch.Tensor) -> torch.Tensor:
        return op(z.mT).mT

    def apply_precon(z: torch.Tensor) -> torch.Tensor:
        return z if precon is None else precon(z.mT).mT

    b = rhs.mT
    n = b.shape[-1]
    dtype, device = b.dtype, b.device
    eps = torch.finfo(dtype).eps
    max_iter = 10 * n if settings.max_iter is None else settings.max_iter
    # A cycle never needs more Arnoldi vectors than the problem size or the iteration budget
    restart = min(restart, n, max_iter)
    norm = _scaled_norm

    b_is_zero = b.eq(0).all(dim=-1)
    b_norm = norm(b)
    threshold = torch.clamp_min(rtol * b_norm, atol)

    # Initial residual. SciPy returns zero for a zero right-hand side.
    matvecs = 0
    if initial_guess is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = initial_guess.mT.clone().masked_fill_(b_is_zero.unsqueeze(-1), 0)
        r = b - apply_op(x)
        matvecs += 1
    r_norm = norm(r).masked_fill_(b_is_zero, 0)

    # A non-finite residual never converges, even against an infinite threshold (e.g. an infinite rhs)
    finite_r_norm = torch.isfinite(r_norm)
    converged = finite_r_norm & r_norm.le(threshold)
    broken = ~finite_r_norm
    running = ~converged & ~broken

    # Tolerance on the preconditioned residual estimate for the inner iterations (scipy/scipy#8400)
    Mb_norm = b_norm if precon is None else norm(apply_precon(b))
    safe_b_norm = b_norm.masked_fill(b_is_zero, 1)
    ptol_max_factor = torch.ones_like(b_norm)
    ptol = Mb_norm * torch.clamp_max(threshold / safe_b_norm, 1)
    # Preconditioned residual estimate, starting from the initial residual so that it is meaningful for columns that
    # never iterate (e.g. max_iter=0, or an initial guess that already converged)
    if precon is None:
        presid = r_norm.clone()
    elif initial_guess is None:
        presid = Mb_norm.masked_fill(b_is_zero, 0)
    else:
        presid = norm(apply_precon(r)).masked_fill_(b_is_zero, 0)
    safe_Mb_norm = Mb_norm.masked_fill(Mb_norm.eq(0), 1)

    batch = b.shape[:-1]
    if not bool(running.any()):
        # Zero budget, zero right-hand sides or an initial guess that already converged: nothing iterates, so do not
        # allocate a restart-sized Krylov workspace for nothing (the loop below does not run)
        restart = 0
    V, QT, R = _krylov_workspace(batch, restart, n, dtype, device)
    eye = torch.eye(restart, dtype=dtype, device=device)
    steps = torch.arange(restart, device=device)
    full_basis = _project_on_full_basis(device)

    iterations = 0
    restarts = 0
    while iterations < max_iter and bool(running.any()):
        cycle_length = min(restart, max_iter - iterations)
        restarts += 1

        v0 = apply_precon(r)
        beta = norm(v0)
        # A preconditioner that maps a nonzero residual to zero leaves nothing to iterate on
        unusable = running & ~(beta.gt(0) & torch.isfinite(beta))
        broken = broken | unusable
        running = running & ~unusable
        V.zero_()
        V[..., 0, :] = torch.where(running.unsqueeze(-1), v0 / beta.masked_fill(~running, 1).unsqueeze(-1), 0)
        QT.copy_(torch.eye(restart + 1, dtype=dtype, device=device))
        R.zero_()

        # Columns still iterating in this cycle, their number of Arnoldi steps, and whether they broke down
        active = running.clone()
        length = torch.zeros(active.shape, dtype=torch.long, device=device)
        cycle_breakdown = torch.zeros_like(active)

        j = 0
        for j in range(cycle_length):
            w = apply_precon(apply_op(V[..., j, :]))
            matvecs += 1
            iterations += 1
            h0 = norm(w)

            # Rows of V past j are zero, so projecting on all of them gives the same coefficients (see
            # _project_on_full_basis)
            basis = V if full_basis else V[..., : j + 1, :]
            h_col = torch.zeros(*batch, restart + 1, dtype=dtype, device=device)
            if settings.orthogonalization == "cgs2":
                h = (basis @ w.unsqueeze(-1)).squeeze(-1)
                w = w - (basis.mT @ h.unsqueeze(-1)).squeeze(-1)
                h2 = (basis @ w.unsqueeze(-1)).squeeze(-1)
                w = w - (basis.mT @ h2.unsqueeze(-1)).squeeze(-1)
                h_col[..., : j + 1] = (h + h2)[..., : j + 1]
            else:
                for i in range(j + 1):
                    hi = (V[..., i, :] * w).sum(dim=-1)
                    w = w - hi.unsqueeze(-1) * V[..., i, :]
                    h_col[..., i] = hi

            h1 = norm(w)
            # Exact solution indicator (lucky breakdown); frozen columns have zero vectors and also land here
            breakdown = h1.le(eps * h0)
            proceed = active & ~breakdown
            V[..., j + 1, :] = torch.where(proceed.unsqueeze(-1), w / h1.masked_fill(~proceed, 1).unsqueeze(-1), 0)
            h_col[..., j + 1] = h1.masked_fill(breakdown, 0)
            h_col = h_col.masked_fill(~active.unsqueeze(-1), 0)

            # Apply all previous rotations at once (on the full, fixed-size matrix), then the new one
            h_rot = (QT @ h_col.unsqueeze(-1)).squeeze(-1)
            c, s, rho = _givens(h_rot[..., j], h_rot[..., j + 1])
            R[..., :j, j] = h_rot[..., :j]
            R[..., j, j] = rho
            row_j = QT[..., j, : j + 2].clone()
            row_j1 = QT[..., j + 1, : j + 2]
            QT[..., j, : j + 2] = c.unsqueeze(-1) * row_j + s.unsqueeze(-1) * row_j1
            QT[..., j + 1, : j + 2] = c.unsqueeze(-1) * row_j1 - s.unsqueeze(-1) * row_j

            # Residual estimate of the (preconditioned) least-squares problem
            presid_j = beta * QT[..., j + 1, 0].abs()
            presid = torch.where(active, presid_j, presid)
            length = length + active.long()
            cycle_breakdown = cycle_breakdown | (active & (breakdown | ~torch.isfinite(presid_j)))
            active = active & ~cycle_breakdown & ~presid_j.le(ptol)

            if (j + 1) % settings.check_every == 0 and not bool(active.any()):
                break
        steps_done = j + 1

        # Solve the triangular system of each column over its own number of steps. Steps past that number become
        # identity rows with a zero right-hand side, and a zero pivot (singular A) gives a zero coefficient as in SciPy.
        within = steps[:steps_done] < length.unsqueeze(-1)
        R_cycle = R[..., :steps_done, :steps_done]
        within_2d = within.unsqueeze(-1) & within.unsqueeze(-2)
        R_cycle = torch.where(within_2d, R_cycle, eye[:steps_done, :steps_done])
        zero_pivot = within & torch.diagonal(R_cycle, dim1=-2, dim2=-1).eq(0)
        R_cycle = R_cycle + torch.diag_embed(zero_pivot.to(dtype))
        g = beta.unsqueeze(-1) * QT[..., :steps_done, 0]
        g = g.masked_fill(~within | zero_pivot, 0)
        y = torch.linalg.solve_triangular(R_cycle, g.unsqueeze(-1), upper=True)
        dx = (V[..., :steps_done, :].mT @ y).squeeze(-1)

        finite_update = torch.isfinite(dx).all(dim=-1)
        cycle_breakdown = cycle_breakdown | (running & ~finite_update)
        x = x + torch.where((running & finite_update).unsqueeze(-1), dx, 0)

        r = b - apply_op(x)
        matvecs += 1
        previous_r_norm = r_norm
        r_norm = torch.where(running, norm(r), r_norm)

        newly_converged = running & torch.isfinite(r_norm) & r_norm.le(threshold)
        # A breakdown found an invariant subspace. If the true residual still misses the tolerance, SciPy stops. A
        # breakdown in finite precision (e.g. once the Krylov space spans everything) can however leave a residual
        # that a restart, acting as iterative refinement, reduces further: only stop when the cycle did not at least
        # halve the true residual, as for an inconsistent singular system.
        stalled = cycle_breakdown & ~r_norm.lt(0.5 * previous_r_norm)
        newly_broken = running & ~newly_converged & (stalled | ~torch.isfinite(r_norm))
        converged = converged | newly_converged
        broken = broken | newly_broken
        inner_passed = presid.le(ptol)
        ptol_max_factor = torch.where(
            inner_passed, torch.clamp_min(0.25 * ptol_max_factor, eps), torch.clamp_max(1.5 * ptol_max_factor, 1)
        )
        ptol = presid * torch.minimum(ptol_max_factor, threshold / r_norm)
        running = running & ~newly_converged & ~newly_broken

    true_relative_residual = (r_norm / safe_b_norm).masked_fill_(b_is_zero, 0)
    if bool(converged.all()):
        reason = "converged"
    elif bool(running.any()):
        reason = "max_iter"
    else:
        reason = "breakdown"

    if not return_info and reason != "converged":
        warnings.warn(
            f"GMRES terminated after {iterations} iterations ({reason}) with maximum true relative residual "
            f"{true_relative_residual.max().item():.3e}: {(~converged).sum().item()} of {converged.numel()} "
            f"right-hand sides did not reach the tolerance (rtol={rtol}, atol={atol}).",
            UserWarning,
            stacklevel=2,
        )

    solution = x.mT
    if squeeze:
        solution = solution.squeeze(-1)
    if not return_info:
        return solution
    info = GMRESInfo(
        iterations=iterations,
        restarts=restarts,
        matvecs=matvecs,
        converged=converged,
        recursive_relative_residual=(presid / safe_Mb_norm).masked_fill_(b_is_zero, 0),
        true_relative_residual=true_relative_residual,
        rtol=rtol,
        atol=atol,
        reason=reason,
    )
    return solution, info
