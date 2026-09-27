# MIT-licensed code imported from https://github.com/cornellius-gp/linear_operator
# Minor modifications for torchsparsegradutils to remove dependencies

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch


class MINRESSettings(NamedTuple):
    max_cg_iterations: int | None = None  # Deprecated alias of max_minres_iterations, kept in first position for
    # positional construction. When set, it takes precedence over max_minres_iterations and a DeprecationWarning is
    # emitted.
    minres_tolerance: float = 1e-4  # Relative tolerance used for terminating MINRES (see minres_convergence).
    verbose_linalg: bool = False  # Print out information whenever running an expensive linear algebra routine
    minres_convergence: Literal["residual", "update"] = "residual"  # "residual" stops once every right-hand side
    # has a relative (preconditioned) residual estimate below minres_tolerance. "update" retains the historical
    # criterion on the mean relative norm of the solution update.
    minres_check_every: int = 10  # Iterations between termination checks, which synchronise with the host, and
    # between true residual checks past n iterations. 1 checks every iteration.
    max_minres_iterations: int = 1000  # The maximum number of MINRES iterations to perform (when computing
    # matrix solves). A higher value rarely results in more accurate solves -- instead, lower the MINRES tolerance.


@dataclass(frozen=True)
class MINRESInfo:
    """Convergence information returned by minres.

    Residual tensors have shape (*batch_shape, num_rhs), with a leading
    num_shifts dimension when several shifts are solved at once. Vector
    right-hand sides therefore produce a length-one residual tensor.

    recursive_relative_residual is the MINRES residual estimate
    ``phi_k / phi_0``, measured in the ``M^{-1}``-norm when a preconditioner
    ``M^{-1}`` is used, and true_relative_residual is the recomputed
    ``||b - (value * A + shift * I) x||_2 / ||b||_2``.

    The reason is "converged" when the recomputed true residual meets
    tolerance, "recursive_converged" or "update_converged" when the internal
    stopping criterion is met but the true residual is not, "breakdown" when
    every unconverged right-hand side hit a Lanczos breakdown or a non-finite
    residual estimate, "stagnated" when, past n iterations, every unconverged
    right-hand side stopped because its true residual no longer improved or
    diverged from the residual estimate, and "max_iter" when the iteration
    limit is reached.
    """

    iterations: int
    matvecs: int
    converged: torch.Tensor
    recursive_relative_residual: torch.Tensor
    true_relative_residual: torch.Tensor
    tolerance: float
    reason: Literal["converged", "recursive_converged", "update_converged", "breakdown", "stagnated", "max_iter"]


def _pad_with_singletons(obj, num_singletons_before=0, num_singletons_after=0):
    """
    Pad obj with singleton dimensions on the left and right
    Example:
        >>> x = torch.randn(10, 5)
        >>> _pad_with_singletons(x, 2, 3).shape
        torch.Size([1, 1, 10, 5, 1, 1, 1])
    """
    new_shape = [1] * num_singletons_before + list(obj.shape) + [1] * num_singletons_after
    return obj.view(*new_shape)


def minres(  # noqa: C901 - inherited solver is intentionally kept as one recurrence
    matmul_closure: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    eps: float = 1e-25,
    shifts: torch.Tensor | None = None,
    value: float | None = None,
    max_iter: int | None = None,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    settings: MINRESSettings = MINRESSettings(),
    tolerance: float | None = None,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, MINRESInfo]:
    """
    Minimum Residual (MINRES) solver for symmetric (Hermitian) linear systems.

    Solves linear systems ``A x = b`` where ``A`` is symmetric (Hermitian) and may
    be indefinite. Supports single/multiple right-hand sides and (optionally)
    multiple shift values to solve ``(A + \\sigma I) x = b`` in one run. Gradually
    minimizes the residual norm ``||A x - b||_2`` via the Lanczos process.

    Parameters
    ----------
    matmul_closure : {torch.Tensor, callable(x) -> A @ x}
        Matrix–vector multiplication operator. If a tensor is provided, its
        ``.matmul`` is used. The operator should represent a symmetric/Hermitian
        matrix for MINRES to behave as intended.
    rhs : torch.Tensor, shape (..., n) or (..., n, k)
        Right-hand side vector(s). Leading batch dimensions are supported; for
        multi-RHS, the last two dims are ``(n, k)``.
    eps : float, optional
        Small constant to prevent division by zero/numerical issues. A Lanczos
        coefficient below ``eps`` is treated as a breakdown: the affected
        right-hand side stops updating. Default: 1e-25.
    shifts : torch.Tensor or scalar, optional
        Shift(s) ``\\sigma`` for solving ``(A + \\sigma I) x = b``. If ``None`` or a
        scalar, a single system is solved. If a tensor with ``s`` elements, the
        solver computes ``s`` shifted systems and stacks their solutions along a
        new leading dimension.
    value : float, optional
        Scalar multiplier ``\\alpha`` applied to the operator (solves ``(\\alpha A) x = b``)
        when provided. Default: ``None`` (no scaling).
    max_iter : int, optional
        Maximum iterations. If ``None``, uses
        ``min(settings.max_minres_iterations, n + 1)`` where ``n`` is the problem size.
        An explicit value is used as given: in finite precision, loss of Lanczos
        orthogonality can require more than ``n`` iterations. Past ``n``
        iterations, the iterate can also drift away from the solution while the
        residual estimate keeps decreasing, so every
        ``settings.minres_check_every`` iterations the true residual is
        recomputed (one operator application per shift) and the best iterate is
        kept. A right-hand side then stops once its true residual meets the
        tolerance, once its residual estimate meets the tolerance while its true
        residual does not, or after 3 checks without improvement. Going past
        ``n`` is only useful while the true residual is still decreasing. With a
        preconditioner and nonzero shifts, this true residual is unavailable, so
        ``max_iter`` is capped at ``n + 1`` with a warning.
    preconditioner : callable, optional
        Symmetric positive definite preconditioner with signature
        ``preconditioner(x) -> M^{-1} x``. If ``None``, no preconditioning is used.
        A positive semi-definite ``M^{-1}`` is acceptable when its null space only
        covers rows where both ``A`` and ``b`` vanish, e.g. a Jacobi preconditioner
        that is zero on the empty rows of a singular graph Laplacian.
    settings : MINRESSettings, optional
        Configuration object controlling iteration caps, the tolerance, the
        convergence criterion (``minres_convergence``) and how often termination
        is checked (``minres_check_every``).
    tolerance : float, optional
        Finite, nonnegative relative tolerance. If ``None``, uses
        ``settings.minres_tolerance``. With the default ``"residual"`` criterion,
        each right-hand side (and shift) stops updating once its residual estimate
        ``phi_k / phi_0`` is at most ``tolerance``, and the solve terminates once all
        of them have. ``phi_k`` is the residual norm estimate provided by the MINRES
        recurrence at no extra cost; it is measured in the ``M^{-1}``-norm when a
        preconditioner is used. The tolerance then does *not* bound
        ``||b - A x||_2 / ||b||_2``, which can differ from the ``M^{-1}``-norm
        relative residual by up to a factor ``sqrt(cond(M))`` either way. With the
        ``"update"`` criterion, the solve terminates when the mean relative norm of
        the latest solution update is below ``tolerance``. Termination is checked
        every ``settings.minres_check_every`` iterations.
    return_info : bool, optional
        Return a ``MINRESInfo`` with iterations, matvecs, per-right-hand-side
        convergence, recursive and recomputed true relative residuals, and the
        termination reason. Recomputing the true residual costs one extra operator
        application per shift. Not supported together with a preconditioner and
        nonzero shifts, since the preconditioned recurrence then solves
        ``(A + \\sigma M) x = b``.

    Returns
    -------
    torch.Tensor or tuple
        If ``shifts`` is ``None`` or a scalar: solution with the **same shape as**
        ``rhs`` (i.e., ``(..., n)`` or ``(..., n, k)``).
        If ``shifts`` has length ``s``: a stacked tensor of shape
        ``(s, *rhs.shape)`` containing solutions for each shift.
        If ``return_info`` is true, ``(solution, info)`` is returned instead.

    Raises
    ------
    ValueError
        If ``tolerance``, ``max_iter``, ``settings.minres_convergence`` or
        ``settings.minres_check_every`` is invalid, or if ``return_info`` is
        combined with a preconditioner and nonzero shifts.

    Warns
    -----
    UserWarning
        If an explicit ``max_iter`` above ``n + 1`` is capped because a
        preconditioner is combined with nonzero shifts.
    UserWarning
        With the ``"residual"`` criterion and ``return_info=False``, if some
        right-hand sides did not reach the tolerance. After more than ``n``
        iterations, this is decided on the recomputed true residual rather than
        on the residual estimate.

    Notes
    -----
    - MINRES [1g]_ is appropriate for symmetric/Hermitian **indefinite** systems; it
      minimizes the Euclidean residual norm rather than the A-norm (as in CG).
    - For symmetric positive definite systems, Conjugate Gradient (CG) typically
      converges faster; prefer CG unless indefiniteness/robustness suggests MINRES.
    - For singular systems with a consistent right-hand side, MINRES started from
      zero converges to a least-squares solution, which is the minimum-norm
      solution in exact arithmetic. It is not guaranteed to return the
      minimum-norm solution when rounding makes the system slightly inconsistent.
    - When multiple shifts are provided, the solver reuses Lanczos information and
      returns one solution per shift value.
    - All inputs should share device and dtype; the implementation normalizes
      ``rhs`` internally and rescales the final solution(s).

    See Also
    --------
    linear_cg : Conjugate Gradient for SPD systems.
    bicgstab : BiCGSTAB for general non-symmetric systems.

    References
    ----------
    .. [1g] Paige, C. C., & Saunders, M. A. (1975). Solution of sparse indefinite
           systems of linear equations. *SIAM Journal on Numerical Analysis*, 12(4), 617–629.

    Examples
    --------
    Basic solve (indefinite, symmetric):

    >>> A = torch.tensor([[2.0, 1.0], [1.0, -1.0]])
    >>> b = torch.tensor([1.0, 2.0])
    >>> x = minres(A.matmul, b)
    >>> x.shape
    torch.Size([2])

    Multiple right-hand sides:

    >>> B = torch.randn(2, 3)
    >>> X = minres(A.matmul, B)
    >>> X.shape
    torch.Size([2, 3])

    Shifted system (regularization):

    >>> x_shifted = minres(A.matmul, b, shifts=torch.tensor(0.1))

    Sparse operator via closure:

    >>> idx = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> val = torch.tensor([2.0, 1.0, 1.0, -1.0])
    >>> A_sp = torch.sparse_coo_tensor(idx, val, (2, 2))
    >>> x = minres(lambda v: A_sp @ v, b)

    With a simple diagonal preconditioner:

    >>> M_diag = torch.abs(torch.diag(A)) + 0.1
    >>> precond = lambda x: x / M_diag.unsqueeze(-1)
    >>> x = minres(A.matmul, b, preconditioner=precond)

    Custom iteration cap/tolerance:

    >>> settings = MINRESSettings(max_minres_iterations=200, minres_tolerance=1e-5)
    >>> x = minres(A.matmul, b, settings=settings)

    Convergence information:

    >>> x, info = minres(A.matmul, b, tolerance=1e-6, return_info=True)
    >>> info.reason
    'converged'
    """
    # Default values
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    mm_ = matmul_closure
    has_preconditioner = preconditioner is not None
    if preconditioner is None:
        preconditioner = lambda x: x.clone()

    if shifts is None:
        shifts = torch.tensor(0.0, dtype=rhs.dtype, device=rhs.device)

    if tolerance is None:
        tolerance = settings.minres_tolerance
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    convergence = settings.minres_convergence
    if convergence not in ("residual", "update"):
        raise ValueError("settings.minres_convergence must be 'residual' or 'update'")
    max_minres_iterations = settings.max_minres_iterations
    if settings.max_cg_iterations is not None:
        warnings.warn(
            "MINRESSettings.max_cg_iterations is deprecated, use MINRESSettings.max_minres_iterations instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        max_minres_iterations = settings.max_cg_iterations
    check_every = settings.minres_check_every
    if check_every < 1:
        raise ValueError("settings.minres_check_every must be at least 1")
    if max_iter is not None and max_iter < 0:
        raise ValueError("max_iter must be nonnegative")
    # With a preconditioner, the shifted recurrence solves (A + sigma M) x = b: its true residual is unavailable
    shifted_preconditioning = has_preconditioner and bool(torch.as_tensor(shifts).ne(0).any())
    if return_info and shifted_preconditioning:
        raise ValueError("return_info is not supported with a preconditioner and nonzero shifts")

    # Scale the rhs
    squeeze = False
    if rhs.dim() == 1:
        rhs = rhs.unsqueeze(-1)
        squeeze = True
    rhs_original = rhs
    size = rhs.size(-2)

    # Use the right number of iterations
    if max_iter is None:
        max_iter = min(max_minres_iterations, size + 1)
    elif shifted_preconditioning and max_iter > size + 1:
        # Past n iterations the true residual guards against drift, but (A + sigma M) x cannot be checked
        # from M^{-1} alone: do not run unguarded past n
        warnings.warn(
            f"max_iter={max_iter} is capped at n + 1 = {size + 1} for MINRES with a preconditioner and nonzero "
            "shifts, since the true residual needed to guard iterations past n cannot be computed.",
            UserWarning,
            stacklevel=2,
        )
        max_iter = size + 1

    rhs_norm = torch.linalg.vector_norm(rhs, ord=2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.eq(0)
    rhs_norm = rhs_norm.masked_fill(rhs_is_zero, 1)
    rhs = rhs.div(rhs_norm)

    # Epsilon (to prevent nans)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    # Create space for matmul product, solution
    prod = mm_(rhs)
    if value is not None:
        prod.mul_(value)
    matvecs = 1

    # Resize shifts
    shifts = _pad_with_singletons(shifts, 0, prod.dim() - shifts.dim() + 1)
    solution = torch.zeros(shifts.shape[:1] + prod.shape, dtype=rhs.dtype, device=rhs.device)

    # Variables for Lanczos terms
    zvec_prev2 = torch.zeros_like(prod)
    zvec_prev1 = rhs.clone().expand_as(prod).contiguous()
    qvec_prev1 = preconditioner(zvec_prev1)
    alpha_curr = torch.empty(prod.shape[:-2] + (1, prod.size(-1)), dtype=rhs.dtype, device=rhs.device)
    alpha_shifted_curr = torch.empty(solution.shape[:-2] + (1, prod.size(-1)), dtype=rhs.dtype, device=rhs.device)
    beta_prev = (zvec_prev1 * qvec_prev1).sum(dim=-2, keepdim=True).sqrt_()
    beta_curr = torch.empty_like(beta_prev)
    tmpvec = torch.empty_like(qvec_prev1)

    # Zero right-hand sides have nothing to solve: keep their Lanczos vectors at zero
    lanczos_is_zero = beta_prev.eq(0)
    beta_prev.masked_fill_(lanczos_is_zero, 1)
    # Initial (preconditioned) residual norm, phi_0
    beta_initial = beta_prev.clone()

    # Divide by beta_prev
    zvec_prev1.div_(beta_prev)
    qvec_prev1.div_(beta_prev)

    # Variables for the QR rotation
    # 1) Components of the Givens rotations
    cos_prev2 = torch.ones(solution.shape[:-2] + (1, rhs.size(-1)), dtype=rhs.dtype, device=rhs.device)
    sin_prev2 = torch.zeros(solution.shape[:-2] + (1, rhs.size(-1)), dtype=rhs.dtype, device=rhs.device)
    cos_prev1 = torch.ones_like(cos_prev2)
    sin_prev1 = torch.zeros_like(sin_prev2)
    radius_curr = torch.empty_like(cos_prev1)
    cos_curr = torch.empty_like(cos_prev1)
    sin_curr = torch.empty_like(cos_prev1)
    # 2) Terms QR decomposition of T
    subsub_diag_term = torch.empty_like(alpha_shifted_curr)
    sub_diag_term = torch.empty_like(alpha_shifted_curr)
    diag_term = torch.empty_like(alpha_shifted_curr)

    # Variables for the solution updates
    # 1) The "search" vectors of the solution
    # Equivalent to the vectors of Q R^{-1}, where Q is the matrix of Lanczos vectors and
    # R is the QR factor of the tridiagonal Lanczos matrix.
    search_prev2 = torch.zeros_like(solution)
    search_prev1 = torch.zeros_like(solution)
    search_curr = torch.empty_like(search_prev1)
    search_update = torch.empty_like(search_prev1)
    # 2) The "scaling" terms of the search vectors
    # Equivalent to the terms of V^T Q^T rhs, where Q is the matrix of Lanczos vectors and
    # V is the QR orthonormal of the tridiagonal Lanczos matrix.
    # The magnitude of the latest scaling term is the residual norm estimate phi_k.
    scale_prev = beta_prev.repeat(shifts.size(0), *([1] * beta_prev.dim()))
    scale_curr = torch.empty_like(scale_prev)

    # Terms for checking for convergence
    solution_norm = torch.zeros(*solution.shape[:-2], solution.size(-1), dtype=solution.dtype, device=solution.device)
    search_update_norm = torch.zeros_like(solution_norm)
    # Per shift and right-hand side: relative residual estimate, whether the column is still updated,
    # and whether it stopped because of a Lanczos breakdown or a non-finite residual estimate
    recursive_relative_residual = torch.ones_like(cos_prev1).masked_fill_(lanczos_is_zero, 0)
    active = ~lanczos_is_zero.expand_as(recursive_relative_residual)
    broken = torch.zeros_like(active)
    stop_reason = None
    # Past n iterations: best iterate found so far, its true relative residual, and stagnation tracking
    shift_values = shifts.reshape(-1)
    best_solution = None
    best_relative_residual = None
    stagnation_count = torch.zeros(active.shape, dtype=torch.int32, device=active.device)
    stagnated = torch.zeros_like(active)

    def _true_residual(normalized_solution):
        # Residual of each shifted system for the unnormalized solution
        nonlocal matvecs
        residuals = []
        for shift, shift_solution in zip(shift_values, normalized_solution):
            shift_solution = shift_solution * rhs_norm
            residual = mm_(shift_solution)
            if value is not None:
                residual = residual * value
            residuals.append(rhs_original - residual - shift * shift_solution)
            matvecs += 1
        return torch.stack(residuals)

    def _tolerance_relative_residual(residual):
        # Relative residual in the norm the tolerance applies to: the M^{-1}-norm with a preconditioner
        if has_preconditioner:
            norm = torch.stack([(r * preconditioner(r)).sum(dim=-2, keepdim=True) for r in residual])
            return norm.clamp_min_(0).sqrt_().div_(beta_initial * rhs_norm)
        return torch.linalg.vector_norm(residual, ord=2, dim=-2, keepdim=True).div_(rhs_norm)

    # Maybe log
    if settings.verbose_linalg:
        # settings.verbose_linalg.logger.debug(
        print(
            f"Running MINRES on a {rhs.shape} RHS for up to {max_iter} iterations "
            f"(tol={tolerance}, convergence={convergence}). Output: {solution.shape}."
        )

    # Perform iterations
    iterations = 0
    for i in range(max_iter):
        # Perform matmul
        prod = mm_(qvec_prev1)
        if value is not None:
            prod.mul_(value)
        matvecs += 1

        # Get next Lanczos terms
        # --> alpha_curr, beta_curr, qvec_curr
        torch.mul(prod, qvec_prev1, out=tmpvec)
        torch.sum(tmpvec, -2, keepdim=True, out=alpha_curr)

        zvec_curr = prod.addcmul_(alpha_curr, zvec_prev1, value=-1).addcmul_(beta_prev, zvec_prev2, value=-1)

        qvec_curr = preconditioner(zvec_curr)
        torch.mul(zvec_curr, qvec_curr, out=tmpvec)
        torch.sum(tmpvec, -2, keepdim=True, out=beta_curr)
        beta_curr.sqrt_()
        lanczos_breakdown = ~beta_curr.gt(eps)
        beta_curr.clamp_min_(eps)

        zvec_curr.div_(beta_curr)
        qvec_curr.div_(beta_curr)

        # Perform the Givens rotation and search vector update
        _minres_updates(
            solution,
            shifts,
            eps,
            qvec_prev1,
            alpha_curr,
            alpha_shifted_curr,
            beta_prev,
            beta_curr,
            cos_prev2,
            cos_prev1,
            cos_curr,
            sin_prev2,
            sin_prev1,
            sin_curr,
            radius_curr,
            subsub_diag_term,
            sub_diag_term,
            diag_term,
            search_prev2,
            search_prev1,
            search_curr,
            search_update,
            scale_prev,
            scale_curr,
            search_update_norm,
            solution_norm,
        )
        iterations = i + 1

        # Freeze columns whose recurrence became non-finite before their update reaches the solution
        relative_residual = scale_curr.abs().div_(beta_initial)
        finite = torch.isfinite(relative_residual) & torch.isfinite(search_update).all(dim=-2, keepdim=True)
        broken = broken | (active & ~finite)
        active = active & ~broken

        # Update the solution of the right-hand sides that are still active
        search_update.masked_fill_(~active, 0)
        solution.add_(search_update)

        # Track the residual estimate and freeze columns that converged or hit a Lanczos breakdown.
        # The update of an exact (finite) Lanczos breakdown is still valid and has been applied above.
        recursive_relative_residual = torch.where(active, relative_residual, recursive_relative_residual)
        broken = broken | (active & lanczos_breakdown)
        active = active & ~broken

        # Check convergence criterion. Freezing columns stays on the device; only the exit tests,
        # every check_every iterations, synchronise with the host.
        past_size = not shifted_preconditioning and iterations > size
        check = iterations % check_every == 0
        recursive_ok = recursive_relative_residual.le(tolerance)
        if convergence == "residual" and not past_size:
            active = active & ~recursive_ok

        if past_size and check:
            # Past n iterations, loss of Lanczos orthogonality can make the iterate drift while the residual
            # estimate keeps decreasing: monitor the true residual and keep the best iterate
            true_relative_residual = _tolerance_relative_residual(_true_residual(solution))
            if best_solution is None:
                best_solution = solution.clone()
                best_relative_residual = torch.full_like(true_relative_residual, math.inf)
            improved = active & true_relative_residual.lt(best_relative_residual)
            best_solution = torch.where(improved, solution, best_solution)
            best_relative_residual = torch.where(improved, true_relative_residual, best_relative_residual)
            stagnation_count = torch.where(improved, 0, stagnation_count + active.int())
            true_ok = true_relative_residual.le(tolerance)
            newly_stagnated = stagnation_count.ge(3)
            if convergence == "residual":
                # The residual estimate is no longer reliable when it has converged but the true residual has not
                newly_stagnated = newly_stagnated | recursive_ok
            newly_stagnated = active & ~true_ok & newly_stagnated
            stagnated = stagnated | newly_stagnated
            active = active & ~true_ok & ~newly_stagnated

        if check:
            if convergence == "update":
                torch.linalg.vector_norm(search_update, dim=-2, out=search_update_norm)
                torch.linalg.vector_norm(solution, dim=-2, out=solution_norm)
                solution_norm.clamp_min_(torch.finfo(solution.dtype).tiny)
                conv = search_update_norm.div_(solution_norm).mean().item()
                if conv < tolerance:
                    stop_reason = "update_converged"
                    break
            if not bool(active.any()):
                break

        # Update terms for next iteration
        # Lanczos terms
        zvec_prev2, zvec_prev1 = zvec_prev1, prod
        qvec_prev1 = qvec_curr
        beta_prev, beta_curr = beta_curr, beta_prev
        # Givens rotations terms
        cos_prev2, cos_prev1, cos_curr = cos_prev1, cos_curr, cos_prev2
        sin_prev2, sin_prev1, sin_curr = sin_prev1, sin_curr, sin_prev2
        # Search vector terms)
        search_prev2, search_prev1, search_curr = search_prev1, search_curr, search_prev2
        scale_prev, scale_curr = scale_curr, scale_prev

    # Past n iterations, trust the recomputed true residual over the residual estimate and return the best iterate
    past_size = not shifted_preconditioning and iterations > size
    final_relative_residual = None
    if past_size:
        final_relative_residual = _tolerance_relative_residual(_true_residual(solution))
        if best_solution is not None:
            use_best = best_relative_residual.lt(final_relative_residual)
            solution = torch.where(use_best, best_solution, solution)
            final_relative_residual = torch.where(use_best, best_relative_residual, final_relative_residual)
        tolerance_converged = final_relative_residual.le(tolerance)
    else:
        tolerance_converged = recursive_relative_residual.le(tolerance)

    if stop_reason is None:
        # The residual estimate is only a stopping criterion in "residual" mode
        if convergence == "residual" and bool(tolerance_converged.all()):
            stop_reason = "recursive_converged"
        elif not bool(active.any()):
            stop_reason = "breakdown" if bool((broken & ~tolerance_converged).any()) else "stagnated"
        else:
            stop_reason = "max_iter"

    if convergence == "residual" and not return_info and stop_reason != "recursive_converged":
        if final_relative_residual is None:
            residual_kind, reported_residual = "relative residual estimate", recursive_relative_residual
        else:
            residual_kind, reported_residual = "true relative residual", final_relative_residual
        warnings.warn(
            f"MINRES terminated after {iterations} iterations ({stop_reason}) with maximum {residual_kind} "
            f"{reported_residual.max().item()} which is larger than the tolerance of {tolerance}. "
            f"{(~tolerance_converged).sum().item()} of {tolerance_converged.numel()} right-hand sides "
            "did not converge.",
            UserWarning,
            stacklevel=2,
        )

    if return_info:
        # Recompute the Euclidean true residual of each shifted system
        true_relative_residual = torch.linalg.vector_norm(_true_residual(solution), ord=2, dim=-2, keepdim=True)
        true_relative_residual = true_relative_residual.div_(rhs_norm).masked_fill_(rhs_is_zero, 0)
        converged = true_relative_residual.le(tolerance)
        reason = "converged" if bool(converged.all()) else stop_reason

        def _info_shape(t):
            t = t.squeeze(-2).detach()
            return t.squeeze(0) if shifts.numel() == 1 else t

        info = MINRESInfo(
            iterations=iterations,
            matvecs=matvecs,
            converged=_info_shape(converged),
            recursive_relative_residual=_info_shape(recursive_relative_residual),
            true_relative_residual=_info_shape(true_relative_residual),
            tolerance=tolerance,
            reason=reason,
        )

    # Set the solution of zero right-hand sides to zero and undo the rhs scaling
    solution = solution.masked_fill(rhs_is_zero, 0).mul_(rhs_norm)

    if squeeze:
        solution = solution.squeeze(-1)

    if shifts.numel() == 1:
        # If we weren't shifting we shouldn't return a batch output
        solution = solution.squeeze(0)

    if return_info:
        return solution, info
    return solution


def _minres_updates(
    solution,
    shifts,
    eps,
    qvec_prev1,
    alpha_curr,
    alpha_shifted_curr,
    beta_prev,
    beta_curr,
    cos_prev2,
    cos_prev1,
    cos_curr,
    sin_prev2,
    sin_prev1,
    sin_curr,
    radius_curr,
    subsub_diag_term,
    sub_diag_term,
    diag_term,
    search_prev2,
    search_prev1,
    search_curr,
    search_update,
    scale_prev,
    scale_curr,
    search_update_norm,
    solution_norm,
):
    # Start givens rotation
    # Givens rotation from 2 steps ago
    torch.mul(sin_prev2, beta_prev, out=subsub_diag_term)
    torch.mul(cos_prev2, beta_prev, out=sub_diag_term)

    # Compute shifted alpha
    torch.add(alpha_curr, shifts, out=alpha_shifted_curr)

    # Givens rotation from 1 step ago
    torch.mul(alpha_shifted_curr, cos_prev1, out=diag_term).addcmul_(sin_prev1, sub_diag_term, value=-1)
    sub_diag_term.mul_(cos_prev1).addcmul_(sin_prev1, alpha_shifted_curr)

    # 3) Compute next Givens terms
    torch.mul(diag_term, diag_term, out=radius_curr).addcmul_(beta_curr, beta_curr).sqrt_()
    cos_curr = torch.div(diag_term, radius_curr, out=cos_curr)
    sin_curr = torch.div(beta_curr, radius_curr, out=sin_curr)
    # 4) Apply current Givens rotation
    diag_term.mul_(cos_curr).addcmul_(sin_curr, beta_curr)

    # Update the solution
    # --> search_curr, scale_curr solution
    # 1) Apply the latest Givens rotation to the Lanczos-rhs ( ||rhs|| e_1 )
    # This is getting the scale terms for the "search" vectors
    torch.mul(scale_prev, sin_curr, out=scale_curr).mul_(-1)
    scale_prev.mul_(cos_curr)
    # 2) Get the new search vector
    torch.addcmul(qvec_prev1, sub_diag_term, search_prev1, value=-1, out=search_curr)
    search_curr.addcmul_(subsub_diag_term, search_prev2, value=-1)
    search_curr.div_(diag_term)

    # 3) Get the solution update (applied by the caller to the right-hand sides that are still active)
    torch.mul(search_curr, scale_prev, out=search_update)
