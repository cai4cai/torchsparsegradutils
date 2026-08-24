# MIT-licensed code imported from https://github.com/cornellius-gp/linear_operator
# Minor modifications for torchsparsegradutils to remove dependencies

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch


class LinearCGSettings(NamedTuple):
    max_cg_iterations: int = 1000  # The maximum number of conjugate gradient iterations to perform (when computing
    # matrix solves). A higher value rarely results in more accurate solves -- instead, lower the CG tolerance.
    max_lanczos_quadrature_iterations: int = (
        20  # The maximum number of Lanczos iterations to perform when doing stochastic
    )
    # Lanczos quadrature. This is ONLY used for log determinant calculations and
    # computing Tr(K^{-1}dK/d\theta)
    cg_tolerance: float = 1e-5  # Relative residual tolerance to use for terminating CG.
    terminate_cg_by_size: bool = False  # If set to true, cg will terminate after n iterations for an n x n matrix.
    verbose_linalg: bool = False  # Print out information whenever running an expensive linear algebra routine


_DEFAULT_LINEAR_CG_SETTINGS = LinearCGSettings()


def _default_preconditioner(x):
    return x.clone()


@dataclass(frozen=True)
class CGInfo:
    """Convergence information returned by linear_cg.

    Residual tensors have shape (batch_shape, num_rhs). Vector right-hand
    sides therefore produce a length-one residual tensor.
    """

    iterations: int
    matvecs: int
    converged: torch.Tensor
    recursive_relative_residual: torch.Tensor
    true_relative_residual: torch.Tensor
    reason: str


def _linear_cg_updates(
    result, alpha, residual_inner_prod, eps, beta, residual, precond_residual, mul_storage, is_zero, curr_conjugate_vec
):
    # # Update result
    # # result_{k} = result_{k-1} + alpha_{k} p_vec_{k-1}
    result = torch.addcmul(result, alpha, curr_conjugate_vec, out=result)

    # beta_{k} = (precon_residual{k}^T r_vec_{k}) / (precon_residual{k-1}^T r_vec_{k-1})
    beta.resize_as_(residual_inner_prod).copy_(residual_inner_prod)
    torch.mul(residual, precond_residual, out=mul_storage)
    torch.sum(mul_storage, -2, keepdim=True, out=residual_inner_prod)

    # Do a safe division here
    torch.lt(beta, eps, out=is_zero)
    beta.masked_fill_(is_zero, 1)
    torch.div(residual_inner_prod, beta, out=beta)
    beta.masked_fill_(is_zero, 0)

    # Update curr_conjugate_vec
    # curr_conjugate_vec_{k} = precon_residual{k} + beta_{k} curr_conjugate_vec_{k-1}
    curr_conjugate_vec.mul_(beta).add_(precond_residual)


def _linear_cg_updates_no_precond(
    mvms,
    result,
    has_converged,
    alpha,
    residual_inner_prod,
    eps,
    beta,
    residual,
    precond_residual,
    mul_storage,
    is_zero,
    curr_conjugate_vec,
):
    torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
    torch.sum(mul_storage, dim=-2, keepdim=True, out=alpha)

    active = ~has_converged
    if bool((active & ((alpha <= 0) | ~torch.isfinite(alpha))).any()):
        raise RuntimeError("CG breakdown: p^T A p must be finite and positive for active right-hand sides")

    # Do a safe division here
    torch.lt(alpha, eps, out=is_zero)
    alpha.masked_fill_(is_zero, 1)
    torch.div(residual_inner_prod, alpha, out=alpha)
    alpha.masked_fill_(is_zero, 0)

    # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
    alpha.masked_fill_(has_converged, 0)

    # Update residual
    # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
    torch.addcmul(residual, -alpha, mvms, out=residual)

    # Update precond_residual
    # precon_residual{k} = M^-1 residual_{k}
    precond_residual = residual.clone()

    _linear_cg_updates(
        result,
        alpha,
        residual_inner_prod,
        eps,
        beta,
        residual,
        precond_residual,
        mul_storage,
        is_zero,
        curr_conjugate_vec,
    )


def linear_cg(  # noqa: C901 - inherited solver is intentionally kept as one recurrence
    matmul_closure: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    n_tridiag: int = 0,
    tolerance: float | None = None,
    eps: float | None = None,
    stop_updating_after: float | None = None,
    max_iter: int | None = None,
    max_tridiag_iter: int | None = None,
    initial_guess: torch.Tensor | None = None,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    settings: LinearCGSettings = _DEFAULT_LINEAR_CG_SETTINGS,
    convergence_reduction: str = "all",
    min_iter: int = 0,
    return_info: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, CGInfo]
    | tuple[torch.Tensor, torch.Tensor, CGInfo]
):
    r"""
    Solve symmetric positive definite linear systems using conjugate gradient (CG).

    Implements linear CG for systems :math:`A x = b` with symmetric positive definite
    operator :math:`A`. Supports single/multiple RHS and optional stochastic Lanczos
    tridiagonalization for eigenvalue/log-determinant estimation.

    Parameters
    ----------
    matmul_closure : {``torch.Tensor``, callable(x) -> ``A x``}
        Matrix–vector multiply. If a tensor is provided, its ``.matmul`` is used.
        The callable must accept inputs shaped like ``rhs`` and return ``A @ rhs``.
    rhs : torch.Tensor, shape ``(..., n)`` or ``(..., n, k)``
        Right-hand side(s). Leading batch dims are supported.
    n_tridiag : int, optional
        Number of Lanczos tridiagonalizations (probe vectors). If ``> 0``,
        tridiagonal matrices are returned in addition to the solution. Default: ``0``.
    tolerance : float, optional
        Relative residual-norm stopping criterion. If ``None``, uses
        ``settings.cg_tolerance``.
    eps : float, optional
        Absolute breakdown threshold for recurrence denominators. If ``None``,
        uses the smallest positive normal value of the RHS dtype. The historical
        fixed default of 1e-10 can stall accurate solves.
    stop_updating_after : float, optional
        Per-vector early-stop threshold for normalized residual norms. If
        ``None``, uses ``tolerance``.
    max_iter : int, optional
        Maximum CG iterations. If ``None``, uses ``settings.max_cg_iterations``.
    max_tridiag_iter : int, optional
        Maximum Lanczos size. If ``None``, uses
        ``settings.max_lanczos_quadrature_iterations``.
    initial_guess : torch.Tensor, optional, shape like ``rhs``
        Initial guess. If ``None``, zeros are used.
    preconditioner : callable, optional
        Preconditioner with signature ``preconditioner(x) -> M^{-1} x``.
        If ``None``, no preconditioning is used.
    settings : LinearCGSettings, optional
        Configuration for iteration caps, tolerances, and logging verbosity.
    convergence_reduction : {"all", "mean"}, optional
        Stop when every right-hand side passes tolerance (default), or retain
        the historical mean-residual behavior.
    min_iter : int, optional
        Minimum number of iterations before tolerance termination. Default: 0.
    return_info : bool, optional
        Return ``CGInfo`` containing recomputed true relative residuals,
        per-right-hand-side convergence, iterations, and matvecs.

    Returns
    -------
    torch.Tensor or tuple
        * If ``n_tridiag == 0``: solution ``x`` with the same shape as ``rhs``.
        * If ``n_tridiag > 0``: ``(x, T)`` where ``T`` has shape
          ``(n_tridiag, *rhs.shape[:-2], r, r)`` with ``r = last_tridiag_iter + 1``
          and ``r <= min(max_tridiag_iter, n)``. Without batch dims this is ``(n_tridiag, r, r)``.
        * If ``return_info`` is true, append a ``CGInfo`` object.

    Raises
    ------
    RuntimeError
        If ``max_tridiag_iter > max_iter``.
    RuntimeError
        If ``matmul_closure`` is neither a tensor nor a callable.

    Notes
    -----
    CG converges in at most ``n`` iterations for SPD matrices, but typically much
    faster if eigenvalues are clustered. Preconditioning (e.g. diagonal or
    incomplete Cholesky) can significantly accelerate convergence. When
    ``n_tridiag > 0``, Lanczos tridiagonalization is accumulated alongside CG for
    spectral / log-determinant estimates.

    This implementation is based on MIT-licensed code from the linear_operator
    library [1e]_.

    Examples
    --------
    Basic CG solve::

        >>> A = torch.tensor([[4.0, -1.0], [-1.0, 4.0]])
        >>> b = torch.tensor([1.0, 2.0])
        >>> x = linear_cg(A.matmul, b)
        >>> x.shape
        torch.Size([2])

    Multiple RHS::

        >>> B = torch.randn(2, 5)  # 5 RHS
        >>> X = linear_cg(A.matmul, B, max_iter=100, tolerance=1e-8)
        >>> X.shape
        torch.Size([2, 5])

    With preconditioning::

        >>> M_inv = torch.diag(1.0 / torch.diag(A))
        >>> x = linear_cg(A.matmul, b, preconditioner=lambda v: M_inv @ v)

    With Lanczos tridiagonalization::

        >>> x, T = linear_cg(A.matmul, b, n_tridiag=1)
        >>> T.shape  # (n_tridiag, r, r) with r <= n
        torch.Size([1, 2, 2])

    Sparse operator via closure::

        >>> indices = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]])
        >>> values = torch.tensor([4.0, -1.0, -1.0, 4.0, 2.0])
        >>> A_sp = torch.sparse_coo_tensor(indices, values, (3, 3))
        >>> x = linear_cg(lambda v: A_sp @ v, torch.randn(3))

    References
    ----------
    .. [1e] linear_operator library. https://github.com/cornellius-gp/linear_operator
    """
    if not isinstance(rhs, torch.Tensor) or not torch.is_floating_point(rhs):
        raise TypeError("rhs must be a real floating-point torch.Tensor")
    if rhs.ndimension() < 1:
        raise ValueError("rhs must have at least one dimension")

    # Unsqueeze vector right-hand sides without later reusing this rank flag.
    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)
    rhs_original = rhs

    # Some default arguments
    if max_iter is None:
        max_iter = settings.max_cg_iterations
    if max_tridiag_iter is None:
        max_tridiag_iter = settings.max_lanczos_quadrature_iterations
    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)
    else:
        if initial_guess.ndimension() == 1:
            initial_guess = initial_guess.unsqueeze(-1)
        if initial_guess.shape != rhs.shape:
            raise ValueError(f"initial_guess must have shape {tuple(rhs.shape)}, got {tuple(initial_guess.shape)}")
        if initial_guess.device != rhs.device or initial_guess.dtype != rhs.dtype:
            raise ValueError("initial_guess must share rhs device and dtype")
    if tolerance is None:
        tolerance = settings.cg_tolerance
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if eps is None:
        eps = torch.finfo(rhs.dtype).tiny
    if eps <= 0:
        raise ValueError("eps must be positive")
    if stop_updating_after is None:
        stop_updating_after = tolerance
    if stop_updating_after < 0:
        raise ValueError("stop_updating_after must be nonnegative")
    if convergence_reduction not in ("all", "mean"):
        raise ValueError("convergence_reduction must be 'all' or 'mean'")
    if min_iter < 0:
        raise ValueError("min_iter must be nonnegative")
    if preconditioner is None:
        preconditioner = _default_preconditioner
        precond = False
    else:
        precond = True

    # If we are running m CG iterations, we obviously can't get more than m Lanczos coefficients
    if max_tridiag_iter > max_iter:
        raise RuntimeError("Getting a tridiagonalization larger than the number of CG iterations run is not possible!")

    # Check matmul_closure object
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    # Get some constants
    num_rows = rhs.size(-2)
    n_iter = min(max_iter, num_rows) if settings.terminate_cg_by_size else max_iter
    n_tridiag_iter = min(max_tridiag_iter, num_rows)
    eps_tensor = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    # Get the norm of the rhs for convergence checks. Replace exact-zero
    # norms with 1 to avoid division by zero, while tracking those RHS columns.
    rhs_norm = torch.linalg.vector_norm(rhs, ord=2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.eq(0)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)

    # Let's normalize. We'll un-normalize afterwards
    rhs = rhs.div(rhs_norm)
    initial_guess = initial_guess.div(rhs_norm)
    # The unique solution of an SPD zero-RHS system is zero. Do not return an
    # arbitrary nonzero initial guess while marking that column converged.
    initial_guess = initial_guess.masked_fill(rhs_is_zero, 0)

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(initial_guess)
    matvecs = 1
    if residual.shape != rhs.shape or residual.device != rhs.device or residual.dtype != rhs.dtype:
        raise RuntimeError("matmul_closure output must match rhs shape, device, and dtype")
    batch_shape = residual.shape[:-2]

    # result <- x_{0}
    result = initial_guess.expand_as(residual).contiguous()

    # Maybe log
    if settings.verbose_linalg:
        # settings.verbose_linalg.logger.debug(
        print(f"Running CG on a {rhs.shape} RHS for {n_iter} iterations (tol={tolerance}). Output: {result.shape}.")

    # Check for NaNs
    if not torch.equal(residual, residual):
        raise RuntimeError("NaNs encountered when trying to perform matrix-vector multiplication")

    # Sometime we're lucky and the preconditioner solves the system right away
    # Check for convergence
    residual_norm = torch.linalg.vector_norm(residual, ord=2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    if has_converged.all() and not n_tridiag:
        n_iter = 0  # Skip the iteration!

    # Otherwise, let's define precond_residual and curr_conjugate_vec
    else:
        # precon_residual{0} = M^-1 residual_{0}
        precond_residual = preconditioner(residual)
        if (
            precond_residual.shape != residual.shape
            or precond_residual.device != residual.device
            or precond_residual.dtype != residual.dtype
        ):
            raise RuntimeError("preconditioner output must match residual shape, device, and dtype")
        curr_conjugate_vec = precond_residual
        residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)
        active = ~has_converged
        if bool((active & ((residual_inner_prod <= 0) | ~torch.isfinite(residual_inner_prod))).any()):
            raise RuntimeError("CG breakdown: preconditioned residual inner product must be finite and positive")

        # Define storage matrices
        mul_storage = torch.empty_like(residual)
        alpha = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=residual.dtype, device=residual.device)
        beta = torch.empty_like(alpha)
        is_zero = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=torch.bool, device=residual.device)

    # Define tridiagonal matrices, if applicable
    if n_tridiag:
        t_mat = torch.zeros(
            n_tridiag_iter, n_tridiag_iter, *batch_shape, n_tridiag, dtype=alpha.dtype, device=alpha.device
        )
        alpha_tridiag_is_zero = torch.empty(*batch_shape, n_tridiag, dtype=torch.bool, device=t_mat.device)
        alpha_reciprocal = torch.empty(*batch_shape, n_tridiag, dtype=t_mat.dtype, device=t_mat.device)
        prev_alpha_reciprocal = torch.empty_like(alpha_reciprocal)
        prev_beta = torch.empty_like(alpha_reciprocal)

    update_tridiag = True
    last_tridiag_iter = 0

    # It's conceivable we reach the tolerance on the last iteration, so can't just check iteration number.
    tolerance_reached = False
    iterations = 0

    # Start the iteration
    for k in range(n_iter):
        # Get next alpha
        # alpha_{k} = (residual_{k-1}^T precon_residual{k-1}) / (p_vec_{k-1}^T mat p_vec_{k-1})
        mvms = matmul_closure(curr_conjugate_vec)
        matvecs += 1
        if mvms.shape != rhs.shape or mvms.device != rhs.device or mvms.dtype != rhs.dtype:
            raise RuntimeError("matmul_closure output must match rhs shape, device, and dtype")
        if precond:
            torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
            torch.sum(mul_storage, -2, keepdim=True, out=alpha)

            active = ~has_converged
            if bool((active & ((alpha <= 0) | ~torch.isfinite(alpha))).any()):
                raise RuntimeError("CG breakdown: p^T A p must be finite and positive for active right-hand sides")

            # Do a safe division here
            torch.lt(alpha, eps_tensor, out=is_zero)
            alpha.masked_fill_(is_zero, 1)
            torch.div(residual_inner_prod, alpha, out=alpha)
            alpha.masked_fill_(is_zero, 0)

            # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
            alpha.masked_fill_(has_converged, 0)

            # Update residual
            # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
            residual = torch.addcmul(residual, alpha, mvms, value=-1, out=residual)

            # Update precond_residual
            # precon_residual{k} = M^-1 residual_{k}
            precond_residual = preconditioner(residual)
            if (
                precond_residual.shape != residual.shape
                or precond_residual.device != residual.device
                or precond_residual.dtype != residual.dtype
            ):
                raise RuntimeError("preconditioner output must match residual shape, device, and dtype")

            _linear_cg_updates(
                result,
                alpha,
                residual_inner_prod,
                eps_tensor,
                beta,
                residual,
                precond_residual,
                mul_storage,
                is_zero,
                curr_conjugate_vec,
            )
        else:
            _linear_cg_updates_no_precond(
                mvms,
                result,
                has_converged,
                alpha,
                residual_inner_prod,
                eps_tensor,
                beta,
                residual,
                precond_residual,
                mul_storage,
                is_zero,
                curr_conjugate_vec,
            )

        torch.linalg.vector_norm(residual, ord=2, dim=-2, keepdim=True, out=residual_norm)
        torch.lt(residual_norm, stop_updating_after, out=has_converged)
        iterations = k + 1

        if convergence_reduction == "all":
            convergence_reached = bool(torch.le(residual_norm, tolerance).all())
        else:
            convergence_reached = bool(residual_norm.mean() <= tolerance)

        if iterations >= min_iter and convergence_reached and not (n_tridiag and k < min(n_tridiag_iter, max_iter - 1)):
            tolerance_reached = True
            break

        # Update tridiagonal matrices, if applicable
        if n_tridiag and k < n_tridiag_iter and update_tridiag:
            alpha_tridiag = alpha.squeeze(-2).narrow(-1, 0, n_tridiag)
            beta_tridiag = beta.squeeze(-2).narrow(-1, 0, n_tridiag)
            torch.eq(alpha_tridiag, 0, out=alpha_tridiag_is_zero)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 1)
            torch.reciprocal(alpha_tridiag, out=alpha_reciprocal)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 0)

            if k == 0:
                t_mat[k, k].copy_(alpha_reciprocal)
            else:
                torch.addcmul(alpha_reciprocal, prev_beta, prev_alpha_reciprocal, out=t_mat[k, k])
                torch.mul(prev_beta.sqrt_(), prev_alpha_reciprocal, out=t_mat[k, k - 1])
                t_mat[k - 1, k].copy_(t_mat[k, k - 1])

                if t_mat[k - 1, k].max() < 1e-6:
                    update_tridiag = False

            last_tridiag_iter = k

            prev_alpha_reciprocal.copy_(alpha_reciprocal)
            prev_beta.copy_(beta_tridiag)

    # Un-normalize and recompute a true residual before reporting convergence.
    result = result.mul(rhs_norm)
    true_residual = rhs_original - matmul_closure(result)
    matvecs += 1
    if true_residual.shape != rhs.shape or true_residual.device != rhs.device or true_residual.dtype != rhs.dtype:
        raise RuntimeError("matmul_closure output must match rhs shape, device, and dtype")
    rhs_original_norm = torch.linalg.vector_norm(rhs_original, ord=2, dim=-2, keepdim=True)
    denominator = rhs_original_norm.masked_fill(rhs_original_norm.eq(0), 1)
    true_relative_residual = torch.linalg.vector_norm(true_residual, ord=2, dim=-2, keepdim=True) / denominator
    recursive_relative_residual = residual_norm
    converged = torch.le(true_relative_residual, tolerance)
    all_true_converged = bool(converged.all())
    if all_true_converged:
        reason = "converged"
    elif tolerance_reached and convergence_reduction == "mean":
        reason = "mean_converged"
    else:
        reason = "max_iter"

    if tolerance > 0 and not all_true_converged and n_iter > 0:
        maximum_residual = true_relative_residual.max().item()
        num_unconverged = (~converged).sum().item()
        warnings.warn(
            f"CG terminated in {iterations} iterations with maximum true relative residual "
            f"{maximum_residual} which is larger than the tolerance of {tolerance}. "
            f"{num_unconverged} of {converged.numel()} right-hand sides did not converge.",
            UserWarning,
            stacklevel=2,
        )

    info = CGInfo(
        iterations=iterations,
        matvecs=matvecs,
        converged=converged.squeeze(-2).detach(),
        recursive_relative_residual=recursive_relative_residual.squeeze(-2).detach(),
        true_relative_residual=true_relative_residual.squeeze(-2).detach(),
        reason=reason,
    )

    if is_vector:
        result = result.squeeze(-1)

    if n_tridiag:
        t_mat = t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1]
        tridiagonal = t_mat.permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()
        if return_info:
            return result, tridiagonal, info
        return result, tridiagonal
    if return_info:
        return result, info
    return result
