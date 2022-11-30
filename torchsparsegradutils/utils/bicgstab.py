# Code imported from https://gist.github.com/bridgesign/f421f69ad4a3858430e5e235bccde8c6
# Modifications to fit torchsparsegradutils

import torch
import warnings

from typing import NamedTuple
import types


class BICGSTABSettings(NamedTuple):
    nsteps: int = None  # Number of steps of calculation
    tol: float = 1e-10  # Tolerance such that if ||r||^2 < tol * ||b||^2 then converged
    atol: float = 1e-16  # Tolerance such that if ||r||^2 < atol then converged
    
        
def _init_params(Ax_gen, b, x=None, nsteps=None, tol=1e-10, atol=1e-16):
    """
    Ax_gen: A tensor or function that takes a 1-D tensor x and output Ax
    b: The R.H.S of the system. 1-D tensor
    nsteps: Number of steps of calculation
    tol: Tolerance such that if ||r||^2 < tol * ||b||^2 then converged
    atol: Tolerance such that if ||r||^2 < atol then converged
    """
    opt = types.SimpleNamespace()
    # Check Ax_gen object
    if torch.is_tensor(Ax_gen):
        opt.Ax_gen = Ax_gen.matmul
    elif callable(Ax_gen):
        opt.Ax_gen = Ax_gen
    else:
        raise RuntimeError("Ax_gen must be a tensor, or a callable object!")
    opt.b = b.detach()
    opt.device = b.device
    opt.dtype = b.dtype
    opt.x = torch.zeros(b.shape[0], dtype=opt.dtype, device=opt.device) if x is None else x
    opt.residual_tol = tol * torch.dot(opt.b, opt.b)
    opt.atol = torch.tensor(atol, dtype=opt.dtype, device=opt.device)
    opt.nsteps = b.shape[0] if nsteps is None else nsteps
    opt.status, opt.r = _check_convergence(opt)
    opt.rho = torch.tensor(1, dtype=opt.dtype, device=opt.device)
    opt.alpha = torch.tensor(1, dtype=opt.dtype, device=opt.device)
    opt.omega = torch.tensor(1, dtype=opt.dtype, device=opt.device)
    opt.v = torch.zeros(b.shape[0], dtype=opt.dtype, device=opt.device)
    opt.p = torch.zeros(b.shape[0], dtype=opt.dtype, device=opt.device)
    opt.r_hat = opt.r.detach().clone()

    return opt


def _check_convergence(opt):
    r = opt.b - opt.Ax_gen(opt.x)
    rdotr = torch.dot(r, r)
    if rdotr < opt.residual_tol or rdotr < opt.atol:
        return True, r
    else:
        return False, r

def _step(opt):
    rho = torch.dot(opt.r, opt.r_hat)  # rho_i <- <r0, r^>
    beta = (rho / opt.rho) * (opt.alpha / opt.omega)  # beta <- (rho_i/rho_{i-1}) x (alpha/omega_{i-1})
    opt.rho = rho  # rho_{i-1} <- rho_i  replaced self value
    opt.p = opt.r + beta * (
        opt.p - opt.omega * opt.v
    )  # p_i <- r_{i-1} + beta x (p_{i-1} - w_{i-1} v_{i-1}) replaced p self value
    opt.v = opt.Ax_gen(opt.p)  # v_i <- Ap_i
    opt.alpha = opt.rho / torch.dot(opt.r_hat, opt.v)  # alpha <- rho_i/<r^, v_i>
    # TODO compute h = opt.x + opt.alpha * opt.p
    s = opt.r - opt.alpha * opt.v  # s <- r_{i-1} - alpha v_i
    t = opt.Ax_gen(s)  # t <- As
    opt.omega = torch.dot(t, s) / torch.dot(t, t)  # w_i <- <t, s>/<t, t>
    opt.x = opt.x + opt.alpha * opt.p + opt.omega * s  # x_i <- x_{i-1} + alpha p + w_i s
    opt.status, opt.res = _check_convergence(opt)
    if opt.status:
        return True
    else:
        opt.r = s - opt.omega * t  # r_i <- s - w_i t
        return False

def bicgstab(
    matmul_closure,
    rhs,
    settings=BICGSTABSettings(),
):
    """
    Method to find the solution.

    Returns the final answer of x

    This is a pytorch implementation of BiCGSTAB or BCGSTAB, a stable version
    of the CGD method, published first by Van Der Vrost.

    For solving ``Ax = b`` system.

    """
    opt = _init_params(matmul_closure, rhs, x=None, nsteps=settings.nsteps, tol=settings.tol, atol=settings.atol)
    if opt.status:
        return opt.x
    while opt.nsteps:
        s = _step(opt)
        if s:
            return opt.x
        if opt.rho == 0:
            break
        opt.nsteps -= 1
    warnings.warn("BICGSTAB convergence has failed :(")
    return opt.x
