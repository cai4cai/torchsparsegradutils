import torch
import warnings

class BiCGSTAB():
    """
    This is a pytorch implementation of BiCGSTAB or BCGSTAB, a stable version
    of the CGD method, published first by Van Der Vrost.

    For solving ``Ax = b`` system.

    Example:

    solver = BiCGSTAB(Ax_gen)
    solver.solve(b, x=intial_x, tol=1e-10, atol=1e-16)

    """
    def __init__(self, Ax_gen, device='cuda'):
        """
        Ax_gen: A function that takes a 1-D tensor x and output Ax

        Note: This structure is follwed as it may not be computationally
        efficient to compute A explicitly.

        """
        self.Ax_gen = Ax_gen
        self.device = device
    
    def init_params(self, b, x=None, nsteps=None, tol=1e-10, atol=1e-16):
        """
        b: The R.H.S of the system. 1-D tensor
        nsteps: Number of steps of calculation
        tol: Tolerance such that if ||r||^2 < tol * ||b||^2 then converged
        atol:  Tolernace such that if ||r||^2 < atol then converged

        """
        self.b = b.clone().detach()
        self.x = torch.zeros(b.shape[0], device=self.device) if x is None else x
        self.residual_tol = tol * torch.dot(self.b, self.b)
        self.atol = torch.tensor(atol, device=self.device)
        self.nsteps = b.shape[0] if nsteps is None else nsteps
        self.status, self.r = self.check_convergence(self.x)
        self.rho = torch.tensor(1, device=self.device)
        self.alpha = torch.tensor(1, device=self.device)
        self.omega = torch.tensor(1, device=self.device)
        self.v = torch.zeros(b.shape[0], device=self.device)
        self.p = torch.zeros(b.shape[0], device=self.device)
        self.r_hat = self.r.clone().detach()
    
    def check_convergence(self, x):
        r = self.Ax_gen(x) - self.b
        rdotr = torch.dot(r,r)
        if rdotr < self.residual_tol or rdotr < self.atol:
            return True, r
        else:
            return False, r
    
    def step(self):
        rho = torch.dot(self.r, self.r_hat)                     # rho_i <- <r0, r^>
        beta = (rho/self.rho)*(self.alpha/self.omega)           # beta <- (rho_i/rho_{i-1}) x (alpha/omega_{i-1})
        self.rho = rho                                          # rho_{i-1} <- rho_i  replaced self value
        self.p = self.r + beta*(self.p - self.omega*self.v)     # p_i <- r_{i-1} + beta x (p_{i-1} - w_{i-1} v_{i-1}) replaced p self value
        self.v = self.Ax_gen(self.p)                            # v_i <- Ap_i
        self.alpha = self.rho/torch.dot(self.r_hat, self.v)     # alpha <- rho_i/<r^, v_i>
        s = self.r - self.alpha*self.v                          # s <- r_{i-1} - alpha v_i
        t = self.Ax_gen(s)                                      # t <- As
        self.omega = torch.dot(t, s)/torch.dot(t, t)            # w_i <- <t, s>/<t, t>
        self.x = self.x + self.alpha*self.p + self.omega*s      # x_i <- x_{i-1} + alpha p + w_i s
        status, res = self.check_convergence(self.x)
        if status:
            return True
        else:
            self.r = s - self.omega*t                           # r_i <- s - w_i t
            return False
    
    def solve(self, *args, **kwargs):
        """
        Method to find the solution.

        Returns the final answer of x

        """
        self.init_params(*args, **kwargs)
        if self.status:
            return self.x
        while self.nsteps:
            s = self.step()
            if s:
                return self.x
            if self.rho == 0:
                break
            self.nsteps-=1
        warnings.warn('Convergence has failed :(')
        return self.x