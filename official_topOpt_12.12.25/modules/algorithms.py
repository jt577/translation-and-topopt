import torch
import math
import numpy as np
import torch.nn.functional as F
import math

def PCG(x0, args, residual_fun, apply_precond,
        maxit=3000, tol=1e-4, outfile=None,
        restart_interval=100, epsilon=1e-8):

    x = x0.clone()
    x -= x.mean()
    b = residual_fun(torch.zeros_like(x), *args)[0]

    if not torch.isfinite(b).all():
        raise RuntimeError("PCG: b has NaN/inf from residual_fun(0).")

    r = residual_fun(x, *args)[0]
    if not torch.isfinite(r).all():
        raise RuntimeError("PCG: initial residual has NaN/inf.")

    r -= r.mean()
    z = apply_precond(r)
    if not torch.isfinite(z).all():
        raise RuntimeError("PCG: preconditioned residual has NaN/inf.")
    z -= z.mean()
    p = z.clone()
    rz_old = torch.dot(r, z)

    tiny = 1e-30

    for k in range(maxit):
        if k and (k % restart_interval == 0):
            r = residual_fun(x, *args)[0]
            r -= r.mean()
            if not torch.isfinite(r).all():
                raise RuntimeError(f"PCG: residual NaN at restart iter {k}.")
            z = apply_precond(r)
            if not torch.isfinite(z).all():
                raise RuntimeError(f"PCG: precond NaN at restart iter {k}.")
            z -= z.mean()
            p = z.clone()
            rz_old = torch.dot(r, z)

        Ap = b - residual_fun(p, *args)[0]
        if epsilon != 0.0:
            Ap = Ap + epsilon * p

        denom = torch.dot(p, Ap)
        if not torch.isfinite(denom) or abs(denom) < tiny:
            raise RuntimeError(f"PCG: breakdown, p^T A p = {denom.item():.3e}")

        alpha = rz_old / denom

        x = x + alpha * p
        x -= x.mean()

        r = r - alpha * Ap
        r -= r.mean()

        if not torch.isfinite(r).all():
            raise RuntimeError(f"PCG: residual became NaN/inf at iter {k}.")

        res_norm = torch.norm(r)
        if outfile and k % 50 == 0:
            with open(outfile, 'a') as f:
                f.write(f" PCG it {k:4d} |r| = {res_norm:.3e}\n")

        if res_norm < tol:
            if outfile:
                with open(outfile, 'a') as f:
                    f.write(f" PCG converged at it {k} |r| = {res_norm:.3e}\n")
            break

        z = apply_precond(r)
        if not torch.isfinite(z).all():
            raise RuntimeError(f"PCG: preconditioner produced NaN/inf at iter {k}.")
        z -= z.mean()

        rz_new = torch.dot(r, z)
        if abs(rz_old) < tiny or not torch.isfinite(rz_new):
            raise RuntimeError(f"PCG: breakdown, r^T z old={rz_old.item():.3e}, new={rz_new.item():.3e}")

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x

def get_Lattice(ucparams):
    """
    Get the lattice vector matrix (columns are lattice vectors).
    ucparams: tensor([a, b, c, α, β, γ]) on self.device, dtype float64
    """

    a, b, c, α, β, γ = ucparams

    zero = torch.zeros((), device=ucparams.device, dtype=torch.float64)

    # e1 = (a, 0, 0)
    e1 = torch.stack([a, zero, zero])

    # e2 = (b cos γ, b sin γ, 0)
    e2 = torch.stack([b * torch.cos(γ),
                    b * torch.sin(γ),
                    zero])

    # e3 components
    e3_0 = c * torch.cos(β)
    e3_1 = c * (torch.cos(α) - torch.cos(β) * torch.cos(γ)) / torch.sin(γ)
    e3_2 = torch.sqrt(c**2 - e3_0**2 - e3_1**2)
    e3   = torch.stack([e3_0, e3_1, e3_2])

    # Columns are lattice vectors
    Lattice = torch.stack((e1, e2, e3), dim=1)  # shape (3,3)
    return Lattice

def get_ucparams(Lattice):
    """
    Get the unit cell parameters from the lattice vector matrix.
    """
    # Calculate lengths of the unit cell vectors
    a = torch.linalg.norm(Lattice[:, 0])
    b = torch.linalg.norm(Lattice[:, 1])
    c = torch.linalg.norm(Lattice[:, 2])
    # Calculate angles between the unit cell vectors
    α = torch.acos(torch.dot(Lattice[:, 1], Lattice[:, 2]) / (b * c))
    β = torch.acos(torch.dot(Lattice[:, 0], Lattice[:, 2]) / (a * c))
    γ = torch.acos(torch.dot(Lattice[:, 0], Lattice[:, 1]) / (a * b))
    return torch.tensor([a, b, c, α, β, γ], device=Lattice.device, dtype=torch.float64)

def project_density(v, c, tol=1e-12, maxit=100):
    """
    Euclidean projection onto {0 ≤ x ≤ 1, sum(x) = c}.
    Michelot (1986); see also Condat (2016).
    """
    if not torch.isfinite(v).all():
        raise RuntimeError("project: input v contains NaN/inf.")
    lo = v.min().item() - 1.0
    hi = v.max().item()
    for i in range(maxit):
        if (hi - lo) < tol:
            return torch.clamp(v - hi, 0, 1)
        mid = 0.5 * (lo + hi)
        s = torch.clamp(v - mid, 0, 1).sum().item()
        if s > c:
            lo = mid
        else:
            hi = mid
    print(f'     Warning: Projections converged to {hi-lo} instead of tol {tol} after maximum {maxit} iterations.')
    return torch.clamp(v - hi, 0, 1)

def project_ucparams(v, tol=1e-12, maxit=100):
    """
    Projection of unit cell parameters.
    """
    print(v)
    v[:3] = torch.clamp(v[:3], min=0.1, max=4.0) # lattice constants cannot be too small or large
    v[3:] = torch.clamp(v[3:], min=math.radians(20), max=math.pi - math.radians(20))
    Lattice = get_Lattice(v)
    vol = torch.abs(torch.det(Lattice)).item()
    ucparams = get_ucparams(Lattice/vol**(1/3))
    return ucparams


class minimizer():
    def __init__(self, device, n):
        """
        Minimizer for topology optimization with unit cell optimization.
        Args:
            device: torch device
            n: number of variables
        """
        # ---- weights ----------------------------------------------------
        self.w = torch.ones(n, device=device, dtype=torch.float64) / n

    def wdot(self, a, b):    # <a,b>_w
        return torch.dot(self.w * a, b)

    def wnorm(self, a):
        return torch.sqrt(self.wdot(a, a) + 1e-30)

    def SPG(self, x0, f0, g0, delta, N, M_inv, SPG_hist, project, proj_args, maxit=1000, tol=1e-5):
        """
        Spectral Projected Gradient for ½‖x-x0‖_H² + g0ᵀ(x-x0) + f0.

        Args:
            x0: Initial point
            f0: Initial function value at x0
            g0: Gradient at x0
            delta, N, M_inv: Matrices for limited memory BFGS update
            c: Inverse volume fraction of the design domain
            SPG_hist: History length for function values
            project: Function to project onto the feasible set
            maxit: Maximum number of iterations
            tol: Tolerance for convergence
        """
        def quad(x):
            # Compute the quadratic function value and gradient
            r = N.t().matmul(x - x0)
            r = M_inv.matmul(r)
            r = N.matmul(r)
            r = delta * (x - x0) - r
            return (0.5 * self.wdot(x - x0, r) + self.wdot(g0, x - x0)).item() + f0, r + g0

        a_max = 1e-2
        a_min = 1e-8
        x = project(x0, *proj_args)           # ensure feasibility
        f, g = quad(x)              # initial function value
        a_bb = 1e-4                 # important to not set large otherwise can obtain weird densities (I think)
        f_hist = [f]  # history of function values

        for k in range(maxit):
            a_k = min(a_max, max(a_min, a_bb)) # spectral step size

            d_k = project(x - a_k * g, *proj_args) - x  # descent direction
            f_b = max(f_hist)
            a, f_new_val, g_new = self.line_min(
                fun_jac=quad, x0=x, f0=f, g0=g,
                p=d_k, f_b=f_b, args=(), amax=1, c1=1e-4, maxiter=10
            )
            
            # if line search failed, use a = 1 (fall back to bb step since it is encoded already in d_k)
            if (a is None) or (a < 0) or isinstance(a, complex):
                a = 1.0
                f_new_val, g_new = quad(x + a * d_k)  # re-evaluate function and gradient

            if len(f_hist) < SPG_hist:
                f_hist.append(f_new_val)
            else:
                f_hist = f_hist[1:] + [f_new_val]

            x_new = x + a * d_k
            s = x_new - x
            y = g_new - g
            num = self.wdot(s, s).item()
            den = self.wdot(s, y).item()
            a_bb = num / den if den > 0 else 1e-4  # update step size for next iteration

            g = g_new.clone()
            x = x_new.clone()

            err = self.wnorm(project(x_new - g_new, *proj_args) - x_new).item()
            if (k-1) % 200 == 0:
                pass
                # print(f"                SPG iteration {k}: fun = {f_new_val:.6e}, |P(x-g)-x| = {err:.6e}, a = {a:.6e}, a_bb = {a_bb:.6e}.")
            if err < tol:
                # print(f"                SPG converged to within {tol} at iteration {k}.")
                return x_new, True
            
            x, g = x_new, g_new  # advance iterate
        
        return x, False

    def PBFGS(self, fun_jac, x0, BFGS_hist, args, proj_args, callback, project, maxit=100, tol=1e-10):
        """
        Perform projected limited memory BFGS.

        Args:
            fun_jac: Function to minimize (along with its gradient)
            x0: Initial guess
            jac: Gradient of function
            BFGS_hist: History length for function values
            args: Extra args passed to fun and jac
            proj_args: Arguments for the projection function
            project: Projection back onto nearest point on valid domain
            maxit: Maximum iterations
            tol: Tolerance of convergence

        Returns:
            x: Final solution
        """
        device = x0.device
        dtype = x0.dtype
        x = x0.clone()
        f_new = None
        g_new = None
        k = 1
        S_list = []
        R_list = []
        converged = False
        reset = True  # flag to indicate if we need to reset the quasi-Hessian (or begin optimization)

        while k <= maxit:
            if reset == True:
                # Perform gradient descent step on first iteration to initialize delta
                if not torch.isfinite(x).all():
                    raise RuntimeError("PBFGS: x has NaN/inf before fun_jac.")
                f, g = fun_jac(x, *args)
                if not torch.isfinite(g).all():
                    raise RuntimeError("PBFGS: gradient g has NaN/inf.")
                err = self.wnorm(project(x - g, *proj_args) - x).item()
                if isinstance(f, torch.Tensor):
                    f = f.item()
                alpha = 1e-3  # initial step size for gradient descent
                x = x - alpha * g
                x = project(x, *proj_args)  # project again to make sure no negative values
                if not torch.isfinite(x).all():
                    raise RuntimeError("PBFGS: x has NaN/inf before fun_jac.")
                f_new, g_new = fun_jac(x, *args)
                if not torch.isfinite(g_new).all():
                    raise RuntimeError("PBFGS: gradient g has NaN/inf.")
                s = -alpha * g
                y = g_new - g
                r = y.clone()  # use y as initial r
                # Set delta, N, M_inv for the first iteration
                delta = (self.wdot(y,y) / self.wdot(y,s)).item() if self.wdot(y,s).item() > 0 else 1.0
                N = torch.zeros((len(x), 1), device=device, dtype=dtype)
                M_inv = torch.zeros((1, 1), device=device, dtype=dtype)
                # set reset to false
                reset = False
            else:
                f = f_new
                g = g_new

                err = self.wnorm(project(x - g, *proj_args) - x).item()
                if err < tol:
                    # converged
                    converged = True
                    # Callback
                    callback(x, f, err, k, converged, *args)
                    break
                
                try:
                    x_star, SPG_converged = self.SPG(
                        x0=x, f0=f, g0=g, delta=delta,
                        N=N, M_inv=M_inv, SPG_hist=10, project=project, proj_args=proj_args
                    )
                except:
                    # SPG failed; reset quasi-Hessian
                    S_list = []
                    R_list = []
                    reset = True
                    continue

                if SPG_converged == False:
                    # SPG did not converge likely due to poor conditioning; reset quasi-Hessian
                    S_list = []
                    R_list = []
                    reset = True
                    continue
                else:
                    x_star = project(x_star, *proj_args)  # project result to be safe
                    d = x_star - x

                    try:
                        alpha, f_temp, g_temp = self.line_min(
                            fun_jac=fun_jac, x0=x, f0=f, g0=g, p=d,
                            f_b=f, args=args, amax=1, c1=1e-4, maxiter=5
                        )
                        if (alpha is None) or (alpha < 0) or isinstance(alpha, complex):
                            # Line search failed. Reset quasi-Hessian
                            reset = True
                            S_list = []
                            R_list = []
                            continue
                    except:
                        # Line search failed. Reset quasi-Hessian
                        reset = True
                        S_list = []
                        R_list = []
                        continue

                    g_new = g_temp
                    f_new = f_temp
                    x = x + alpha * d
                    x = project(x, *proj_args)  # project again to make sure no negative values
                    if not torch.isfinite(x).all():
                        raise RuntimeError("PBFGS: x became NaN/inf after line search + projection.")
                    s = alpha * d
                    y = g_new - g
                    r = self.construct_r(delta, N, M_inv, s, y) # construct r for damped update

                    # Construct BFGS matrix
                    delta, N, M_inv, S_list, R_list = self.construct_B(delta, N, M_inv, s, r, S_list, R_list, BFGS_hist)

            # Callback
            callback(x, f, err, k, converged, *args)
            k += 1

        return f, x


    def construct_B(self, delta, N, M_inv, s, y, S_list, Y_list, m):
        """
        Construct the matrices needed to obtain the limited memory BFGS matrix.

        Args:
            delta, N, M_inv: Previous matrices
            s: Current step (descent direction)
            y: Gradient difference
            S_list: List of previous steps
            Y_list: List of previous gradient differences
            m: History length for function values
        Returns:
            delta, N, M_inv: Matrices used to update the BFGS matrix by B = delta - N @ np.linalg.inv(M) @ N.T
        """
        device = s.device
        dtype = s.dtype

        curv = self.wdot(s, y).item()
        yy   = self.wdot(y, y).item()
        if yy <= 0.0 or curv <= 1e-12 * yy:
            # just skip this (s,y) pair, keep previous delta, N, M_inv
            return delta, N, M_inv, S_list, Y_list
        
        device = s.device
        dtype = s.dtype
        if len(S_list) < m:
            S_list.append(s.clone())
            Y_list.append(y.clone())
        else:
            S_list.pop(0)
            S_list.append(s.clone())
            Y_list.pop(0)
            Y_list.append(y.clone())

        S = torch.stack(S_list, dim=1)
        Y = torch.stack(Y_list, dim=1)
        gamma = curv / yy
        delta = 1.0 / gamma

        L = torch.zeros((S.shape[1], S.shape[1]), device=device, dtype=dtype)
        for i in range(S.shape[1]):
            for j in range(S.shape[1]):
                if i > j:
                    L[i, j] = self.wdot(S_list[i], Y_list[j]).item()

        D = torch.zeros((S.shape[1], S.shape[1]), device=device, dtype=dtype)
        for i in range(S.shape[1]):
            D[i, i] = self.wdot(S_list[i], Y_list[i]).item()

        N = torch.cat([delta * S, Y], dim=1)
        A = delta * S.t().matmul(S)
        top = torch.cat([A, L], dim=1)
        bottom = torch.cat([L.t(), -D], dim=1)
        M = torch.cat([top, bottom], dim=0)
        M = M + 1e-8 * torch.eye(M.shape[0], device=device, dtype=dtype)     # regularize
        try:
            M_inv = torch.inverse(M)
        except RuntimeError:
            # Fallback: reset history if still singular
            S_list.clear()
            Y_list.clear()
            delta = 1.0
            N = torch.zeros((len(s), 0), device=device, dtype=dtype)
            M_inv = torch.zeros((0, 0), device=device, dtype=dtype)
            return delta, N, M_inv, S_list, Y_list

        return delta, N, M_inv, S_list, Y_list

    def construct_r(self, delta, N, M_inv, s, y, tiny=1e-12):
        """
        Construct the damped BFGS update version of y.
        Falls back to r = y if anything looks numerically sketchy.
        """
        # If we have no quasi-Hessian info, don't try to damp
        if N.numel() == 0 or M_inv.numel() == 0:
            return y.clone()

        # Compute B s
        t = N.t() @ s          # (m,)
        u = M_inv @ t          # (m,)
        v = N @ u              # (n,)
        Bs = delta * s - v     # (n,)

        # Inner products
        sy  = self.wdot(s, y)
        sBs = self.wdot(s, Bs)

        # Guard against NaNs/infs
        if (not torch.isfinite(sy)) or (not torch.isfinite(sBs)):
            return y.clone()

        # If s^T Bs is tiny, damping is meaningless – use y
        if sBs.abs() < tiny:
            return y.clone()

        # Damping parameter θ
        if sy >= 0.2 * sBs:
            theta = 1.0
        else:
            denom = sBs - sy
            if abs(denom) < tiny:
                # avoid 0/0 or huge θ
                theta = 1.0
            else:
                theta = 0.8 * sBs / denom

        r = theta * y + (1.0 - theta) * Bs

        # Final safety: if r is not finite, fall back
        if not torch.isfinite(r).all():
            return y.clone()

        return r

    def line_min(self, fun_jac, x0, f0, g0, p, f_b, args, amax=1, c1=0.0001, maxiter=10):
        """
        Perfoms line minimization to find optimal step size

        Args:
            fun_jac: objective function and gradient
            x0: current point
            f0: function value at x0
            g0: gradient value at x0
            p: search direction
            f_b: function value for armijio condition (typically f0)
            args: objective function arguments
            amax: maximum step size
            c1: Armijo condition parameter
            maxiter: maximum iterations

        Returns:
            a: best step size
            f: objective function at best step size
            g: objective gradient at best step size
        """
        a_list = [0.0, float(amax)]
        f_list = [f0]
        g_list = [g0]
        i = 1
        while i <= maxiter:
            x_new = x0 + a_list[i] * p
            if not torch.isfinite(x_new).all():
                raise RuntimeError("PBFGS: x has NaN/inf before fun_jac.")
            f_val, g_val = fun_jac(x_new, *args)
            if not torch.isfinite(g_val).all():
                raise RuntimeError("PBFGS: gradient g has NaN/inf.")
            if isinstance(f_val, torch.Tensor):
                f_val = f_val.item()
            f_list.append(f_val)
            g_list.append(g_val)

            if f_list[i] <= f_b + c1 * a_list[i] * self.wdot(g_list[0], p).item():
                return a_list[i], f_list[i], g_list[i]

            a_new = self.cubic_interp(
                a_list[i], f_list[i], self.wdot(g_list[i], p).item(),
                a_list[0], f_list[0], self.wdot(g_list[0], p).item()
            )

            if isinstance(a_new, complex) or a_new < 0:
                a_new = a_list[i] / 2.0 # just do back tracking if failure occurs
            a_list.append(a_new)
            i += 1

        return None, f_list[maxiter], g_list[maxiter]

    def cubic_interp(self, an, fn, gn, ao, fo, go):
        """
        Obtain new step size by cubic interpolation

        Args:
            an: newest step size
            fn: fun evaluated at newest step size
            gn: grad evaluated at newest step size
            ao: previous step size
            fo: fun evaluated at previous step size
            go: grad evaluated at previous step size
        """
        # If error occurs, return step size an
        try:
            d1 = go + gn - 3.0 * (fo - fn) / (ao - an)
            d2 = (1.0 if (an - ao) >= 0 else -1.0) * ((d1 ** 2 - go * gn) ** 0.5)
            res = an - (an - ao) * ((gn + d2 - d1) / (gn - go + 2.0 * d2))
        except:
            res = an
        return res


def density_filter(
    rho: torch.Tensor,
    rmin: float
) -> torch.Tensor:
    """
    3D SIMP-style density filter with periodic boundary conditions,
    generalized so that `rho` can have any number of leading dimensions.
    
    Args:
      rho    Tensor of shape (..., Nx, Ny, Nz), any device/dtype (float).
      rmin   positive float, filter radius (in grid-units).
    
    Returns:
      filtered Tensor of shape (..., Nx, Ny, Nz), same device/dtype as `rho`.
    
    The filter weights w(dx,dy,dz) = max(0, rmin - sqrt(dx^2+dy^2+dz^2)),
    for dx,dy,dz in [-R..R],  R = floor(rmin).  We normalize so that
    sum_{dx,dy,dz} w = 1.  Then we do a 3D convolution with `padding='circular'`.
    """
    # --- 1) Extract spatial dimensions ---
    #    The last three dims of `rho` are (Nx,Ny,Nz); anything before is "batch".
    *leading_shape, Nx, Ny, Nz = rho.shape
    
    # Compute integer radius R and kernel size K = 2R + 1
    R = math.floor(rmin)
    K = 2 * R + 1

    # --- 2) Build (K,K,K) grid of offsets and distances ---
    device = rho.device
    ar = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(ar, ar, ar, indexing='ij')  # each is (K,K,K)
    dist3d = torch.sqrt(xx**2 + yy**2 + zz**2)              # (K,K,K)

    # --- 3) Compute unnormalized weights: w = max(0, rmin - dist) ---
    w_unnorm = torch.clamp(rmin - dist3d, min=0.0)  # shape (K,K,K)
    wsum = w_unnorm.sum()
    if wsum.item() == 0:
        raise ValueError(f"No nonzero weights (rmin={rmin}, R={R}).")

    # --- 4) Normalize so that sum(w) = 1 ---
    w_kernel = (w_unnorm / wsum).to(rho.dtype)       # shape (K,K,K)

    # --- 5) Reshape into a conv3d kernel: (1, 1, K, K, K) ---
    # We will do a conv3d on input of shape (batch, 1, Nx, Ny, Nz) 
    # with weight=w_kernel.unsqueeze(0).unsqueeze(0).
    w_kernel = w_kernel.unsqueeze(0).unsqueeze(0)   # now (1,1,K,K,K)

    # --- 6) Flatten all leading dims into one batch dimension ---
    #     If leading_shape = [A,B,C], then batch_size = A*B*C.
    #     If there are no leading dims, batch_size = 1.
    batch_size = math.prod(leading_shape)  # math.prod(()) == 1
    # Now reshape rho into shape (batch_size, Nx, Ny, Nz).
    rho_flat = rho.view(batch_size, Nx, Ny, Nz)

    # --- 7) Add channel dim → shape (batch_size, 1, Nx, Ny, Nz) ---
    rho_in = rho_flat.unsqueeze(1)

    # --- 8) Circularly pad by R on each side ---
    # pad format = (W_left, W_right, H_left, H_right, D_left, D_right)
    pad = (R, R, R, R, R, R)
    rho_padded = F.pad(rho_in, pad=pad, mode='circular')

    # --- 9) Convolve with stride=1 ---
    # Input channels=1, out channels=1, so conv3d output = (batch_size, 1, Nx, Ny, Nz).
    filtered_flat = F.conv3d(rho_padded, w_kernel, stride=1)

    # --- 10) Remove the channel dim → shape (batch_size, Nx, Ny, Nz) ---
    filtered_flat = filtered_flat.squeeze(1)

    # --- 11) Reshape back to original leading dims: (..., Nx, Ny, Nz) ---
    filtered = filtered_flat.view(*leading_shape, Nx, Ny, Nz)
    return filtered