import torch
import numpy as np
import cvxpy as cp

class TranslationMinimizer:
    """
    TranslationMinimizer class implements optimization algorithm for minimizing functions associated with Translation method using projected limited memory BFGS method.
    This class provides methods to perform optimization on a given function with its gradient. The class is designed to handle constraints by projecting 
    the iterates back onto the feasible set.
    Methods:
        PBFGS(fun_jac, x0, BFGS_hist, args, proj_args, callback, project, maxit=100, tol=1e-10):
            Performs projected limited memory BFGS optimization.
        _construct_B(s, y, S_list, Y_list):
            Constructs the matrices needed to obtain the limited memory BFGS matrix.
        _construct_r(delta, N, M_inv, s, y):
            Constructs the damped BFGS update version of the gradient difference.
        _solve_local(x0, f0, g0, delta, N, M_inv, SPG_hist, maxit=1000, tol=1e-3):
            Solves local convex quadratic subproblem subject to constraints using CVXPY.
        _project(x0):
            Projects input x0 onto the feasible set using CVXPY.
        _line_min(fun_jac, x0, f0, g0, p, f_b, args, amax=1, c1=0.0001, maxiter=10):
            Performs line minimization to find the optimal step size.
        _cubic_interp(an, fn, gn, ao, fo, go):
            Obtains a new step size by cubic interpolation.
    """

    def __init__(self, fun_jac, x0, args, phase_props, plotter, logger, helper, callback=None, BFGS_hist=50, maxit=100, tol=1e-10, max_line_search=10):
        """
        Initializes the minimizer class.

        Attributes:
            fun_jac: Function to minimize (along with its gradient)
            x0: Initial guess
            args: Extra args passed to fun and jac
            phase_props: Microscopic phase properties for materials
            plotter: Plotter object for plotting results
            logger: Logger object for logging information
            helper: Translation helper class
            callback: Callback function for displaying information at each iteration
            BFGS_hist: History length for function values
            maxit: Maximum iterations
            tol: Tolerance of convergence
            max_line_search: Maximum line search iterations
        """
        self.fun_jac = fun_jac
        self.x0 = x0
        self.BFGS_hist = BFGS_hist
        self.args = args
        self.callback = callback
        self.maxit = maxit
        self.tol = tol
        self.max_line_search = max_line_search
        self.helper = helper
        self.phase_props = phase_props
        self.plotter = plotter
        self.logger = logger
        self.device = x0.device
        self.dtype = x0.dtype
    
    def PBFGS(self):
        """
        Perform projected limited memory BFGS.
        """
        device, dtype = self.x0.device, self.x0.dtype
        self.logger.specific_log("Using device: " + str(device))
        x = self.x0.clone()
        f_new = None
        g_new = None
        k = 1
        S_list = []
        R_list = []
        converged = False
        reset = True  # flag to indicate if we need to reset the quasi-Hessian (or begin optimization)

        # Initial projection of input
        self.logger.specific_log("Projecting initial guess onto feasible set...") 
        x = self._project(x)[0]
        # Run optimization loop
        self.logger.specific_log("Starting optimization with projected L-BFGS...")
        while k <= self.maxit:
            self.logger.specific_log(f"\nIteration {k}")
            if reset == True:
                # Perform gradient descent step on first iteration to initialize delta
                f, g = self.fun_jac(x, *self.args)
                err = torch.norm(self._project(x - g)[0] - x).item()
                if isinstance(f, torch.Tensor):
                    f = f.item()
                alpha = 1e-3  # initial step size for gradient descent (TODO: optimize this automatically instead of hardcoding)
                x = x - alpha * g
                x = self._project(x)[0]  # project again to make sure no negative values
                f_new, g_new = self.fun_jac(x, *self.args)
                s = -alpha * g
                y = g_new - g
                r = y.clone()  # use y as initial r
                # Set delta, N, M_inv for the first iteration
                delta = (y.dot(y) / y.dot(s)).item() if y.dot(s).item() > 0 else 1.0
                N = torch.zeros((len(x), 1), device=device, dtype=dtype)
                M_inv = torch.zeros((1, 1), device=device, dtype=dtype)
                # set reset to false
                reset = False
            else:
                f = f_new
                g = g_new

                err = torch.norm(self._project(x - g)[0] - x).item()
                if err < self.tol:
                    # converged
                    converged = True
                    break
                
                x_star, CVXPY_status = self._solve_local(
                    x0=x, f0=f, g0=g, delta=delta,
                    N=N, M_inv=M_inv
                )

                if CVXPY_status not in ('optimal', 'optimal_inaccurate'):
                    # CVXPY did not converge likely due to poor conditioning; reset quasi-Hessian
                    self.logger.specific_log("     CVXPY failed to solve local QP; resetting quasi-Hessian and performing gradient descent step...")
                    S_list = []
                    R_list = []
                    reset = True
                    continue
                else:
                    x_star = self._project(x_star)[0]  # project result to be safe
                    d = x_star - x

                    alpha, f_temp, g_temp = self._line_min(
                        fun_jac=self.fun_jac, x0=x, f0=f, g0=g, p=d,
                        f_b=f, args=self.args, amax=1, c1=1e-4, 
                        maxiter=self.max_line_search,
                    )
                    if (alpha is None) or (alpha < 0) or isinstance(alpha, complex):        # Line search failed.
                        alpha = 1 # fluke case when optimization has bad conditioning, assume large alpha to get out of there

                    g_new = g_temp
                    f_new = f_temp
                    x = x + alpha * d
                    x = self._project(x)[0]  # project again to make sure no negative values
                    s = alpha * d
                    y = g_new - g
                    r = self._construct_r(delta, N, M_inv, s, y) # construct r for damped update

                    # Construct BFGS matrix
                    delta, N, M_inv, S_list, R_list = self._construct_B(delta, N, M_inv, s, r, S_list, R_list)
            # Plot and print results
            self.plotter.plot(x, k)
            self.logger.normal_log(k, f_new, err, converged, x)
            f = f_new
            g = g_new
            k += 1
        if not converged:
            self.logger.specific_log("\nOptimization reached maximum number of iterations.")
        else:
            self.logger.specific_log(f"\nOptimization converged within tol = {self.tol}.")
        self.logger.normal_log(k, f_new, err, converged, x)

        return f, x

    def _construct_B(self, delta, N, M_inv, s, y, S_list, Y_list):
        """
        Construct the matrices needed to obtain the limited memory BFGS matrix.

        Args:
            s: Current step (descent direction)
            y: Gradient difference
            S_list: List of previous steps
            Y_list: List of previous gradient differences
        Returns:
            delta, N, M_inv: Matrices used to update the BFGS matrix by B = delta - N @ inv(M) @ N.T
        """

        device = s.device
        dtype = s.dtype

        curv = torch.dot(s, y).item()
        yy   = torch.dot(y, y).item()
        if yy <= 0.0 or curv <= 1e-12 * yy:
            # just skip this (s,y) pair, keep previous delta, N, M_inv
            return delta, N, M_inv, S_list, Y_list
        
        device = s.device
        dtype = s.dtype
        if len(S_list) < self.BFGS_hist:
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
                    L[i, j] = torch.dot(S_list[i], Y_list[j]).item()

        D = torch.zeros((S.shape[1], S.shape[1]), device=device, dtype=dtype)
        for i in range(S.shape[1]):
            D[i, i] = torch.dot(S_list[i], Y_list[i]).item()

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

    def _construct_r(self, delta, N, M_inv, s, y, tiny=1e-12):
        """
        Construct the damped BFGS update version of y.
        Falls back to r = y if anything looks numerically sketchy.

        Args:
            delta, N, M_inv: Necessary to build quasi-Hessian
            s: Current step (descent direction)
            y: Gradient difference
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
        sy  = torch.dot(s, y)
        sBs = torch.dot(s, Bs)

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
    
    def _project(self, x0, solver="MOSEK", verbose=False):
        """
        Project input x0 onto feasible set using CVXPY.
        Minimize ||x - x0||^2
        s.t. T == T^T
                B_k^T T B_k  >>  0       for k=1..n_samples   (quasi-convexity)
                D_i - T      >>  0       for i=1,2            (phase bounds)
        x is vec(T) with T in R^{15x15} (symmetric).
        """
        def _torch_to_numpy(t):
            return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

        # ---------- Torch -> NumPy ----------
        x0 = _torch_to_numpy(x0).reshape(-1)

        # Sizes
        nT = int(np.sqrt(x0.shape[0]))                   # T is 15x15

        D1 = self.helper.D1
        D2 = self.helper.D2

        # ---------- Decision variable ----------
        T = cp.Variable((nT, nT), symmetric=True)
        x = cp.vec(T, order='C')  # vec(T) \in R^{nvec}

        # ---------- Objective ----------
        obj = 0.5 * cp.sum_squares(x - x0)

        # ---------- Constraints ----------
        cons = []

        # Phase LMIs: D_i - T  ⪰  eps*I
        cons += [ (D1 - T) >> 0,
                (D2 - T) >> 0 ]

        # Quasi-convexity / translated positivity LMIs:
        for B in self.helper.Bs:
            Xk = B.T @ T @ B
            cons.append( Xk >> 0 )

        # ---------- Solve ----------
        print("         Projecting onto feasible set using CVXPY...")
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=solver, verbose=verbose)
        if prob.status in ('optimal', 'optimal_inaccurate'):
            print(f"         Projection successful.")
        else:
            print(f"         Projection failed with status {prob.status}.")

        return torch.tensor(x.value.reshape(-1), dtype=self.dtype, device=self.device), prob.status

    def _solve_local(self,
        x0, f0, g0, delta, N, M_inv,
        solver="MOSEK", verbose=False
    ):
        """
        Minimize  0.5*(x-x0)^T H (x-x0) + g0^T (x-x0) + f0
        s.t.  T == T^T
                B_k^T T B_k  >>  0       for k=1..n_samples   (quasi-convexity)
                D_i - T      >>  0       for i=1,2            (phase bounds)

        x is vec(T) with T in R^{15x15} (symmetric).
        """
        def _torch_to_numpy(t):
            return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

        def _make_psd(H, eps=1e-12):
            """Symmetrize and project to PSD (shift tiny negative eigs to zero)."""
            H = 0.5 * (H + H.T)
            w, V = np.linalg.eigh(H)
            w = np.maximum(w, 0.0)
            # add a tiny ridge so quad_form is strictly convex on the range
            return (V * w) @ V.T + eps * np.eye(H.shape[0])

        # ---------- Torch -> NumPy ----------
        x0 = _torch_to_numpy(x0).reshape(-1)
        g0 = _torch_to_numpy(g0).reshape(-1)
        delta = float(delta) if np.ndim(delta) == 0 else float(_torch_to_numpy(delta))
        N = _torch_to_numpy(N)      # shape: (n, m)
        M_inv = _torch_to_numpy(M_inv)  # shape: (m, m)

        # Sizes
        nT = int(np.sqrt(x0.shape[0]))                   # T is 15x15
        nvec = nT * nT              # vec(T) length

        # ---------- L-BFGS compact Hessian ----------
        # H = delta*I - N M_inv N^T
        H = delta * np.eye(nvec)
        if N.size > 0:
            # N is defined in the vector space of x = vec(T)
            H -= N @ (M_inv @ N.T)
        H = _make_psd(H)            # ensure PSD for cvxpy.dcp

        # ---------- Phase properties ----------
        D1 = self.helper.D1
        D2 = self.helper.D2

        # ---------- Decision variable ----------
        T = cp.Variable((nT, nT), symmetric=True)
        x = cp.vec(T, order='C')  # vec(T) \in R^{nvec}

        # ---------- Objective ----------
        obj = 0.5 * cp.quad_form(x - x0, H) + g0 @ (x - x0) + f0

        # ---------- Constraints ----------
        cons = []

        # Phase LMIs: D_i - T  ⪰  eps*I
        cons += [ (D1 - T) >> 0,
                (D2 - T) >> 0 ]

        # Quasi-convexity / translated positivity LMIs:
        for B in self.helper.Bs:
            Xk = B.T @ T @ B
            cons.append( Xk >> 0 )

        # ---------- Solve ----------
        print("         Solving local QP subproblem using CVXPY...")
        prob = cp.Problem(cp.Minimize(obj), cons)
        try:
            prob.solve(solver=solver, verbose=verbose)
        except:
            # Treat this as "quasi-Hessian is bad" and let PBFGS reset it.
            print(f"         Local QP solve raised an error.")
            print("         Treating this as Hessian failure and asking PBFGS to reset.")
            # Return x0 and a special status string
            return (
                torch.tensor(x0.reshape(-1), dtype=self.dtype, device=self.device),
                "DCPError",
            )
        if prob.status in ('optimal', 'optimal_inaccurate'):
            print(f"         Local QP solve successful.")
            return (
                torch.tensor(x.value.reshape(-1), dtype=self.dtype, device=self.device),
                prob.status,
            )
        else:
            print(f"         Local QP solve failed with status {prob.status}.")
            return (
                torch.tensor(x0.reshape(-1), dtype=self.dtype, device=self.device),
                prob.status,
            )

    def _line_min(self, fun_jac, x0, f0, g0, p, f_b, args, amax=1, c1=0.0001, maxiter=10):
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
            verbal: whether to print information

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
            self.logger.specific_log(f"     Line search iteration {i} with step size {a_list[i]:.4e}...")
            x_new = x0 + a_list[i] * p
            f_val, g_val = fun_jac(x_new, *args)
            if isinstance(f_val, torch.Tensor):
                f_val = f_val.item()
            f_list.append(f_val)
            g_list.append(g_val)

            if f_list[i] <= f_b + c1 * a_list[i] * torch.dot(g_list[0], p).item():
                self.logger.specific_log("     Line search successful.")
                return a_list[i], f_list[i], g_list[i]

            a_new = self._cubic_interp(
                a_list[i], f_list[i], torch.dot(g_list[i], p).item(),
                a_list[0], f_list[0], torch.dot(g_list[0], p).item()
            )
            if isinstance(a_new, complex) or a_new < 0:
                a_new = a_list[i] / 2.0 # just do back tracking if failure occurs
            a_list.append(a_new)
            i += 1

        self.logger.specific_log("     Line search failed.")

        return None, f_list[maxiter], g_list[maxiter]

    def _cubic_interp(self, an, fn, gn, ao, fo, go):
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