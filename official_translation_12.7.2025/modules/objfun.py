import numpy as np
import torch
from scipy.optimize import brentq


def objfun(t, helper):
    """
    Example objective:
      average of lambda_min over the region where lambda_min >= 0,
      and its gradient wrt T (same as wrt A because A = Y(z)+T).

    Parameters
    ----------
    t : ndarray, shape (n*n,)
        Flattened symmetric T (we will reshape to (n, n)).

    Returns
    -------
    f : float
        Objective value: mean of lambda_min over feasible (>=0) nodes. If no feasible nodes, returns 0.
    gT : ndarray, shape (n, n)
        Gradient wrt T: mean projector over feasible nodes (zeros if no feasible nodes).
    """

    t = np.array(t.detach().cpu())
    n = int(np.sqrt(t.size))
    T = t.reshape(n, n)
    # Ensure symmetry (robustness if t is slightly off)
    T = 0.5 * (T + T.T)

    # Get diagonal matrices needed
    D_k = helper.D_k
    D_mu = helper.D_mu
    D_sig = helper.D_sig
    ZHIGH = 100 # something large so it doesn't affect the bound
    alphas = helper.alphas
    if helper.calc_type == 'bulk':
        zmu = ZHIGH 
        A = D_mu * zmu
        B = D_k[:, :, None] + alphas[None, None, :] * D_sig[:, :, None]
    elif helper.calc_type == 'shear':
        zk = ZHIGH
        A = D_k * zk
        B = D_mu[:, :, None] + alphas[None, None, :] * D_sig[:, :, None]
    
    # Solve for where minimum eigenvalue equals zero (at z_sol)
    def nonlinfun(x, A, B, T):
        eigvals, eigvecs = np.linalg.eigh(A + B * x + T)
        lam_min = eigvals[0]
        vmin   = eigvecs[:, 0]  
        return lam_min

    z_sols = np.zeros(len(alphas))
    for i in range(len(alphas)):
        Bi = B[:, :, i]
        z_sol, r = brentq(f=nonlinfun, args=(A, Bi, T), a=0.0, b=500.0, xtol=1e-10, maxiter=100, disp=True, full_output=True)
        print(f'alpha {alphas[i]}: Found z_sol = ', z_sol, ' with r.converged = ', r.converged)
        z_sols[i] = z_sol

    f = -np.mean(z_sols) # we maximize sum of z_sol's so need neg sign since our optimizer minimizes

    # Get gradient of objective function
    dfs = np.zeros((n, n, len(alphas)))
    for i in range(len(alphas)):
        z_sol = z_sols[i]
        Bi = B[:, :, i]
        eigvals, eigvecs = np.linalg.eigh(A + Bi * z_sol + T)
        lam_min = eigvals[0]
        vmin   = eigvecs[:, 0]
        vv = vmin[:, None] * vmin[None, :]
        gT = -vv / (vmin.T @ Bi @ vmin)
        dfs[:, :, i] = -gT # since f has negative sign
    df = np.mean(dfs, axis=2)

    return f, torch.tensor(df.flatten(), dtype=torch.double)