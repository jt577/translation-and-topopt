import os
import math
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.algorithms import PCG

torch.set_default_dtype(torch.float64)

def _build_k_grids(N, device, dtype=torch.float64):
    dx, dy, dz = (1 / N)
    kx = 2*math.pi * torch.fft.fftfreq(N[0], d=dx, device=device, dtype=dtype)
    ky = 2*math.pi * torch.fft.fftfreq(N[1], d=dy, device=device, dtype=dtype)
    kz = 2*math.pi * torch.fft.rfftfreq(N[2], d=dz, device=device, dtype=dtype)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K = torch.stack((KX, KY, KZ), dim=0)
    return K, dx

def _make_preconditioner(N, K, metric):
    const = 1.0
    k_g_inv_k = torch.einsum('ixyz,ij,jxyz->xyz', K, metric.g_inv, K) # account for cell shape in laplacian
    denom = (k_g_inv_k + const)
    def apply_precond(r_flat):
        r = r_flat.view(N[0],N[1],N[2])
        z_hat = torch.fft.rfftn(r, dim=(0,1,2)) / denom
        z = torch.real(torch.fft.irfftn(z_hat, s=(N[0],N[1],N[2]), dim=(0,1,2)))
        return z.reshape(-1)
    return apply_precond

def get_residual(x, sigma, K, cond_comp, dx, metric):
    phi_tilde = x.reshape(sigma.shape)
    phi_tilde_hat = torch.fft.rfftn(phi_tilde, dim=(0,1,2))
    dphi_tilde_hat = torch.stack([-1j*K[0]*phi_tilde_hat, -1j*K[1]*phi_tilde_hat, -1j*K[2]*phi_tilde_hat], dim=0)
    dphi_tilde = torch.fft.irfftn(dphi_tilde_hat, s=sigma.shape, dim=(1,2,3))

    # build E in lab frame
    if cond_comp == 'xx':
        E = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=sigma.device)
    elif cond_comp == 'yy':
        E = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=sigma.device)
    elif cond_comp == 'zz':
        E = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=sigma.device)

    # build dphi in lab frame
    dphi = -E.view(-1, 1, 1, 1) + torch.einsum('ij,jxyz->ixyz', metric.Lattice_inv.T, dphi_tilde)

    # build J in lab frame
    J = -sigma.unsqueeze(0) * dphi

    # Find lab frame divergence of J_lab
    Lattice_inv_J = torch.einsum('ij,jxyz->ixyz', metric.Lattice_inv, J)
    div_J = dx**2 * torch.fft.irfftn(-1j*(K[0]*torch.fft.rfftn(Lattice_inv_J[0], dim=(0,1,2)) +
                                            K[1]*torch.fft.rfftn(Lattice_inv_J[1], dim=(0,1,2)) +
                                            K[2]*torch.fft.rfftn(Lattice_inv_J[2], dim=(0,1,2))),
                                    s=sigma.shape, dim=(0,1,2))
    residual = div_J.flatten()

    # Find conductivity tensor in lab frame
    idx = {'xx':0,'yy':1,'zz':2}[cond_comp]
    eff = torch.zeros((3,3), dtype=torch.float64, device=sigma.device)
    eff[idx, 0] = torch.mean(J[0])
    eff[idx, 1] = torch.mean(J[1])
    eff[idx, 2] = torch.mean(J[2])
    
    return residual, eff, dphi


def solve_conductivity(sigma, outfile, metric, phi0_dict=None, parallel=True):
    """
    sigma    : 3D torch.Tensor on CPU
    outfile  : path to append logs to
    metric   : metric object with lattice related attributes
    phi0_dict: dictionary of initial guesses for each direction, keys are 'xx', 'yy', 'zz'
    parallel : if True, on CPU will spawn 3 threads
    """
    # prep data
    device = sigma.device
    sigma = sigma.to(torch.float64)
    # --- sanity check ---
    if not torch.isfinite(sigma).all():
        raise RuntimeError("solve_conductivity: sigma has NaN/inf before PCG.")
    N     = torch.tensor(sigma.shape, device=device)
    K, dx = _build_k_grids(N, device)
    apply_precond = _make_preconditioner(N, K, metric)

    directions = ['xx','yy','zz']
    phi_dict = {}
    results = {}

    def _solve_one(cond_comp):
        with open(outfile, 'a') as f:
            print(f"[{cond_comp}] starting PCG...", file=f)
        # Initial guess
        if phi0_dict is None:
            phi0 = torch.randn(*sigma.shape, dtype=torch.float64, device=device)
        else:
            phi0 = phi0_dict[cond_comp] # phi0 is a dictionary of initial guesses
        x0 = phi0.flatten()
        phi = PCG(x0,
                  args=(sigma, K, cond_comp, dx, metric),
                  residual_fun=get_residual,
                  apply_precond=apply_precond,
                  outfile=outfile).reshape(phi0.shape)

        _, eff, dphi = get_residual(phi, sigma, K, cond_comp, dx, metric)

        # store resulting phi in dictionary
        phi_dict[cond_comp] = phi
        with open(outfile, 'a') as f:
            print(f"[{cond_comp}] done.", file=f)
        return cond_comp, eff, dphi

    if device.type=='cpu' and parallel:
        with ThreadPoolExecutor(max_workers=3) as exec:
            futures = { exec.submit(_solve_one, d): d for d in directions }
            for fut in as_completed(futures):
                comp, eff, dphi = fut.result()
                results[comp] = [eff, dphi]
    else:
        # serial on GPU (or if parallel=False)
        for d in directions:
            comp, eff, dphi = _solve_one(d)
            results[comp] = [eff, dphi]

    # assemble and symmetrize
    Sigma = results['xx'][0] + results['yy'][0] + results['zz'][0]
    Sigma = 0.5*(Sigma + Sigma.T)

    return Sigma, results, phi_dict
