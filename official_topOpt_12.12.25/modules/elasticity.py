import os, math, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.algorithms import PCG

torch.set_default_dtype(torch.float64)

def get_stress_strain(u, bulk, shear, K, E, metric):
    u_stack = u.permute(3,0,1,2)                           # u_stack is shape (3,Nx,Ny,Nz) since u is shape (Nx,Ny,Nz,3)
    u_hat   = torch.fft.rfftn(u_stack, dim=(1,2,3))
    Kc = K.to(torch.complex128)  # ensure K is complex for multiplication
    Lattice_inv_C = metric.Lattice_inv.to(torch.complex128)  # ensure Lattice_inv is complex for multiplication
    d_u_hat = -1j * 0.5 * (torch.einsum('kxyz,ki,jxyz->ijxyz', Kc, Lattice_inv_C, u_hat) + torch.einsum('kxyz,kj,ixyz->ijxyz', Kc, Lattice_inv_C, u_hat))
    d_u     = torch.fft.irfftn(d_u_hat, s=u.shape[:3], dim=(2,3,4))
    
    # fill strain tensor
    strain = torch.zeros((3,3,*u.shape[:3]), dtype=torch.float64, device=bulk.device)
    strain[0,0] = E[0,0] + d_u[0,0]
    strain[1,1] = E[1,1] + d_u[1,1]
    strain[2,2] = E[2,2] + d_u[2,2]
    for i,j in ((0,1),(0,2),(1,2)):
        strain[i,j] = E[i,j] + d_u[i,j]
        strain[j,i] = strain[i,j]

    Id = torch.eye(3, device=u.device)
    tr = torch.einsum('kkxyz->xyz', strain)
    stress =  ((bulk - 2/3*shear) * torch.einsum('ij,xyz->ijxyz', Id, tr)
            + shear*(strain + strain.permute(1,0,2,3,4)))
    
    return stress, strain

def get_residual(x, bulk, shear, K, E, dx, metric):
    u      = x.reshape(*bulk.shape, 3)  # [Nx,Ny,Nz,3]
    stress, strain = get_stress_strain(u, bulk, shear, K, E, metric)
    stress_hat   = torch.fft.rfftn(stress, dim=(2,3,4))
    Kc = K.to(torch.complex128)  # ensure K is complex for multiplication
    Lattice_inv_C = metric.Lattice_inv.to(torch.complex128)  # ensure Lattice_inv is complex for multiplication
    div_stress_hat = -1j * torch.einsum('kxyz,ki,ijxyz->jxyz', Kc, Lattice_inv_C, stress_hat)
    div_stress = torch.fft.irfftn(div_stress_hat, s=u.shape[:3], dim=(1,2,3))  # [3,Nx,Ny,Nz]
    residual = (dx**2)*(-1) * div_stress.permute(1,2,3,0).reshape(-1) # [Nx*Ny*Nz*3,]
    mean_stress = torch.mean(stress, dim=(2,3,4))

    # build 4th-order C for this E
    C = torch.zeros((3,3,3,3), dtype=torch.float64, device=u.device)
    # off-diagonals
    if E[0,1]==1:
        C[:, :, 0,1] = mean_stress/2; C[:,:,1,0] = mean_stress/2
    elif E[0,2]==1:
        C[:, :, 0,2] = mean_stress/2; C[:,:,2,0] = mean_stress/2
    elif E[1,2]==1:
        C[:, :, 1,2] = mean_stress/2; C[:,:,2,1] = mean_stress/2
    # diagonals
    elif E[0,0]==1:
        C[:,:,0,0] = mean_stress
    elif E[1,1]==1:
        C[:,:,1,1] = mean_stress
    elif E[2,2]==1:
        C[:,:,2,2] = mean_stress

    return residual, C, strain


def solve_elasticity(bulk, shear, outfile, metric, u0_dict=None, parallel=True):
    """
    bulk   : Tensor [Nx,Ny,Nz]
    shear  : Tensor [Nx,Ny,Nz]
    outfile: path for logs
    metric: object with lattice info
    u0_dict: initial guess for each component, dict with keys 'xy', 'xz', 'yz', 'xx', 'yy', 'zz'
    parallel: if True, on CPU uses 6 threads
    """
    # choose device
    device = bulk.device

    # prep
    bulk  = bulk.to(torch.float64).to(device)
    shear = shear.to(torch.float64).to(device)
    N = torch.tensor(bulk.shape, device=device)
    dx, dy, dz = 1 / N

    # wave-numbers
    kx = 2*math.pi*torch.fft.fftfreq(N[0], d=dx, device=device)
    ky = 2*math.pi*torch.fft.fftfreq(N[1], d=dy, device=device)
    kz = 2*math.pi*torch.fft.rfftfreq(N[2], d=dz, device=device)
    KX, KY, KZ = torch.meshgrid(kx,ky,kz, indexing='ij')
    K = torch.stack((KX, KY, KZ), dim=0)

    # preconditioner M_inv
    const = 1.0
    Id = torch.eye(3, device=device)
    C_const = 1/3 * torch.einsum('ij,kl->ijkl', Id, Id) + torch.einsum('ik,jl->ijkl', Id, Id) + torch.einsum('il,jk->ijkl', Id, Id)
    M = torch.einsum('ijkl,ni,mk,nxyz,mxyz->xyzjl', C_const, metric.Lattice_inv, metric.Lattice_inv, K, K) + const * Id.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    M_inv = torch.linalg.inv((dx**2)*M).to(torch.complex128)

    def apply_precond(r_flat):
        r = r_flat.view(*bulk.shape,3)
        r_hat = torch.fft.rfftn(r, dim=(0,1,2))
        z_hat = torch.einsum('...ij,...j->...i', M_inv, r_hat)
        z = torch.fft.irfftn(z_hat, s=tuple(bulk.shape), dim=(0,1,2))
        return z.reshape(-1)

    comps = ['xy','xz','yz','xx','yy','zz']
    results = {}
    u_dict = {}

    def _solve(cmp):
        # build E
        E = torch.zeros((3,3), dtype=torch.float64, device=device)
        if cmp=='xy': E[0,1]=E[1,0]=1
        elif cmp=='xz': E[0,2]=E[2,0]=1
        elif cmp=='yz': E[1,2]=E[2,1]=1
        elif cmp=='xx': E[0,0]=1
        elif cmp=='yy': E[1,1]=1
        elif cmp=='zz': E[2,2]=1

        with open(outfile,'a') as f:
            print(f'[{cmp}] starting PCG...', file=f)

        # initial guess
        if u0_dict is None:
            u0 = torch.randn(*bulk.shape, 3, dtype=torch.float64, device=device)
        else:
            u0 = u0_dict[cmp]

        x0 = u0.flatten()
        u = PCG(x0,
                args=(bulk, shear, K, E, dx, metric),
                residual_fun=get_residual,
                apply_precond=apply_precond,
                outfile=outfile).reshape(u0.shape)

        Cc, strain = get_residual(u, bulk, shear, K, E, dx, metric)[1:]

        u_dict[cmp] = u
        with open(outfile,'a') as f:
            print(f'[{cmp}] done.', file=f)
        return cmp, Cc, strain

    if device.type=='cpu' and parallel:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_solve, c):c for c in comps}
            for f in as_completed(futs):
                cmp, Cc, strain = f.result()
                results[cmp] = [Cc, strain]
    else:
        for c in comps:
            cmp, Cc, strain = _solve(c)
            results[cmp] = [Cc, strain]

    # assemble & symmetrize
    C = sum(v[0] for v in results.values())
    C = 0.25*(C
              + C.permute(1,0,2,3)
              + C.permute(0,1,3,2)
              + C.permute(1,0,3,2))
    # then enforce major symmetry
    C = 0.5*(C + C.permute(2,3,0,1))
    
    return C, results, u_dict
