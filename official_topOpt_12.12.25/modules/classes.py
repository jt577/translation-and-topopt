import torch
import numpy as np
import os
import math
import shutil
from modules.conductivity import solve_conductivity
from modules.elasticity import solve_elasticity
from modules.algorithms import density_filter
from modules.dump import projection, isosurface
from modules.seed import SEED
from scipy.optimize import brentq, least_squares

class INFILEclass:
    """
    Class to hold input file parameters.
    """
    def __init__(self, infile_path, out_dir_path):
        self.gridShape = None 
        self.rmin = None            
        self.SIMP_exp = None        
        self.volFrac = None  
        self.k1 = None
        self.k2 = None
        self.mu1 = None
        self.mu2 = None
        self.sig1 = None
        self.sig2 = None      
        self.weight_conductivity = None
        self.weight_bulk = None
        self.weight_shear = None
        self.cutoff = None
        self.maxit = None     
        self.tol = None     
        self.hist = None
        self.density_file = None
        self.dumpFreq = None
        self.Lattice = None
        self.ucparams = None
        self.run_type = 'opt'
        self.dir_name = out_dir_path
        self.infile_path = infile_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = None
        self.invert = False

        with open(self.infile_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line:
                split_line = stripped_line.split()
                if split_line[0] == 'gridShape':
                    self.gridShape = [int(split_line[1]), int(split_line[2]), int(split_line[3])]
                elif split_line[0] == 'rmin':
                    self.rmin = int(split_line[1])
                elif split_line[0] == 'SIMP_exp':
                    self.SIMP_exp = [float(split_line[1]), float(split_line[2]), float(split_line[3]), float(split_line[4]), float(split_line[5])]
                elif split_line[0] == 'volFrac':
                    self.volFrac = float(split_line[1])
                elif split_line[0] == 'weight_conductivity':
                    self.weight_conductivity = float(split_line[1])
                elif split_line[0] == 'weight_bulk':
                    self.weight_bulk = float(split_line[1])
                elif split_line[0] == 'weight_shear':
                    self.weight_shear = float(split_line[1])
                elif split_line[0] == 'k1':
                    self.k1 = float(split_line[1])
                elif split_line[0] == 'k2':
                    self.k2 = float(split_line[1])
                elif split_line[0] == 'mu1':
                    self.mu1 = float(split_line[1])
                elif split_line[0] == 'mu2':
                    self.mu2 = float(split_line[1])
                elif split_line[0] == 'sig1':
                    self.sig1 = float(split_line[1])
                elif split_line[0] == 'sig2':
                    self.sig2 = float(split_line[1])
                elif split_line[0] == 'cutoff':
                    self.cutoff = float(split_line[1])
                elif split_line[0] == 'maxit':
                    self.maxit = int(split_line[1])
                elif split_line[0] == 'tol':
                    self.tol = float(split_line[1])
                elif split_line[0] == 'hist':
                    self.hist = int(split_line[1])
                elif split_line[0] == 'density_file':
                    self.density_file = split_line[1]
                elif split_line[0] == 'dumpFreq':
                    self.dumpFreq = int(split_line[1])
                elif split_line[0] == 'Lattice':
                    self.Lattice = torch.zeros((3, 3), device=self.device, dtype=torch.float64) 
                    split_line1 = lines[i+1].strip().split()
                    split_line2 = lines[i+2].strip().split()
                    split_line3 = lines[i+3].strip().split()
                    self.Lattice[0, :] = torch.tensor([float(split_line1[0]), float(split_line1[1]), float(split_line1[2])], device=self.device, dtype=torch.float64)
                    self.Lattice[1, :] = torch.tensor([float(split_line2[0]), float(split_line2[1]), float(split_line2[2])], device=self.device, dtype=torch.float64)
                    self.Lattice[2, :] = torch.tensor([float(split_line3[0]), float(split_line3[1]), float(split_line3[2])], device=self.device, dtype=torch.float64)
                elif split_line[0] == 'ucparams':
                    a = float(split_line[1])
                    b = float(split_line[2])
                    c = float(split_line[3])
                    alpha = float(split_line[4])
                    beta = float(split_line[5])
                    gamma = float(split_line[6])
                    self.ucparams = torch.tensor([a, b, c, alpha, beta, gamma], device=self.device, dtype=torch.float64)

                elif split_line[0] == 'run_type':
                    self.run_type = split_line[1]
                elif split_line[0] == 'seed':
                    self.seed = split_line[1]
                    # Safely check for the optional third token
                    if len(split_line) > 2 and split_line[2] == 'invert':
                        self.invert = True

        # when doing topology optimization
        if self.run_type == 'opt':
            if self.gridShape is None:
                raise ValueError("gridShape is missing")
            if self.rmin is None:
                raise ValueError("rmin is missing")
            if self.SIMP_exp is None:
                raise ValueError("SIMP_exp is missing")
            if self.volFrac is None:
                raise ValueError("volFrac is missing")
            if self.weight_conductivity is None:
                raise ValueError("weight_conductivity is missing")
            if self.weight_bulk is None:
                raise ValueError("weight_bulk is missing")
            if self.weight_shear is None:
                raise ValueError("weight_shear is missing")
            if self.k1 is None:
                raise ValueError("k1 is missing")
            if self.k2 is None:
                raise ValueError("k2 is missing")
            if self.mu1 is None:
                raise ValueError("mu1 is missing")
            if self.mu2 is None:
                raise ValueError("mu2 is missing")
            if self.sig1 is None:
                raise ValueError("sig1 is missing")
            if self.sig2 is None:
                raise ValueError("sig2 is missing")
            if self.cutoff is None and self.density_file is None and self.seed is None:
                raise ValueError("cutoff or density_file or seed is missing")
            if self.maxit is None:
                raise ValueError("maxit is missing")
            if self.tol is None:
                raise ValueError("tol is missing")
            if self.hist is None:
                raise ValueError("hist is missing")
            if self.dumpFreq is None:
                raise ValueError("dumpFreq is missing")
            if self.Lattice is None and self.ucparams is None:
                raise ValueError("Must include lattice or unit cell parameters")
        
        # if doing a single conductivity/elasticity calculation on an existing density
        elif self.run_type == 'single':
            if self.density_file is None:
                raise ValueError("density_file is missing")
            if self.Lattice is None:
                raise ValueError("Lattice is missing")
            if self.k1 is None:
                raise ValueError("k1 is missing")
            if self.k2 is None:
                raise ValueError("k2 is missing")
            if self.mu1 is None:
                raise ValueError("mu1 is missing")
            if self.mu2 is None:
                raise ValueError("mu2 is missing")
            if self.sig1 is None:
                raise ValueError("sig1 is missing")
            if self.sig2 is None:
                raise ValueError("sig2 is missing")
        
    def write_input(self, DIR):
        # Read the input file lines
        with open(self.infile_path, "r") as file:
            in_lines = file.readlines()
        # Open the output file for appending
        with open(DIR.outfile, "a") as file:
            for line in in_lines:
                # Write the original line
                file.write(line)


class phase_properties:
    def __init__(self, INFILE):
        self.k1, self.k2 = INFILE.k1, INFILE.k2
        self.mu1, self.mu2 = INFILE.mu1, INFILE.mu2
        self.sig1, self.sig2 = INFILE.sig1, INFILE.sig2
        self.f = INFILE.volFrac


class METRICclass:
    """
    Class to hold lattice related stuff.
    """
    def __init__(self, INFILE):
        self.device = INFILE.device
        self.Lattice = INFILE.Lattice  # Lattice matrix
        self.ucparams = INFILE.ucparams  # Unit cell parameters
        if self.Lattice is None:
            self.Lattice = self._get_Lattice(self.ucparams)
        self.Lattice_inv = torch.linalg.inv(self.Lattice)  # Inverse of the lattice matrix
        self.g = self.Lattice.T @ self.Lattice
        self.g_inv = torch.linalg.inv(self.g)

        # DEBUG: sanity checks
        for name, T in [
            ("Lattice", self.Lattice),
            ("Lattice_inv", self.Lattice_inv),
            ("g", self.g),
            ("g_inv", self.g_inv),
        ]:
            if not torch.isfinite(T).all():
                raise RuntimeError(f"{name} contains NaN/inf; check input Lattice.")
        
        self.ucparams = self._get_ucparams().to(INFILE.device).double()
        self.device = INFILE.device
        self.ucparams.requires_grad_(True)

        self.init_ucparams = self.ucparams.clone().detach()
        self.rebuild()

    def rebuild(self):
        # ---- 1) fresh leaf uc (no ties to any prior graph) ----
        uc = self.ucparams.detach().to(self.device, dtype=torch.float64).clone()
        uc.requires_grad_(True)  # local grad-enabled copy only for J

        def f(uc_):
            # uc_ is (6,), returns L (3,3)
            return self._get_Lattice(uc_)

        # ---- 2) compute L and J without building higher-order graph ----
        # vectorize=True is usually faster/more memory-efficient for small outputs
        L = f(uc)
        J = torch.autograd.functional.jacobian(
            f, uc, create_graph=False, vectorize=True
        )  # J has shape (3,3,6)

        # ---- 3) DETACH everything you will store ----
        L = L.detach()
        J = J.detach()
        uc = uc.detach()  # stored uc has no grad

        # Write back plain tensors (no grad)
        self.ucparams     = uc
        self.Lattice      = L
        self.Lattice_grad = J  # dL_{ij}/duc_k, shape (3,3,6)

        # ---- 4) Do all downstream metric ops without grad tracking ----
        with torch.no_grad():
            self.Lattice_inv      = torch.linalg.inv(self.Lattice)
            # L_inv_grad = - L_inv * (dL) * L_inv
            self.Lattice_inv_grad = -torch.einsum(
                'ik,klp,lj->ijp', self.Lattice_inv, self.Lattice_grad, self.Lattice_inv
            )
            self.g     = self.Lattice.T @ self.Lattice
            self.g_inv = torch.linalg.inv(self.g)
            self.V     = torch.linalg.det(self.Lattice)

    def _get_Lattice(self, ucparams=None):
        """
        Get the lattice vector matrix (columns are lattice vectors).
        ucparams: tensor([a, b, c, α, β, γ]) on self.device, dtype float64
        """
        if ucparams is None:
            ucparams = self.ucparams

        a, b, c, α, β, γ = ucparams

        zero = torch.zeros((), device=self.device, dtype=torch.float64)

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

    def _get_ucparams(self):
        """
        Get the unit cell parameters from the lattice vector matrix.
        """
        # Calculate lengths of the unit cell vectors
        a = torch.linalg.norm(self.Lattice[:, 0])
        b = torch.linalg.norm(self.Lattice[:, 1])
        c = torch.linalg.norm(self.Lattice[:, 2])
        # Calculate angles between the unit cell vectors
        α = torch.acos(torch.dot(self.Lattice[:, 1], self.Lattice[:, 2]) / (b * c))
        β = torch.acos(torch.dot(self.Lattice[:, 0], self.Lattice[:, 2]) / (a * c))
        γ = torch.acos(torch.dot(self.Lattice[:, 0], self.Lattice[:, 1]) / (a * b))
        return torch.tensor([a, b, c, α, β, γ], device=self.device, dtype=torch.float64)



class SOLclass:
    """
    Class to hold quantities relavant to the iterate.
    """
    def __init__(self, INFILE, METRIC, DIR):
        self.x0 = None
        self.initial_iter = None # specify initial_iter in main.py
        self.iter = 1
        self.gridShape = INFILE.gridShape
        self.rmin = INFILE.rmin           
        self.SIMP_exp_conductivity = None  
        self.SIMP_exp_shear = None
        self.volFrac = INFILE.volFrac  
        self.volFrac_binarized = None  # This will be set later in final_obs      
        self.weight_conductivity = INFILE.weight_conductivity
        self.weight_bulk = INFILE.weight_bulk
        self.weight_shear = INFILE.weight_shear
        self.phase_props = phase_properties(INFILE)
        self.cutoff = INFILE.cutoff  # fraction of Nyquist; lower → smoother
        self.maxit = INFILE.maxit
        self.tol = INFILE.tol
        self.hist = INFILE.hist
        self.device = INFILE.device
        self.dtype = torch.float64
        self.dumpFreq = INFILE.dumpFreq
        self.run_type = INFILE.run_type
        self.seed = INFILE.seed
        self.invert = INFILE.invert
    
        if INFILE.density_file is not None and self.cutoff is None:
            density = torch.load(INFILE.density_file, map_location=self.device, weights_only=False)
            self.gridShape = density.shape  # Update gridShape based on the loaded density
            if self.run_type == 'single':
                self.x0 = density.flatten().to(self.device, dtype=self.dtype)
                self.volFrac = torch.mean(density).item()
                self.phase_props.f = self.volFrac
            elif self.run_type == 'opt':
                density = density / density.sum() * (self.volFrac * self.gridShape[0]*self.gridShape[1]*self.gridShape[2])
                self.x0 = density.flatten().to(self.device, dtype=self.dtype)
        elif self.cutoff is not None and INFILE.density_file is None:
            # 1) make white noise
            noise = torch.randn(*self.gridShape, device=self.device, dtype=self.dtype)

            # 2) FFT it
            noise_fft = torch.fft.fftn(noise)

            # 3) build a spherical low-pass mask (only keep a few low-k modes)
            freqx = torch.fft.fftfreq(self.gridShape[0], d=1/self.gridShape[0], device=self.device)
            freqy = torch.fft.fftfreq(self.gridShape[1], d=1/self.gridShape[1], device=self.device)
            freqz = torch.fft.fftfreq(self.gridShape[2], d=1/self.gridShape[2], device=self.device)
            fx, fy, fz = torch.meshgrid(freqx, freqy, freqz, indexing="ij")
            r = torch.sqrt(fx**2 + fy**2 + fz**2)
            mask = (r < self.cutoff).to(self.dtype)
            num_modes = int((r < self.cutoff).sum().item())

            # 4) inverse FFT to get a smooth real field
            smooth = torch.fft.ifftn(noise_fft * mask).real

            # 5) stretch to full [0,1] range so you get large variation
            smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())

            # 6) enforce the total sum = volFrac * gridShape[0]*gridShape[1]*gridShape[2]
            density = smooth
            density = density / density.sum() * (self.volFrac * self.gridShape[0]*self.gridShape[1]*self.gridShape[2])
            self.x0 = density.flatten().to(self.device, dtype=self.dtype)
        elif self.cutoff is not None and INFILE.density_file is not None:
            # 1) Load density file
            density_init = torch.load(INFILE.density_file, map_location=self.device, weights_only=False)

            # 2) FFT it
            density_fft = torch.fft.fftn(density_init)

            # 3) build a spherical low-pass mask (only keep a few low-k modes)
            freqx = torch.fft.fftfreq(self.gridShape[0], d=1/self.gridShape[0], device=self.device)
            freqy = torch.fft.fftfreq(self.gridShape[1], d=1/self.gridShape[1], device=self.device)
            freqz = torch.fft.fftfreq(self.gridShape[2], d=1/self.gridShape[2], device=self.device)
            fx, fy, fz = torch.meshgrid(freqx, freqy, freqz, indexing="ij")
            r = torch.sqrt(fx**2 + fy**2 + fz**2)
            mask = (r < self.cutoff).to(self.dtype)

            # 4) inverse FFT to get a smooth real field
            smooth = torch.fft.ifftn(density_fft * mask).real

            # 5) stretch to full [0,1] range so you get large variation
            smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())

            # 6) enforce the total sum = volFrac * gridShape[0]*gridShape[1]*gridShape[2]
            density = smooth
            density = density / density.sum() * (self.volFrac * self.gridShape[0]*self.gridShape[1]*self.gridShape[2])
            self.x0 = density.flatten().to(self.device, dtype=self.dtype)
        elif self.seed is not None:
            morphology = self.seed
            # make density seed
            seed_maker = SEED(self.gridShape)
            density = seed_maker.make_seed(morphology, self.invert)
            # enforce the total sum = volFrac * gridShape[0]*gridShape[1]*gridShape[2]
            density = density / density.sum() * (self.volFrac * self.gridShape[0]*self.gridShape[1]*self.gridShape[2])
            self.x0 = density.flatten().to(self.device, dtype=self.dtype)
            
        DIR.log('\n--------------------------------------------------------------------------\n')
        
        # Dump initial density plots
        DIR.dumpInit(self)

        # Log beginning of optimization
        if INFILE.run_type == 'opt':
            DIR.log('Beginning optimization...')
        elif INFILE.run_type == 'single':
            DIR.log('Beginning single calculation...\n')
    

    def unflatten(self, array):
        """
        Reshape a flattened tensor to grid shape.
        """
        return array.reshape(self.gridShape)

    def fun_jac_density(self, x, CONDUCTIVITY, ELASTICITY, DIR, METRIC):
        """
        Objective function (to be minimized wrt density field) and its gradient.
        """
        # Compute fields
        CONDUCTIVITY.get_conductivity_field(x, self)
        ELASTICITY.get_elasticity_field(x, self)
        # Run PDE Solvers
        pde_outfile = DIR.add_pde_outfile(int(self.iter + self.initial_iter))
        CONDUCTIVITY.meso_Sigma, CONDUCTIVITY.results_conductivity, CONDUCTIVITY.phi_dict = solve_conductivity(CONDUCTIVITY.conductivity_field, pde_outfile, phi0_dict=CONDUCTIVITY.phi_dict, metric=METRIC)
        ELASTICITY.meso_C, ELASTICITY.results_elasticity, ELASTICITY.u_dict = solve_elasticity(ELASTICITY.bulk_field, ELASTICITY.shear_field, pde_outfile, u0_dict=ELASTICITY.u_dict, metric=METRIC)
        # Find macroscopic observables
        CONDUCTIVITY.get_macro_conductivity()
        ELASTICITY.get_macro_shear()
        # Gradient of observables wrt field
        CONDUCTIVITY.get_grad_macro_conductivity(x, self)
        ELASTICITY.get_grad_macro_shear(x, self)
        # Gradient of observables wrt uc parameters
        CONDUCTIVITY.get_grad_macro_conductivity_ucparams(self, METRIC)
        ELASTICITY.get_grad_macro_shear_ucparams(self, METRIC)
        # Objective function
        fun = -(self.weight_conductivity * CONDUCTIVITY.macro_conductivity + self.weight_bulk * ELASTICITY.macro_bulk + self.weight_shear * ELASTICITY.macro_shear)
        # Gradient of objective function
        jac = -(self.weight_conductivity * CONDUCTIVITY.grad_macro_conductivity + self.weight_bulk * ELASTICITY.grad_macro_bulk + self.weight_shear * ELASTICITY.grad_macro_shear)
        jac_density = jac.view(-1)  # Flatten the gradient to match the input shape for optimization
        
        return fun, jac_density

    def fun_jac_ucparams(self, ucparams, x, CONDUCTIVITY, ELASTICITY, DIR, METRIC):
        """
        Objective function (to be minimized wrt unit cell parameters) and its gradient.
        """
        # Update METRIC with new ucparams
        METRIC.ucparams = ucparams
        METRIC.rebuild()
        # Compute fields
        CONDUCTIVITY.get_conductivity_field(x, self)
        ELASTICITY.get_elasticity_field(x, self)
        # Run PDE Solvers
        pde_outfile = DIR.add_pde_outfile(int(self.iter + self.initial_iter))
        CONDUCTIVITY.meso_Sigma, CONDUCTIVITY.results_conductivity, CONDUCTIVITY.phi_dict = solve_conductivity(CONDUCTIVITY.conductivity_field, pde_outfile, phi0_dict=CONDUCTIVITY.phi_dict, metric=METRIC)
        ELASTICITY.meso_C, ELASTICITY.results_elasticity, ELASTICITY.u_dict = solve_elasticity(ELASTICITY.bulk_field, ELASTICITY.shear_field, pde_outfile, u0_dict=ELASTICITY.u_dict, metric=METRIC)
        # Find macroscopic observables
        CONDUCTIVITY.get_macro_conductivity()
        ELASTICITY.get_macro_shear()
        # Gradient of observables wrt field
        CONDUCTIVITY.get_grad_macro_conductivity(x, self)
        ELASTICITY.get_grad_macro_shear(x, self)
        # Gradient of observables wrt uc parameters
        CONDUCTIVITY.get_grad_macro_conductivity_ucparams(self, METRIC)
        ELASTICITY.get_grad_macro_shear_ucparams(self, METRIC)
        # Objective function 
        fun = -(self.weight_conductivity * CONDUCTIVITY.macro_conductivity + self.weight_bulk * ELASTICITY.macro_bulk + self.weight_shear * ELASTICITY.macro_shear)
        # Gradient wrt objective function
        jac_ucparams = -(self.weight_conductivity * CONDUCTIVITY.grad_macro_conductivity_ucparams + self.weight_bulk * ELASTICITY.grad_macro_bulk_ucparams + self.weight_shear * ELASTICITY.grad_macro_shear_ucparams)

        return fun, jac_ucparams

    def callback_density(self, x, f, err, iteration, converged, CONDUCTIVITY, ELASTICITY, DIR, METRIC):
        """
        Callback function for density to be called after each iteration.
        """
        self.iter = iteration
        DIR.log(f"Density iteration {int(self.iter + self.initial_iter)}: fun = {f:.6e}, |P(x-g)-x| = {err:.6e}, σ = {CONDUCTIVITY.macro_conductivity:.6f}, κ = {ELASTICITY.macro_bulk:.6f}, μ = {ELASTICITY.macro_shear:.6f}")
        if self.iter == self.maxit:
            DIR.log("Maximum number of iterations reached before density optimization converged.")
        else: 
            DIR.dumpIteration(self, x, CONDUCTIVITY, ELASTICITY, METRIC)
        if converged:
            DIR.log("Density optimization converged successfully.")
    
    def callback_ucparams(self, x, f, err, iteration, converged, x0, CONDUCTIVITY, ELASTICITY, DIR, METRIC):
        """
        Callback function for unit cell parameters to be called after each iteration.
        """
        self.iter = iteration
        DIR.log(f"Unit cell iteration {int(self.iter + self.initial_iter)}: fun = {f:.6e}, |P(x-g)-x| = {err:.6e}, σ = {CONDUCTIVITY.macro_conductivity:.6f}, κ = {ELASTICITY.macro_bulk:.6f}, μ = {ELASTICITY.macro_shear:.6f}")
        if self.iter == self.maxit:
            DIR.log("Maximum number of iterations reached before unit cell optimization converged.")
        else: 
            DIR.dumpIteration(self, x0, CONDUCTIVITY, ELASTICITY, METRIC)
        if converged:
            DIR.log("Unit cell optimization converged successfully.")
    
    def final_obs(self, x, CONDUCTIVITY, ELASTICITY, DIR, METRIC):
        """
        Calculate final conductivity and elasticity using the optimal density field without any power law exponent.
        """
        DIR.log("\nRunning observation calculation using binarized density field...")
        # Reshape the flattened density field back to its original shape
        rho = self.unflatten(x)
        volFrac_original = torch.mean(rho)
        rho_binarized = torch.zeros_like(rho, dtype=self.dtype, device=self.device)
        rho_binarized[rho >= 0.5] = 1
        rho_binarized[rho < 0.5] = 0
        volFrac_binarized = torch.mean(rho_binarized)
        self.volFrac_binarized = volFrac_binarized
        DIR.log(f'Volume fraction before binarization: {volFrac_original}, Volume fraction after binarization: {volFrac_binarized}')
        # Obtain fields while binarizing density field
        conductivity_field = torch.zeros_like(rho, dtype=self.dtype, device=self.device)
        bulk_field = torch.zeros_like(rho, dtype=self.dtype, device=self.device)
        shear_field = torch.zeros_like(rho, dtype=self.dtype, device=self.device)
        conductivity_field[rho_binarized == 1] = self.phase_props.sig1
        conductivity_field[rho_binarized == 0] = self.phase_props.sig2
        bulk_field[rho_binarized == 1] = self.phase_props.k1
        bulk_field[rho_binarized == 0] = self.phase_props.k2
        shear_field[rho_binarized == 1] = self.phase_props.mu1
        shear_field[rho_binarized == 0] = self.phase_props.mu2
        # Run PDE Solvers
        pde_outfile = DIR.add_pde_outfile(int(self.iter + self.initial_iter))
        CONDUCTIVITY.meso_Sigma, CONDUCTIVITY.results_conductivity, CONDUCTIVITY.phi_dict = solve_conductivity(conductivity_field, pde_outfile, phi0_dict=CONDUCTIVITY.phi_dict, metric=METRIC)
        ELASTICITY.meso_C, ELASTICITY.results_elasticity, ELASTICITY.u_dict = solve_elasticity(bulk_field, shear_field, pde_outfile, u0_dict=ELASTICITY.u_dict, metric=METRIC)
        # Find macroscopic observables
        CONDUCTIVITY.get_macro_conductivity()
        ELASTICITY.get_macro_shear()


class CONDUCTIVITYclass:
    """
    Class to perform and hold conductivity related stuff.
    """
    def __init__(self, SOL):
        self.meso_Sigma = None
        self.meso_C = None
        self.conductivity_field = None
        self.phi_dict = None
        self.results_conductivity = None
        self.phase_props = SOL.phase_props
    
    def get_conductivity_field(self, x, SOL):
        """
        Compute the conductivity field based on the density field using the SIMP model.
        """
        rho = SOL.unflatten(x)
        rho_filtered = density_filter(rho, SOL.rmin)
        eps = 1e-6
        rho_filtered = torch.clamp(rho_filtered, eps, 1.0 - eps) # to ensure rho_filtered and (1-rho_filtered) are never slightly negative
        p = SOL.SIMP_exp_conductivity
        self.conductivity_field = self.phase_props.sig1 * (rho_filtered)**p + self.phase_props.sig2 * (1 - rho_filtered)**p
    
    def get_macro_conductivity(self):
        """
        Compute the macroscopic conductivity from the mesoscopic conductivity tensor.
        """
        meso_Sigma = self.meso_Sigma
        meso_Sigma_inv = torch.linalg.inv(meso_Sigma)
        self.voigt_conductivity = 1/3 * torch.einsum('ii->', meso_Sigma)
        self.reuss_conductivity = (1/3 * torch.einsum('ii->', meso_Sigma_inv))**(-1)
        self.hill_conductivity = 1/2 * (self.voigt_conductivity + self.reuss_conductivity)
        self.EMA_conductivity = self._get_EMA_conductivity()
        self.macro_conductivity = self.EMA_conductivity # choice for optimization

    def get_grad_macro_conductivity(self, x, SOL):
        """
        Compute the gradient of the macroscopic conductivity with respect to the raw density field.
        """
        meso_Sigma = self.meso_Sigma
        p = SOL.SIMP_exp_conductivity
        rmin = SOL.rmin
        rho = SOL.unflatten(x)
        rho_filtered = density_filter(rho, rmin)
        eps = 1e-6
        rho_filtered = torch.clamp(rho_filtered, eps, 1.0 - eps) # to ensure rho_filtered and (1-rho_filtered) are never slightly negative
        results_conductivity = self.results_conductivity
        # Extract relevant gradients of electrostatic potential
        dphi_stack = torch.zeros((3, 3, *SOL.gridShape), device=SOL.device, dtype=torch.float64)
        for key, value in results_conductivity.items():
            if key == 'xx':
                dphi_stack[0, :] = value[1] # This is E - h_inv.T @ dphi_tilde
            elif key == 'yy':
                dphi_stack[1, :] = value[1]
            elif key == 'zz':
                dphi_stack[2, :] = value[1]
        # First index of dphi_stack is the direction of the applied field, second index is the component of the gradient

        # gradient of meso_Sigma wrt filtered density field
        outer_derivative = self.phase_props.sig1 * p * (rho_filtered)**(p-1) - self.phase_props.sig2 * p * (1 - rho_filtered)**(p-1)
        d_meso_Sigma = torch.einsum('inxyz, jnxyz->ijxyz', dphi_stack, dphi_stack) * outer_derivative # no explicit division by volume since L = 1 and so V = 1 for cubic cell
        # gradient wrt raw density field (must use this below!)
        d_meso_Sigma_unfiltered = density_filter(d_meso_Sigma, rmin)
        
        # gradient of EMA
        Id = torch.eye(3, device=SOL.device, dtype=SOL.dtype)
        EMA_meso_Sigma = self.EMA_conductivity * Id
        inv_Sigma_plus_2EMA_Sigma = torch.linalg.inv(meso_Sigma + 2 * EMA_meso_Sigma)
        trace_numerator = torch.einsum('ij,jkxyz,ki->xyz', Id - meso_Sigma @ inv_Sigma_plus_2EMA_Sigma, d_meso_Sigma_unfiltered, inv_Sigma_plus_2EMA_Sigma)
        trace_denomenator = torch.trace(meso_Sigma @ inv_Sigma_plus_2EMA_Sigma @ inv_Sigma_plus_2EMA_Sigma)
        self.grad_EMA_conductivity = 1/2 * trace_numerator / trace_denomenator
        self.grad_macro_conductivity = self.grad_EMA_conductivity

    def get_grad_macro_conductivity_ucparams(self, SOL, METRIC):
        """
        Compute the gradient of the macroscopic conductivity with respect to the unit cell params.
        """
        meso_Sigma = self.meso_Sigma
        results_conductivity = self.results_conductivity
        phi_dict = self.phi_dict
        # Build k-vectors
        dx, dy, dz = (1 / SOL.gridShape[0], 1 / SOL.gridShape[1], 1 / SOL.gridShape[2])
        kx = 2*math.pi * torch.fft.fftfreq(SOL.gridShape[0], d=dx, device=SOL.device, dtype=torch.float64)
        ky = 2*math.pi * torch.fft.fftfreq(SOL.gridShape[1], d=dy, device=SOL.device, dtype=torch.float64)
        kz = 2*math.pi * torch.fft.rfftfreq(SOL.gridShape[2], d=dz, device=SOL.device, dtype=torch.float64)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K = torch.stack((KX, KY, KZ), dim=0)
        # directions to process, in order
        dirs = ['xx', 'yy', 'zz']

        # allocate stacks: (dir, component, x, y, z)
        dphi_stack        = torch.empty((3, 3, *SOL.gridShape), device=SOL.device, dtype=torch.float64)
        dphi_tilde_stack  = torch.empty_like(dphi_stack)

        for i, key in enumerate(dirs):
            if key not in results_conductivity:
                raise KeyError(f"Missing '{key}' in results_conductivity")

            # value[1] is the real-space gradient: E - h_inv.T @ dphi_tilde  (shape: 3×Nx×Ny×Nz)
            dphi_stack[i] = results_conductivity[key][1].to(device=SOL.device, dtype=torch.float64)

            # Rebuild ∇_tilde phi from the saved potential phi_tilde (shape: Nx×Ny×Nz)
            phi_tilde = phi_dict[key].to(device=SOL.device, dtype=torch.float64)

            # FFTs
            phi_tilde_hat   = torch.fft.rfftn(phi_tilde, dim=(0, 1, 2))  # (Nx, Ny, Nz/2+1)
            dphi_tilde_hat  = torch.stack((
                -1j * K[0] * phi_tilde_hat,
                -1j * K[1] * phi_tilde_hat,
                -1j * K[2] * phi_tilde_hat
            ), dim=0)  # (3, Nx, Ny, Nz/2+1)

            # Back to real space
            dphi_tilde = torch.fft.irfftn(dphi_tilde_hat, s=SOL.gridShape, dim=(1, 2, 3)).real  # (3, Nx, Ny, Nz)
            dphi_tilde_stack[i] = dphi_tilde

        # First index of dphi_stack's is the direction of the applied field, second index is the component of the gradient

        # gradient of meso_Sigma wrt uc params
        n_gridpts = SOL.gridShape[0] * SOL.gridShape[1] * SOL.gridShape[2]
        d_meso_Sigma = 1/n_gridpts * (torch.einsum('knp, ikxyz, jnxyz, xyz -> ijp', METRIC.Lattice_inv_grad, dphi_tilde_stack, dphi_stack, self.conductivity_field) 
                            + torch.einsum('inxyz, knp, jkxyz, xyz -> ijp', dphi_stack, METRIC.Lattice_inv_grad, dphi_tilde_stack, self.conductivity_field))

        # gradient of EMA
        Id = torch.eye(3, device=SOL.device, dtype=SOL.dtype)
        EMA_meso_Sigma = self.EMA_conductivity * Id
        inv_Sigma_plus_2EMA_Sigma = torch.linalg.inv(meso_Sigma + 2 * EMA_meso_Sigma)
        trace_numerator = torch.einsum('ij,jkp,ki->p', Id - meso_Sigma @ inv_Sigma_plus_2EMA_Sigma, d_meso_Sigma, inv_Sigma_plus_2EMA_Sigma)
        trace_denomenator = torch.trace(meso_Sigma @ inv_Sigma_plus_2EMA_Sigma @ inv_Sigma_plus_2EMA_Sigma)
        self.grad_EMA_conductivity_ucparams = 1/2 * trace_numerator / trace_denomenator
        self.grad_macro_conductivity_ucparams = self.grad_EMA_conductivity_ucparams
    
    def _get_EMA_conductivity(self):
        """
        Evaluate the effective medium approximation (EMA) conductivity.
        """
        meso_sigma = self.meso_Sigma.cpu().numpy()
        # Conductivity
        def objfun(x, sigma):
            # 3x3 Identity
            δ = np.eye(3)
            # Assemble matrices
            M = sigma + 2 * x * δ
            N = sigma @ np.linalg.inv(M)
            # Objective function
            f = np.linalg.trace(N) - 1
            return f
        a = 1e-6
        b = self.phase_props.f * self.phase_props.sig1 + (1 - self.phase_props.f) * self.phase_props.sig2 + 0.1 # add some buffer to avoid issues
        EMA_sigma, r = brentq(f=objfun, a=a, b=b, args=(meso_sigma,), xtol=1e-10, maxiter=100, disp=True, full_output=True)
        return EMA_sigma



class ELASTICITYclass:
    """
    Class to perform and hold elasticity related stuff.
    """
    def __init__(self, SOL):
        self.meso_C = None
        self.bulk_field = None
        self.shear_field = None
        self.results_elasticity = None
        self.u_dict = None
        self.phase_props = SOL.phase_props
        self.device = SOL.device
        self.dtype = SOL.dtype
        # Define basis tensors for elasticity calculations
        ah = (1/np.sqrt(3)) * torch.tensor([[1,0,0],[0,1,0],[0,0,1]], device=SOL.device, dtype=SOL.dtype)
        a1 = (1/np.sqrt(2)) * torch.tensor([[ 1,0,0],[0,-1,0],[0,0,0]], device=SOL.device, dtype=SOL.dtype)
        a2 = (1/np.sqrt(6)) * torch.tensor([[ 1,0,0],[0, 1,0],[0,0,-2]], device=SOL.device, dtype=SOL.dtype)
        a3 = (1/np.sqrt(2)) * torch.tensor([[ 0,1,0],[1, 0,0],[0,0,0]], device=SOL.device, dtype=SOL.dtype)
        a4 = (1/np.sqrt(2)) * torch.tensor([[ 0,0,1],[0, 0,0],[1,0,0]], device=SOL.device, dtype=SOL.dtype)
        a5 = (1/np.sqrt(2)) * torch.tensor([[ 0,0,0],[0, 0,1],[0,1,0]], device=SOL.device, dtype=SOL.dtype)
        b1 = (1/np.sqrt(2)) * torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], device=SOL.device, dtype=SOL.dtype) # b1, b2, b3 are for antisymmetric and so we don't need them
        b2 = (1/np.sqrt(2)) * torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], device=SOL.device, dtype=SOL.dtype)
        b3 = (1/np.sqrt(2)) * torch.tensor([[0, 0, 0], [0, 0, 1], [0, -1, 0]], device=SOL.device, dtype=SOL.dtype)
        self.bases = [ah, a1, a2, a3, a4, a5]
        # Define elasticity projection tensors
        δ = torch.eye(3, device=SOL.device, dtype=SOL.dtype)
        self.I_prime = torch.einsum('ij,kl->ijkl', δ, δ) / 3.0
        self.I_sym   = 0.5*(torch.einsum('ik,jl->ijkl', δ, δ) +
                        torch.einsum('il,jk->ijkl', δ, δ))
        self.I_doublePrime   = self.I_sym - self.I_prime
    
    def get_elasticity_field(self, x, SOL):
        """
        Compute the elasticity fields based on the density field using the SIMP model. Bulk field is set to one everywhere for anion exchange membranes.
        """
        rho = SOL.unflatten(x)
        rho_filtered = density_filter(rho, SOL.rmin)
        eps = 1e-6
        rho_filtered = torch.clamp(rho_filtered, eps, 1.0 - eps) # to ensure rho_filtered and (1-rho_filtered) are never slightly negative
        p = SOL.SIMP_exp_shear
        self.bulk_field = self.phase_props.k1 * (rho_filtered)**p + self.phase_props.k2 * (1-rho_filtered)**p
        self.shear_field = self.phase_props.mu1 * (rho_filtered)**p + self.phase_props.mu2 * (1-rho_filtered)**p

    def get_macro_shear(self):
        """
        Compute the macroscopic shear modulus from the mesoscopic elasticity tensor.
        """
        meso_C = self.meso_C
        meso_C_inv = self.invert_C(meso_C)
        self.voigt_bulk = 1/9 * torch.einsum('iijj->', meso_C)
        self.reuss_bulk = (torch.einsum('iijj->', meso_C_inv))**(-1)
        self.hill_bulk = 1/2 * (self.voigt_bulk + self.reuss_bulk)
        self.voigt_shear = 1/10 * (torch.einsum('ijij->', meso_C) - 1/3 * torch.einsum('iijj->', meso_C))
        self.reuss_shear = 5/2 * (torch.einsum('ijij->', meso_C_inv) - 1/3 * torch.einsum('iijj->', meso_C_inv))**(-1)
        self.hill_shear = 1/2 * (self.voigt_shear + self.reuss_shear)
        self.EMA_bulk, self.EMA_shear = self._get_EMA_bulk_shear()
        self.macro_bulk = self.EMA_bulk
        self.macro_shear = self.EMA_shear
    
    def invert_C(self, C):
        """
        Invert an elastic tensor.
        """
        # Write C in matrix form
        C_matrix = torch.zeros((6, 6), device=C.device, dtype=C.dtype)
        for i in range(6):
            for j in range(6):
                C_matrix[i,j] = torch.einsum('ijkl,ij,kl->', C, self.bases[i], self.bases[j])
        C_matrix_inv = torch.linalg.inv(C_matrix)
        # Convert back to tensor form
        C_inv = torch.zeros((3, 3, 3, 3), device=C.device, dtype=C.dtype)
        for i in range(6):
            for j in range(6):
                C_inv += torch.einsum('ij,kl->ijkl', self.bases[i], self.bases[j]) * C_matrix_inv[i,j]
        return C_inv

    def get_grad_macro_shear(self, x, SOL):
        """
        Compute the gradient of the macroscopic shear modulus with respect to the raw density field.
        """
        meso_C = self.meso_C
        p = SOL.SIMP_exp_shear
        rmin = SOL.rmin
        rho = SOL.unflatten(x)
        rho_filtered = density_filter(rho, rmin)
        eps = 1e-6
        rho_filtered = torch.clamp(rho_filtered, eps, 1.0 - eps) # to ensure rho_filtered and (1-rho_filtered) are never slightly negative
        results_elasticity = self.results_elasticity
        # Extract relevant strains
        all_strains = torch.zeros((6, 3, 3, *SOL.gridShape), device=SOL.device, dtype=torch.float64)
        for key, value in results_elasticity.items():
            if key == 'xy':
                all_strains[0] = value[1]
            elif key == 'xz':
                all_strains[1] = value[1]
            elif key == 'yz':
                all_strains[2] = value[1]
            elif key == 'xx':
                all_strains[3] = value[1]
            elif key == 'yy':
                all_strains[4] = value[1]
            elif key == 'zz':
                all_strains[5] = value[1]

        # assemble matrix of derivatives for shear
        h_mu = torch.zeros((6, 6, *SOL.gridShape), device=SOL.device, dtype=torch.float64)
        for a in range(6):
            for b in range(6):
                if a < 3 and b < 3:
                    prefactor = 1/4
                elif a < 3 and b >= 3:
                    prefactor = 1/2
                elif a >= 3 and b < 3:
                    prefactor = 1/2
                else:
                    prefactor = 1
                h_mu[a, b] = prefactor * (-2/3 * torch.einsum('iixyz->xyz', all_strains[a]) * torch.einsum('iixyz->xyz', all_strains[b])
                                            + torch.einsum('ijxyz, ijxyz->xyz', all_strains[a], all_strains[b])
                                            + torch.einsum('ijxyz, jixyz->xyz', all_strains[a], all_strains[b]))
        
        # assemble matrix of derivatives for bulk
        h_k = torch.zeros((6, 6, *SOL.gridShape), device=SOL.device, dtype=torch.float64)
        for a in range(6):
            for b in range(6):
                h_k[a, b] = torch.einsum('iixyz->xyz', all_strains[a]) * torch.einsum('iixyz->xyz', all_strains[b])

        # assemble dCdmu 
        d_meso_C = torch.zeros((3, 3, 3, 3, *SOL.gridShape), device=SOL.device, dtype=torch.float64)
        zipped_indices = [(0, 1), (0, 2), (1, 2), (0, 0), (1, 1), (2, 2)]
        for a, inda in enumerate(zipped_indices):
            for b, indb in enumerate(zipped_indices):
                a1 = inda[0]
                a2 = inda[1]
                b1 = indb[0]
                b2 = indb[1]
                d_meso_C[a1, a2, b1, b2] = d_meso_C[a2, a1, b1, b2] = d_meso_C[a1, a2, b2, b1] = d_meso_C[a2, a1, b2, b1] = (h_mu[a, b] * (self.phase_props.mu1 * p * (rho_filtered)**(p-1) + self.phase_props.mu2 * -p * (1-rho_filtered)**(p-1)) 
                                                                                                                                + h_k[a, b] * (self.phase_props.k1 * p * (rho_filtered)**(p-1) + self.phase_props.k2 * -p * (1-rho_filtered)**(p-1)))
        # Unfilter the gradient (use this one below!)
        d_meso_C_unfiltered = density_filter(d_meso_C, rmin)

        # Compute gradient of EMA
        C_macro = 3 * self.EMA_bulk * self.I_prime + 2 * self.EMA_shear * self.I_doublePrime
        diff_meso_C = meso_C - C_macro
        EMA_bulk = self.EMA_bulk; EMA_shear = self.EMA_shear
        p1 = 1/(3*EMA_bulk + 4*EMA_shear); p2 = 3*(EMA_bulk + 2*EMA_shear)/(5*EMA_shear*(3*EMA_bulk + 4*EMA_shear))
        A = self.I_sym + torch.einsum('ijmn,mnkl->ijkl', (p1 * self.I_prime + p2 * self.I_doublePrime), diff_meso_C); A_inv = self.invert_C(A)
        dp1_dbulk = -3/(3*EMA_bulk + 4*EMA_shear)**2; dp2_dbulk = -6/(5*(3*EMA_bulk + 4*EMA_shear)**2); dp1_dshear = -4/(3*EMA_bulk + 4*EMA_shear)**2; dp2_shear = -1/(5*EMA_shear**2) - 8/(5*(3*EMA_bulk + 4*EMA_shear)**2)
        dA_dbulk = torch.einsum('ijmn,mnkl->ijkl', (dp1_dbulk * self.I_prime + dp2_dbulk * self.I_doublePrime), diff_meso_C) - 3 * p1 * self.I_prime; dA_dshear = torch.einsum('ijmn,mnkl->ijkl', (dp1_dshear * self.I_prime + dp2_shear * self.I_doublePrime), diff_meso_C) - 2 * p2 * self.I_doublePrime
        D = torch.einsum('ijmn,mnop,opqrxyz,qrkl->ijklxyz', A_inv, p1*self.I_prime + p2*self.I_doublePrime, d_meso_C_unfiltered, A_inv)
        D_hydro = torch.einsum('ijkl,ijmn,mnop,opqrxyz,qrkl->xyz', self.I_prime, A_inv, p1*self.I_prime + p2*self.I_doublePrime, d_meso_C_unfiltered, A_inv);      D_dev = torch.einsum('ijkl,ijmn,mnop,opqrxyz,qrkl->xyz', self.I_doublePrime, A_inv, p1*self.I_prime + p2*self.I_doublePrime, d_meso_C_unfiltered, A_inv)
        B_bulk = -torch.einsum('ijmn,mnop,opkl->ijkl', A_inv, dA_dbulk, A_inv);     B_shear = -torch.einsum('ijmn,mnop,opkl->ijkl', A_inv, dA_dshear, A_inv)
        B_bulk_hydro = torch.einsum('ijkl,ijkl->', self.I_prime, B_bulk);       B_shear_hydro = torch.einsum('ijkl,ijkl->', self.I_prime, B_shear);     B_bulk_dev = torch.einsum('ijkl,ijkl->', self.I_doublePrime, B_bulk);       B_shear_dev = torch.einsum('ijkl,ijkl->', self.I_doublePrime, B_shear)
        det_Bmtx = B_bulk_hydro * B_shear_dev - B_shear_hydro * B_bulk_dev
        self.grad_EMA_bulk = (D_hydro * B_shear_dev - D_dev * B_shear_hydro) / (det_Bmtx + 1e-10)
        self.grad_EMA_shear = (-D_hydro * B_bulk_dev + D_dev * B_bulk_hydro) / (det_Bmtx + 1e-10)
        self.grad_macro_bulk = self.grad_EMA_bulk
        self.grad_macro_shear = self.grad_EMA_shear

    def get_grad_macro_shear_ucparams(self, SOL, METRIC):
        """
        Compute the gradient of the macroscopic shear modulus with respect to the unit cell parameters.
        """
        meso_C = self.meso_C
        results_elasticity = self.results_elasticity
        u_dict = self.u_dict

        # Define strain component order
        strain_keys = ['xy', 'xz', 'yz', 'xx', 'yy', 'zz']

        # Allocate tensors
        all_strains = torch.zeros((6, 3, 3, *SOL.gridShape), device=SOL.device, dtype=torch.float64)

        # Precompute constant
        Lattice_inv_grad_C = METRIC.Lattice_inv_grad.to(torch.complex128)

        # Build k-vectors
        dx, dy, dz = (1 / SOL.gridShape[0], 1 / SOL.gridShape[1], 1 / SOL.gridShape[2])
        kx = 2*math.pi * torch.fft.fftfreq(SOL.gridShape[0], d=dx, device=SOL.device, dtype=torch.float64)
        ky = 2*math.pi * torch.fft.fftfreq(SOL.gridShape[1], d=dy, device=SOL.device, dtype=torch.float64)
        kz = 2*math.pi * torch.fft.rfftfreq(SOL.gridShape[2], d=dz, device=SOL.device, dtype=torch.float64)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K = torch.stack((KX, KY, KZ), dim=0)
        Kc = K.to(torch.complex128)

        dC_temp = torch.zeros((6, 6, 6), device=SOL.device, dtype=torch.float64)
        bulk = self.bulk_field
        shear = self.shear_field
        for a, key_a in enumerate(strain_keys):
            for b, key_b in enumerate(strain_keys):
                # Get values
                value_a = results_elasticity[key_a]
                value_b = results_elasticity[key_b]
                # Compute prefactor to scale
                if a < 3 and b < 3:
                    prefactor = 1/4
                elif a < 3 and b >= 3:
                    prefactor = 1/2
                elif a >= 3 and b < 3:
                    prefactor = 1/2
                else:
                    prefactor = 1
                # Get strain
                strain_a = value_a[1]
                strain_b = value_b[1]
                # Get u's
                u_a = u_dict[key_a]
                u_b = u_dict[key_b]
                # FT u's
                u_stack_a = u_a.permute(3,0,1,2)                           # u_stack is shape (3,Nx,Ny,Nz) since u is shape (Nx,Ny,Nz,3)
                u_hat_a   = torch.fft.rfftn(u_stack_a, dim=(1,2,3))
                u_stack_b = u_b.permute(3,0,1,2)
                u_hat_b   = torch.fft.rfftn(u_stack_b, dim=(1,2,3))
                # Compute d_u_hat_d_uc and transform back
                d_u_hat_a_d_uc = -1j * 0.5 * (
                    torch.einsum('kxyz,kip,jxyz->ijpxyz', Kc, Lattice_inv_grad_C, u_hat_a)
                    + torch.einsum('kxyz,kjp,ixyz->ijpxyz', Kc, Lattice_inv_grad_C, u_hat_a)
                )
                d_strain_a_d_uc = torch.fft.irfftn(d_u_hat_a_d_uc, s=SOL.gridShape, dim=(3, 4, 5))
                d_u_hat_b_d_uc = -1j * 0.5 * (
                    torch.einsum('kxyz,kip,jxyz->ijpxyz', Kc, Lattice_inv_grad_C, u_hat_b)
                    + torch.einsum('kxyz,kjp,ixyz->ijpxyz', Kc, Lattice_inv_grad_C, u_hat_b)
                )
                d_strain_b_d_uc = torch.fft.irfftn(d_u_hat_b_d_uc, s=SOL.gridShape, dim=(3, 4, 5))
                
                n_gridpts = SOL.gridShape[0] * SOL.gridShape[1] * SOL.gridShape[2]
                dC_temp[a, b] = prefactor/n_gridpts * (torch.einsum('iixyz, jjpxyz, xyz -> p', strain_a, d_strain_b_d_uc, (bulk - 2/3 * shear))
                                + torch.einsum('iipxyz, jjxyz, xyz -> p', d_strain_a_d_uc, strain_b, (bulk - 2/3 * shear))
                                + torch.einsum('ijxyz, ijpxyz, xyz -> p', strain_a, d_strain_b_d_uc, shear)
                                + torch.einsum('ijpxyz, jixyz, xyz -> p', d_strain_a_d_uc, strain_b, shear)
                                + torch.einsum('ijxyz, ijpxyz, xyz -> p', strain_a, d_strain_b_d_uc, shear)
                                + torch.einsum('ijpxyz, jixyz, xyz -> p', d_strain_a_d_uc, strain_b, shear)
                                )

        # assemble dC_duc
        d_meso_C = torch.zeros((3, 3, 3, 3, 6), device=SOL.device, dtype=torch.float64)
        zipped_indices = [(0, 1), (0, 2), (1, 2), (0, 0), (1, 1), (2, 2)]
        for a, inda in enumerate(zipped_indices):
            for b, indb in enumerate(zipped_indices):
                a1 = inda[0]
                a2 = inda[1]
                b1 = indb[0]
                b2 = indb[1]
                d_meso_C[a1, a2, b1, b2] = d_meso_C[a2, a1, b1, b2] = d_meso_C[a1, a2, b2, b1] = d_meso_C[a2, a1, b2, b1] = dC_temp[a, b]

        # Compute gradient of EMA
        C_macro = 3 * self.EMA_bulk * self.I_prime + 2 * self.EMA_shear * self.I_doublePrime
        diff_meso_C = meso_C - C_macro
        EMA_bulk = self.EMA_bulk; EMA_shear = self.EMA_shear
        p1 = 1/(3*EMA_bulk + 4*EMA_shear); p2 = 3*(EMA_bulk + 2*EMA_shear)/(5*EMA_shear*(3*EMA_bulk + 4*EMA_shear))
        A = self.I_sym + torch.einsum('ijmn,mnkl->ijkl', (p1 * self.I_prime + p2 * self.I_doublePrime), diff_meso_C); A_inv = self.invert_C(A)
        dp1_dbulk = -3/(3*EMA_bulk + 4*EMA_shear)**2; dp2_dbulk = -6/(5*(3*EMA_bulk + 4*EMA_shear)**2); dp1_dshear = -4/(3*EMA_bulk + 4*EMA_shear)**2; dp2_shear = -1/(5*EMA_shear**2) - 8/(5*(3*EMA_bulk + 4*EMA_shear)**2)
        dA_dbulk = torch.einsum('ijmn,mnkl->ijkl', (dp1_dbulk * self.I_prime + dp2_dbulk * self.I_doublePrime), diff_meso_C) - 3 * p1 * self.I_prime; dA_dshear = torch.einsum('ijmn,mnkl->ijkl', (dp1_dshear * self.I_prime + dp2_shear * self.I_doublePrime), diff_meso_C) - 2 * p2 * self.I_doublePrime
        D = torch.einsum('ijmn,mnop,opqrd,qrkl->ijkld', A_inv, p1*self.I_prime + p2*self.I_doublePrime, d_meso_C, A_inv)
        D_hydro = torch.einsum('ijkl,ijmn,mnop,opqrd,qrkl->d', self.I_prime, A_inv, p1*self.I_prime + p2*self.I_doublePrime, d_meso_C, A_inv);      D_dev = torch.einsum('ijkl,ijmn,mnop,opqrd,qrkl->d', self.I_doublePrime, A_inv, p1*self.I_prime + p2*self.I_doublePrime, d_meso_C, A_inv)
        B_bulk = -torch.einsum('ijmn,mnop,opkl->ijkl', A_inv, dA_dbulk, A_inv);     B_shear = -torch.einsum('ijmn,mnop,opkl->ijkl', A_inv, dA_dshear, A_inv)
        B_bulk_hydro = torch.einsum('ijkl,ijkl->', self.I_prime, B_bulk);       B_shear_hydro = torch.einsum('ijkl,ijkl->', self.I_prime, B_shear);     B_bulk_dev = torch.einsum('ijkl,ijkl->', self.I_doublePrime, B_bulk);       B_shear_dev = torch.einsum('ijkl,ijkl->', self.I_doublePrime, B_shear)
        det_Bmtx = B_bulk_hydro * B_shear_dev - B_shear_hydro * B_bulk_dev
        self.grad_EMA_bulk_ucparams = (D_hydro * B_shear_dev - D_dev * B_shear_hydro) / (det_Bmtx + 1e-10)
        self.grad_EMA_shear_ucparams = (-D_hydro * B_bulk_dev + D_dev * B_bulk_hydro) / (det_Bmtx + 1e-10)
        self.grad_macro_bulk_ucparams = self.grad_EMA_bulk_ucparams
        self.grad_macro_shear_ucparams = self.grad_EMA_shear_ucparams

    def _get_EMA_bulk_shear(self):
        """
        Evaluate the effective medium approximation (EMA) shear.
        """
        meso_C = self.meso_C.cpu().numpy()
        # Projection tensors
        I_prime = self.I_prime.cpu().numpy()
        I_doublePrime = self.I_doublePrime.cpu().numpy()

        # Bulk and Shear moduli
        def objfun(x, C):
            # 3x3 Identity
            δ = np.eye(3)
            # Bulk and shear moduli of isotropic material
            K = x[0]
            mu = x[1]
            # Isotropic elasticity tensor
            Cstar = 3 * K * I_prime + 2 * mu * I_doublePrime
            # Difference between elasticity tensors
            ΔC = C - Cstar
            # Constants
            p1 = 1 / (3 * K + 4 * mu)
            p2 = 3 * (K + 2 * mu) / (5 * mu * (3 * K + 4 * mu))
            # Assemble A
            δδ     = np.einsum('mk,nl->mnkl', δ, δ)
            A = (np.einsum('ijmn,mnkl->ijkl', I_prime, δδ + p1*ΔC) +
                np.einsum('ijmn,mnkl->ijkl', I_doublePrime, δδ + p2*ΔC))
            A_inv = self.invert_C(torch.tensor(A, device=self.device, dtype=self.dtype)).cpu().numpy()
            # Hydrostatic and deviatoric invariants
            t_hydro = np.einsum('ijkl,ijkl->', I_prime,      A_inv)
            t_dev   = np.einsum('ijkl,ijkl->', I_doublePrime, A_inv)

            f = np.zeros(2)
            f[0] = t_hydro - 1.0
            f[1] = t_dev   - 5.0
            return f

        # Good initial guess: midpoints of phase moduli
        x0 = np.array([0.5 * (self.phase_props.k1 + self.phase_props.k2), 0.5 * (self.phase_props.mu1 + self.phase_props.mu2)])

        res = least_squares(
            objfun,
            x0,
            args=(meso_C,),
            bounds=([0.0, 0.0], [np.inf, np.inf]),   # ← bounds: K≥0, μ≥0
            xtol=1e-12, ftol=1e-12, gtol=1e-12,
            max_nfev=200
        )

        EMA_bulk, EMA_shear = res.x

        return EMA_bulk, EMA_shear


class DIRclass:
    """
    Class to hold directories.
    """
    def __init__(self, INFILE):
        self.mainDir = INFILE.dir_name
        self.isoDir = f"{INFILE.dir_name}/iso"
        self.projDir = f"{INFILE.dir_name}/proj"
        self.pdeDir = f"{INFILE.dir_name}/pde"
        self.obsDir = f"{INFILE.dir_name}/obs"
        self.densDir = f"{INFILE.dir_name}/dens"
        self.outfile = f"{INFILE.dir_name}/outfile.txt"
        # Clean up/create directories
        if os.path.exists(INFILE.dir_name):
            raise FileExistsError(
                f"Directory '{INFILE.dir_name}' already exists. Please choose a different name."
            )
        else:
            os.makedirs(INFILE.dir_name)
        for d in [self.isoDir, self.projDir, self.pdeDir, self.obsDir, self.densDir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        # Clean up/create outfile
        file_path = os.path.join(INFILE.dir_name, self.outfile)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    def add_pde_outfile(self, iter):
        """
        Add a PDE output file for the current iteration.
        """
        pde_file_path = os.path.join(self.pdeDir, f"pde_iter_{iter}.txt")
        with open(pde_file_path, "w") as file:
            file.write(f"PDE output for iteration {iter}\n")
        return pde_file_path
    
    def log(self, message):
        """
        Log a message to the output file.
        """
        with open(self.outfile, "a") as file:
            print(message, file=file)
    
    def dumpInit(self, SOL):
        """
        Dump initial state.
        """
        rho = SOL.unflatten(SOL.x0)
        self.log("Dumping initial density plots...")
        projection(rho, out_dir=self.projDir, iter_num=0)
        isosurface(rho, out_dir=self.isoDir, iter_num=0, tau=0.8)
        
    def dumpIteration(self, SOL, x, CONDUCTIVITY, ELASTICITY, METRIC):
        """
        Dump the current state of the simulation.
        """
        rho = SOL.unflatten(x)
        if SOL.iter % SOL.dumpFreq == 0:
            self.log("\nDumping plots...")
            projection(rho, out_dir=self.projDir, iter_num=int(SOL.iter + SOL.initial_iter))
            isosurface(rho, out_dir=self.isoDir, iter_num=int(SOL.iter + SOL.initial_iter), tau=0.8)
            self.log("Dumping density...")
            density_file_path = os.path.join(self.densDir, f"density_iter_{int(SOL.iter + SOL.initial_iter)}.pt")
            torch.save(rho, density_file_path)
            self.log("Dumping observables...")
            with open(os.path.join(self.obsDir, f"obs_iter_{int(SOL.iter + SOL.initial_iter)}.txt"), "a") as file:
                file.write(f"Conductivity tensor:\n{CONDUCTIVITY.meso_Sigma}\n")
                file.write(f"Voigt Conductivity: {CONDUCTIVITY.voigt_conductivity:.6f}\n")
                file.write(f"Reuss Conductivity: {CONDUCTIVITY.reuss_conductivity:.6f}\n")
                file.write(f"Hill Conductivity: {CONDUCTIVITY.hill_conductivity:.6f}\n")
                file.write(f"EMA Conductivity: {CONDUCTIVITY.EMA_conductivity:.6f}\n\n")
                file.write(f"Elasticity tensor:\n{ELASTICITY.meso_C}\n")
                file.write(f"Voigt bulk: {ELASTICITY.voigt_bulk:.6f}\n")
                file.write(f"Reuss bulk: {ELASTICITY.reuss_bulk:.6f}\n")
                file.write(f"Hill bulk: {ELASTICITY.hill_bulk:.6f}\n")
                file.write(f"EMA bulk: {ELASTICITY.EMA_bulk:.6f}\n\n")
                file.write(f"Voigt shear: {ELASTICITY.voigt_shear:.6f}\n")
                file.write(f"Reuss shear: {ELASTICITY.reuss_shear:.6f}\n")
                file.write(f"Hill shear: {ELASTICITY.hill_shear:.6f}\n")
                file.write(f"EMA shear: {ELASTICITY.EMA_shear:.6f}\n")
                file.write("\nLattice\n")
                for row in METRIC.Lattice:
                    file.write(f"{row[0]:.3f} {row[1]:.3f} {row[2]:.3f}\n")
            self.log("Resuming optimization...\n")


    def dumpEnd(self, SOL, x, CONDUCTIVITY, ELASTICITY, METRIC):
        """
        Dump the end state of the simulation.
        """
        rho = SOL.unflatten(x)
        if SOL.run_type == 'opt':
            self.log("\nDumping plots...")
            projection(rho, out_dir=self.projDir, iter_num=int(SOL.iter + SOL.initial_iter))
            isosurface(rho, out_dir=self.isoDir, iter_num=int(SOL.iter + SOL.initial_iter), tau=0.8)
        elif SOL.run_type == 'single':
            rho_binarized = torch.zeros_like(rho, dtype=SOL.dtype, device=SOL.device)
            rho_binarized[rho >= 0.5] = 1
            rho_binarized[rho < 0.5] = 0
            self.log("\nDumping plots of binarized density...")
            projection(rho_binarized, out_dir=self.projDir, iter_num='binarized')
            isosurface(rho_binarized, out_dir=self.isoDir, iter_num='binarized', tau=0.8)
        self.log("Dumping observables...")
        with open(os.path.join(self.obsDir, f"obs_iter_{int(SOL.iter + SOL.initial_iter)}.txt"), "a") as file:
            file.write(f"Conductivity tensor:\n{CONDUCTIVITY.meso_Sigma}\n")
            file.write(f"Voigt Conductivity: {CONDUCTIVITY.voigt_conductivity:.6f}\n")
            file.write(f"Reuss Conductivity: {CONDUCTIVITY.reuss_conductivity:.6f}\n")
            file.write(f"Hill Conductivity: {CONDUCTIVITY.hill_conductivity:.6f}\n")
            file.write(f"EMA Conductivity: {CONDUCTIVITY.EMA_conductivity:.6f}\n\n")
            file.write(f"Elasticity tensor:\n{ELASTICITY.meso_C}\n")
            file.write(f"Voigt bulk: {ELASTICITY.voigt_bulk:.6f}\n")
            file.write(f"Reuss bulk: {ELASTICITY.reuss_bulk:.6f}\n")
            file.write(f"Hill bulk: {ELASTICITY.hill_bulk:.6f}\n")
            file.write(f"EMA bulk: {ELASTICITY.EMA_bulk:.6f}\n\n")
            file.write(f"Voigt shear: {ELASTICITY.voigt_shear:.6f}\n")
            file.write(f"Reuss shear: {ELASTICITY.reuss_shear:.6f}\n")
            file.write(f"Hill shear: {ELASTICITY.hill_shear:.6f}\n")
            file.write(f"EMA shear: {ELASTICITY.EMA_shear:.6f}\n")
            file.write("\nLattice\n")
            for row in METRIC.Lattice:
                file.write(f"{row[0]:.3f} {row[1]:.3f} {row[2]:.3f}\n")
        self.log('Dumping density...')
        if SOL.run_type == 'opt':
            density_file_path = os.path.join(self.mainDir, "density_final.pt")
            torch.save(rho, density_file_path)
        elif SOL.run_type == 'single':
            density_file_path = os.path.join(self.mainDir, "density_binarized.pt")
            torch.save(rho_binarized, density_file_path)
        self.log('Dumping conductivity and shear modulus... (Note: these will likely not match the previous shown values since we binarize the density and use SIMP exponent = 1 for the final observable calculations.)')
        with open(os.path.join(self.mainDir, "observables.txt"), "w") as file:
            file.write(f"Observables for binarized density field with volume fraction {SOL.volFrac_binarized:.6f}\n")
            file.write(f"Voigt conductivity: {CONDUCTIVITY.voigt_conductivity:.6f}\n")
            file.write(f"Reuss conductivity: {CONDUCTIVITY.reuss_conductivity:.6f}\n")
            file.write(f"Hill conductivity: {CONDUCTIVITY.hill_conductivity:.6f}\n")
            file.write(f"EMA conductivity: {CONDUCTIVITY.EMA_conductivity:.6f}\n\n")
            file.write(f"Voigt bulk: {ELASTICITY.voigt_bulk:.6f}\n")
            file.write(f"Reuss bulk: {ELASTICITY.reuss_bulk:.6f}\n")
            file.write(f"Hill bulk: {ELASTICITY.hill_bulk:.6f}\n")
            file.write(f"EMA bulk: {ELASTICITY.EMA_bulk:.6f}\n\n")
            file.write(f"Voigt shear: {ELASTICITY.voigt_shear:.6f}\n")
            file.write(f"Reuss shear: {ELASTICITY.reuss_shear:.6f}\n")
            file.write(f"Hill shear: {ELASTICITY.hill_shear:.6f}\n")
            file.write(f"EMA shear: {ELASTICITY.EMA_shear:.6f}\n")
            file.write("\nLattice\n")
            for row in METRIC.Lattice:
                file.write(f"{row[0]:.3f} {row[1]:.3f} {row[2]:.3f}\n")
        self.log("All done!")