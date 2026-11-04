import os
import argparse
import shutil
import numpy as np
# ask PyTorch to use expandable segments for its CUDA allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # expose only the second GPU to this process
import torch
from modules.algorithms import minimizer, project_density, project_ucparams
from modules.classes import INFILEclass, SOLclass, METRICclass, CONDUCTIVITYclass, ELASTICITYclass, DIRclass
from modules.dump import projection, isosurface

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# limit intra-op threads to avoid oversubscription for cpu calcs
if device.type == 'cpu':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

# Obtain paths of input file and output directory from command line arguments
parser = argparse.ArgumentParser(
    description="Run topology optimization an input file and dump results into a directory"
)
parser.add_argument(
    "infile_path",
    help="path to the input file"
)
parser.add_argument(
    "out_dir_path",
    help="directory where results will be written"
)
args = parser.parse_args()
infile_path  = args.infile_path
out_dir_path  = args.out_dir_path

# Initialize INFILE and DIR classes
INFILE = INFILEclass(infile_path, out_dir_path)
DIR = DIRclass(INFILE)
# Write input to outfile
INFILE.write_input(DIR)
# Initialize other classes
METRIC = METRICclass(INFILE)
SOL = SOLclass(INFILE, METRIC, DIR)
CONDUCTIVITY = CONDUCTIVITYclass(SOL)
ELASTICITY = ELASTICITYclass(SOL)


if INFILE.run_type == 'opt':
    # Volume fraction constraint for conductive species
    c = SOL.volFrac*SOL.gridShape[0]*SOL.gridShape[1]*SOL.gridShape[2]
    # Anneal SIMP_exp
    SIMP_exp_conductivity_init, SIMP_exp_conductivity_final, SIMP_exp_shear_init, SIMP_exp_shear_final, SIMP_exp_step = INFILE.SIMP_exp[0], INFILE.SIMP_exp[1], INFILE.SIMP_exp[2], INFILE.SIMP_exp[3], INFILE.SIMP_exp[4]
    SIMP_exp_conductivity_list = np.arange(SIMP_exp_conductivity_init, SIMP_exp_conductivity_final + SIMP_exp_step, SIMP_exp_step)
    SIMP_exp_shear_list = np.arange(SIMP_exp_shear_init, SIMP_exp_shear_final + SIMP_exp_step, SIMP_exp_step)
    if len(SIMP_exp_conductivity_list) != len(SIMP_exp_shear_list):
        raise ValueError("SIMP exponent lists must have the same length")
    SOL.initial_iter = 0
    for SIMP_exp_conductivity, SIMP_exp_shear in zip(SIMP_exp_conductivity_list, SIMP_exp_shear_list):
        DIR.log(f"\nSetting SIMP exponent for conductivity to {SIMP_exp_conductivity:.2f} and for shear modulus to {SIMP_exp_shear:.2f}\n")
        SOL.SIMP_exp_conductivity = SIMP_exp_conductivity
        SOL.SIMP_exp_shear = SIMP_exp_shear
        # Run optimization
        density_minimizer = minimizer(device, SOL.x0.numel())
        DIR.log("Optimizing density field...")
        _, SOL.x0 = density_minimizer.PBFGS(fun_jac=SOL.fun_jac_density, x0=SOL.x0, BFGS_hist=SOL.hist, args=(CONDUCTIVITY, ELASTICITY, DIR, METRIC), project=project_density, proj_args=(c,), callback=SOL.callback_density, maxit=SOL.maxit, tol=SOL.tol)
        # ucparams_minimizer = minimizer(device, 1)
        # DIR.log("\nOptimizing unit cell parameters...")
        # _, METRIC.ucparams = ucparams_minimizer.PBFGS(fun_jac=SOL.fun_jac_ucparams, x0=METRIC.ucparams, BFGS_hist=SOL.hist, args=(SOL.x0, CONDUCTIVITY, ELASTICITY, DIR, METRIC), project=project_ucparams, proj_args=(), callback=SOL.callback_ucparams, maxit=SOL.maxit, tol=SOL.tol)
        SOL.final_obs(SOL.x0, CONDUCTIVITY, ELASTICITY, DIR, METRIC)

        # Dump results after optimization
        if abs(SIMP_exp_conductivity - SIMP_exp_conductivity_final) < 1e-6 and abs(SIMP_exp_shear - SIMP_exp_shear_final) < 1e-6:
            # Dump final results
            DIR.dumpEnd(SOL, SOL.x0, CONDUCTIVITY, ELASTICITY, METRIC)
        else:
            ucparams = METRIC.ucparams
            rho = SOL.unflatten(SOL.x0)
            DIR.log("\nDumping plots...")
            projection(rho, out_dir=DIR.projDir, iter_num=int(SOL.iter + SOL.initial_iter))
            isosurface(rho, out_dir=DIR.isoDir, iter_num=int(SOL.iter + SOL.initial_iter), tau=0.8)
            DIR.log("Dumping density...")
            density_file_path = os.path.join(DIR.densDir, f"density_iter_{int(SOL.iter + SOL.initial_iter)}.pt")
            torch.save(rho, density_file_path)
            DIR.log("Dumping observables...")
            with open(os.path.join(DIR.obsDir, f"obs_iter_{int(SOL.iter + SOL.initial_iter)}.txt"), "a") as file:
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
            DIR.log("Resuming optimization...")
            # Update iteration
            SOL.initial_iter += SOL.iter
    
elif INFILE.run_type == 'single':
    SOL.initial_iter = 0
    # Run single conductivity/ealsticity calculation
    SOL.final_obs(SOL.x0, CONDUCTIVITY, ELASTICITY, DIR, METRIC)
    DIR.dumpEnd(SOL, SOL.x0, CONDUCTIVITY, ELASTICITY, METRIC)