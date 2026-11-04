import os, shutil, argparse
import numpy as np
import torch
from modules.logging import log_results
from modules.minimizer import TranslationMinimizer
from modules.helper import TranslationHelper
from modules.objfun import objfun
from modules.plotting import plot_results
from modules.misc import phase_props_holder

# Obtain paths of input file and output directory from command line arguments
parser = argparse.ArgumentParser(
    description="Run translation method on an input file and dump results into a directory"
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

# Check if output directory already exists
if os.path.exists(out_dir_path):
    raise FileExistsError(
        f"Error: The output directory '{out_dir_path}' already exists.\n"
        "Please choose a different directory or delete the existing one to proceed."
    )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

if os.path.exists("out.txt"):
    os.remove("out.txt")

if os.path.exists(f"{out_dir_path}/z_plots"):
    shutil.rmtree(f"{out_dir_path}/z_plots")
if os.path.exists(f"{out_dir_path}/star_plots"):
    shutil.rmtree(f"{out_dir_path}/star_plots")
if os.path.exists(f"{out_dir_path}/translation_matrices"):
    shutil.rmtree(f"{out_dir_path}/translation_matrices")
if os.path.exists(f"{out_dir_path}/translation_matrix_visuals"):
    shutil.rmtree(f"{out_dir_path}/translation_matrix_visuals")

os.makedirs(f"{out_dir_path}/z_plots")
os.makedirs(f"{out_dir_path}/star_plots")
os.makedirs(f"{out_dir_path}/translation_matrices")
os.makedirs(f"{out_dir_path}/translation_matrix_visuals")

# read input file
with open(infile_path, 'r') as openfile:
    lines = openfile.readlines()
    T0_path = None
    for line in lines:
        # Remove comments and trailing whitespace
        line = line.split('#', 1)[0].strip()
        
        # Skip empty or whitespace-only lines
        if not line:
            continue

        parts = line.split()
        key = parts[0]

        if key == 'k1':
            k1 = float(parts[1])
        elif key == 'k2':
            k2 = float(parts[1])
        elif key == 'mu1':
            mu1 = float(parts[1])
        elif key == 'mu2':
            mu2 = float(parts[1])
        elif key == 'sig1':
            sig1 = float(parts[1])
        elif key == 'sig2':
            sig2 = float(parts[1])
        elif key == 'f':
            f = float(parts[1])
        elif key == 'upper':
            upper = float(parts[1])
        elif key == 'npts':
            npts = int(parts[1])
        elif key == 'n_samples':
            n_samples = int(parts[1])
        elif key == 'calc_type':
            calc_type = parts[1]
        elif key == 'bound_type':
            bound_type = parts[1]
        elif key == 'T0':
            T0_path = parts[1]
        elif key == 'alphas':
            alphas = [float(val) for val in parts[1:]]
            alphas = np.array(alphas)


# write input to outfile
with open(f"{out_dir_path}/out.txt", 'w') as openfile:
    openfile.write("Input parameters:\n")
    openfile.write(f"k1 = {k1}\n")
    openfile.write(f"k2 = {k2}\n")
    openfile.write(f"mu1 = {mu1}\n")
    openfile.write(f"mu2 = {mu2}\n")
    openfile.write(f"sig1 = {sig1}\n")
    openfile.write(f"sig2 = {sig2}\n")
    openfile.write(f"f = {f}\n")
    openfile.write(f"upper = {upper}\n")
    openfile.write(f"npts = {npts}\n")
    openfile.write(f"n_samples = {n_samples}\n")
    openfile.write(f"calc_type = {calc_type}\n")
    openfile.write(f"bound_type = {bound_type}\n")
    if T0_path is not None:
        openfile.write(f"T0 = {T0_path}\n")
    openfile.write('--------------------------------------------------\n')

if calc_type == 'bulk':
    n_shear = 1
elif calc_type == 'shear':
    n_shear = 5
phase_props = phase_props_holder(k1, k2, mu1, mu2, sig1, sig2, f)
logger = log_results(out_dir_path)
helper = TranslationHelper(n_shear, phase_props, npts, upper, n_samples, alphas, seed=2387, calc_type=calc_type, bound_type=bound_type)
plotter = plot_results(out_dir_path, helper)

# Load initial translation matrix
if T0_path is not None:
    T0 = np.load(T0_path)
    t0 = torch.tensor(T0.flatten(), dtype=torch.double, device=device) # initial flattened translation matrix
else:
    n = n_shear*6 + 9 # size of translation matrix
    T0 =  np.random.randn(n, n)
    t0 = torch.tensor(T0.flatten(), dtype=torch.double, device=device)

# Run minimization
minimizer = TranslationMinimizer(fun_jac=objfun, x0=t0, args=(helper,), plotter=plotter, logger=logger, phase_props=phase_props, helper=helper, callback=None, BFGS_hist=20, maxit=1000, tol=1e-3, max_line_search=5)
obj, t = minimizer.PBFGS()
T_opt = t.cpu().numpy().reshape(n_shear*6 + 9, n_shear*6 + 9)
np.save(f"{out_dir_path}/T_final.npy", T_opt)