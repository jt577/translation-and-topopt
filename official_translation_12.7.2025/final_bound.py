# ============================================================
# Use a collection of optimal translation matrices to assemble
# a SINGLE OVERLAY plot of ALL bounds (no duplication)
# WITH ZOOMED DATASET PANEL + POINT LABELS
# ============================================================

from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, argparse
import torch

from modules.helper import TranslationHelper
from modules.misc import phase_props_holder

# ============================================================
# ARGUMENT PARSING
# ============================================================

parser = argparse.ArgumentParser(
    description="Overlay all translation bounds into a single plot"
)
parser.add_argument("infile_path", help="path to the input file")
parser.add_argument("out_dir_path", help="directory where results will be written")
args = parser.parse_args()

infile_path   = args.infile_path
out_dir_path  = args.out_dir_path
os.makedirs(out_dir_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# READ INPUT FILE
# ============================================================

with open(infile_path, 'r') as openfile:
    lines = openfile.readlines()
    for line in lines:
        line = line.split('#', 1)[0].strip()
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
        elif key == 'bound_types':
            bound_types = parts[1:]
        elif key == 'calc_type':
            calc_type = parts[1]
        elif key == 'T_paths':
            T_paths = parts[1:]
        elif key == 'dataset_path':
            dataset_path = parts[1]

if calc_type == 'bulk':
    n_shear = 1
elif calc_type == 'shear':
    n_shear = 5

# ============================================================
# ASSEMBLER CLASS
# ============================================================

class Assembler():
    def __init__(self, T_path, helper, bound_type):
        self.T = np.load(T_path)
        self.helper = helper
        self.bound_type = bound_type
        self.zopt = self._solve_nonlinear()

    def _solve_nonlinear(self):
        helper = self.helper
        zk = 100
        D_k = helper.D_k
        D_mu = helper.D_mu
        D_sig = helper.D_sig

        def nonlinfun(x, alpha, T):
            A = D_k * zk
            B = D_mu + alpha * D_sig
            eigvals = np.linalg.eigh(A + B * x + T)[0]
            return eigvals[0]

        N = 100
        alpha_min = 0.01
        alpha_max = 100.0
        t = np.linspace(0, 1, N)
        alphas = alpha_min * (alpha_max/alpha_min)**t

        zopt = []
        for alpha in alphas:
            z_sol = brentq(
                f=nonlinfun,
                args=(alpha, self.T),
                a=0.0,
                b=500.0,
                xtol=1e-10,
                maxiter=100
            )
            zopt.append([z_sol, alpha * z_sol])

        return np.array(zopt)

    def convert_to_physical(self, phase_props):
        mu1 = self.helper.phase_props.mu1
        mu2 = self.helper.phase_props.mu2
        sig1 = self.helper.phase_props.sig1
        sig2 = self.helper.phase_props.sig2

        if self.bound_type == 'upper_1':
            yopt = 1/self.zopt
        elif self.bound_type == 'upper_2':
            yopt = np.zeros(self.zopt.shape)
            yopt[:,0] = self.zopt[:,0]
            yopt[:,1] = 1/self.zopt[:,1]
        elif self.bound_type == 'lower_1':
            yopt = np.zeros(self.zopt.shape)
            yopt[:,0] = 1/self.zopt[:,0]
            yopt[:,1] = self.zopt[:,1]
        elif self.bound_type == 'lower_2':
            yopt = self.zopt

        mu_eff = self.helper.inv_y_transform(yopt[:,0], phase_props.f, mu1, mu2)
        sig_eff = self.helper.inv_y_transform(yopt[:,1], phase_props.f, sig1, sig2)

        return mu_eff, sig_eff

# ============================================================
# TWO-PANEL OVERLAY + ZOOM + LABELS
# ============================================================

def plot_all_bounds_overlay(assemblers, phase_props, dataset=None, out_path="translation_bounds_overlay.png"):
    k1, k2 = phase_props.k1, phase_props.k2
    mu1, mu2 = phase_props.mu1, phase_props.mu2
    sig1, sig2 = phase_props.sig1, phase_props.sig2
    f = phase_props.f

    klow  = 1/(f*1/k1  + (1-f)*1/k2)
    mulow = 1/(f*1/mu1 + (1-f)*1/mu2)
    siglow= 1/(f*1/sig1+ (1-f)*1/sig2)

    khigh  = f*k1  + (1-f)*k2
    muhigh = f*mu1 + (1-f)*mu2
    sighigh= f*sig1+ (1-f)*sig2

    mu_lower_HS, sig_lower_HS = phase_props.mu_lower_HS, phase_props.sig_lower_HS
    mu_upper_HS, sig_upper_HS = phase_props.mu_upper_HS, phase_props.sig_upper_HS

    midpt_mu_lower  = 0.5*(mulow + mu_lower_HS)
    midpt_mu_upper  = 0.5*(muhigh + mu_upper_HS)
    midpt_sig_lower = 0.5*(siglow + sig_lower_HS)
    midpt_sig_upper = 0.5*(sighigh + sig_upper_HS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # ============================
    # LEFT PANEL: FULL BOUNDS
    # ============================
    for assem in assemblers.values():
        mu_eff, sig_eff = assem.convert_to_physical(phase_props)
        ax1.plot(mu_eff, sig_eff, linewidth=2, color='blue')

    # ---- BERRYMAN-MILTON WITH FILTERING (LEFT) ----
    counter = 0
    for bound_type in assemblers.keys():
        if bound_type == 'upper_1':
            sig_BM, mu_BM = phase_props.sig_BM1, phase_props.mu_BM1
        elif bound_type == 'lower_2':
            sig_BM, mu_BM = phase_props.sig_BM2, phase_props.mu_BM2
        elif bound_type == 'upper_2':
            sig_BM, mu_BM = phase_props.sig_BM3, phase_props.mu_BM3
        elif bound_type == 'lower_1':
            sig_BM, mu_BM = phase_props.sig_BM4, phase_props.mu_BM4
        else:
            continue

        if counter == 0:
            ax1.plot(mu_BM, sig_BM, 'r--', label="Berryman-Milton")
        else:
            ax1.plot(mu_BM, sig_BM, 'r--')
        counter += 1

    # ---- HASHIN–SHTRIKMAN (LEFT) ----
    ax1.axvline(mu_lower_HS, linestyle="--", color='black', label="Hashin-Shtrikman")
    ax1.axhline(sig_lower_HS, linestyle="--", color='black')
    ax1.axvline(mu_upper_HS, linestyle="--", color='black')
    ax1.axhline(sig_upper_HS, linestyle="--", color='black')

    if dataset is not None:
        ax1.scatter(dataset['shear_modulus'], dataset['conductivity'], 
                    color='orange', marker='x', s=60, alpha=0.6,
                    label="Topology-optimized designs")

    ax1.set_ylim(midpt_sig_lower, midpt_sig_upper)
    ax1.set_xlim(midpt_mu_lower, midpt_mu_upper)
    ax1.set_ylabel(r'$\sigma_{\mathrm{eff}}$')
    ax1.set_xlabel(r'$\mu_{\mathrm{eff}}$')
    ax1.legend()

    # ============================
    # RIGHT PANEL: ZOOM + LABELS
    # ============================
    for assem in assemblers.values():
        mu_eff, sig_eff = assem.convert_to_physical(phase_props)
        ax2.plot(mu_eff, sig_eff, linewidth=2, color='blue')

    # ---- BERRYMAN-MILTON WITH FILTERING (RIGHT, NO LABELS) ----
    for bound_type in assemblers.keys():
        if bound_type == 'upper_1':
            sig_BM, mu_BM = phase_props.sig_BM1, phase_props.mu_BM1
        elif bound_type == 'lower_2':
            sig_BM, mu_BM = phase_props.sig_BM2, phase_props.mu_BM2
        elif bound_type == 'upper_2':
            sig_BM, mu_BM = phase_props.sig_BM3, phase_props.mu_BM3
        elif bound_type == 'lower_1':
            sig_BM, mu_BM = phase_props.sig_BM4, phase_props.mu_BM4
        else:
            continue

        ax2.plot(mu_BM, sig_BM, 'r--')

    # ---- HASHIN–SHTRIKMAN (RIGHT) ----
    ax2.axvline(mu_lower_HS, linestyle="--", color='black')
    ax2.axhline(sig_lower_HS, linestyle="--", color='black')
    ax2.axvline(mu_upper_HS, linestyle="--", color='black')
    ax2.axhline(sig_upper_HS, linestyle="--", color='black')

    if dataset is not None:
        y = dataset['conductivity'].values
        x = dataset['shear_modulus'].values
        names = dataset['morphology'].values

        ax2.scatter(x, y, color='orange', marker='x', s=60, alpha=0.9)

        # --- Zoom bounds with padding ---
        pad_x = 0.15*(x.max() - x.min())
        pad_y = 0.15*(y.max() - y.min())

        ax2.set_xlim(x.min()-pad_x, x.max()+pad_x)
        ax2.set_ylim(y.min()-pad_y, y.max()+pad_y)

        # --- Label each morphology ---
        for xi, yi, name in zip(x, y, names):
            ax2.text(xi, yi, name, fontsize=9, ha='left', va='bottom')

    ax2.set_ylabel(r'$\sigma_{\mathrm{eff}}$')
    ax2.set_xlabel(r'$\mu_{\mathrm{eff}}$')
    ax2.set_title("Zoom on Topology-Optimized Designs")

    fig.suptitle(
        rf'$\sigma_1={sig1}, \sigma_2={sig2}, \kappa_1={k1}, \kappa_2={k2}, \mu_1={mu1}, \mu_2={mu2}, f={f}$',
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\n✅ Final two-panel overlay plot saved to: {out_path}")

# ============================================================
# BUILD ALL ASSEMBLERS
# ============================================================

phase_props = phase_props_holder(k1, k2, mu1, mu2, sig1, sig2, f)
dataset = pd.read_csv(dataset_path) if dataset_path else None

assemblers = {}

for bound_type, T_path in zip(bound_types, T_paths):

    helper = TranslationHelper(
        n_shear=n_shear,
        phase_props=phase_props,
        npts=10,
        upper=40,
        n_samples=50,
        alphas=(),
        seed=2387,
        calc_type=calc_type,
        bound_type=bound_type
    )

    assemblers[bound_type] = Assembler(T_path, helper, bound_type)

# ============================================================
# FINAL OVERLAY PLOT
# ============================================================

plot_all_bounds_overlay(
    assemblers,
    phase_props,
    dataset=dataset,
    out_path=f"{out_dir_path}/translation_bounds_overlay.png"
)
