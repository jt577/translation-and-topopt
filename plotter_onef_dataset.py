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

from helper import TranslationHelper
from misc import phase_props_holder

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

dataset_path = None
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
        elif key == 'dataset_paths':
            dataset_paths = parts[1:]
        elif key == 'special_dataset_paths':
            special_dataset_paths = parts[1:]

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

        N = 600
        alpha_min = 0.001
        alpha_max = 200.0
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

def plot_all_bounds_overlay(assemblers, phase_props, datasets=None, special_datasets=None, out_path="translation_bounds_overlay.png"):
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

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))

    # ----------------------------
    # helper: plot only in-box segments
    # ----------------------------
    def plot_masked_segments(ax, x, y, mask, **plot_kwargs):
        """
        Plot contiguous segments where mask==True.
        Avoids connecting across gaps.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.asarray(mask, dtype=bool)

        # find contiguous True runs
        idx = np.where(mask)[0]
        if idx.size == 0:
            return

        # split where gaps > 1
        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)

        first = True
        for run in runs:
            if run.size < 2:
                continue
            if first:
                ax.plot(x[run], y[run], **plot_kwargs)
                first = False
            else:
                # don't repeat label
                if "label" in plot_kwargs:
                    tmp = dict(plot_kwargs)
                    tmp.pop("label", None)
                    ax.plot(x[run], y[run], **tmp)
                else:
                    ax.plot(x[run], y[run], **plot_kwargs)

    # ============================
    # TRANSLATION BOUNDS (BLUE) + legend entry + CLIP to HS box
    # ============================
    first_translation = True
    for assem in assemblers.values():
        mu_eff, sig_eff = assem.convert_to_physical(phase_props)

        in_box = (
            (mu_eff >= mu_lower_HS) & (mu_eff <= mu_upper_HS) &
            (sig_eff >= sig_lower_HS) & (sig_eff <= sig_upper_HS)
        )

        if first_translation:
            plot_masked_segments(
                ax1, sig_eff, mu_eff,  in_box,
                linewidth=1.8, color="blue"
            )
            first_translation = False
        else:
            plot_masked_segments(
                ax1, sig_eff, mu_eff,  in_box,
                linewidth=1.8, color="blue"
            )

    # ============================
    # HASHIN–SHTRIKMAN: draw finite segments (rectangle)
    # ============================
    # bottom edge
    ax1.plot([sig_lower_HS, sig_lower_HS], [mu_lower_HS, mu_upper_HS], 
             linestyle="-", color="black", linewidth=2)
    # top edge
    ax1.plot([sig_upper_HS, sig_upper_HS], [mu_lower_HS, mu_upper_HS], 
             linestyle="-", color="black", linewidth=2)
    # left edge
    ax1.plot([sig_lower_HS, sig_upper_HS], [mu_lower_HS, mu_lower_HS], 
             linestyle="-", color="black", linewidth=2)
    # right edge
    ax1.plot([sig_lower_HS, sig_upper_HS], [mu_upper_HS, mu_upper_HS],
             linestyle="-", color="black", linewidth=2)

    # ---- Dataset (unchanged) ----
    if datasets is not None:
        counter = 0
        for dataset in datasets:
            if counter == 0:
                ax1.scatter(dataset['conductivity'], dataset['shear_modulus'],
                            color='orange', marker='o', s=50, alpha=0.3,
                            label="Truss")
            else:
                ax1.scatter(dataset['conductivity'], dataset['shear_modulus'],
                            color='orange', marker='o', s=50, alpha=0.3)
            counter += 1
                    
    if special_datasets is not None:
        colors = ['purple', 'red', 'cyan', 'green', 'magenta']
        markers = ['P', 'D', 's', 'X', '*']
        labels = ["TPMS", "P-Bridged"]
        for idx_sd, special_dataset in enumerate(special_datasets):
            x = np.array(special_dataset['conductivity'])
            y = np.array(special_dataset['shear_modulus'])

            idx = np.argsort(x)  # sort by x
            x, y = x[idx], y[idx]
            ax1.scatter(x, y, color=colors[idx_sd], marker=markers[idx_sd], s=50, alpha=1.0, label=labels[idx_sd])

    ax1.set_xlim(midpt_sig_lower, midpt_sig_upper)
    ax1.set_ylim(midpt_mu_lower, midpt_mu_upper)
    ax1.set_xlabel(r'$\sigma^{\star}$', fontsize=20)
    ax1.set_ylabel(r'$\mu^{\star}$', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    print(f"\n✅ Final overlay plot saved to: {out_path}")


# ============================================================
# BUILD ALL ASSEMBLERS
# ============================================================

phase_props = phase_props_holder(k1, k2, mu1, mu2, sig1, sig2, f)
datasets = []
for dataset_path in dataset_paths:
    dataset = pd.read_csv(dataset_path) if dataset_path else None
    datasets.append(dataset)

special_datasets = []
for special_dataset_path in special_dataset_paths:
    special_dataset = pd.read_csv(special_dataset_path) if special_dataset_path else None
    special_datasets.append(special_dataset)

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
    datasets=datasets,
    special_datasets=special_datasets,
    out_path=f"{out_dir_path}/bounds_f={phase_props.f}_truss_datasets.png"
)
