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
        elif key == 'dataset_path':
            dataset_path = parts[1]

n_shear = 1

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
        zmu = 100
        D_k = helper.D_k
        D_mu = helper.D_mu
        D_sig = helper.D_sig

        def nonlinfun(x, alpha, T):
            A = D_mu * zmu
            B = D_k + alpha * D_sig
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
        k1 = self.helper.phase_props.k1
        k2 = self.helper.phase_props.k2
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

        k_eff = self.helper.inv_y_transform(yopt[:,0], phase_props.f, k1, k2)
        sig_eff = self.helper.inv_y_transform(yopt[:,1], phase_props.f, sig1, sig2)

        return k_eff, sig_eff

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

    k_lower_HS, sig_lower_HS = phase_props.k_lower_HS, phase_props.sig_lower_HS
    k_upper_HS, sig_upper_HS = phase_props.k_upper_HS, phase_props.sig_upper_HS

    midpt_k_lower  = 0.5*(klow + k_lower_HS)
    midpt_k_upper  = 0.5*(khigh + k_upper_HS)
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
        k_eff, sig_eff = assem.convert_to_physical(phase_props)

        in_box = (
            (k_eff >= k_lower_HS) & (k_eff <= k_upper_HS) &
            (sig_eff >= sig_lower_HS) & (sig_eff <= sig_upper_HS)
        )

        if first_translation:
            plot_masked_segments(
                ax1, sig_eff, k_eff,  in_box,
                linewidth=2.8, color="blue", label="Translation"
            )
            first_translation = False
        else:
            plot_masked_segments(
                ax1, sig_eff, k_eff,  in_box,
                linewidth=2.8, color="blue"
            )

    # ---- Gibiansky-Torquato (kept as you had it) ----
    sig_GT1, k_GT1 = phase_props.sig_GT1, phase_props.k_GT1
    sig_GT2, k_GT2 = phase_props.sig_GT2, phase_props.k_GT2
    sig_GT3, k_GT3 = phase_props.sig_GT3, phase_props.k_GT3
    sig_GT4, k_GT4 = phase_props.sig_GT4, phase_props.k_GT4
    sig_GT5, k_GT5 = phase_props.sig_GT5, phase_props.k_GT5
    # # use 2 and 4 for well our ordered
    # ax1.plot(sig_GT2, k_GT2, 'r--', label="Gibiansky-Torquato", linewidth=1.8)
    # ax1.plot(sig_GT4, k_GT4, 'r--', linewidth=1.8)
    # use 4 and 5 for badly ordered
    ax1.plot(sig_GT4, k_GT4, 'r--', label="Gibiansky-Torquato", linewidth=1.8)
    ax1.plot(sig_GT5, k_GT5, 'r--', linewidth=1.8)

    # ax1.plot(sig_GT1, k_GT1, label="Gibiansky-Torquato 1", linewidth=1.8)
    # ax1.plot(sig_GT2, k_GT2, label="Gibiansky-Torquato 2", linewidth=1.8)
    # ax1.plot(sig_GT3, k_GT3, label="Gibiansky-Torquato 3", linewidth=1.8)
    # ax1.plot(sig_GT4, k_GT4, label="Gibiansky-Torquato 4", linewidth=1.8)
    # ax1.plot(sig_GT5, k_GT5, label="Gibiansky-Torquato 5", linewidth=1.8)



    # ============================
    # HASHIN–SHTRIKMAN: draw finite segments (rectangle)
    # ============================
    # bottom edge
    ax1.plot([sig_lower_HS, sig_lower_HS], [k_lower_HS, k_upper_HS], 
             linestyle="-", color="black", label="Hashin-Shtrikman", linewidth=2)
    # top edge
    ax1.plot([sig_upper_HS, sig_upper_HS], [k_lower_HS, k_upper_HS], 
             linestyle="-", color="black", linewidth=2)
    # left edge
    ax1.plot([sig_lower_HS, sig_upper_HS], [k_lower_HS, k_lower_HS], 
             linestyle="-", color="black", linewidth=2)
    # right edge
    ax1.plot([sig_lower_HS, sig_upper_HS], [k_upper_HS, k_upper_HS],
             linestyle="-", color="black", linewidth=2)

    # ---- Dataset (unchanged) ----
    if dataset is not None:
        ax1.scatter(dataset['conductivity'], dataset['shear_modulus'],
                    color='orange', marker='o', s=60, alpha=0.4,
                    label="Topology-optimized")

    ax1.set_xlim(midpt_sig_lower, midpt_sig_upper)
    ax1.set_ylim(midpt_k_lower, midpt_k_upper)
    ax1.set_xlabel(r'$\sigma^{\star}$', fontsize=28)
    ax1.set_ylabel(r'$\kappa^{\star}$', fontsize=28)
    ax1.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\n✅ Final overlay plot saved to: {out_path}")


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
    out_path=f"{out_dir_path}/bounds_f={phase_props.f}.png"
)
