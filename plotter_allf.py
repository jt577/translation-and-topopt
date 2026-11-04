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

if calc_type == 'bulk':
    n_shear = 1
elif calc_type == 'shear':
    n_shear = 5


# class to hold phase properties
class phase_props_holder:
    def __init__(self, k1, k2, mu1, mu2, sig1, sig2):
        self.k1 = k1
        self.k2 = k2
        self.mu1 = mu1
        self.mu2 = mu2
        self.sig1 = sig1
        self.sig2 = sig2

    def _hashin_shtrikman(self, f):
        """
        Compute Hashin-Shtrikman upper and lower bound.
        """
        # Elasticity
        k_min_star = 4/3 * min(self.mu1, self.mu2)
        k_max_star = 4/3 * max(self.mu1, self.mu2)
        H1 = self.mu1 * (3/2 * self.k1 + 4/3 * self.mu1) / (self.k1 + 2 * self.mu1)
        H2 = self.mu2 * (3/2 * self.k2 + 4/3 * self.mu2) / (self.k2 + 2 * self.mu2)
        mu_min_star = min(H1, H2)
        mu_max_star = max(H1, H2)
        self.k_lower_HS = 1/(f / (k_min_star + self.k1) + (1-f) / (k_min_star + self.k2)) - k_min_star
        self.k_upper_HS = 1/(f / (k_max_star + self.k1) + (1-f) / (k_max_star + self.k2)) - k_max_star
        self.mu_lower_HS = 1/(f / (mu_min_star + self.mu1) + (1-f) / (mu_min_star + self.mu2)) - mu_min_star
        self.mu_upper_HS = 1/(f / (mu_max_star + self.mu1) + (1-f) / (mu_max_star + self.mu2)) - mu_max_star

        # Conductivity
        sig_min = min(self.sig1, self.sig2)
        sig_max = max(self.sig1, self.sig2)
        self.sig_upper_HS = f*self.sig1 + (1-f)*self.sig2 - f*(1-f)*(self.sig2-self.sig1)**2/(f*self.sig2 + (1-f)*self.sig1 + 2*sig_max)
        self.sig_lower_HS = f*self.sig1 + (1-f)*self.sig2 - f*(1-f)*(self.sig2-self.sig1)**2/(f*self.sig2 + (1-f)*self.sig1 + 2*sig_min)

    def _berryman_milton(self, f):
        """
        Compute shear-conductivity bound derived from Berryman-Milton.
        """
        def sig_upper(sig1, sig2, f, eta):
            # Assumes sig1 > sig2
            sig_max = max(sig1, sig2)
            sig_min = min(sig1, sig2)
            sig_tilde = (1-f)*sig1 + f*sig2
            sig_avg = f*sig1 + (1-f)*sig2
            sig_eta = sig_max*(1-eta) + sig_min*eta
            return sig_avg - f*(1-f)*(sig1 - sig2)**2 / (sig_tilde + 2*sig_eta)
        
        def sig_lower(sig1, sig2, f, eta):
            # Assumes sig1 > sig2
            sig_max = max(sig1, sig2)
            sig_min = min(sig1, sig2)
            sig_tilde = (1-f)*sig1 + f*sig2
            sig_avg = f*sig1 + (1-f)*sig2
            sig_eta = 1/((1-eta)/sig_max + eta/sig_min)
            return sig_avg - f*(1-f)*(sig1 - sig2)**2 / (sig_tilde + 2*sig_eta)
        
        def mu_upper(mu1, mu2, k1, k2, f, eta):
            # Assumes mu2 > mu1
            max_mu = max(mu1, mu2)
            def eta_avg(expr1, expr2):
                return (1-eta)*expr1 + eta*expr2
            eta_avg_mu = eta_avg(mu1, mu2)
            eta_avg_k = eta_avg(k1, k2)
            eta_avg_6k7mu = eta_avg(6*k1 + 7*mu1, 6*k2 + 7*mu2)
            THETA = 1/6 * (24*max_mu*eta_avg_6k7mu + 45*eta_avg_mu*eta_avg_k)/(21*eta_avg_k + 2*eta_avg_mu + 40*max_mu)
            mu_tilde = f*mu2 + (1-f)*mu1
            mu_avg = f*mu1 + (1-f)*mu2
            return mu_avg - f*(1-f)*(mu1 - mu2)**2 / (mu_tilde + THETA)
        
        def mu_lower(mu1, mu2, k1, k2, f, eta):
            mu1_hat = 1/mu1
            mu2_hat = 1/mu2
            k1_hat = 1/k1
            k2_hat = 1/k2
            # Assumes mu2 > mu1
            min_mu = min(mu1, mu2)
            min_mu_hat = 1/min_mu
            def eta_avg(expr1, expr2):
                return (1-eta)*expr1 + eta*expr2
            eta_avg_mu_hat = eta_avg(mu1_hat, mu2_hat)
            eta_avg_k_hat = eta_avg(k1_hat, k2_hat)
            eta_avg_2k21mu_hat = eta_avg(2*k1_hat + 21*mu1_hat, 2*k2_hat + 21*mu2_hat)
            XI = 2 * (min_mu_hat * eta_avg_2k21mu_hat + 40 * eta_avg_mu_hat * eta_avg_k_hat) / (56*eta_avg_k_hat + 48*eta_avg_mu_hat + 15*min_mu_hat)
            mu_tilde = f*mu2 + (1-f)*mu1
            mu_avg = f*mu1 + (1-f)*mu2
            return mu_avg - f*(1-f)*(mu1 - mu2)**2 / (mu_tilde + 1/XI)

        # CRUCIAL: When you have "well-ordered" material, that is when the phase with higher mechanical moduli also has higher conductivity, you must SWAP the role of one of conductivity or elastic moduli
        # So then you would make self.sig_lower_BM correspond to the upper bound formula, and vice versa
        # So we take the outer-most of the four curves here as our bounds!
        eta = np.linspace(0,1,10001)
        self.sig_BM1 = sig_upper(self.sig1, self.sig2, f, eta)
        self.sig_BM2 = sig_lower(self.sig1, self.sig2, f, eta)
        if self.sig1 >= self.sig2:
            self.mu_BM1 = mu_upper(self.mu1, self.mu2, self.k1, self.k2, f, eta)
            self.mu_BM2 = mu_lower(self.mu1, self.mu2, self.k1, self.k2, f, eta)
        else:
            self.mu_BM1 = mu_upper(self.mu1, self.mu2, self.k1, self.k2, f, 1-eta)
            self.mu_BM2 = mu_lower(self.mu1, self.mu2, self.k1, self.k2, f, 1-eta)
        self.sig_BM3 = sig_lower(self.sig1, self.sig2, f, eta)
        self.sig_BM4 = sig_upper(self.sig1, self.sig2, f, eta)
        if self.sig1 >= self.sig2:
            self.mu_BM3 = mu_upper(self.mu1, self.mu2, self.k1, self.k2, f, eta)
            self.mu_BM4 = mu_lower(self.mu1, self.mu2, self.k1, self.k2, f, eta)
        else:
            self.mu_BM3 = mu_upper(self.mu1, self.mu2, self.k1, self.k2, f, 1-eta)
            self.mu_BM4 = mu_lower(self.mu1, self.mu2, self.k1, self.k2, f, 1-eta)

def make_diag(bound_type):
    # Make diagonal matrices for use in objective function
    if bound_type == 'upper_1':
        D_k_list_elastic = [1/3] +  [0] * 5
        D_k_list_conductive = [0] * 9
        D_mu_list_elastic = [0] +  [1/2] * 5
        D_mu_list_conductive = [0] * 9
        D_sig_list_elastic = [0] * (6 * 5)
        D_sig_list_conductive = [1] * 9
    elif bound_type == 'upper_2':
        D_k_list_elastic = [3] +  [0] * 5
        D_k_list_conductive = [0] * 9
        D_mu_list_elastic = [0] +  [2] * 5
        D_mu_list_conductive = [0] * 9
        D_sig_list_elastic = [0] * (6 * 5)
        D_sig_list_conductive = [1] * 9
    elif bound_type == 'lower_1':
        D_k_list_elastic = [1/3] +  [0] * 5
        D_k_list_conductive = [0] * 9
        D_mu_list_elastic = [0] +  [1/2] * 5
        D_mu_list_conductive = [0] * 9
        D_sig_list_elastic = [0] * (6 * 5)
        D_sig_list_conductive = [1] * 9
    elif bound_type == 'lower_2':
        D_k_list_elastic = [3] +  [0] * 5
        D_k_list_conductive = [0] * 9
        D_mu_list_elastic = [0] +  [2] * 5
        D_mu_list_conductive = [0] * 9
        D_sig_list_elastic = [0] * (6 * 5)
        D_sig_list_conductive = [1] * 9
    
    D_k_list = D_k_list_elastic * 5  + D_k_list_conductive
    D_mu_list = D_mu_list_elastic * 5  + D_mu_list_conductive
    D_sig_list = D_sig_list_elastic + D_sig_list_conductive
    D_k = np.diag(D_k_list)
    D_mu = np.diag(D_mu_list)
    D_sig = np.diag(D_sig_list)
    return D_sig, D_k, D_mu

def inv_y_transform(y, f, x1, x2):
    return f*x1 + (1-f)*x2 - f*(1-f)*(x2-x1)**2 / ((1-f)*x1 + f*x2 + y)

# ============================================================
# ASSEMBLER CLASS
# ============================================================

class Assembler():
    def __init__(self, T_path, phase_props, bound_type):
        self.T = np.load(T_path)
        self.bound_type = bound_type
        self.phase_props = phase_props
        self.zopt = self._solve_nonlinear()

    def _solve_nonlinear(self):
        D_sig, D_k, D_mu = make_diag(self.bound_type)
        zk = 100

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

    def convert_to_physical(self):
        mu1 = self.phase_props.mu1
        mu2 = self.phase_props.mu2
        sig1 = self.phase_props.sig1
        sig2 = self.phase_props.sig2

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

        return yopt

def collect_translation_points(assemblers, phase_props, f_grid, clip_to_hs=True):
    """
    Returns point cloud arrays (sig_pts, mu_pts) for the UNION over:
      - bound_types (assemblers)
      - volume fractions f in f_grid
      - alpha parameter along each curve (via y_opt arrays)
    Optionally clipped to HS box (same logic you already use).
    """
    mu1, mu2 = phase_props.mu1, phase_props.mu2
    sig1, sig2 = phase_props.sig1, phase_props.sig2

    sig_pts = []
    mu_pts  = []

    for assem in assemblers.values():
        y_opt = assem.convert_to_physical()  # shape (Nalpha, 2) in "y-space"

        for f in f_grid:
            mu_eff  = inv_y_transform(y_opt[:, 0], f, mu1,  mu2)
            sig_eff = inv_y_transform(y_opt[:, 1], f, sig1, sig2)

            if clip_to_hs:
                phase_props._hashin_shtrikman(f)
                in_box = (
                    (mu_eff  >= phase_props.mu_lower_HS)  & (mu_eff  <= phase_props.mu_upper_HS) &
                    (sig_eff >= phase_props.sig_lower_HS) & (sig_eff <= phase_props.sig_upper_HS)
                )
                mu_eff  = mu_eff[in_box]
                sig_eff = sig_eff[in_box]

            if mu_eff.size:
                mu_pts.append(mu_eff)
                sig_pts.append(sig_eff)

    if len(mu_pts) == 0:
        return np.array([]), np.array([])

    return np.concatenate(sig_pts), np.concatenate(mu_pts)

def collect_bm_points(assemblers, phase_props, f_grid, clip_to_hs=True):
    """
    Returns point cloud arrays (sig_pts, mu_pts) for the UNION over:
      - bound_types present in assemblers.keys()
      - volume fractions f in f_grid
      - eta parameter along BM curve (inside _berryman_milton)
    Optionally clipped to HS box (consistent with translation plot logic).
    """
    sig_pts = []
    mu_pts  = []

    for bound_type in assemblers.keys():
        for f in f_grid:
            phase_props._berryman_milton(f)

            # Match your existing mapping from bound_type -> which BM branch to plot
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

            sig_eff = np.asarray(sig_BM)
            mu_eff  = np.asarray(mu_BM)

            if clip_to_hs:
                phase_props._hashin_shtrikman(f)
                in_box = (
                    (mu_eff  >= phase_props.mu_lower_HS)  & (mu_eff  <= phase_props.mu_upper_HS) &
                    (sig_eff >= phase_props.sig_lower_HS) & (sig_eff <= phase_props.sig_upper_HS)
                )
                sig_eff = sig_eff[in_box]
                mu_eff  = mu_eff[in_box]

            if mu_eff.size:
                sig_pts.append(sig_eff)
                mu_pts.append(mu_eff)

    if len(mu_pts) == 0:
        return np.array([]), np.array([])

    return np.concatenate(sig_pts), np.concatenate(mu_pts)



def envelope_by_xbins(x, y, nbins=20, x_min=None, x_max=None, smooth_window=15):
    """
    Compute lower/upper envelope y(x) by binning in x and taking min/max y in each bin.
    Returns:
      x_centers, y_lower, y_upper (with NaN bins removed)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x_min is None: x_min = np.nanmin(x)
    if x_max is None: x_max = np.nanmax(x)

    # bins
    edges = np.linspace(x_min, x_max, nbins + 1)
    bin_id = np.digitize(x, edges) - 1
    valid = (bin_id >= 0) & (bin_id < nbins)
    x = x[valid]; y = y[valid]; bin_id = bin_id[valid]

    y_lo = np.full(nbins, np.nan)
    y_hi = np.full(nbins, np.nan)

    # aggregate extrema per bin
    for b in range(nbins):
        m = (bin_id == b)
        if np.any(m):
            yy = y[m]
            y_lo[b] = np.min(yy)
            y_hi[b] = np.max(yy)

    x_cent = 0.5 * (edges[:-1] + edges[1:])

    # drop empty bins
    keep = ~np.isnan(y_lo) & ~np.isnan(y_hi)
    x_cent = x_cent[keep]
    y_lo = y_lo[keep]
    y_hi = y_hi[keep]

    # light smoothing to make it "clean" (optional)
    if smooth_window is not None and smooth_window >= 3:
        # enforce odd window
        if smooth_window % 2 == 0:
            smooth_window += 1
        # simple moving average via convolution
        w = np.ones(smooth_window) / smooth_window
        y_lo = np.convolve(y_lo, w, mode="same")
        y_hi = np.convolve(y_hi, w, mode="same")

    return x_cent, y_lo, y_hi


# ============================================================
# TWO-PANEL OVERLAY + ZOOM + LABELS
# ============================================================

def plot_all_bounds_overlay(assemblers, phase_props, dataset=None, out_path="translation_bounds_overlay.png"):
    k1, k2 = phase_props.k1, phase_props.k2
    mu1, mu2 = phase_props.mu1, phase_props.mu2
    sig1, sig2 = phase_props.sig1, phase_props.sig2

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
    # TRANSLATION: build volume-fraction-free envelope
    # ============================
    f_grid = np.linspace(0.001, 0.999, 10000)  # increase for tighter envelope

    sig_pts, mu_pts = collect_translation_points(
        assemblers=assemblers,
        phase_props=phase_props,
        f_grid=f_grid,
        clip_to_hs=True  # keep consistent with your current plot logic
    )

    if sig_pts.size == 0:
        print("⚠️ No translation points collected (check clipping / inputs).")
    else:
        x_env, mu_lo, mu_hi = envelope_by_xbins(
            sig_pts, mu_pts,
            nbins=1000,          # increase if you want sharper boundary
            smooth_window=None     # set to None to disable smoothing
        )

        ax1.plot(x_env, mu_hi, color="blue", linewidth=1.8, label="Translation")
        ax1.plot(x_env, mu_lo, color="blue", linewidth=1.8)


    # ============================
    # BERRYMAN–MILTON: build volume-fraction-free envelope
    # ============================
    f_grid_bm = np.linspace(0.001, 0.999, 1000)  # tune like you did for translation

    sig_bm_pts, mu_bm_pts = collect_bm_points(
        assemblers=assemblers,
        phase_props=phase_props,
        f_grid=f_grid_bm,
        clip_to_hs=True
    )

    if sig_bm_pts.size == 0:
        print("⚠️ No BM points collected (check clipping / inputs).")
    else:
        x_bm, mu_bm_lo, mu_bm_hi = envelope_by_xbins(
            sig_bm_pts, mu_bm_pts,
            nbins=1000,
            smooth_window=None
        )

        ax1.plot(x_bm, mu_bm_hi, 'r--', linewidth=1.8, label="Berryman-Milton")
        ax1.plot(x_bm, mu_bm_lo, 'r--', linewidth=1.8)


    ax1.set_xlabel(r'$\sigma^{\star}$', fontsize=28)
    ax1.set_ylabel(r'$\mu^{\star}$', fontsize=28)
    ax1.tick_params(axis='both', which='major', labelsize=24)
    # ax1.legend(fontsize=22)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\n✅ Final overlay plot saved to: {out_path}")


# ============================================================
# BUILD ALL ASSEMBLERS
# ============================================================

phase_props = phase_props_holder(k1, k2, mu1, mu2, sig1, sig2)
dataset = pd.read_csv(dataset_path) if dataset_path else None

assemblers = {}

for bound_type, T_path in zip(bound_types, T_paths):
    assemblers[bound_type] = Assembler(T_path, phase_props, bound_type)

# ============================================================
# FINAL OVERLAY PLOT
# ============================================================

plot_all_bounds_overlay(
    assemblers,
    phase_props,
    dataset=dataset,
    out_path=f"{out_dir_path}/bounds_allf.png"
)
