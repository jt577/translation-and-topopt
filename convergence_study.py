import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-translation-convergence")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from modules.helper import TranslationHelper
from modules.misc import phase_props_holder


RUN_DIR_RE = re.compile(r"^out_(?P<n_rot>\d+)_run(?P<run>\d+)$")


def parse_input_file(infile_path):
    params = {}
    with open(infile_path, "r") as openfile:
        for line in openfile:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            key = parts[0]
            vals = parts[1:]
            if key in {"k1", "k2", "mu1", "mu2", "sig1", "sig2", "f", "upper"}:
                params[key] = float(vals[0])
            elif key in {"npts", "n_samples"}:
                params[key] = int(vals[0])
            elif key == "alphas":
                params[key] = np.array([float(val) for val in vals], dtype=float)
            elif key in {"calc_type", "bound_type"}:
                params[key] = vals[0]

    required = {
        "k1",
        "k2",
        "mu1",
        "mu2",
        "sig1",
        "sig2",
        "f",
        "upper",
        "npts",
        "alphas",
        "calc_type",
        "bound_type",
    }
    missing = sorted(required - params.keys())
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")
    return params


def discover_completed_runs(runs_root):
    completed = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_DIR_RE.match(run_dir.name)
        if match is None:
            continue
        t_path = run_dir / "T_final.npy"
        if not t_path.exists():
            continue
        completed.append(
            {
                "n_rotations": int(match.group("n_rot")),
                "run": int(match.group("run")),
                "run_dir": run_dir,
                "T_path": t_path,
            }
        )
    return completed


def find_input_file(bundle_dir):
    preferred = bundle_dir / "infile.txt"
    if preferred.exists():
        return preferred

    candidates = sorted(bundle_dir.glob("infile*.txt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No infile*.txt found inside {bundle_dir}")
    raise ValueError(
        f"Multiple infile*.txt files found inside {bundle_dir}; "
        "rename the one to use to infile.txt or use the three-argument form."
    )


def make_helper(params, n_samples, alpha):
    n_shear = 1 if params["calc_type"] == "bulk" else 5
    phase_props = phase_props_holder(
        params["k1"],
        params["k2"],
        params["mu1"],
        params["mu2"],
        params["sig1"],
        params["sig2"],
        params["f"],
    )
    helper = TranslationHelper(
        n_shear=n_shear,
        phase_props=phase_props,
        npts=params["npts"],
        upper=params["upper"],
        n_samples=n_samples,
        alphas=np.array([alpha], dtype=float),
        seed=2387,
        calc_type=params["calc_type"],
        bound_type=params["bound_type"],
    )
    return helper, phase_props


def solve_z_sol(T, helper, alpha, zhigh=100.0, bracket_hi=500.0):
    if helper.calc_type == "bulk":
        A = helper.D_mu * zhigh
        B = helper.D_k + alpha * helper.D_sig
    elif helper.calc_type == "shear":
        A = helper.D_k * zhigh
        B = helper.D_mu + alpha * helper.D_sig
    else:
        raise ValueError(f"Unsupported calc_type: {helper.calc_type}")

    def nonlinfun(x):
        eigvals = np.linalg.eigh(A + B * x + T)[0]
        return eigvals[0]

    f_lo = nonlinfun(0.0)
    f_hi = nonlinfun(bracket_hi)
    if f_lo == 0.0:
        return 0.0
    if f_lo * f_hi > 0.0:
        raise RuntimeError(
            "Root is not bracketed on [0, "
            f"{bracket_hi}]: lambda_min(0)={f_lo:.6e}, "
            f"lambda_min({bracket_hi})={f_hi:.6e}"
        )
    return brentq(nonlinfun, a=0.0, b=bracket_hi, xtol=1e-10, maxiter=100)


def z_line_to_star_point(z_mu, alpha, helper):
    """Map one z-line point into star space, matching modules/plotting.py."""
    if helper.calc_type != "shear":
        raise ValueError("This convergence study currently reports physical mu-sigma coordinates for shear runs.")

    z_sig = alpha * z_mu
    phase_props = helper.phase_props
    if helper.bound_type == "upper_1":
        y_mu = 1.0 / z_mu
        y_sig = 1.0 / z_sig
    elif helper.bound_type == "upper_2":
        y_mu = z_mu
        y_sig = 1.0 / z_sig
    elif helper.bound_type == "lower_1":
        y_mu = 1.0 / z_mu
        y_sig = z_sig
    elif helper.bound_type == "lower_2":
        y_mu = z_mu
        y_sig = z_sig
    else:
        raise ValueError(f"Unsupported bound_type: {helper.bound_type}")

    mu_eff = helper.inv_y_transform(y_mu, phase_props.f, phase_props.mu1, phase_props.mu2)
    sig_eff = helper.inv_y_transform(y_sig, phase_props.f, phase_props.sig1, phase_props.sig2)
    return mu_eff, sig_eff


def star_lambda_min(T, helper, mu_eff, sig_eff):
    """Evaluate the same star-space slice used by the star plots."""
    phase_props = helper.phase_props
    yk = helper._y_transform(helper.kstar_point, phase_props.f, phase_props.k1, phase_props.k2)
    ymu = helper._y_transform(mu_eff, phase_props.f, phase_props.mu1, phase_props.mu2)
    ysig = helper._y_transform(sig_eff, phase_props.f, phase_props.sig1, phase_props.sig2)

    if helper.bound_type == "upper_1":
        y_list_elastic = [1.0 / (3.0 * yk)] + [1.0 / (2.0 * ymu)] * 5
        y_list_conductive = [1.0 / ysig] * 9
    elif helper.bound_type == "upper_2":
        y_list_elastic = [3.0 * yk] + [2.0 * ymu] * 5
        y_list_conductive = [1.0 / ysig] * 9
    elif helper.bound_type == "lower_1":
        y_list_elastic = [1.0 / (3.0 * yk)] + [1.0 / (2.0 * ymu)] * 5
        y_list_conductive = [ysig] * 9
    elif helper.bound_type == "lower_2":
        y_list_elastic = [3.0 * yk] + [2.0 * ymu] * 5
        y_list_conductive = [ysig] * 9
    else:
        raise ValueError(f"Unsupported bound_type: {helper.bound_type}")

    y_diag = y_list_elastic * helper.n_shear + y_list_conductive
    return float(np.linalg.eigh(np.diag(y_diag) + T)[0][0])


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    ddof = 1 if arr.size > 1 else 0
    return float(np.mean(arr)), float(np.std(arr, ddof=ddof))


def write_csv(path, rows, columns):
    with open(path, "w") as openfile:
        openfile.write(",".join(columns) + "\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    vals.append(f"{val:.16g}")
                else:
                    vals.append(str(val))
            openfile.write(",".join(vals) + "\n")


def build_run_rows(completed_runs, params, alpha, zhigh, bracket_hi):
    helper_cache = {}
    rows = []
    for run_info in completed_runs:
        n_rot = run_info["n_rotations"]
        if n_rot not in helper_cache:
            helper_cache[n_rot] = make_helper(params, n_rot, alpha)[0]
        helper = helper_cache[n_rot]

        T = np.load(run_info["T_path"])
        T = 0.5 * (T + T.T)
        z_mu = solve_z_sol(T, helper, alpha, zhigh=zhigh, bracket_hi=bracket_hi)
        z_sig = alpha * z_mu
        z_mu_star = solve_z_sol(T, helper, alpha, zhigh=helper.zk_point, bracket_hi=bracket_hi)
        z_sig_star = alpha * z_mu_star
        mu_eff, sig_eff = z_line_to_star_point(z_mu_star, alpha, helper)
        star_residual = star_lambda_min(T, helper, mu_eff, sig_eff)
        rows.append(
            {
                "n_rotations": n_rot,
                "inv_n_rotations": 1.0 / n_rot,
                "run": run_info["run"],
                "run_dir": str(run_info["run_dir"]),
                "z_mu": z_mu,
                "z_sigma": z_sig,
                "z_mu_star_slice": z_mu_star,
                "z_sigma_star_slice": z_sig_star,
                "mu": mu_eff,
                "sigma": sig_eff,
                "star_lambda_min": star_residual,
            }
        )
    rows.sort(key=lambda row: (row["n_rotations"], row["run"]))
    return rows


def build_summary_rows(run_rows):
    summary_rows = []
    n_values = sorted({row["n_rotations"] for row in run_rows})
    for n_rot in n_values:
        rows = [row for row in run_rows if row["n_rotations"] == n_rot]
        z_mu_mean, z_mu_std = summarize([row["z_mu"] for row in rows])
        z_sig_mean, z_sig_std = summarize([row["z_sigma"] for row in rows])
        z_mu_star_mean, z_mu_star_std = summarize([row["z_mu_star_slice"] for row in rows])
        z_sig_star_mean, z_sig_star_std = summarize([row["z_sigma_star_slice"] for row in rows])
        mu_mean, mu_std = summarize([row["mu"] for row in rows])
        sig_mean, sig_std = summarize([row["sigma"] for row in rows])
        summary_rows.append(
            {
                "n_rotations": n_rot,
                "inv_n_rotations": 1.0 / n_rot,
                "n_completed_runs": len(rows),
                "runs": ";".join(str(row["run"]) for row in rows),
                "z_mu_mean": z_mu_mean,
                "z_mu_std": z_mu_std,
                "z_sigma_mean": z_sig_mean,
                "z_sigma_std": z_sig_std,
                "z_mu_star_slice_mean": z_mu_star_mean,
                "z_mu_star_slice_std": z_mu_star_std,
                "z_sigma_star_slice_mean": z_sig_star_mean,
                "z_sigma_star_slice_std": z_sig_star_std,
                "mu_mean": mu_mean,
                "mu_std": mu_std,
                "sigma_mean": sig_mean,
                "sigma_std": sig_std,
            }
        )
    return summary_rows


def plot_z_convergence(summary_rows, out_path):
    x = np.array([row["inv_n_rotations"] for row in summary_rows], dtype=float)
    z_mean = np.array([row["z_mu_mean"] for row in summary_rows], dtype=float)
    z_std = np.array([row["z_mu_std"] for row in summary_rows], dtype=float)
    n_rot = [row["n_rotations"] for row in summary_rows]

    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(x[order], z_mean[order], yerr=z_std[order], fmt="o-", capsize=4, linewidth=1.8)
    for xi, yi, n in zip(x[order], z_mean[order], np.array(n_rot)[order]):
        ax.annotate(f"N={n}", (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("1 / number of rotations")
    ax.set_ylabel(r"$z_\mu$ on $z_\sigma=\alpha z_\mu$")
    ax.set_title("Translation Bound Convergence in z-space")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_property_convergence(summary_rows, out_path):
    x = np.array([row["inv_n_rotations"] for row in summary_rows], dtype=float)
    n_rotations = np.array([row["n_rotations"] for row in summary_rows], dtype=int)
    mu_mean = np.array([row["mu_mean"] for row in summary_rows], dtype=float)
    mu_std = np.array([row["mu_std"] for row in summary_rows], dtype=float)
    sig_mean = np.array([row["sigma_mean"] for row in summary_rows], dtype=float)
    sig_std = np.array([row["sigma_std"] for row in summary_rows], dtype=float)
    order = np.argsort(x)

    if np.allclose(mu_mean, sig_mean, rtol=1e-12, atol=1e-12):
        y_mean = 0.5 * (mu_mean + sig_mean)
        y_std = 0.5 * (mu_std + sig_std)
        y_label = r"$\sigma^\star,\ \mu^\star$"
    else:
        y_mean = sig_mean
        y_std = sig_std
        y_label = r"$\sigma^\star$"

    rel_std_pct = 100.0 * y_std / np.abs(y_mean)

    fig, (ax, ax_std) = plt.subplots(
        2,
        1,
        figsize=(6, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08},
    )
    ax.errorbar(
        x[order],
        y_mean[order],
        yerr=y_std[order],
        fmt="o-",
        markersize=8,
        capsize=4,
        linewidth=1.8,
        color="blue",
    )
    ax.set_xscale("log")
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis="y", which="major", labelsize=14)
    ax.tick_params(axis="x", which="both", labelbottom=False)

    ax_std.plot(
        x[order],
        rel_std_pct[order],
        "s-",
        markersize=6,
        linewidth=1.5,
        color="black",
    )
    ax_std.set_xscale("log")
    ax_std.set_xticks(x[order])
    ax_std.set_xticklabels([rf"$\frac{{1}}{{{n}}}$" for n in n_rotations[order]])
    ax_std.set_xlabel(r"$1/N_R$", fontsize=18)
    ax_std.set_ylabel("Error (%)", fontsize=15)
    ax_std.tick_params(axis="x", which="major", labelsize=18)
    ax_std.tick_params(axis="y", which="major", labelsize=13)
    if np.all(rel_std_pct > 0.0):
        ax_std.set_yscale("log")

    # ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sigma_mu(summary_rows, run_rows, out_path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    n_values = [row["n_rotations"] for row in summary_rows]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.12, 0.9, len(n_values)))

    for color, summary in zip(colors, summary_rows):
        n_rot = summary["n_rotations"]
        rows = [row for row in run_rows if row["n_rotations"] == n_rot]
        ax.scatter(
            [row["mu"] for row in rows],
            [row["sigma"] for row in rows],
            color=color,
            alpha=0.28,
            s=24,
            edgecolors="none",
        )
        ax.errorbar(
            summary["mu_mean"],
            summary["sigma_mean"],
            xerr=summary["mu_std"],
            yerr=summary["sigma_std"],
            fmt="o",
            capsize=4,
            color=color,
            label=f"N={n_rot}",
        )

    ax.set_xlabel(r"$\mu_{\mathrm{eff}}$")
    ax.set_ylabel(r"$\sigma_{\mathrm{eff}}$")
    ax.set_title(r"Convergence of Bound Point in $(\mu,\sigma)$")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Rotations", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence of a one-line translation bound point with rotation samples."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Either a self-contained convergence folder, or the legacy "
            "three arguments: infile_path runs_root out_dir"
        ),
    )
    parser.add_argument("--alpha", type=float, default=None, help="override the input alpha")
    parser.add_argument("--zhigh", type=float, default=100.0, help="fixed large z coordinate used in objfun")
    parser.add_argument("--bracket-hi", type=float, default=500.0, help="upper bracket for z root solve")
    args = parser.parse_args()

    if len(args.paths) == 1:
        runs_root = Path(args.paths[0])
        infile_path = find_input_file(runs_root)
        out_dir = runs_root
    elif len(args.paths) == 3:
        infile_path = Path(args.paths[0])
        runs_root = Path(args.paths[1])
        out_dir = Path(args.paths[2])
    else:
        parser.error("Pass either one convergence folder or three paths: infile_path runs_root out_dir")

    out_dir.mkdir(parents=True, exist_ok=True)

    params = parse_input_file(infile_path)
    if args.alpha is None:
        if len(params["alphas"]) != 1:
            raise ValueError(
                "The convergence study expects one line alpha. "
                "Pass --alpha to select a specific alpha from the input."
            )
        alpha = float(params["alphas"][0])
    else:
        alpha = args.alpha

    completed_runs = discover_completed_runs(runs_root)
    if not completed_runs:
        raise FileNotFoundError(f"No completed runs with T_final.npy found under {runs_root}")

    run_rows = build_run_rows(completed_runs, params, alpha, args.zhigh, args.bracket_hi)
    summary_rows = build_summary_rows(run_rows)

    run_columns = [
        "n_rotations",
        "inv_n_rotations",
        "run",
        "run_dir",
        "z_mu",
        "z_sigma",
        "z_mu_star_slice",
        "z_sigma_star_slice",
        "mu",
        "sigma",
        "star_lambda_min",
    ]
    summary_columns = [
        "n_rotations",
        "inv_n_rotations",
        "n_completed_runs",
        "runs",
        "z_mu_mean",
        "z_mu_std",
        "z_sigma_mean",
        "z_sigma_std",
        "z_mu_star_slice_mean",
        "z_mu_star_slice_std",
        "z_sigma_star_slice_mean",
        "z_sigma_star_slice_std",
        "mu_mean",
        "mu_std",
        "sigma_mean",
        "sigma_std",
    ]
    write_csv(out_dir / "convergence_runs.csv", run_rows, run_columns)
    write_csv(out_dir / "convergence_summary.csv", summary_rows, summary_columns)
    plot_z_convergence(summary_rows, out_dir / "z_convergence.png")
    plot_property_convergence(summary_rows, out_dir / "property_convergence_vs_inv_rotations.png")
    plot_sigma_mu(summary_rows, run_rows, out_dir / "sigma_mu_convergence.png")

    print(f"alpha = {alpha}")
    print(f"completed runs used = {len(run_rows)}")
    print("z-space standard deviations by rotation count:")
    for row in summary_rows:
        print(
            f"  N={row['n_rotations']:>4}: runs={row['n_completed_runs']} "
            f"z_mu_mean={row['z_mu_mean']:.12g} z_mu_std={row['z_mu_std']:.12g} "
            f"z_sigma_mean={row['z_sigma_mean']:.12g} z_sigma_std={row['z_sigma_std']:.12g}"
        )
    print(f"wrote {out_dir / 'convergence_runs.csv'}")
    print(f"wrote {out_dir / 'convergence_summary.csv'}")
    print(f"wrote {out_dir / 'z_convergence.png'}")
    print(f"wrote {out_dir / 'property_convergence_vs_inv_rotations.png'}")
    print(f"wrote {out_dir / 'sigma_mu_convergence.png'}")


if __name__ == "__main__":
    main()
