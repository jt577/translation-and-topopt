#!/usr/bin/env python3
"""Plot violation percentages and minimum eigenvalues in the paper style.

Mean and sample standard deviation are across independent saved optimization
runs at each training rotation count, using each run's refined worst eigenvalue.
This script only reads audit results; it performs no new rotation searches.
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-translation-convergence")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def summary_rows(document):
    if "completed_utc" not in document["metadata"]:
        raise ValueError("The input audit is incomplete")
    runs = document["results"]
    if len(runs) != document["metadata"]["matrix_count"]:
        raise ValueError("The audit result count does not match its metadata")
    rows = []
    for count in sorted({r["n_rotations"] for r in runs}):
        group = sorted((r for r in runs if r["n_rotations"] == count), key=lambda r: r["run"])
        values = np.array([r["lambda_worst"] for r in group])
        percentages = np.array([100*r["random_violations"]/r["random_samples"] for r in group])
        rows.append({
            "n_rotations": count, "inv_n_rotations": 1.0/count,
            "n_runs": len(group), "runs": ";".join(str(r["run"]) for r in group),
            "minimum_eigenvalue_mean": float(values.mean()),
            "minimum_eigenvalue_sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            "minimum_eigenvalue_min": float(values.min()),
            "minimum_eigenvalue_max": float(values.max()),
            "violating_rotations_percent_mean": float(percentages.mean()),
            "violating_rotations_percent_min": float(percentages.min()),
            "violating_rotations_percent_max": float(percentages.max()),
        })
    return rows


def plot(rows, output_stem, linthresh):
    # Match convergence_study.py::plot_property_convergence dimensions,
    # line/marker styles, fonts, and reciprocal-N ticks.
    x = np.array([r["inv_n_rotations"] for r in rows])
    counts = np.array([r["n_rotations"] for r in rows])
    means = np.array([r["minimum_eigenvalue_mean"] for r in rows])
    percentages = np.array([r["violating_rotations_percent_mean"] for r in rows])
    percentage_min = np.array([r["violating_rotations_percent_min"] for r in rows])
    percentage_max = np.array([r["violating_rotations_percent_max"] for r in rows])
    order = np.argsort(x)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, (ax_percentage, ax) = plt.subplots(
            2, 1, figsize=(6, 5.2), sharex=True,
            gridspec_kw={"height_ratios": [1.4, 2.0], "hspace": .08},
        )
        ax_percentage.plot(x[order], percentages[order], "s-", markersize=6,
                           linewidth=1.5, color="black")
        ax_percentage.fill_between(x[order], percentage_min[order], percentage_max[order],
                                   color="black", alpha=.12, linewidth=0)
        ax_percentage.axhline(0, color="0.7", linewidth=.7, zorder=0)
        ax_percentage.set_xscale("log")
        ax_percentage.set_ylim(-4, 100)
        ax_percentage.set_yticks([0, 25, 50, 75, 100])
        ax_percentage.set_ylabel("Violating\nrotations (%)", fontsize=15)
        ax_percentage.tick_params(axis="y", which="major", labelsize=14)
        ax_percentage.tick_params(axis="x", which="both", labelbottom=False)

        ax.plot(x[order], means[order], "o-",
                markersize=8, linewidth=1.8, color="blue")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.axhline(0, color="0.55", linewidth=.7, zorder=0)
        ax.set_ylabel(r"$\lambda_{\mathrm{worst}}$", fontsize=18)
        ax.tick_params(axis="y", which="major", labelsize=14)
        # Keep the signed-log ticks legible at the original paper figure size.
        ax.set_yticks([-1, -1e-2, -1e-4, -1e-6, 0, 1e-5])
        ax.yaxis.set_minor_locator(NullLocator())
        # Linear autoscaling otherwise leaves excessive positive
        # whitespace when its negative data span many more orders of magnitude.
        lower = float(np.min(means))
        upper = float(np.max(means))
        ax.set_ylim(min(lower*5, -linthresh*2), max(upper*3, linthresh*2))

        ax.set_xticks(x[order])
        ax.set_xticklabels([rf"$\frac{{1}}{{{n}}}$" for n in counts[order]])
        ax.set_xlabel(r"$1/N_R$", fontsize=18)
        ax.tick_params(axis="x", which="major", labelsize=18)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path(__file__).parent/"convergence_audit/results.json")
    parser.add_argument("--output-stem", type=Path,
                        default=ROOT/"convergence_study/minimum_eigenvalue_convergence_vs_inv_rotations")
    parser.add_argument("--linthresh", type=float, default=1e-6,
                        help="Half-width of the linear interval around zero in the lower signed-log panel")
    args = parser.parse_args()
    if not np.isfinite(args.linthresh) or args.linthresh <= 0:
        parser.error("linthresh must be finite and positive")
    document = json.loads(args.audit.read_text())
    rows = summary_rows(document)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, args.output_stem, args.linthresh)
    with args.output_stem.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    note = (
        "Numerical quasiconvexity convergence. Each run contributes its smallest "
        "eigenvalue found over the independent rotation audit and local searches. "
        "The upper panel shows the percentage of independently sampled audit "
        "rotations whose minimum eigenvalue is below the negative audit tolerance "
        f"(-{document['metadata']['atol']:g}). The black line shows the mean percentage "
        "across saved optimization runs; shading spans the full run range. This "
        "counts violating rotations, rather than pooling all eigenvalues within "
        "each projected matrix. The lower panel shows the mean worst eigenvalue "
        "across saved optimization runs. "
        "The lower panel uses a symmetric logarithmic scale "
        f"with a linear interval [-{args.linthresh:g}, +{args.linthresh:g}] to preserve "
        "the signs of eigenvalues crossing zero. The horizontal gray line marks zero. "
        "Both panels use the original paper's reciprocal-rotation horizontal axis. "
        "Positive mean values do not establish positivity for every run or rotation. "
        "This plot is not the mean or standard deviation of eigenvalues over the "
        "audit rotations within a single run.\n\n"
        f"Source audit: {args.audit.resolve()}\n"
        f"Audit rotations per matrix: {document['metadata']['samples']}\n"
        f"Local starts per matrix: {document['metadata']['local_starts']}\n"
        "Included run counts: "+", ".join(f"N={r['n_rotations']}: {r['n_runs']}" for r in rows)+".\n"
    )
    args.output_stem.with_suffix(".txt").write_text(note)
    print(f"Wrote {args.output_stem}.{{png,pdf,svg,csv,txt}}")
    for row in rows:
        print(f"N={row['n_rotations']:3d}: mean={row['minimum_eigenvalue_mean']:+.8e}, "
              f"sample std={row['minimum_eigenvalue_sample_std']:.8e}, runs={row['n_runs']}")


if __name__ == "__main__":
    main()
