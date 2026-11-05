#!/usr/bin/env python3
"""Audit saved convergence matrices for the paper quasiconvexity figure."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-translation-quasiconvexity")
import numpy as np

from check_quasiconvexity import ROOT, HELPER_PATH, audit_matrix

# Reuse the original convergence analysis, including the physical plotting slice.
sys.path.insert(0, str(ROOT / "official_translation_12.7.2025"))
spec = importlib.util.spec_from_file_location("original_convergence_analysis", ROOT / "convergence_study.py")
convergence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convergence)


def summarize(results):
    summary = []
    for count in sorted({r["n_rotations"] for r in results}):
        group = [r for r in results if r["n_rotations"] == count]
        worst = np.asarray([r["lambda_worst"] for r in group])
        summary.append({
            "n_rotations": count, "n_runs": len(group),
            "worst_eigenvalue_mean": float(worst.mean()),
            "worst_eigenvalue_std": float(worst.std()),
            "worst_eigenvalue_min": float(worst.min()),
            "worst_eigenvalue_max": float(worst.max()),
            "random_violation_fraction_mean": float(np.mean([r["random_violations"]/r["random_samples"] for r in group])),
            "random_violation_fraction_min": float(min(r["random_violations"]/r["random_samples"] for r in group)),
            "random_violation_fraction_max": float(max(r["random_violations"]/r["random_samples"] for r in group)),
            "runs_without_detected_violation": sum(r["no_violation_found"] for r in group),
            "mu_mean": float(np.mean([r["mu"] for r in group])),
            "mu_std": float(np.std([r["mu"] for r in group])),
            "sigma_mean": float(np.mean([r["sigma"] for r in group])),
            "sigma_std": float(np.std([r["sigma"] for r in group])),
        })
    return summary


def save_results(output, metadata, results):
    (output / "results.json").write_text(json.dumps({
        "metadata": metadata, "summary": summarize(results), "results": results,
    }, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "convergence_study")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "convergence_audit")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--local-starts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    if args.samples < 1 or args.local_starts < 0 or not np.isfinite(args.atol) or args.atol <= 0:
        parser.error("samples and atol must be positive; local-starts must be nonnegative")
    args.study = args.study.resolve()
    params = convergence.parse_input_file(args.study / "infile.txt")
    if len(params["alphas"]) != 1:
        parser.error("study must specify one alpha")
    completed = sorted(convergence.discover_completed_runs(args.study), key=lambda r:(r["n_rotations"],r["run"]))
    if not completed:
        parser.error("no completed matrices found")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("output directory must be empty")
    # Small display grids leave the original slice selection and transforms intact.
    display_params = dict(params, npts=2)
    alpha = float(params["alphas"][0])
    helper, _ = convergence.make_helper(display_params, 0, alpha)
    original_csv = {}
    if (args.study / "convergence_runs.csv").exists():
        with (args.study / "convergence_runs.csv").open() as f:
            original_csv = {(int(r["n_rotations"]),int(r["run"])):r for r in csv.DictReader(f)}
    metadata = {
        "study": str(args.study), "samples": args.samples, "local_starts": args.local_starts,
        "seed": args.seed, "atol": args.atol, "matrix_count": len(completed),
        "calc_type": params["calc_type"], "bound_type": params["bound_type"], "alpha": alpha,
        "phase_properties": {k:params[k] for k in ("k1","k2","mu1","mu2","sig1","sig2","f")},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256((Path(__file__).parent/"check_quasiconvexity.py").read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest(),
        "sampling": "Identical independent Haar audit stream for every matrix; original training RNG uses scalar-first quaternions with seed 2387",
        "method": "Existing audit_matrix routine: random sampling, seven canonical rotations, Powell local search, original-Q witness validation",
    }
    results = []
    # The shared auditor infers bound type from the parent directory. A temporary
    # symlink view supplies that name without copying or changing any source data.
    with tempfile.TemporaryDirectory(prefix="translation-convergence-audit-") as temporary:
        view = Path(temporary)
        options = SimpleNamespace(samples=args.samples, batch_size=2048, local_starts=args.local_starts,
                                  maxiter=300, seed=args.seed, atol=args.atol, rtol=0., results_root=view)
        for i, info in enumerate(completed, 1):
            source = info["T_path"].resolve()
            link = view / info["run_dir"].name / params["bound_type"] / "T_final.npy"
            link.parent.mkdir(parents=True)
            link.symlink_to(source)
            print(f"[{i}/{len(completed)}] {source.parent.name}", flush=True)
            result = audit_matrix(link, options)
            result.update({"matrix": str(source.relative_to(args.study)), "n_rotations":info["n_rotations"],
                           "run":info["run"], "source_path":str(source)})
            t = np.load(source)
            z = convergence.solve_z_sol(t,helper,alpha,zhigh=100.)
            z_slice = convergence.solve_z_sol(t,helper,alpha,zhigh=helper.zk_point)
            mu, sigma = convergence.z_line_to_star_point(z_slice,alpha,helper)
            result.update({"z_mu":float(z),"z_mu_star_slice":float(z_slice),"mu":float(mu),"sigma":float(sigma)})
            previous = original_csv.get((info["n_rotations"], info["run"]))
            if previous:
                result["existing_csv_mu_difference"] = float(mu-float(previous["mu"]))
                result["existing_csv_sigma_difference"] = float(sigma-float(previous["sigma"]))
                np.testing.assert_allclose([mu,sigma],[float(previous["mu"]),float(previous["sigma"])],atol=1e-10,rtol=1e-10)
            results.append(result)
            save_results(args.output,metadata,results)
            print(f"  worst={result['lambda_worst']:.8e}; violating audit rotations="
                  f"{100*result['random_violations']/args.samples:.3f}%; mu={mu:.12g}; "
                  f"{result['seconds']:.1f}s",flush=True)
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save_results(args.output,metadata,results)
    print(f"Wrote {args.output}",flush=True)


if __name__ == "__main__":
    main()
