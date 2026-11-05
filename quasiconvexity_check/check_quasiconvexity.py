#!/usr/bin/env python3
"""Search SO(3) for negative eigenvalues of the saved translations' Q.T T Q.

Finite sampling and local optimization give numerical evidence, not a proof.
Only NumPy and SciPy are required; no solver, Torch, or GPU is used.
"""

import hashlib
import importlib.util
import os
from pathlib import Path
import time

# Small dense eigenproblems are much faster without threaded BLAS overhead.
# Explicit settings in the calling environment take precedence.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "official_translation_12.7.2025/modules/helper.py"
_spec = importlib.util.spec_from_file_location("translation_q_helper", HELPER_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class FourierHelper(_module.TranslationHelper):
    """Reuse original basis/maps and Q assembly, skipping unrelated bound grids."""

    def _create_Y(self):
        pass

    def _create_Z(self):
        pass

    def _create_D(self):
        pass

    def _make_diag(self):
        pass


def make_helper(bound_type, n_shear):
    return FourierHelper(
        n_shear=n_shear, phase_props=None, npts=0, upper=0, n_samples=0,
        alphas=np.array([]), seed=0,
        calc_type="bulk" if n_shear == 1 else "shear", bound_type=bound_type,
    )


def q_batch(rotations, helper):
    """Vectorized original _assemble_H, _assemble_G, and _block_rot algebra."""
    rotations = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    bases = np.asarray(helper.bases)
    rotated = np.einsum("nik,bkj->nbij", rotations, bases, optimize=True)
    h = np.einsum("aij,nbij->nab", bases, rotated, optimize=True)
    if helper.bound_type in ("upper_1", "lower_1"):
        rotated_symmetric = np.einsum(
            "nbil,njl->nbij", rotated[:, :6], rotations, optimize=True,
        )
        g = np.einsum("aij,nbij->nab", bases[:6], rotated_symmetric, optimize=True)
        elastic = g @ helper.M
    else:
        elastic = (h @ helper.L)[:, :6]
    electric = h @ (helper.N if helper.bound_type.startswith("upper") else helper.L)
    m = helper.n_shear
    q = np.zeros((len(rotations), 6 * m + 9, 3 * m + electric.shape[-1]))
    for block in range(m):
        q[:, 6 * block:6 * block + 6, 3 * block:3 * block + 3] = elastic
    q[:, 6 * m:, 3 * m:] = electric
    return q


def reference_q(rotation, helper):
    """Evaluate the original implementation at a specified rotation."""
    original = helper._random_rotation
    try:
        helper._random_rotation = lambda: rotation
        return helper._block_rot()
    finally:
        helper._random_rotation = original


def projected_matrices(rotations, translation, helper):
    q = q_batch(rotations, helper)
    a = q.swapaxes(-1, -2) @ (translation @ q)
    return 0.5 * (a + a.swapaxes(-1, -2))


def minimum_eigenvalues(rotations, translation, helper):
    return np.linalg.eigvalsh(projected_matrices(rotations, translation, helper))[:, 0]


def validate_q(helper, seed):
    rotations = Rotation.random(8, random_state=np.random.default_rng(seed)).as_matrix()
    actual = q_batch(rotations, helper)
    expected = np.stack([reference_q(r, helper) for r in rotations])
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    return float(np.max(np.abs(actual - expected)))


def select_starts(values, rotations, count):
    """Prioritize the lowest eigenvalues, with separated starts where possible."""
    if count == 0:
        return []
    candidates = np.argsort(values)
    selected = []
    for index in candidates:
        r = Rotation.from_matrix(rotations[index])
        if all((r * Rotation.from_matrix(rotations[j]).inv()).magnitude() >= 0.2
               for j in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    for index in candidates:
        if len(selected) == count:
            break
        if int(index) not in selected:
            selected.append(int(index))
    return selected


def local_search(translation, helper, starts, maxiter):
    """Optimize three unconstrained incremental rotation-vector coordinates."""
    outcomes = []
    for start in starts:
        seed_value = float(minimum_eigenvalues(start[None], translation, helper)[0])
        best_value, best_rotation = seed_value, start.copy()

        def objective(delta):
            nonlocal best_value, best_rotation
            r = Rotation.from_rotvec(delta).as_matrix() @ start
            value = float(minimum_eigenvalues(r[None], translation, helper)[0])
            if value < best_value:
                best_value, best_rotation = value, r.copy()
            return value

        result = minimize(
            objective, np.zeros(3), method="Powell",
            options={"xtol": 1e-9, "ftol": 1e-11, "maxiter": maxiter},
        )
        outcomes.append({
            "seed_lambda": seed_value, "lambda_min": best_value,
            "rotation": best_rotation.tolist(), "success": bool(result.success),
            "message": str(result.message), "nfev": int(result.nfev),
            "nit": int(result.nit), "endpoint_lambda": float(result.fun),
        })
    return outcomes


def audit_matrix(path, args):
    started = time.perf_counter()
    raw = np.load(path, allow_pickle=False)
    if raw.shape not in ((15, 15), (39, 39)) or not np.isfinite(raw).all():
        raise ValueError(f"Invalid translation shape or entries: {path}: {raw.shape}")
    asymmetry = float(np.max(np.abs(raw - raw.T)))
    if asymmetry > 1e-12 * max(1.0, float(np.max(np.abs(raw)))):
        raise ValueError(f"Translation is not symmetric: {path}: {asymmetry}")
    translation = np.asarray(0.5 * (raw + raw.T), dtype=np.float64)
    bound_type = path.parent.name
    if bound_type not in ("upper_1", "upper_2", "lower_1", "lower_2"):
        raise ValueError(f"Unrecognized bound type: {path}")
    n_shear = (len(translation) - 9) // 6
    helper = make_helper(bound_type, n_shear)
    reference_error = validate_q(helper, args.seed + 1)
    norm = float(np.linalg.norm(translation, ord=2))
    epsilon = args.atol + args.rtol * norm
    rng = np.random.default_rng(args.seed)
    values = np.empty(args.samples)
    pool_values, pool_rotations = np.empty(0), np.empty((0, 3, 3))
    pool_size = min(args.samples, max(args.local_starts * 16, 1))
    for offset in range(0, args.samples, args.batch_size):
        count = min(args.batch_size, args.samples - offset)
        rotations = Rotation.random(count, random_state=rng).as_matrix()
        batch_values = minimum_eigenvalues(rotations, translation, helper)
        if not np.isfinite(batch_values).all():
            raise FloatingPointError(f"Nonfinite eigenvalues: {path}")
        values[offset:offset + count] = batch_values
        joined_values = np.concatenate((pool_values, batch_values))
        joined_rotations = np.concatenate((pool_rotations, rotations))
        keep = min(pool_size, len(joined_values))
        indices = np.argpartition(joined_values, keep - 1)[:keep]
        pool_values, pool_rotations = joined_values[indices], joined_rotations[indices]

    canonical = Rotation.from_rotvec(
        np.vstack((np.zeros(3), np.eye(3) * np.pi / 2, np.eye(3) * np.pi)),
    ).as_matrix()
    canonical_values = minimum_eigenvalues(canonical, translation, helper)
    candidate_values = np.concatenate((pool_values, canonical_values))
    candidate_rotations = np.concatenate((pool_rotations, canonical))
    starts = select_starts(candidate_values, candidate_rotations, args.local_starts)
    local = local_search(translation, helper, candidate_rotations[starts], args.maxiter)
    i = int(np.argmin(candidate_values))
    worst_value, worst_rotation = float(candidate_values[i]), candidate_rotations[i]
    for outcome in local:
        if outcome["lambda_min"] < worst_value:
            worst_value = outcome["lambda_min"]
            worst_rotation = np.asarray(outcome["rotation"])

    # Independent scalar assembly validates the actual worst-case witness.
    q = reference_q(worst_rotation, helper)
    a = q.T @ translation @ q
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (a + a.T))
    np.testing.assert_allclose(eigenvalues[0], worst_value, atol=1e-12 * max(1, norm), rtol=1e-10)
    witness = eigenvectors[:, 0]
    field = q @ witness
    relative_path = str(path.relative_to(args.results_root))
    return {
        "matrix": relative_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bound_type": bound_type, "calc_type": helper.calc_type,
        "shape": list(raw.shape), "q_shape": list(q.shape),
        "spectral_norm_T": norm, "input_asymmetry_max": asymmetry,
        "q_reference_max_abs_error": reference_error,
        "epsilon": epsilon, "random_samples": args.samples,
        "random_min": float(values.min()), "random_max": float(values.max()),
        "random_quantiles": dict(zip(
            ("0", "0.01", "0.5", "0.99", "1"),
            np.quantile(values, [0, .01, .5, .99, 1]).tolist(),
        )),
        "random_violations": int(np.count_nonzero(values < -epsilon)),
        "random_violations_by_absolute_tolerance": {
            f"{tol:.0e}": int(np.count_nonzero(values < -tol))
            for tol in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
        },
        "canonical_rotations": canonical.tolist(), "canonical_eigenvalues": canonical_values.tolist(),
        "canonical_violations": int(np.count_nonzero(canonical_values < -epsilon)),
        "local_starts": len(local),
        "local_violations": sum(r["lambda_min"] < -epsilon for r in local),
        "local_successes": sum(r["success"] for r in local), "local_results": local,
        "lambda_worst": float(eigenvalues[0]),
        "lambda_worst_over_norm_T": float(eigenvalues[0] / norm) if norm else None,
        "no_violation_found": bool(eigenvalues[0] >= -epsilon),
        "worst_rotation": worst_rotation.tolist(),
        "worst_quaternion_xyzw": Rotation.from_matrix(worst_rotation).as_quat().tolist(),
        "worst_wave_direction": worst_rotation[:, 0].tolist(),
        "worst_eigenvector": witness.tolist(), "worst_field": field.tolist(),
        "witness_quadratic_form": float(field @ translation @ field),
        "eigenpair_residual": float(np.linalg.norm(a @ witness - eigenvalues[0] * witness)),
        "rotation_orthogonality_error": float(np.linalg.norm(worst_rotation.T @ worst_rotation - np.eye(3))),
        "rotation_determinant": float(np.linalg.det(worst_rotation)),
        "seconds": time.perf_counter() - started,
    }
