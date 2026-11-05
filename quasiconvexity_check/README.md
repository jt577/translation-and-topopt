# Quasiconvexity convergence figure

This directory contains the code and audit data used to make
`../convergence_study/minimum_eigenvalue_convergence_vs_inv_rotations.png`
and its PDF/SVG exports. It covers the 21 saved matrices in
`../convergence_study/`: the badly ordered shear `upper_1` formulation at
`alpha=1` and phase fraction `f1=0.5`.

## Files

- `plot_eigenvalue_convergence.py`: makes the final figure and its CSV/text
  companions from the saved audit, without running new rotation searches.
- `convergence_audit/results.json`: complete per-run audit results, including
  negative eigenvalues, violation counts, worst rotations/field witnesses,
  local solver outcomes, source matrix hashes, and settings.
- `check_convergence.py`: regenerates this audit from the saved convergence
  matrices and their input file. It also reuses `../convergence_study.py` to
  check the physical bound points against the original convergence CSVs.
- `check_quasiconvexity.py`: shared numerical routines used by the audit,
  including batched Q assembly, local searches, and original-Q validation.
  It imports the basis and maps from
  `../official_translation_12.7.2025/modules/helper.py`.

## Reproduce the figure

Run from the repository root using Python with NumPy, SciPy, and Matplotlib:

```bash
python3 quasiconvexity_check/plot_eigenvalue_convergence.py
```

The upper panel shows the percentage of audit rotations whose minimum
eigenvalue is below `-1e-8`, averaged across optimization runs. Shading spans
the full run range. The lower panel shows the mean smallest eigenvalue found
per matrix, using a signed logarithmic axis with a linear interval of
`[-1e-6, 1e-6]`. Both panels use the reciprocal optimization rotation count.
The run counts at 10, 20, 50, 100, and 200 rotations are 4, 3, 5, 4, and 5.

## Regenerate the audit

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 quasiconvexity_check/check_convergence.py \
  --samples 100000 --local-starts 24 \
  --output /tmp/translation-convergence-audit-new

python3 quasiconvexity_check/plot_eigenvalue_convergence.py \
  --audit /tmp/translation-convergence-audit-new/results.json
```

The audit output directory must be empty. The saved audit uses the same
seeded stream of 100,000 independent Haar rotations for every matrix, seven
canonical rotations, and 24 local Powell searches. Worst-case witnesses are
recomputed with the original scalar Q construction. Historical source hashes
in the saved JSON describe the scripts used at audit time; subsequent removal
of unused commands and alternate plotting functions did not change the
numerical audit routines or the saved results.

Finite sampling and local minimization provide numerical evidence, not a
proof over all rotations. Averaging across runs does not establish positivity
of every individual matrix.
