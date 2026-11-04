# Supporting data and code

This repository contains supporting data and code for the paper:

**Translation Method Bounds for 3D Isotropic Conductivity–Shear Modulus and the Search for Attaining Microstructures**  
Justin Tahmassebpur and Tomás Arias, Cornell University.

The work computes numerical translation-method bounds on effective conductivity and shear modulus for three-dimensional isotropic two-phase composites, and uses periodic topology optimization to search for microstructures that approach these bounds. An effective-medium approximation estimates the response of isotropic aggregates of randomly oriented grains.

## Repository contents

| Path | Contents |
| --- | --- |
| `official_translation_12.7.2025/` | Numerical translation-method implementation and example input parameters. |
| `official_topOpt_12.12.25/` | Periodic topology optimization, conductivity and elasticity calculations, microstructure initialization, and output utilities. |
| `translation_results/` | Saved translation matrices, calculation logs, input parameters, and bound plots. |
| `topopt_results/` | Microstructure density fields and tabulated effective properties, including TPMS and truss comparisons. |
| `convergence_study/` | Saved translation matrices, summary tables, and plots for the convergence study. |
| `convergence_study.py` | Convergence analysis script. |
| `plotter_allf.py`, `plotter_onef.py`, `plotter_onef_dataset.py` | Scripts for plotting bounds and comparing microstructure data. |
| `helper.py`, `misc.py` | Supporting numerical utilities. |

Data are provided as CSV and text tables, NumPy arrays (`.npy`), PyTorch tensors (`.pt`), and PNG plots. Calculation directories include input files documenting the associated material properties and numerical parameters.

Please cite the accompanying paper when using this code or data.
