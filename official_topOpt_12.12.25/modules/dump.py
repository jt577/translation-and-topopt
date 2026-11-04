import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import measure
import pyvista as pv
# start a headless X server via Xvfb (note: doesn't work on cluster nodes - only workstations)
try:
    pv.start_xvfb()
except Exception:
    pv.OFF_SCREEN = True

def projection(
    rho,
    out_dir=".",
    iter_num=None,
    axis="z",
    slice_indices=None,
    cmap="viridis"
):
    """
    Plot and save three 2D slices of a 3D density field in one figure.

    Args:
      rho          (Tensor or ndarray) shape (N,N,N)
      out_dir      directory to save plots
      iter_num     optional integer for filename
      axis         'x','y', or 'z' slice orientation
      slice_indices list of indices; default [N//4, N//2, 3*N//4]
      cmap         matplotlib colormap
    """
    # ensure NumPy array on CPU
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()

    # pick slice indices if not provided
    N = rho.shape["xyz".index(axis)]
    if slice_indices is None:
        slice_indices = [N//4, N//2, 3*N//4]

    # prepare output
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_iter{iter_num}" if iter_num is not None else ""

    # make one figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, idx in zip(axes, slice_indices):
        if axis == "z":
            img = rho[:, :, idx].T
            xlabel, ylabel = "x", "y"
        elif axis == "y":
            img = rho[:, idx, :].T
            xlabel, ylabel = "x", "z"
        elif axis == "x":
            img = rho[idx, :, :].T
            xlabel, ylabel = "y", "z"
        else:
            raise ValueError("axis must be 'x','y', or 'z'")

        im = ax.imshow(img,
                       origin="lower",
                       cmap=cmap,
                       aspect="auto")
        ax.set_title(f"{axis.upper()} = {idx}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # one shared colorbar for all
    cbar = fig.colorbar(im, ax=axes, label="density")
    
    # save and clean up
    fname = os.path.join(out_dir, f"slices{tag}_{axis}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

def isosurface(
    rho,
    out_dir=".",
    iter_num=None,
    tau=0.7,
    rel_tau=0.5,
    spacing=1.0,
    colour_cycle=('red','cyan','orange','lime','violet'),
    camera_position='iso'
):
    """
    Extract and save isosurfaces from a 3D density field.

    Args:
      rho             (Tensor or ndarray) shape (N,N,N)
      out_dir         directory to save meshes
      iter_num        optional integer for filename
      tau             preferred iso-level
      rel_tau         fallback fraction if tau out of [min,max]
      spacing         voxel size
      colour_cycle    iterable of colors
      camera_position PyVista view ('iso', '+x', etc.) or 3-tuple
    """
    # ensure NumPy array on CPU
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    tag = f"_iter{iter_num}" if iter_num is not None else ""

    vmin, vmax = float(rho.min()), float(rho.max())
    if vmax == vmin:
        print("Uniform field—no isosurface.")
        return

    level = tau if vmin <= tau <= vmax else vmin + rel_tau*(vmax-vmin)
    try:
        verts, faces, normals, values = measure.marching_cubes(
            rho, level=level, spacing=(spacing,)*3
        )
    except RuntimeError:
        print(f"No surface at level={level:.3f}")
        return

    if faces.size == 0:
        print("Empty surface—skipped")
        return

    # convert faces to PyVista “prepend count” format
    faces_pv = np.hstack([
        np.full((faces.shape[0],1), 3, dtype=np.int32),
        faces.astype(np.int32)
    ])

    mesh = pv.PolyData(verts, faces_pv)
    p = pv.Plotter(off_screen=True, window_size=(800,800))
    p.set_background("white")
    color = colour_cycle[0 % len(colour_cycle)]
    p.add_mesh(mesh, color=color, opacity=0.6, smooth_shading=True)
    p.show_axes()
    p.camera_position = camera_position
    p.add_text(f"level = {level:.3f}", font_size=24)

    fname = os.path.join(out_dir, f"isosurface{tag}.png")
    p.screenshot(fname)
    p.close()
