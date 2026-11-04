import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import gridspec
from matplotlib.colors import Normalize
from modules.misc import batch_min_eigval

class plot_results:
    def __init__(self, out_dir_path, helper):
        self.out_dir_path = out_dir_path
        self.helper = helper
        
    def _plot_negative_region_2d(self,
        Z1, Z2, lam_min_grid, level=0.0,
        dot_line_values=None,
        label_names=(r"$z_\mu$", r"$z_\sigma$"),
        curves=None, 
        pct=98,                  # robust symmetric scaling uses this percentile of |λ|
        cmap="RdBu"              # red→neg, blue→pos (centered at white)
    ):
        """
        2D heat map of λ_min(Y(z)+T) on the (z_mu, z_sig) plane for a fixed z_k (= zk).

        Parameters
        ----------
        ZMU, ZSIG : 2D arrays
            Meshgrids of z_mu and z_sig with shape (n_mu, n_sig).
        lam_min_grid : 2D array
            λ_min evaluated on the same (z_mu,z_sig) grid. Shape (n_mu, n_sig).
        level : float
            Contour level for the zero boundary (default 0.0).
        dot_line_values : tuple or None
            (k_star, mu_star, sig_star) to draw dashed guides at those coordinates.
        label_names : tuple of str
            (label for k, label for mu, label for sig); the k label is used in the
            slice annotation only.
        pct : float
            Percentile used for robust symmetric color scaling around 0.
        cmap : str
            Matplotlib colormap name.
        """
        # axis vectors from meshgrids (assume proper monotone grids)
        z1 = Z1[:, 0] if Z1.ndim == 2 and Z1.shape[1] > 1 else Z1.ravel()
        z2 = Z2[0, :] if Z2.ndim == 2 and Z2.shape[0] > 1 else Z2.ravel()

        # Robust, symmetric color scaling centered at 0
        vals = lam_min_grid[np.isfinite(lam_min_grid)]
        if vals.size == 0:
            vabs = 1.0
        else:
            vabs = np.nanpercentile(np.abs(vals), pct)
            if not np.isfinite(vabs) or vabs <= 0:
                vabs = max(np.nanmax(np.abs(vals)), 1.0)
        norm = Normalize(vmin=-float(vabs), vmax=float(vabs))

        # Figure
        fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=160)

        # Heat map
        im = ax.contourf(
            z1, z2, lam_min_grid.T, levels=64, cmap=cmap, norm=norm, antialiased=True
        )
        # Zero-level contour (feasible boundary)
        ax.contour(z1, z2, lam_min_grid.T, levels=[level], colors="k", linewidths=1.2, label='Translation')

        # Labels and title (annotate fixed k)
        ax.set_xlabel(label_names[0])
        ax.set_ylabel(label_names[1])

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
        cbar.set_label(r"$\lambda_{\min}$")

        # Optional dashed guides
        if dot_line_values is not None:
            star_1_upper, star_2_upper = dot_line_values[0]
            star_1_lower, star_2_lower = dot_line_values[1]
            # On (z_mu, z_sig) plane we draw lines at mu_star and sig_star
            if np.isfinite(star_1_upper):
                ax.axvline(star_1_upper, color="k", linestyle="--", alpha=0.6, linewidth=1.0, label='Hashin-Shtrikman')
            if np.isfinite(star_2_upper):
                ax.axhline(star_2_upper, color="k", linestyle="--", alpha=0.6, linewidth=1.0)
            if np.isfinite(star_1_lower):
                ax.axvline(star_1_lower, color="k", linestyle="--", alpha=0.6, linewidth=1.0)
            if np.isfinite(star_2_lower):
                ax.axhline(star_2_lower, color="k", linestyle="--", alpha=0.6, linewidth=1.0)

        # --- Other bound overlay ---
        if curves is not None:
            indic = False
            for curve in curves:
                curve_1, curve_2 = curve
                if self.helper.calc_type == 'shear':
                    label = "Berryman-Milton"
                elif self.helper.calc_type == 'bulk':
                    label = "Gibiansky-Torquato"
                if indic is False:
                    indic = True
                    ax.plot(
                        curve_1, curve_2,
                        linestyle=":", linewidth=2.0, color='red',
                        label=label
                    )
                else:
                    ax.plot(
                        curve_1, curve_2,
                        linestyle=":", linewidth=2.0, color='red'
                    )

        # --- Correct curve: image of z_sigma = alpha z_mu under inv_y ---
        if (Z1 is self.helper.mustar_mesh or Z1 is self.helper.kstar_mesh) and (Z2 is self.helper.sigstar_mesh):
            # We're in STAR space (μ*, σ*)
            # choose z_mu range that maps inside current axes
            z1_vals = np.linspace(1e-3, 100, 10000)
            f = self.helper.phase_props.f
            counter = 0
            for alpha in self.helper.alphas:
                if self.helper.bound_type == 'upper_1':
                    if self.helper.calc_type == 'shear':
                        mu1, mu2 = self.helper.phase_props.mu1, self.helper.phase_props.mu2
                        sig1, sig2 = self.helper.phase_props.sig1, self.helper.phase_props.sig2
                        mu_star_curve  = self.helper.inv_y_transform(1.0 / z1_vals, f, mu1,  mu2)
                        sig_star_curve = self.helper.inv_y_transform(1.0 / (alpha * z1_vals), f, sig1, sig2)
                        if counter == 0:
                            ax.plot(mu_star_curve, sig_star_curve, color='orange', linewidth=1.0,
                                    label=r"Line $z_\sigma = \alpha\, z_\mu$ mapped to $(\mu^*,\sigma^*)$")
                        else:
                            ax.plot(mu_star_curve, sig_star_curve, color='orange', linewidth=1.0)
                    elif self.helper.calc_type == 'bulk':
                        k1, k2 = self.helper.phase_props.k1, self.helper.phase_props.k2
                        sig1, sig2 = self.helper.phase_props.sig1, self.helper.phase_props.sig2
                        k_star_curve  = self.helper.inv_y_transform(1.0 / z1_vals, f, k1,  k2)
                        sig_star_curve = self.helper.inv_y_transform(1.0 / (alpha * z1_vals), f, sig1, sig2)
                        if counter == 0:
                            ax.plot(k_star_curve, sig_star_curve, color='orange', linewidth=1.0,
                                    label=r"Line $z_\sigma = \alpha\, z_\kappa$ mapped to $(\kappa^*,\sigma^*)$")
                        else:
                            ax.plot(k_star_curve, sig_star_curve, color='orange', linewidth=1.0)
                
                elif self.helper.bound_type == 'upper_2':
                    if self.helper.calc_type == 'shear':
                        mu1, mu2 = self.helper.phase_props.mu1, self.helper.phase_props.mu2
                        sig1, sig2 = self.helper.phase_props.sig1, self.helper.phase_props.sig2
                        mu_star_curve  = self.helper.inv_y_transform(z1_vals, f, mu1,  mu2)
                        sig_star_curve = self.helper.inv_y_transform(1.0 / (alpha * z1_vals), f, sig1, sig2)
                        if counter == 0:
                            ax.plot(mu_star_curve, sig_star_curve, color='orange', linewidth=1.0,
                                    label=r"Line $z_\sigma = \alpha\, z_\mu$ mapped to $(\mu^*,\sigma^*)$")
                        else:
                            ax.plot(mu_star_curve, sig_star_curve, color='orange', linewidth=1.0)
                    elif self.helper.calc_type == 'bulk':
                        k1, k2 = self.helper.phase_props.k1, self.helper.phase_props.k2
                        sig1, sig2 = self.helper.phase_props.sig1, self.helper.phase_props.sig2
                        k_star_curve  = self.helper.inv_y_transform(z1_vals, f, k1,  k2)
                        sig_star_curve = self.helper.inv_y_transform(1.0 / (alpha * z1_vals), f, sig1, sig2)
                        if counter == 0:
                            ax.plot(k_star_curve, sig_star_curve, color='orange', linewidth=1.0,
                                    label=r"Line $z_\sigma = \alpha\, z_\kappa$ mapped to $(\kappa^*,\sigma^*)$")
                        else:
                            ax.plot(k_star_curve, sig_star_curve, color='orange', linewidth=1.0)

                elif self.helper.bound_type == 'lower_1':
                    if self.helper.calc_type == 'shear':
                        mu1, mu2 = self.helper.phase_props.mu1, self.helper.phase_props.mu2
                        sig1, sig2 = self.helper.phase_props.sig1, self.helper.phase_props.sig2
                        mu_star_curve  = self.helper.inv_y_transform(1/z1_vals, f, mu1,  mu2)
                        sig_star_curve = self.helper.inv_y_transform(alpha * z1_vals, f, sig1, sig2)
                        if counter == 0:
                            ax.plot(mu_star_curve, sig_star_curve, color='orange', linewidth=1.0,
                                    label=r"Line $z_\sigma = \alpha\, z_\mu$ mapped to $(\mu^*,\sigma^*)$")
                        else:
                            ax.plot(mu_star_curve, sig_star_curve, color='orange', linewidth=1.0)

                    elif self.helper.calc_type == 'bulk':
                        k1, k2 = self.helper.phase_props.k1, self.helper.phase_props.k2
                        sig1, sig2 = self.helper.phase_props.sig1, self.helper.phase_props.sig2
                        k_star_curve  = self.helper.inv_y_transform(1/z1_vals, f, k1,  k2)
                        sig_star_curve = self.helper.inv_y_transform(alpha * z1_vals, f, sig1, sig2)
                        if counter == 0:
                            ax.plot(k_star_curve, sig_star_curve, color='orange', linewidth=1.0,
                                label=r"Line $z_\sigma = \alpha\, z_\kappa$ mapped to $(\kappa^*,\sigma^*)$")
                        else:
                            ax.plot(k_star_curve, sig_star_curve, color='orange', linewidth=1.0)

                elif self.helper.bound_type == 'lower_2':
                    if self.helper.calc_type == 'shear':
                        mu1, mu2 = self.helper.phase_props.mu1, self.helper.phase_props.mu2
                        sig1, sig2 = self.helper.phase_props.sig1, self.helper.phase_props.sig2
                        mu_star_curve  = self.helper.inv_y_transform(z1_vals, f, mu1,  mu2)
                        sig_star_curve = self.helper.inv_y_transform(alpha * z1_vals, f, sig1, sig2)
                        if counter == 0:
                            ax.plot(mu_star_curve, sig_star_curve, color='orange', linewidth=1.0,
                                    label=r"Line $z_\sigma = \alpha\, z_\mu$ mapped to $(\mu^*,\sigma^*)$")
                        else:
                            ax.plot(mu_star_curve, sig_star_curve, color='orange', linewidth=1.0)
                    elif self.helper.calc_type == 'bulk':
                        k1, k2 = self.helper.phase_props.k1, self.helper.phase_props.k2
                        sig1, sig2 = self.helper.phase_props.sig1, self.helper.phase_props.sig2
                        k_star_curve  = self.helper.inv_y_transform(z1_vals, f, k1,  k2)
                        sig_star_curve = self.helper.inv_y_transform(alpha * z1_vals, f, sig1, sig2)
                        if counter == 0:
                            ax.plot(k_star_curve, sig_star_curve, color='orange', linewidth=1.0,
                                    label=r"Line $z_\sigma = \alpha\, z_\kappa$ mapped to $(\kappa^*,\sigma^*)$")
                        else:
                            ax.plot(k_star_curve, sig_star_curve, color='orange', linewidth=1.0)
                else:
                    # We're in Z-plane (z_mu, z_sig), a straight line is correct there:
                    z1 = np.linspace(np.min(Z1), np.max(Z1), 200)
                    z2 = self.helper.alpha * z1
                    if counter == 0:
                        if self.helper.calc_type == 'shear':
                            ax.plot(z1, z2, color='orange', linewidth=2.0, label=r"$z_\sigma=\alpha z_\mu$")
                        elif self.helper.calc_type == 'bulk':
                            ax.plot(z1, z2, color='orange', linewidth=2.0, label=r"$z_\sigma=\alpha z_\kappa$")   
                    else:
                        if self.helper.calc_type == 'shear':
                            ax.plot(z1, z2, color='orange', linewidth=2.0)
                        elif self.helper.calc_type == 'bulk':
                            ax.plot(z1, z2, color='orange', linewidth=2.0)       
                counter += 1  
        ax.legend(loc='best', fontsize=12)
        ax.set_xlim(np.min(z1), np.max(z1))
        ax.set_ylim(np.min(z2), np.max(z2))
        fig.tight_layout()
        return fig, ax

    
    def _plot_YpT(self, t, iteration):
        """
        Plots the negative λ_min region of A(star) = Y + T over a grid in effective properties kstar, mustar, sigstar.
        """
        t_np = np.array(t.detach().cpu())
        n = int(np.sqrt(t_np.size))
        T = t_np.reshape(n, n)
        # Ensure symmetry (robustness if t is slightly off)
        T = 0.5 * (T + T.T)

        # Get Y matrix meshgrid
        Y = self.helper.Y

        # A(z) = Y(z) + T, with T broadcast across the grid
        A = Y + T[..., None, None]  # shape (n, n, npts, npts)

        # Move grid axes to the front and collapse to a batch: (M, n, n)
        # where M = npts^2
        A_batch = np.moveaxis(A, (2,3), (0,1)).reshape(-1, n, n)

        # Batched eigensolve
        lam_min = batch_min_eigval(A_batch)[0]          # lam_min: (M,)
        lam_grid = lam_min.reshape(self.helper.kstar_mesh.shape)  # (npts, npts)

        if self.helper.calc_type == 'shear':
            z1_mesh = self.helper.mustar_mesh
            z2_mesh = self.helper.sigstar_mesh
            dot_line_values = [
                (self.helper.phase_props.mu_upper_HS, self.helper.phase_props.sig_upper_HS),
                (self.helper.phase_props.mu_lower_HS, self.helper.phase_props.sig_lower_HS)
            ]
            curves = [
                (self.helper.phase_props.mu_BM1, self.helper.phase_props.sig_BM1),
                (self.helper.phase_props.mu_BM2, self.helper.phase_props.sig_BM2),
                (self.helper.phase_props.mu_BM3, self.helper.phase_props.sig_BM3),
                (self.helper.phase_props.mu_BM4, self.helper.phase_props.sig_BM4)
            ]
            label_names = (r"$\mu^*$", r"$\sigma^*$")
        elif self.helper.calc_type == 'bulk':
            z1_mesh = self.helper.kstar_mesh
            z2_mesh = self.helper.sigstar_mesh
            dot_line_values = [
                (self.helper.phase_props.k_upper_HS, self.helper.phase_props.sig_upper_HS),
                (self.helper.phase_props.k_lower_HS, self.helper.phase_props.sig_lower_HS)
            ]
            curves = [
                (self.helper.phase_props.k_GT1, self.helper.phase_props.sig_GT1),
                (self.helper.phase_props.k_GT2, self.helper.phase_props.sig_GT2),
                (self.helper.phase_props.k_GT3, self.helper.phase_props.sig_GT3),
                (self.helper.phase_props.k_GT4, self.helper.phase_props.sig_GT4),
                (self.helper.phase_props.k_GT5, self.helper.phase_props.sig_GT5)
            ]
            label_names = (r"$\kappa^*$", r"$\sigma^*$")

        fig, axs = self._plot_negative_region_2d(z1_mesh, z2_mesh,
                                            lam_grid, 
                                            dot_line_values=dot_line_values,
                                            curves=curves, 
                                            label_names=label_names)
        plt.savefig(f"{self.out_dir_path}/star_plots/feasible_star_iter{iteration}.png", dpi=300)
        plt.close()

    def _plot_ZpT(self, t, iteration):
        """
        Plots the negative λ_min region of A(z) = Z + T over a grid in inverse y transforms: zk, zmu, zsig.
        """
        t_np = np.array(t.detach().cpu())
        n = int(np.sqrt(t_np.size))
        T = t_np.reshape(n, n)
        # Ensure symmetry (robustness if t is slightly off)
        T = 0.5 * (T + T.T)

        # Get Z matrix meshgrid
        Z = self.helper.Z

        # A(z) = Y(z) + T, with T broadcast across the grid
        A = Z + T[..., None, None]  # shape (n, n, npts, npts)

        # Move grid axes to the front and collapse to a batch: (M, n, n)
        # where M = npts^2
        A_batch = np.moveaxis(A, (2,3), (0,1)).reshape(-1, n, n)

        # Batched eigensolve
        lam_min = batch_min_eigval(A_batch)[0]        # lam_min: (M,)
        lam_grid = lam_min.reshape(self.helper.ZK.shape)  # (npts, npts)
        if self.helper.calc_type == 'shear':
            z1_mesh = self.helper.ZMU
            z2_mesh = self.helper.ZSIG
            label_names = (r"$z_\mu$", r"$z_\sigma$")
        elif self.helper.calc_type == 'bulk':
            z1_mesh = self.helper.ZK
            z2_mesh = self.helper.ZSIG
            label_names = (r"$z_\kappa$", r"$z_\sigma$")
        fig, axs = self._plot_negative_region_2d(z1_mesh, z2_mesh, lam_grid, label_names=label_names)
        plt.savefig(f"{self.out_dir_path}/z_plots/feasible_z_iter{iteration}.png", dpi=300)
        plt.close()

    def plot(self, t, iteration):
        self._plot_ZpT(t, iteration)
        self._plot_YpT(t, iteration) # f can be set to anything within 0 to 1, since bound is valid for any f