import numpy as np

class TranslationHelper:
    def __init__(self, n_shear, phase_props, npts, upper, n_samples, alphas, seed, calc_type, bound_type):
        self.L = np.array([
                            [1/np.sqrt(3),              0,                              0],
                            [1/np.sqrt(2),              0,                              0],
                            [1/np.sqrt(6),              0,                              0],
                            [0,                         1/np.sqrt(2),                   0],
                            [0,                         0,                   1/np.sqrt(2)],
                            [0,                         0,                              0],
                            [0,                         1/np.sqrt(2),                   0],
                            [0,                         0,                   1/np.sqrt(2)],
                            [0,                         0,                              0]
        ])
        self.M = np.array([
                            [ 1/np.sqrt(3),     0.0,               1/np.sqrt(3)],
                            [-1/np.sqrt(2),     0.0,               0.0         ],
                            [ 1/np.sqrt(6),     0.0,              -np.sqrt(2/3)],
                            [ 0.0,              0.0,               0.0         ],
                            [ 0.0,              0.0,               0.0         ],
                            [ 0.0,              np.sqrt(2.0),      0.0         ],
                        ], float)
        self.N = np.array([
                            [ 0.0,               1/np.sqrt(3),   0.0,             0.0,             0.0,              1/np.sqrt(3)],
                            [ 0.0,              -1/np.sqrt(2),   0.0,             0.0,             0.0,              0.0         ],
                            [ 0.0,               1/np.sqrt(6),   0.0,             0.0,             0.0,             -np.sqrt(2/3)],
                            [ 1/np.sqrt(2),      0.0,            0.0,             0.0,             0.0,              0.0         ],
                            [ 0.0,               0.0,            0.0,             1/np.sqrt(2),    0.0,              0.0         ],
                            [ 0.0,               0.0,            1/np.sqrt(2),    0.0,             1/np.sqrt(2),     0.0         ],     
                            [-1/np.sqrt(2),      0.0,            0.0,             0.0,             0.0,              0.0         ], 
                            [ 0.0,               0.0,            0.0,            -1/np.sqrt(2),    0.0,              0.0         ], 
                            [ 0.0,               0.0,            1/np.sqrt(2),    0.0,            -1/np.sqrt(2),     0.0         ],
                        ], float)

        ah = (1/np.sqrt(3)) * np.array([[1,0,0],[0,1,0],[0,0,1]], float)
        a1 = (1/np.sqrt(2)) * np.array([[ 1,0,0],[0,-1,0],[0,0,0]], float)
        a2 = (1/np.sqrt(6)) * np.array([[ 1,0,0],[0, 1,0],[0,0,-2]], float)
        a3 = (1/np.sqrt(2)) * np.array([[ 0,1,0],[1, 0,0],[0,0,0]], float)
        a4 = (1/np.sqrt(2)) * np.array([[ 0,0,1],[0, 0,0],[1,0,0]], float)
        a5 = (1/np.sqrt(2)) * np.array([[ 0,0,0],[0, 0,1],[0,1,0]], float)
        b1 = (1/np.sqrt(2)) * np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], float)
        b2 = (1/np.sqrt(2)) * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float)
        b3 = (1/np.sqrt(2)) * np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], float)
        self.bases_reduced = [ah, a1, a2, a3, a4, a5]
        self.bases = [ah, a1, a2, a3, a4, a5, b1, b2, b3]
        self.n_shear = n_shear
        self.phase_props = phase_props
        self.npts = npts
        self.upper = upper
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.calc_type = calc_type
        self.bound_type = bound_type
        self.alphas = alphas

        self._get_Bs()
        self._create_Y()
        self._create_Z()
        self._create_D()
        self._make_diag()

    def _frob(self, A, B):
        """Frobenius inner product <A,B> = trace(A^T B)."""
        return float(np.tensordot(A, B))

    def _random_rotation(self):
        """Haar-uniform random rotation via a random unit quaternion."""
        q = self.rng.normal(size=4)
        q = q / np.linalg.norm(q)
        w, x, y, z = q  # scalar-first
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=float)
    
    def _assemble_G(self, R, bases):
        """G_{ij}(R) = <a_i, R a_j R^T>."""
        k = len(bases)
        G = np.empty((k, k), float)
        for j in range(k):
            RajRt = R @ bases[j] @ R.T
            for i in range(k):
                G[i, j] = self._frob(bases[i], RajRt)
        return G
    
    def _assemble_H(self, R, bases):
        """H_{ij}(R) = <a_i, R a_j>."""
        k = len(bases)
        H = np.empty((k, k), float)
        for i in range(k):
            for j in range(k):
                Raj = R @ bases[j]
                H[i, j] = self._frob(bases[i], Raj)
        return H

    def _get_Bs(self):
        """Generate a random B matrix for the current n_shear."""
        self.Bs = [self._block_rot() for _ in range(self.n_samples)]
    
    def _block_rot(self):
        """Block diagonal [V(R) 0; 0 Q(R)] without resampling R inside."""
        R = self._random_rotation()
        if self.bound_type == 'upper_1':
            G = self._assemble_G(R, self.bases_reduced)
            H = self._assemble_H(R, self.bases)
            V = G @ self.M
            Q = H @ self.N
        elif self.bound_type == 'upper_2':
            H = self._assemble_H(R, self.bases)
            V = (H @ self.L)[:6]
            Q = H @ self.N
        elif self.bound_type == 'lower_1':
            G = self._assemble_G(R, self.bases_reduced)
            H = self._assemble_H(R, self.bases)
            V = G @ self.M
            Q = H @ self.L
        elif self.bound_type == 'lower_2':
            H = self._assemble_H(R, self.bases)
            V = (H @ self.L)[:6]
            Q = H @ self.L
        r, c   = V.shape
        qr, qc = Q.shape
        dtype  = V.dtype

        # Block-diagonal m copies of V
        # shape: (m*r, m*c)
        m = self.n_shear
        VL = np.kron(np.eye(m, dtype=dtype), V)

        # Append Q in the bottom-right, zeros elsewhere
        top    = np.hstack([VL, np.zeros((m*r, qc), dtype=dtype)])
        bottom = np.hstack([np.zeros((qr, m*c), dtype=dtype), Q])

        block = np.vstack([top, bottom])
        return block

    def _y_transform(self, xstar, f, x1, x2):
        if x1 == x2:
            return -x1
        else:
            return -f*x2 - (1-f)*x1 - f*(1-f)*(x1-x2)**2 / (xstar - f*x1 - (1-f)*x2)
    
    def inv_y_transform(self, y, f, x1, x2):
        return f*x1 + (1-f)*x2 - f*(1-f)*(x2-x1)**2 / ((1-f)*x1 + f*x2 + y)

    def _create_Y(self):
        # Phases
        f = self.phase_props.f
        k1, k2, mu1, mu2, sig1, sig2 = self.phase_props.k1, self.phase_props.k2, self.phase_props.mu1, self.phase_props.mu2, self.phase_props.sig1, self.phase_props.sig2
        klow, mulow, siglow = 1/(f*1/k1 + (1-f)*1/k2), 1/(f*1/mu1 + (1-f)*1/mu2), 1/(f*1/sig1 + (1-f)*1/sig2)
        khigh, muhigh, sighigh = f*k1 + (1-f)*k2, f*mu1 + (1-f)*mu2, f*sig1 + (1-f)*sig2
        k_lower_HS = self.phase_props.k_lower_HS
        k_upper_HS = self.phase_props.k_upper_HS
        mu_lower_HS = self.phase_props.mu_lower_HS
        mu_upper_HS = self.phase_props.mu_upper_HS
        sig_upper_HS = self.phase_props.sig_upper_HS
        sig_lower_HS = self.phase_props.sig_lower_HS
        midpt_k_lower = 0.5 * (klow + k_lower_HS)
        midpt_k_upper = 0.5 * (khigh + k_upper_HS)
        midpt_mu_lower = 0.5 * (mulow + mu_lower_HS)
        midpt_mu_upper = 0.5 * (muhigh + mu_upper_HS)
        midpt_sig_lower = 0.5 * (siglow + sig_lower_HS)
        midpt_sig_upper = 0.5 * (sighigh + sig_upper_HS)
        corr = 1e-4  # small correction to avoid singularities at Reuss/Voigt bounds
        if self.bound_type == 'upper_1':
            if self.calc_type == 'bulk':
                kstar, sigstar = np.linspace(midpt_k_lower, midpt_k_upper, self.npts), np.linspace(midpt_sig_lower, midpt_sig_upper, self.npts)
                self.mustar_point = mulow + corr  # slightly above Reuss so no singularity and so result holds for general mustar
                # Build meshgrids
                self.kstar_mesh, self.sigstar_mesh = np.meshgrid(kstar, sigstar, indexing='ij')
                self.mustar_mesh = np.full(self.kstar_mesh.shape, self.mustar_point)
            elif self.calc_type == 'shear':
                mustar, sigstar = np.linspace(midpt_mu_lower, midpt_mu_upper, self.npts), np.linspace(midpt_sig_lower, midpt_sig_upper, self.npts)
                self.kstar_point = klow + corr  # slightly above Reuss so no singularity and so result holds for general kstar
                # Build meshgrids
                self.mustar_mesh, self.sigstar_mesh = np.meshgrid(mustar, sigstar, indexing='ij')
                self.kstar_mesh = np.full(self.mustar_mesh.shape, self.kstar_point)
            yk = self._y_transform(self.kstar_mesh, f, k1, k2)
            ymu = self._y_transform(self.mustar_mesh, f, mu1, mu2)
            ysig = self._y_transform(self.sigstar_mesh, f, sig1, sig2)

            # build y_list
            y_list_elastic = [1/3 * 1/yk] +  [1/2 * 1/ymu] * 5
            y_list_conductive = [1/ysig] * 9
            y_list = y_list_elastic * self.n_shear  + y_list_conductive

        elif self.bound_type == 'upper_2':
            if self.calc_type == 'bulk':
                kstar, sigstar = np.linspace(midpt_k_lower, midpt_k_upper, self.npts), np.linspace(midpt_sig_lower, midpt_sig_upper, self.npts)
                self.mustar_point = muhigh - corr  # slightly below Voigt so no singularity and so result holds for general mustar
                # Build meshgrids
                self.kstar_mesh, self.sigstar_mesh = np.meshgrid(kstar, sigstar, indexing='ij')
                self.mustar_mesh = np.full(self.kstar_mesh.shape, self.mustar_point)
            elif self.calc_type == 'shear':
                mustar, sigstar = np.linspace(midpt_mu_lower, midpt_mu_upper, self.npts), np.linspace(midpt_sig_lower, midpt_sig_upper, self.npts)
                self.kstar_point = khigh - corr  # slightly below Voigt so no singularity and so result holds for general kstar
                # Build meshgrids
                self.mustar_mesh, self.sigstar_mesh = np.meshgrid(mustar, sigstar, indexing='ij')
                self.kstar_mesh = np.full(self.mustar_mesh.shape, self.kstar_point)
            yk = self._y_transform(self.kstar_mesh, f, k1, k2)
            ymu = self._y_transform(self.mustar_mesh, f, mu1, mu2)
            ysig = self._y_transform(self.sigstar_mesh, f, sig1, sig2)
            # build y_list
            y_list_elastic = [3 * yk] +  [2 * ymu] * 5
            y_list_conductive = [1/ysig] * 9
            y_list = y_list_elastic * self.n_shear  + y_list_conductive
        
        elif self.bound_type == 'lower_1':
            if self.calc_type == 'bulk':
                kstar, sigstar = np.linspace(midpt_k_lower, midpt_k_upper, self.npts), np.linspace(midpt_sig_lower, midpt_sig_upper, self.npts)
                self.mustar_point = mulow + corr  # slightly above Reuss so no singularity and so result holds for general mustar
                # Build meshgrids
                self.kstar_mesh, self.sigstar_mesh = np.meshgrid(kstar, sigstar, indexing='ij')
                self.mustar_mesh = np.full(self.kstar_mesh.shape, self.mustar_point)
            elif self.calc_type == 'shear':
                mustar, sigstar = np.linspace(midpt_mu_lower, midpt_mu_upper, self.npts), np.linspace(midpt_sig_lower, midpt_sig_upper, self.npts)
                self.kstar_point = klow + corr  # slightly above Reuss so no singularity and so result holds for general kstar
                # Build meshgrids
                self.mustar_mesh, self.sigstar_mesh = np.meshgrid(mustar, sigstar, indexing='ij')
                self.kstar_mesh = np.full(self.mustar_mesh.shape, self.kstar_point)
            yk = self._y_transform(self.kstar_mesh, f, k1, k2)
            ymu = self._y_transform(self.mustar_mesh, f, mu1, mu2)
            ysig = self._y_transform(self.sigstar_mesh, f, sig1, sig2)
            # build y_list
            y_list_elastic = [1/3 * 1/yk] +  [1/2 * 1/ymu] * 5
            y_list_conductive = [ysig] * 9
            y_list = y_list_elastic * self.n_shear  + y_list_conductive
        
        elif self.bound_type == 'lower_2':
            if self.calc_type == 'bulk':
                kstar, sigstar = np.linspace(midpt_k_lower, midpt_k_upper, self.npts), np.linspace(midpt_sig_lower, midpt_sig_upper, self.npts)
                self.mustar_point = muhigh - corr  # slightly below Voigt so no singularity and so result holds for general mustar
                # Build meshgrids
                self.kstar_mesh, self.sigstar_mesh = np.meshgrid(kstar, sigstar, indexing='ij')
                self.mustar_mesh = np.full(self.kstar_mesh.shape, self.mustar_point)
            elif self.calc_type == 'shear':
                mustar, sigstar = np.linspace(midpt_mu_lower, midpt_mu_upper, self.npts), np.linspace(midpt_sig_lower, midpt_sig_upper, self.npts)
                self.kstar_point = khigh - corr  # slightly below Voigt so no singularity and so result holds for general kstar
                # Build meshgrids
                self.mustar_mesh, self.sigstar_mesh = np.meshgrid(mustar, sigstar, indexing='ij')
                self.kstar_mesh = np.full(self.mustar_mesh.shape, self.kstar_point)
            yk = self._y_transform(self.kstar_mesh, f, k1, k2)
            ymu = self._y_transform(self.mustar_mesh, f, mu1, mu2)
            ysig = self._y_transform(self.sigstar_mesh, f, sig1, sig2)
            # build y_list
            y_list_elastic = [3 * yk] +  [2 * ymu] * 5
            y_list_conductive = [ysig] * 9
            y_list = y_list_elastic * self.n_shear  + y_list_conductive

        # Build Y as a diagonal tensor of shape (n, n, npts, npts, npts)
        n = 9 + 6 * self.n_shear
        self.Y = np.zeros((n, n) + yk.shape)
        for i in range(n):
            self.Y[i, i, ...] = y_list[i]   # broadcasts over (...)=grid

    def _create_Z(self):
        # Build meshgrids
        if self.bound_type == 'upper_1':
            if self.calc_type == 'bulk':
                zk = np.linspace(0.0, self.upper, self.npts)
                zsig = np.linspace(0.0, self.upper, self.npts)
                self.ZK, self.ZSIG = np.meshgrid(zk, zsig, indexing='ij')
                self.zmu_point = 1/self._y_transform(self.mustar_point, self.phase_props.f, self.phase_props.mu1, self.phase_props.mu2) # this corr to self.mustar_point; if you want HS point, use 3/(4*self.phase_props.mu2)
                self.ZMU = np.full(self.ZK.shape, self.zmu_point)
            elif self.calc_type == 'shear':
                zmu = np.linspace(0.0, self.upper, self.npts)
                zsig = np.linspace(0.0, self.upper, self.npts)
                self.ZMU, self.ZSIG = np.meshgrid(zmu, zsig, indexing='ij')
                self.zk_point = 1/self._y_transform(self.kstar_point, self.phase_props.f, self.phase_props.k1, self.phase_props.k2) # this corr to self.kstar_point; if you want HS point, use 3/(4*self.phase_props.mu2)
                self.ZK = np.full(self.ZMU.shape, self.zk_point)
            # Build Z_list
            Z_list_elastic = [1/3 * self.ZK] +  [1/2 * self.ZMU] * 5
            Z_list_conductive = [self.ZSIG] * 9
            Z_list = Z_list_elastic * self.n_shear  + Z_list_conductive
        
        elif self.bound_type == 'upper_2':
            if self.calc_type == 'bulk':
                zk = np.linspace(0.0, self.upper, self.npts)
                zsig = np.linspace(0.0, self.upper, self.npts)
                self.ZK, self.ZSIG = np.meshgrid(zk, zsig, indexing='ij')
                self.zmu_point = self._y_transform(self.mustar_point, self.phase_props.f, self.phase_props.mu1, self.phase_props.mu2) # this corr to self.mustar_point; if you want HS point, use 3/(4*self.phase_props.mu2)
                self.ZMU = np.full(self.ZK.shape, self.zmu_point)
            elif self.calc_type == 'shear':
                zmu = np.linspace(0.0, self.upper, self.npts)
                zsig = np.linspace(0.0, self.upper, self.npts)
                self.ZMU, self.ZSIG = np.meshgrid(zmu, zsig, indexing='ij')
                self.zk_point = self._y_transform(self.kstar_point, self.phase_props.f, self.phase_props.k1, self.phase_props.k2) # this corr to self.kstar_point; if you want HS point, use 3/(4*self.phase_props.mu2)
                self.ZK = np.full(self.ZMU.shape, self.zk_point)
            # Build Z_list
            Z_list_elastic = [3 * self.ZK] +  [2 * self.ZMU] * 5
            Z_list_conductive = [self.ZSIG] * 9
            Z_list = Z_list_elastic * self.n_shear  + Z_list_conductive
        
        elif self.bound_type == 'lower_1':
            if self.calc_type == 'bulk':
                zk = np.linspace(0.0, self.upper, self.npts)
                zsig = np.linspace(0.0, self.upper, self.npts)
                self.ZK, self.ZSIG = np.meshgrid(zk, zsig, indexing='ij')
                self.zmu_point = 1/self._y_transform(self.mustar_point, self.phase_props.f, self.phase_props.mu1, self.phase_props.mu2) # this corr to self.mustar_point; if you want HS point, use 3/(4*self.phase_props.mu2)
                self.ZMU = np.full(self.ZK.shape, self.zmu_point)
            elif self.calc_type == 'shear':
                zmu = np.linspace(0.0, self.upper, self.npts)
                zsig = np.linspace(0.0, self.upper, self.npts)
                self.ZMU, self.ZSIG = np.meshgrid(zmu, zsig, indexing='ij')
                self.zk_point = 1/self._y_transform(self.kstar_point, self.phase_props.f, self.phase_props.k1, self.phase_props.k2) # this corr to self.kstar_point; if you want HS point, use 3/(4*self.phase_props.mu2)
                self.ZK = np.full(self.ZMU.shape, self.zk_point)
            # Build Z_list
            Z_list_elastic = [1/3 * self.ZK] +  [1/2 * self.ZMU] * 5
            Z_list_conductive = [self.ZSIG] * 9
            Z_list = Z_list_elastic * self.n_shear  + Z_list_conductive
        
        elif self.bound_type == 'lower_2':
            if self.calc_type == 'bulk':
                zk = np.linspace(0.0, self.upper, self.npts)
                zsig = np.linspace(0.0, self.upper, self.npts)
                self.ZK, self.ZSIG = np.meshgrid(zk, zsig, indexing='ij')
                self.zmu_point = self._y_transform(self.mustar_point, self.phase_props.f, self.phase_props.mu1, self.phase_props.mu2) # this corr to self.mustar_point; if you want HS point, use 3/(4*self.phase_props.mu2)
                self.ZMU = np.full(self.ZK.shape, self.zmu_point)
            elif self.calc_type == 'shear':
                zmu = np.linspace(0.0, self.upper, self.npts)
                zsig = np.linspace(0.0, self.upper, self.npts)
                self.ZMU, self.ZSIG = np.meshgrid(zmu, zsig, indexing='ij')
                self.zk_point = self._y_transform(self.kstar_point, self.phase_props.f, self.phase_props.k1, self.phase_props.k2) # this corr to self.kstar_point; if you want HS point, use 3/(4*self.phase_props.mu2)
                self.ZK = np.full(self.ZMU.shape, self.zk_point)
            # Build Z_list
            Z_list_elastic = [3 * self.ZK] +  [2 * self.ZMU] * 5
            Z_list_conductive = [self.ZSIG] * 9
            Z_list = Z_list_elastic * self.n_shear  + Z_list_conductive
        
        # Build Z as a diagonal tensor of shape (n, n, npts, npts, npts)
        n = 9 + 6 * self.n_shear
        self.Z = np.zeros((n, n) + self.ZK.shape)
        for i in range(n):
            self.Z[i, i, ...] = Z_list[i]   # broadcasts over (...)=grid

    def _create_D(self):
        k1, k2, mu1, mu2, sig1, sig2 = self.phase_props.k1, self.phase_props.k2, self.phase_props.mu1, self.phase_props.mu2, self.phase_props.sig1, self.phase_props.sig2

        if self.bound_type == 'upper_1':
            # Elastic compliances
            D_el1 = [1.0/(3.0*k1)] + [1.0/(2.0*mu1)] * 5
            D_el2 = [1.0/(3.0*k2)] + [1.0/(2.0*mu2)] * 5
            # Electric compliances
            D_elc1 = [1.0/sig1] * 9
            D_elc2 = [1.0/sig2] * 9
        elif self.bound_type == 'upper_2':
            # Elastic compliances
            D_el1 = [3.0*k1] + [2.0*mu1] * 5
            D_el2 = [3.0*k2] + [2.0*mu2] * 5
            # Electric compliances
            D_elc1 = [1.0/sig1] * 9
            D_elc2 = [1.0/sig2] * 9
        elif self.bound_type == 'lower_1':
            # Elastic compliances
            D_el1 = [1.0/(3.0*k1)] + [1.0/(2.0*mu1)] * 5
            D_el2 = [1.0/(3.0*k2)] + [1.0/(2.0*mu2)] * 5
            # Electric compliances
            D_elc1 = [sig1] * 9
            D_elc2 = [sig2] * 9
        elif self.bound_type == 'lower_2':
            # Elastic compliances
            D_el1 = [3.0*k1] + [2.0*mu1] * 5
            D_el2 = [3.0*k2] + [2.0*mu2] * 5
            # Electric compliances
            D_elc1 = [sig1] * 9
            D_elc2 = [sig2] * 9
            
        self.D1 = np.diag(D_el1 * self.n_shear + D_elc1)
        self.D2 = np.diag(D_el2 * self.n_shear + D_elc2)

    def _make_diag(self):
        # Make diagonal matrices for use in objective function
        if self.bound_type == 'upper_1':
            D_k_list_elastic = [1/3] +  [0] * 5
            D_k_list_conductive = [0] * 9
            D_mu_list_elastic = [0] +  [1/2] * 5
            D_mu_list_conductive = [0] * 9
            D_sig_list_elastic = [0] * (6 * self.n_shear)
            D_sig_list_conductive = [1] * 9
        elif self.bound_type == 'upper_2':
            D_k_list_elastic = [3] +  [0] * 5
            D_k_list_conductive = [0] * 9
            D_mu_list_elastic = [0] +  [2] * 5
            D_mu_list_conductive = [0] * 9
            D_sig_list_elastic = [0] * (6 * self.n_shear)
            D_sig_list_conductive = [1] * 9
        elif self.bound_type == 'lower_1':
            D_k_list_elastic = [1/3] +  [0] * 5
            D_k_list_conductive = [0] * 9
            D_mu_list_elastic = [0] +  [1/2] * 5
            D_mu_list_conductive = [0] * 9
            D_sig_list_elastic = [0] * (6 * self.n_shear)
            D_sig_list_conductive = [1] * 9
        elif self.bound_type == 'lower_2':
            D_k_list_elastic = [3] +  [0] * 5
            D_k_list_conductive = [0] * 9
            D_mu_list_elastic = [0] +  [2] * 5
            D_mu_list_conductive = [0] * 9
            D_sig_list_elastic = [0] * (6 * self.n_shear)
            D_sig_list_conductive = [1] * 9
        
        D_k_list = D_k_list_elastic * self.n_shear  + D_k_list_conductive
        D_mu_list = D_mu_list_elastic * self.n_shear  + D_mu_list_conductive
        D_sig_list = D_sig_list_elastic + D_sig_list_conductive
        self.D_k = np.diag(D_k_list)
        self.D_mu = np.diag(D_mu_list)
        self.D_sig = np.diag(D_sig_list)