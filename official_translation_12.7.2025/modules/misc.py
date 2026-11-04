import numpy as np
from scipy.linalg import eigh

def batch_min_eigval(A, tol=1e-10):
    """
    A: (..., n, n) symmetric
    Returns:
      lam_min: (...,)
      G: (..., n, n)  projector onto min-eigenspace (averaged if multiplicity>1)
    """
    eigvals, eigvecs = np.linalg.eigh(A)      # eigvals (..., n), eigvecs (..., n, n)
    lam_min = eigvals[..., 0]
    vmin   = eigvecs[..., :, 0]               # (..., n)
    G = vmin[..., :, None] * vmin[..., None, :]  # (..., n, n)

    # multiplicity handling
    mult_mask = (eigvals - lam_min[..., None]) <= tol   # (..., n)
    n = eigvals.shape[-1]
    eye_mask = np.eye(n, dtype=bool)[(None,)* (mult_mask.ndim-1) + (slice(None), slice(None))]
    has_mult = np.any(mult_mask[..., None] & ~eye_mask, axis=(-2, -1))  # (...,)

    # Only loop over the (usually rare) degenerate cases
    if np.any(has_mult):
        it = np.ndindex(has_mult.shape)
        for idx in it:
            if has_mult[idx]:
                V = eigvecs[idx][..., mult_mask[idx]]   # (n, m)
                G[idx] = (V @ V.T) / V.shape[1]
    return lam_min, G


# class to hold phase properties
class phase_props_holder:
    def __init__(self, k1, k2, mu1, mu2, sig1, sig2, f):
        self.k1 = k1
        self.k2 = k2
        self.mu1 = mu1
        self.mu2 = mu2
        self.sig1 = sig1
        self.sig2 = sig2
        self.f = f

        self._hashin_shtrikman()
        self._berryman_milton()
        self._gibianksy_torquato()

    def _hashin_shtrikman(self):
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
        self.k_lower_HS = 1/(self.f / (k_min_star + self.k1) + (1-self.f) / (k_min_star + self.k2)) - k_min_star
        self.k_upper_HS = 1/(self.f / (k_max_star + self.k1) + (1-self.f) / (k_max_star + self.k2)) - k_max_star
        self.mu_lower_HS = 1/(self.f / (mu_min_star + self.mu1) + (1-self.f) / (mu_min_star + self.mu2)) - mu_min_star
        self.mu_upper_HS = 1/(self.f / (mu_max_star + self.mu1) + (1-self.f) / (mu_max_star + self.mu2)) - mu_max_star

        # Conductivity
        sig_min = min(self.sig1, self.sig2)
        sig_max = max(self.sig1, self.sig2)
        self.sig_upper_HS = self.f*self.sig1 + (1-self.f)*self.sig2 - self.f*(1-self.f)*(self.sig2-self.sig1)**2/(self.f*self.sig2 + (1-self.f)*self.sig1 + 2*sig_max)
        self.sig_lower_HS = self.f*self.sig1 + (1-self.f)*self.sig2 - self.f*(1-self.f)*(self.sig2-self.sig1)**2/(self.f*self.sig2 + (1-self.f)*self.sig1 + 2*sig_min)

    def _berryman_milton(self):
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
        self.sig_BM1 = sig_upper(self.sig1, self.sig2, self.f, eta)
        self.sig_BM2 = sig_lower(self.sig1, self.sig2, self.f, eta)
        if self.sig1 >= self.sig2:
            self.mu_BM1 = mu_upper(self.mu1, self.mu2, self.k1, self.k2, self.f, eta)
            self.mu_BM2 = mu_lower(self.mu1, self.mu2, self.k1, self.k2, self.f, eta)
        else:
            self.mu_BM1 = mu_upper(self.mu1, self.mu2, self.k1, self.k2, self.f, 1-eta)
            self.mu_BM2 = mu_lower(self.mu1, self.mu2, self.k1, self.k2, self.f, 1-eta)
        self.sig_BM3 = sig_lower(self.sig1, self.sig2, self.f, eta)
        self.sig_BM4 = sig_upper(self.sig1, self.sig2, self.f, eta)
        if self.sig1 >= self.sig2:
            self.mu_BM3 = mu_upper(self.mu1, self.mu2, self.k1, self.k2, self.f, eta)
            self.mu_BM4 = mu_lower(self.mu1, self.mu2, self.k1, self.k2, self.f, eta)
        else:
            self.mu_BM3 = mu_upper(self.mu1, self.mu2, self.k1, self.k2, self.f, 1-eta)
            self.mu_BM4 = mu_lower(self.mu1, self.mu2, self.k1, self.k2, self.f, 1-eta)

    def _gibianksy_torquato(self):
        """
        Compute bulk-conductivity bound from Gibiansky-Torquato 1996.
        """
        self.f1 = self.f
        self.f2 = 1 - self.f
        def F(self, d1, d2, y):
            return self.f1*d1 + self.f2*d2 - self.f1*self.f2*(d1-d2)**2 / (self.f2*d1+self.f1*d2+y)
        def sig1_star(self):
            return F(self, self.sig1, self.sig2, 2*self.sig1)
        def sig2_star(self):
            return F(self, self.sig1, self.sig2, 2*self.sig2)
        def sig1_sharp(self):
            return F(self, self.sig1, self.sig2, -2*self.sig1)
        def sig2_sharp(self):
            return F(self, self.sig1, self.sig2, -2*self.sig2)
        def k1_star(self):
            return F(self, self.k1, self.k2, 4/3*self.mu1)
        def k2_star(self):
            return F(self, self.k1, self.k2, 4/3*self.mu2)
        def sig_a(self):
            return self.f1*self.sig1 + self.f2*self.sig2
        def sig_h(self):
            return 1/(self.f1/self.sig1 + self.f2/self.sig2)
        def k_a(self):
            return self.f1*self.k1 + self.f2*self.k2
        def k_h(self):
            return 1/(self.f1/self.k1 + self.f2/self.k2)
        def Hyp(p1:tuple, p2:tuple, p3:tuple):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            gam = np.linspace(0,1,10001)
            x_star = gam * x1 + (1-gam) * x2 - gam * (1-gam) * (x1-x2)**2 / ((1-gam)*x1 + gam*x2 - x3)
            y_star = gam * y1 + (1-gam) * y2 - gam * (1-gam) * (y1-y2)**2 / ((1-gam)*y1 + gam*y2 - y3)
            return x_star, y_star
        Hyp_1 = Hyp((sig1_star(self), k1_star(self)), (sig2_star(self), k2_star(self)), (self.sig1, self.k1))
        Hyp_2 = Hyp((sig1_star(self), k1_star(self)), (sig2_star(self), k2_star(self)), (self.sig2, self.k2))
        Hyp_3 = Hyp((sig1_star(self), k1_star(self)), (sig2_star(self), k2_star(self)), (sig1_sharp(self), k_h(self)))
        Hyp_4 = Hyp((sig1_star(self), k1_star(self)), (sig2_star(self), k2_star(self)), (sig2_sharp(self), k_h(self)))
        Hyp_5 = Hyp((sig1_star(self), k1_star(self)), (sig2_star(self), k2_star(self)), (sig_a(self), k_a(self)))

        self.sig_GT1 = Hyp_1[0]
        self.k_GT1 = Hyp_1[1]
        self.sig_GT2 = Hyp_2[0]
        self.k_GT2 = Hyp_2[1]
        self.sig_GT3 = Hyp_3[0]
        self.k_GT3 = Hyp_3[1]
        self.sig_GT4 = Hyp_4[0]
        self.k_GT4 = Hyp_4[1]
        self.sig_GT5 = Hyp_5[0]
        self.k_GT5 = Hyp_5[1]