import numpy as np
import torch
from itertools import product

class SEED:
    def __init__(self, gridShape):
        """
        Initialize real-space grid and precompute trig functions.
        gridShape : number of grid points per dim
        """
        r = self.make_grid(gridShape)
        self.x, self.y, self.z = r[...,0], r[...,1], r[...,2]
        # All trig functions up to 3rd harmonic
        self.Cx = self.cosine(self.x, 1)
        self.Cy = self.cosine(self.y, 1)
        self.Cz = self.cosine(self.z, 1)
        self.C2x = self.cosine(self.x, 2)
        self.C2y = self.cosine(self.y, 2)
        self.C2z = self.cosine(self.z, 2)
        self.C3x = self.cosine(self.x, 3)
        self.C3y = self.cosine(self.y, 3)
        self.C3z = self.cosine(self.z, 3)
        self.Sx = self.sine(self.x, 1)
        self.Sy = self.sine(self.y, 1)
        self.Sz = self.sine(self.z, 1)
        self.S2x = self.sine(self.x, 2)
        self.S2y = self.sine(self.y, 2)
        self.S2z = self.sine(self.z, 2)
        self.S3x = self.sine(self.x, 3)
        self.S3y = self.sine(self.y, 3)
        self.S3z = self.sine(self.z, 3)

    def make_grid(self, gridShape):
        """
        Build real-space grid r with shape (N, N, N, 3) over [0,L)^3.
        """
        x = np.linspace(0, 1, gridShape[0], endpoint=False)
        y = np.linspace(0, 1, gridShape[1], endpoint=False)
        z = np.linspace(0, 1, gridShape[2], endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return np.stack((X, Y, Z), axis=-1)  # (*gridShape,3)

    def make_seed(self, morphology, invert=False):
        """
        Build flattened torch.Tensor initial guess for topology optimization.
        
        morphology: 'Gyroid', 'P', 'D', etc
        invert: whether to invert the field (1-f)
        """
        # choose field generator
        if morphology == 'Gyroid':
            f = self.Gyroid()
        elif morphology == 'D':
            f = self.D()
        elif morphology == 'P':
            f = self.P()
        elif morphology == 'IWP':
            f = self.IWP()
        elif morphology == 'Neovius':
            f = self.Neovius()
        elif morphology == 'CY':
            f = self.CY()
        elif morphology == 'Lidinoid':
            f = self.Lidinoid()
        elif morphology == 'OCTO':
            f = self.OCTO()
        elif morphology == 'FRD':
            f = self.FRD()
        elif morphology == 'S':
            f = self.S()
        elif morphology == 'CD':
            f = self.CD()
        elif morphology == "G'":
            f = self.G_prime()
        elif morphology == "G'2":
            f = self.G_prime2()
        elif morphology == "D'":
            f = self.D_prime()
        elif morphology == 'K':
            f = self.K()
        elif morphology == 'CS':
            f = self.CS()
        elif morphology == 'Y':
            f = self.Y()
        elif morphology == 'pmY':
            f = self.pmY()
        elif morphology == 'CpmY':
            f = self.CpmY()
        elif morphology == 'C_I2_Y':
            f = self.C_I2_Y()
        elif morphology == 'W':
            f = self.W()
        elif morphology == 'Q':
            f = self.Q()
        elif morphology == 'CG':
            f = self.CG()
        elif morphology == 'Slotted_P':
            f = self.Slotted_P()
        else:
            raise ValueError(f"Unknown morphology '{morphology}'")

        f = f - np.min(f) # make non-negative
        f = f / np.max(f) # normalize to [0,1]
        if invert:
            f = 1-f # invert
        
        return torch.tensor(f, dtype=torch.float64)
    
    def cosine(self, x, n):
        return np.cos(2*np.pi*x*n) 

    def sine(self, x, n):
        return np.sin(2*np.pi*x*n)

    def cosine_specific(self, x, c_x, y, c_y, z, c_z):
        return np.cos(2*np.pi*(c_x*x + c_y*y + c_z*z))
    
    def sine_specific(self, x, c_x, y, c_y, z, c_z):
        return np.sin(2*np.pi*(c_x*x + c_y*y + c_z*z))

    def Gyroid(self):
        """
        Gyroid analytic form.
        f = CxSy + CySz + CzSx
        """
        return (
            self.Cx*self.Sy + 
            self.Cy*self.Sz +
            self.Cz*self.Sx
        )
    
    def D(self):
        """
        Diamond analytic form.
        f = SxSySz + SxCyCz + CxSyCz + CxCySz
        """
        return (
            self.Sx*self.Sy*self.Sz +
            self.Sx*self.Cy*self.Cz +
            self.Cx*self.Sy*self.Cz +
            self.Cx*self.Cy*self.Sz
        )
    
    def P(self):
        """
        Primitive analytic form.
        f = Cx + Cy + Cz
        """
        return (
            self.Cx +
            self.Cy +
            self.Cz
        )
    
    def IWP(self):
        """
        IWP analytic form.
        f = 2(CxCy + CyCz + CzCx) - (C2x + C2y + C2z)
        """
        return (
            2*(self.Cx*self.Cy + self.Cy*self.Cz + self.Cz*self.Cx) -
            (self.C2x + self.C2y + self.C2z)
        )

    def Neovius(self):
        """
        Neovius analytic form.
        f = 3(Cx + Cy + Cz) + 4CxCyCz
        """
        return (
            3*(self.Cx + self.Cy + self.Cz) +
            4*self.Cx*self.Cy*self.Cz
        )
    
    def CY(self):
        """
        C(Y) analytic form.
        f = -SxSySz + S2xSy + S2ySz + SxS2z - CxCyCz + S2xCz + CxS2y + CyS2z
        """
        return (
            -self.Sx*self.Sy*self.Sz +
            self.S2x*self.Sy +
            self.S2y*self.Sz +
            self.Sx*self.S2z -
            self.Cx*self.Cy*self.Cz +
            self.S2x*self.Cz +
            self.Cx*self.S2y +
            self.Cy*self.S2z
        )
    
    def Lidinoid(self):
        """
        Lidinoid analytic form.
        f = (S2xCySz + S2yCzSx + S2zCxSy) - (C2xC2y + C2yC2z + C2zC2x) + 0.3
        """
        return (
            (self.S2x*self.Cy*self.Sz +
             self.S2y*self.Cz*self.Sx +
             self.S2z*self.Cx*self.Sy) -
            (self.C2x*self.C2y +
             self.C2y*self.C2z +
             self.C2z*self.C2x) + 0.3
        )
    
    def OCTO(self):
        """
        OCTO analytical form.
        f = 0.6(CxCy + CyCz + CzCx) - 0.4(Cx + Cy + Cz) + 0.25
        """
        return (
            0.6*(self.Cx*self.Cy + self.Cy*self.Cz + self.Cz*self.Cx) -
            0.4*(self.Cx + self.Cy + self.Cz) + 0.25
        )

    def FRD(self):
        """
        FRD analytical form.
        f = 8CxCyCz + C2xC2yC2z - (C2xC2y + C2yC2z + C2zC2x)
        """
        return (
            8*self.Cx*self.Cy*self.Cz +
            self.C2x*self.C2y*self.C2z -
            (self.C2x*self.C2y + self.C2y*self.C2z + self.C2z*self.C2x)
        )
    
    def S(self):
        """
        Fisher-Koch S analytical form.
        f = C2xSyCz + CxC2ySz + SxCyC2z
        """
        return (
            self.C2x*self.Sy*self.Cz +
            self.Cx*self.C2y*self.Sz +
            self.Sx*self.Cy*self.C2z
        )
    
    def PplusCP(self):
        """
        P+C(P) analytical form.
        f = 0.3(CxCyCz) + 0.2(Cx + Cy + Cz) + 0.1(C2xC2yC2z) +0.1(C2x + C2y + C2z) + 0.05(C3x + C3y + C3z) + 0.1(CxCy + CyCz + CzCx)
        """
        return (
            0.3*(self.Cx*self.Cy*self.Cz) +
            0.2*(self.Cx + self.Cy + self.Cz) +
            0.1*(self.C2x*self.C2y*self.C2z) +
            0.1*(self.C2x + self.C2y + self.C2z) +
            0.05*(self.C3x + self.C3y + self.C3z) +
            0.1*(self.Cx*self.Cy + self.Cy*self.Cz + self.Cz*self.Cx)
        )
    
    def Split_P(self):
        """
        Split-P analytical form.
        f = 1.1(S2xCySz + S2yCzSx + S2zCxSy) - 0.2(C2xC2y + C2yC2z + C2zC2x) - 0.4(C2x + C2y + C2z)
        """
        return (
            1.1*(self.S2x*self.Cy*self.Sz +
                    self.S2y*self.Cz*self.Sx +
                    self.S2z*self.Cx*self.Sy) -
            0.2*(self.C2x*self.C2y + self.C2y*self.C2z + self.C2z*self.C2x) -
            0.4*(self.C2x + self.C2y + self.C2z)
        )
    
    def F(self):
        """
        F analytical form.
        f = CxCyCz
        """
        return (
            self.Cx*self.Cy*self.Cz
        )
    
    def CD(self):
        """
        C(D) analytical form.
        f = C3xpyCz - S3xmySz + Cxp3yCz + Sxm3ySz + CxmyC3z - SxpyS3z
        """
        C3xpy = self.cosine_specific(self.x, 3, self.y, 1, self.z, 0)
        S3xmy = self.sine_specific(self.x, 3, self.y, -1, self.z, 0)
        Cxp3y = self.cosine_specific(self.x, 1, self.y, 3, self.z, 0)
        Sxm3y = self.sine_specific(self.x, -1, self.y, 3, self.z, 0)
        Cxmy = self.cosine_specific(self.x, 1, self.y, -1, self.z, 0)
        Sxpy = self.sine_specific(self.x, 1, self.y, 1, self.z, 0)
        return (
            C3xpy*self.Cz - S3xmy*self.Sz +
            Cxp3y*self.Cz + Sxm3y*self.Sz +
            Cxmy*self.C3z - Sxpy*self.S3z
        )
    
    def G_prime(self):
        """
        G' analytical form.
        f = S2xCySz + SxS2yCz + CxSyS2z + 0.32
        """
        return (
            self.S2x*self.Cy*self.Sz +
            self.Sx*self.S2y*self.Cz +
            self.Cx*self.Sy*self.S2z + 0.32
        )
    
    def G_prime2(self):
        """
        G'2 analytical form.
        f = 5(S2xCySz + SxS2yCz + CxSyS2z) + (C2xC2y + C2yC2z + C2zC2x)
        """
        return (
            5*(self.S2x*self.Cy*self.Sz +
               self.Sx*self.S2y*self.Cz +
               self.Cx*self.Sy*self.S2z) +
            (self.C2x*self.C2y +
             self.C2y*self.C2z +
             self.C2z*self.C2x)
        )
    
    def D_prime(self):
        """
        D' analytical form.
        f = 0.5(CxCyCz + CxSySz + SxCySz + SxSyCz) - 0.5(S2xS2y + S2yS2z + S2zS2x) - 0.2
        """
        return (
            0.5*(self.Cx*self.Cy*self.Cz +
                 self.Cx*self.Sy*self.Sz +
                 self.Sx*self.Cy*self.Sz +
                 self.Sx*self.Sy*self.Cz) -
            0.5*(self.S2x*self.S2y +
                 self.S2y*self.S2z +
                 self.S2z*self.S2x) - 0.2
        )
    
    def K(self):
        """
        Karcher-K analytical form.
        f = 0.3(Cx + Cy + Cz) + 0.3(CxCy + CyCz + CzCx) - 0.4(C2x + C2y + C2z) + 0.2
        """
        return (
            0.3*(self.Cx + self.Cy + self.Cz) +
            0.3*(self.Cx*self.Cy + self.Cy*self.Cz + self.Cz*self.Cx) -
            0.4*(self.C2x + self.C2y + self.C2z) + 0.2
        )
    
    def CS(self):
        """
        C(S) analytical form.
        f = (C2x + C2y + C2z) + 2(S3xS2yCz + CxS3yS2z + S2xCyS3z) + 2(S2xC3ySz + SxS2yC3z + C3xSyS2z)
        """
        return (
            (self.C2x + self.C2y + self.C2z) +
            2*(self.S3x*self.S2y*self.Cz +
               self.Cx*self.S3y*self.S2z +
               self.S2x*self.Cy*self.S3z) +
            2*(self.S2x*self.C3y*self.Sz +
               self.Sx*self.S2y*self.C3z +
               self.C3x*self.Sy*self.S2z)
        )
    
    def Y(self):
        """
        Fisher-Koch Y analytical form.
        f = CxCyCz + SxSySz + (S2xSy + S2ySz + SxS2z) + (CxS2y + CyS2z + S2xCz)
        """
        return (
            self.Cx*self.Cy*self.Cz +
            self.Sx*self.Sy*self.Sz +
            (self.S2x*self.Sy + self.S2y*self.Sz + self.Sx*self.S2z) +
            (self.Cx*self.S2y + self.Cy*self.S2z + self.S2x*self.Cz)
        )
    
    def pmY(self):
        """
        plus-minus Y analytical form.
        f = 2(CxCyCz) + (S2xSy + S2ySz + SxS2z)
        """
        return (
            2*(self.Cx*self.Cy*self.Cz) +
            (self.S2x*self.Sy + self.S2y*self.Sz + self.Sx*self.S2z)
        )
    
    def CpmY(self):
        """
        C(plus-minus Y) analytical form.
        f = -2(CxCyCz) + (S2xSy + S2ySz + SxS2z)
        """
        return (
            -2*(self.Cx*self.Cy*self.Cz) +
            (self.S2x*self.Sy + self.S2y*self.Sz + self.Sx*self.S2z)
        )
    
    def C_I2_Y(self):
        """
        C(I2-Y**) analytical form.
        f = 2(S2xCySz + SxS2yCz + CxSyS2z) + (C2xC2y + C2yC2z + C2xC2z)
        """
        return (
            2*(self.S2x*self.Cy*self.Sz +
               self.Sx*self.S2y*self.Cz +
               self.Cx*self.Sy*self.S2z) +
            (self.C2x*self.C2y +
             self.C2y*self.C2z +
             self.C2x*self.C2z)
        )
    
    def W(self):
        """
        W analytical form.
        f = (C2xCy + C2yCz + C2zCx) - (CxC2y + CyC2z + C2xCz)
        """
        return (
            (self.C2x*self.Cy + self.C2y*self.Cz + self.C2z*self.Cx) -
            (self.Cx*self.C2y + self.Cy*self.C2z + self.C2x*self.Cz)
        )
    
    def Q(self):
        """
        Q analytical form.
        f = (Cx - 2Cy)Cz - sqrt(3)Sz(Cxmy - Cx) + CxmyCz
        """
        Cxmy = self.cosine_specific(self.x, 1, self.y, -1, self.z, 0)
        return (
            (self.Cx - 2*self.Cy)*self.Cz -
            np.sqrt(3)*self.Sz*(Cxmy - self.Cx) +
            Cxmy*self.Cz
        )

    def CG(self):
        """
        C(G) analytical form.
        f = 3(SxCy + SyCz + CxSz) + 2(S3xCy + S3yCz + CxS3z) - 2(SxC3y + SyC3z + C3xSz)
        """
        return (
            3*(self.Sx*self.Cy + self.Sy*self.Cz + self.Cx*self.Sz) +
            2*(self.S3x*self.Cy + self.S3y*self.Cz + self.Cx*self.S3z) -
            2*(self.Sx*self.C3y + self.Sy*self.C3z + self.C3x*self.Sz)
        )
    
    def Slotted_P(self):
        """
        Slotted-P analytical form.
        f = -2(CxCy + CyCz + CzCx) - 2(C2x + C2y + C2z) + (C2xCy + C2yCz + C2zCx) - (CxC2y + CyC2z + C2xCz)
        """
        return (
            -2*(self.Cx*self.Cy + self.Cy*self.Cz + self.Cz*self.Cx) -
            2*(self.C2x + self.C2y + self.C2z) +
            (self.C2x*self.Cy + self.C2y*self.Cz + self.C2z*self.Cx) -
            (self.Cx*self.C2y + self.Cy*self.C2z + self.C2x*self.Cz)
        )