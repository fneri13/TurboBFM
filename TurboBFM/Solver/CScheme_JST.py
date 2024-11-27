import numpy as np
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.euler_functions import GetPrimitivesFromConservatives, EulerFluxFromConservatives


class CSchemeJST():
    """
    Class for the evaluation of JST 3d euler flux, Formulation taken from 
    `The Origins and Further Development of the Jameson-Schmidt-Turkel (JST) Scheme`, by Jameson.
    """
    def __init__(self, fluid: FluidIdeal, Ull: np.ndarray, Ul: np.ndarray, Ur: np.ndarray, Urr: np.ndarray, S: np.ndarray):
        """
        The left to right orientation follows the orientation of the normal

        Parameters
        --------------------------

        `fluid`: fluid object

        `Ull`: left left point

        `Ul`: left point

        `Ur`: right point

        `Urr`: right right point

        `S`: surface vector between the left and right point
        """
        self.fluid = fluid
        self.Ull = Ull
        self.Ul = Ul
        self.Ur = Ur
        self.Urr = Urr
        self.S = S
        self.area = np.linalg.norm(S)
        self.S_dir = S/np.linalg.norm(S)
        self.U_avg = 0.5*(self.Ul+self.Ur)
    
    def ComputePrimitives(self):
        Wll = GetPrimitivesFromConservatives(self.Ull)
        Wl = GetPrimitivesFromConservatives(self.Ul)
        Wr = GetPrimitivesFromConservatives(self.Ur)
        Wrr = GetPrimitivesFromConservatives(self.Urr)
        return Wll, Wl, Wr, Wrr
    
    def ComputeFlux(self):
        kappa2 = 1
        kappa4 = 1/32
        c4 = 2

        Wll, Wl, Wr, Wrr = self.ComputePrimitives()

        r_factors = np.array([self.Compute_r(Wl), self.Compute_r(Wr)])
        s_factors = np.array([self.Compute_s(Wll, Wl, Wr), self.Compute_s(Wl, Wr, Wrr)])
        
        r = np.max(r_factors)
        s = np.max(s_factors)
        psi2 = kappa2*s*r
        psi4 = np.max(np.array([0, kappa4*r-c4*psi2]))

        flux_density = EulerFluxFromConservatives(self.U_avg, self.S, self.fluid)
        dissipation = psi2*(Wr-Wl)-psi4*((Wrr-Wr)-2*(Wr-Wl)+(Wl-Wll))
        flux_density -= dissipation 

        return flux_density*self.area
    
    def Compute_r(self, prim):
        """
        r coefficients
        """
        u_mag = np.linalg.norm(prim[1:-1])
        a = self.fluid.ComputeSoundSpeed_rho_u_et(prim[0], prim[1:-1], prim[4])
        return np.abs(u_mag)+a
    
    def Compute_s(self, prim1, prim2, prim3):
        """
        pressure sensors
        """
        p1 = self.fluid.ComputePressure_rho_u_et(prim1[0], prim1[1:-1], prim1[4])
        p2 = self.fluid.ComputePressure_rho_u_et(prim2[0], prim2[1:-1], prim2[4])
        p3 = self.fluid.ComputePressure_rho_u_et(prim3[0], prim3[1:-1], prim3[4])
        s = np.abs((p1-2*p2+p3)/(p1+2*p2+p3))
        return s
    

        