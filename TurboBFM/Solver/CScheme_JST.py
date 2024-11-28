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
        # coefficients for the scheme
        self.kappa2 = 1
        self.kappa4 = 1/32
        self.c4 = 2

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
    
    def ComputeFluxJameson(self):
        

        Wll, Wl, Wr, Wrr = self.ComputePrimitives()

        r_factors = np.array([self.Compute_r(Wl), self.Compute_r(Wr)])
        s_factors = np.array([self.Compute_s(Wll, Wl, Wr), self.Compute_s(Wl, Wr, Wrr)])
        
        r = np.max(r_factors)
        s = np.max(s_factors)
        psi2 = self.kappa2*s*r
        psi4 = np.max(np.array([0, self.kappa4*r-self.c4*psi2]))

        flux = EulerFluxFromConservatives(self.U_avg, self.S, self.fluid)
        dissipation = psi2*(self.Ur-self.Ul)-psi4*((self.Urr-self.Ur)-2*(self.Ur-self.Ul)+(self.Ul-self.Ull))
        flux -= dissipation 

        return flux
    
    def ComputeFluxBlazek(self):
        """
        Compute the flux (density) using the formulation given in the book by blazek
        """
        
        Wll, Wl, Wr, Wrr = self.ComputePrimitives()

        def compute_lambda(W):
            """
            Compute the scaling term given in the book, but divided the surface area. W is the primitive variable vector
            """
            u = W[1:-1]
            a = self.fluid.ComputeSoundSpeed_rho_u_et(W[0], W[1:-1], W[-1])
            un = np.dot(u, self.S_dir)
            lmbda = np.abs(un)+a
            return lmbda

        lambda_l = compute_lambda(Wl)
        lambda_r = compute_lambda(Wr)
        lambda_avg = 0.5*(lambda_l+lambda_r)

        def pressure_sensor(W1, W2, W3):
            p1 = self.fluid.ComputePressure_rho_u_et(W1[0], W1[1:-1], W1[-1])
            p2 = self.fluid.ComputePressure_rho_u_et(W2[0], W2[1:-1], W2[-1])
            p3 = self.fluid.ComputePressure_rho_u_et(W3[0], W3[1:-1], W3[-1])
            return np.abs(p3-2*p2+p1)/(p3+2*p2+p1)

        psi_2 = self.kappa2*np.maximum(pressure_sensor(Wll, Wl, Wr),
                                       pressure_sensor(Wl, Wr, Wrr))
        psi_4 = self.kappa2*np.maximum(0, 
                                       self.kappa4-psi_2)
        
        diss = lambda_avg*(psi_2*(self.Ur-self.Ul) - psi_4*(self.Urr-3*self.Ur+3*self.Ul-self.Ull))
        U_avg = 0.5*(self.Ul+self.Ur)
        flux = EulerFluxFromConservatives(U_avg, self.S, self.fluid)

        return flux-diss


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
    

        