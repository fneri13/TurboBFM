import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt


class FluidIdeal():
    """
    Ideal Fluid Class, where thermodynamic properties and transformation are computed with ideal gas laws
    """
    def __init__(self, gmma, R):
        """
        Initialize fluid element. 

        Parameters
        -----------------------

        `gmma`: cp/cv ratio

        `R`" gas constant [J/kgK]
        """
        self.gmma = gmma
        self.R = R
        self.cp = (gmma*R)/(gmma-1)
        self.cv = self.cp/R
    

    def ComputeStaticEnergy_p_rho(self, p, rho):
        return (p / (self.gmma - 1) / rho)
    

    def ComputePressure_rho_e(self, rho, e):
        return (self.gmma-1)*rho*e
    

    def ComputeSoundSpeed_p_rho(self, p, rho):
        return np.sqrt(self.gmma*p/rho)
    

    def ComputeStaticEnergy_u_et(self, vel, et):
        if isinstance(vel, np.ndarray):
            vel = np.linalg.norm(vel)
        else:
            pass
        e = et - 0.5*vel**2
        return e
    

    def ComputeSoundSpeed_rho_u_et(self, rho, u, et):
        e = self.ComputeStaticEnergy_u_et(u, et)
        p = self.ComputePressure_rho_e(rho, e)
        a = self.ComputeSoundSpeed_p_rho(p, rho)
        return a
    

    def ComputePressure_rho_u_et(self, rho, u, et):
        e = self.ComputeStaticEnergy_u_et(u, et)
        p = self.ComputePressure_rho_e(rho, e)
        return p
    
    
    def ComputeTotalEnthalpy_rho_u_et(self, rho, u, et):
        e = self.ComputeStaticEnergy_u_et(u, et)
        p = self.ComputePressure_rho_e(rho, e)
        ht = et+p/rho
        return ht
    

    def ComputeStaticPressure_pt_M(self, pt, M):
        p = pt*(1+(self.gmma-1)/2*M**2)**(-self.gmma/(self.gmma-1))
        return p 
    

    def ComputeStaticTemperature_rho_u_et(self, rho, u, et):
        p = self.ComputePressure_rho_u_et(rho, u, et)
        T = p/self.R/rho
        return T
    

    def ComputeStaticTemperature_Tt_M(self, Tt, M):
        T = Tt*(1+(self.gmma-1)/2*M**2)**(-1)
        return T
    

    def ComputeEntropy_rho_u_et(self, rho, u, et):
        p = self.ComputePressure_rho_u_et(rho, u, et)
        s = p/(rho**self.gmma)
        return s
    

    def ComputeDensity_p_T(self, p, T):
        rho = p/(self.R*T)
        return rho


