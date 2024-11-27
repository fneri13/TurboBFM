import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt


class FluidIdeal():
    """
    Ideal Fluid Class, where thermodynamic properties and transformation are computed with ideal gas laws
    """
    def __init__(self, gmma):
        self.gmma = gmma
    
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
