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
            if len(vel.shape)==1: # it is a velocity array for one point
                vel = np.linalg.norm(vel)
            elif len(vel.shape)==2: # it is an array of velocity magnitudes already:
                pass
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
    

    def ComputeTotalPressure_rho_u_et(self, rho, u, et):
        p = self.ComputePressure_rho_u_et(rho, u, et)
        M = self.ComputeMachNumber_rho_u_et(rho, u, et)
        pt = p*(1+(self.gmma-1)/2*M**2)**(self.gmma/(self.gmma-1))
        return pt
    

    def ComputeTotalTemperature_rho_u_et(self, rho, u, et):
        T = self.ComputeTemperature_rho_u_et(rho, u, et)
        M = self.ComputeMachNumber_rho_u_et(rho, u, et)
        Tt = T*(1+(self.gmma-1)/2*M**2)
        return Tt
    

    def ComputeTemperature_rho_u_et(self, rho, u, et):
        p = self.ComputePressure_rho_u_et(rho, u, et)
        T = p/rho/self.R
        return T
    
    
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
    

    def ComputeMachNumber_rho_u_et(self, rho, u, et):
        a = self.ComputeSoundSpeed_rho_u_et(rho, u, et)
        umag = np.linalg.norm(u)
        M = umag/a
        return M


    def ComputeMachNumber_rho_umag_et(self, rho, umag, et):
        a = self.ComputeSoundSpeed_rho_u_et(rho, umag, et)
        M = umag/a
        return M
    
    def ComputeTotalPressure_p_M(self, pressure, mach):
        totalPressure = pressure*(1+(self.gmma-1)/2*mach**2)**(self.gmma/(self.gmma-1))
        return totalPressure
    
    def ComputeTotalTemperature_T_M(self, temperature, mach):
        totalTemp = temperature*(1+(self.gmma-1)/2*mach**2)
        return totalTemp
    
    def ComputeEntropy_p_rho(self, pressure, density):
        entropy = pressure/(density**self.gmma)
        return entropy


