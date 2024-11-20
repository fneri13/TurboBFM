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

class FluidReal():
    """
    Real Fluid Class, where thermodynamic properties and transformations are taken from coolprop
    """
    def __init__(self, fluid_name):
        self.fluid_name = fluid_name
    
    def ComputeStaticEnergy_p_rho(self, p, rho):
        e = CP.PropsSI('U', 'P', p, 'D', rho, self.fluid_name)
        return e
    
    def ComputePressure_rho_e(self, rho, e):
        p = CP.PropsSI('P', 'D', rho, 'U', e, self.fluid_name)
        return p
    
    def ComputeSoundSpeed_p_rho(self, p, rho):
        try:
            a = CP.PropsSI("A", "P", p, "D", rho, self.fluid_name)
            return a
        except:
            # two phase region (or close) 
            T = self.ComputeTemperature_p_rho(p, rho)
            try:
                Q = CP.PropsSI("Q", "T", T, "P", p, self.fluid_name)
            except:
                # if the state is very close to saturation line it fails to find the quality -> set artifically to 1
                Q = 1

            # Speed of sound in liquid and vapor phases at the given T and P
            a_liquid = CP.PropsSI("A", "T", T, "Q", 0, self.fluid_name)  # sound speed for liquid phase
            a_vapor = CP.PropsSI("A", "T", T, "Q", 1, self.fluid_name)   # sound speed for vapor phase

            # Calculate weighted speed of sound based on quality
            a = (1 - Q) * a_liquid + Q * a_vapor
            return a
    
    def ComputeTemperature_p_rho(self, p, rho):
        T = CP.PropsSI('T', 'P', p, 'D', rho, self.fluid_name)
        return T