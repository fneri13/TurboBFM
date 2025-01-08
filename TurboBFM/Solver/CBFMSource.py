import numpy as np
from TurboBFM.Solver.math import ComputeCylindricalVectorFromCartesian

class CBFMSource():
    """
    BFM source Class, where all the source terms are computed
    """
    def __init__(self, config):
        """
        Initialize the BFM class with config file info
        """
        self.blockageActive = config.GetBlockageActive()
        self.model = config.GetBFMModel()
    

    def ComputeBlockageSource(self, b: float, bgrad: np.ndarray, rho: float, u: np.ndarray, p: float, ht: float, Vol: float) -> np.ndarray:
        """
        Use the formulation of Magrini (pressure based) to compute the source terms related to the blockage.

        Parameters
        -------------------------

        `b`: blockage value

        `bgrad`: blockage gradient array (x,y,z) ref frame

        `rho`: density

        `u`: velocity array (x,y,z) ref frame

        `p`: pressure

        `ht`: total enthalpy

        `Vol`: volume of the cell
        """
        if self.blockageActive:
            F = np.array([rho*u[0], 
                            rho*u[0]**2 + p,
                            rho*u[0]*u[1],
                            rho*u[0]*u[2],
                            ht*u[0]])
            
            G = np.array([rho*u[1], 
                            rho*u[0]*u[1],
                            rho*u[1]**2 + p,
                            rho*u[1]*u[2],
                            ht*u[1]])
            
            Sb = np.array([0,
                            p*bgrad[0],
                            p*bgrad[1],
                            0,
                            0])
            source = (-1/b*bgrad[0]*F -1/b*bgrad[1]*G + 1/b*Sb)*Vol
        else:
            source = np.zeros(5, dtype=float)
        
        return source
    

    def ComputeForceSource(self, P: tuple, rho: float, u: np.ndarray, p: float, ht: float, Vol: float):
        """
        For a certain BFM model, compute the fource source terms as a np.ndarray of size 5 (continuity, momentum, tot. energy)
        
        Parameters
        -------------------------

        `P`: tuple with the cell coordinates array (x,y,z) ref frame

        `rho`: density

        `u`: velocity array (x,y,z) ref frame

        `p`: pressure

        `ht`: total enthalpy

        `Vol`: volume of the cell
        """
        x, y, z = P
        r = np.sqrt(y**2 + z**2)
        theta = np.arctan2(z, y)
        u_cyl = ComputeCylindricalVectorFromCartesian(x, y, z, u)
        
        if self.model.lower()=='thollet':
            force = self.ComputeTholletForce()
        else:
            raise ValueError('BFM Model %s is not supported' %self.model)

        return force

    def ComputeTholletForce(self):
        """
        Compute the force terms related to the Thollet model based on lift-drag airfoil analogy
        """
        return np.zeros(5)



        
    

    