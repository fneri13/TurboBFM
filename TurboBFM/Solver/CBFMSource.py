import numpy as np

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
        Use the formulation of Magrini to compute the source terms related to the blockage
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



        
    

    