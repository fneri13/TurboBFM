import numpy as np


class CScheme_Upwind():
    """
    Class for the evaluation of upwind flux, defined for scalar transport equations.
    """
    def __init__(self, ul: float, ur: float, S: np.ndarray, u_advection: np.ndarray) -> None:
        """
        The left to right orientation follows the orientation of the normal. 

        Parameters
        --------------------------

        `ul`: left point scalar value

        `ur`: right point scalar value

        `S`: surface vector oriented from left to right points

        `u_advection`: advection velocity for the scalar transport
        """
        self.ul = ul
        self.ur = ur
        self.S = S
        self.S_dir = S/np.linalg.norm(S)
        self.u_adv = u_advection
    

    def ComputeFlux(self) -> float:
        """
        Compute the upwind flux
        """
        un = np.dot(self.S_dir, self.u_adv)
        if un>=0:
            flux = un*self.ul
        else:
            flux = un*self.ur
        
        return flux
            


    
    
    

        