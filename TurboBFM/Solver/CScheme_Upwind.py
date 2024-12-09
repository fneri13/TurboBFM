import numpy as np


class CScheme_Upwind():
    """
    Class for the evaluation of upwind flux, defined for scalar transport equations.
    """
    def __init__(self, ul, ur, S, u_advection):
        """
        The left to right orientation follows the orientation of the normal. 

        Parameters
        --------------------------

        """
        self.ul = ul
        self.ur = ur
        self.S = S
        self.S_dir = S/np.linalg.norm(S)
        self.u_adv = u_advection
    
    def ComputeFlux(self):
        un = np.dot(self.S_dir, self.u_adv)
        if un>=0:
            flux = un*self.ul
        else:
            flux = un*self.ur
        
        return flux
            


    
    
    

        