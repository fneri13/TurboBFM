import numpy as np

class CScheme_Central():
    """
    Class for the evaluation of central flux, defined for scalar laplace equation.
    """
    def __init__(self, ul: np.ndarray, ur: np.ndarray, S: np.ndarray, diffusivity: float = 1.0) -> None:
        """
        The left to right orientation follows the orientation of the normal. 

        Parameters
        --------------------------

        `ul`: left point scalar value

        `ur`: right point scalar value

        `S`: surface vector oriented from left to right points

        `diffusivity`: term used to mimick thermal diffusivity for laplace solver. 1 as default
        """
        self.ul = ul
        self.ur = ur
        self.S = S
        self.S_dir = S/np.linalg.norm(S) 
        self.alpha = diffusivity   

    def ComputeFlux(self) -> float:
        """
        Compute the upwind flux
        """
        fl = self.alpha*np.dot(self.ul, self.S_dir)
        fr = self.alpha*np.dot(self.ur, self.S_dir)
        
        return 0.5*(fl+fr)
            


    
    
    

        