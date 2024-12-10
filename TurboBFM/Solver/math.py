import numpy as np

def GreenGaussGradient(S, U, V):
        """
        For the 6 surfaces enclosing an element, compute the volume using green gauss theorem.

        Paramters
        --------------------------------

        `S`: tuple of 6 surface vectors surrounding the element

        `U`: value of the variable at the surface midpoints surrounding the element. Same order of `S`

        `V`: volume of the element
        """
        assert len(S)==6, 'The number of surfaces must be 6'
        assert len(U)==6, 'The number of surface midpoints must be 6'
                
        grad = np.zeros(3)
        for iFace in range(len(S)):
                grad += U[iFace]*S[iFace]
        grad /= V
        return grad