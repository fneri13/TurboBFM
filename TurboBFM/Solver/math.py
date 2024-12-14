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


def GetProjectedVector(u, n):
        """
        From an initial vector `u`, get back the vector directed along `n`. 
        WARNING: `n` is assumed to be of magnitude 1. No checks are done.
        """
        un = np.dot(u,n)*n
        return un


def GetTangentialVector(u, n):
        """
        From an initial vector `u`, get back the vector obtained by removing the aligned component of it along `n`. 
        WARNING: `n` is assumed to be of magnitude 1. No checks are done.
        """
        un = GetProjectedVector(u, n)
        ut = u-un
        return ut