import numpy as np
from numpy import cos, sin

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


def rotate_xyz_vector_along_x(v, theta):
        """
        Rotate a vector in cartesian components (x,y,z) around the first axis by an angle theta [rad]
        """
        vnew = np.zeros_like(v)
        vnew[0] = v[0]
        vnew[1] = np.cos(theta)*v[1]-np.sin(theta)*v[2]
        vnew[2] = np.sin(theta)*v[1]+np.cos(theta)*v[2]
        return vnew


def ComputeCylindricalVectorFromCartesian(x, y, z, u):
        """
        Get the vector in cylindrical coordinates (z,r,theta) starting from the one in cartesian coords (x,y,z). 
        The convention used in the solver is that x is the axial coordinates.
        """
        r = np.sqrt(y**2 + z**2)
        theta = np.arctan2(z, y)

        u_cyl = np.zeros_like(u)
        u_cyl[0] = u[0] # axial component
        u_cyl[1] = cos(theta)*u[1]+sin(theta)*u[2]  # radial component
        u_cyl[2] = -sin(theta)*u[1]+cos(theta)*u[2]  # tangential component
        return u_cyl
        
        
