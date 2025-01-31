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


def ComputeCartesianVectorFromCylindrical(x, y, z, u):
        """
        Get the vector in cartesian coordinates (x, y, z) starting from the one in cylindrical coords (axial, radial, tangential). 
        The convention used in the solver is that x is the axial coordinates.
        """
        r = np.sqrt(y**2 + z**2)
        theta = np.arctan2(z, y)

        u_car = np.zeros_like(u)
        u_car[0] = u[0] # axial component
        u_car[1] = cos(theta)*u[1]-sin(theta)*u[2]  # radial component
        u_car[2] = +sin(theta)*u[1]+cos(theta)*u[2]  # tangential component
        return u_car
        

def IntegrateVectorFlux(vec: np.ndarray, S: np.ndarray):
        """
        Integration of the vector field on the Surface, obtaning the integral flux of a vectorial quantity.

        Parameters:
        ----------------------------------

        `vec`: array of vectorial quantities defined on the surfaces centers (ni,nj,3)

        `S: array of surfaces vectors (ni,nj,3)
        """
        assert vec[:,:,0].shape[0] == S[:,:,0].shape[0]
        assert vec[:,:,0].shape[1] == S[:,:,0].shape[1]

        ni, nj = S[:,:,0].shape
        flux = 0
        for i in range(ni):
            for j in range(nj):
                tmp = np.dot(vec[i,j,:], S[i,j,:])
                flux += tmp
        return flux


def IntegrateScalarFlux(phi: np.ndarray, S: np.ndarray):
        """
        Integration of a scalar field on the Surface, obtaning the summation of value of the scalar quantity times the surface area.

        Parameters:
        ----------------------------------

        `phi`: array of vectorial quantities defined on the surfaces centers (ni,nj)

        `S: array of surfaces vectors (ni,nj,3)
        """
        assert phi[:,:].shape[0] == S[:,:,0].shape[0]
        assert phi[:,:].shape[1] == S[:,:,0].shape[1]

        ni, nj = S[:,:,0].shape
        sum = 0
        for i in range(ni):
            for j in range(nj):
                area = np.linalg.norm(S[i,j,:])
                tmp = phi[i,j]*area
                sum += tmp
        return sum

