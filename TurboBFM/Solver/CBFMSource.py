import numpy as np
from TurboBFM.Solver.math import ComputeCylindricalVectorFromCartesian, ComputeCartesianVectorFromCylindrical

class CBFMSource():
    """
    BFM source Class, where all the source terms are computed
    """
    def __init__(self, config):
        """
        Initialize the BFM class with config file info
        """
        self.config = config
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
    

    def ComputeForceSource(self, P: tuple, rho: float, u: np.ndarray, p: float, ht: float, Vol: float, b: float, normal: np.ndarray, omega: float):
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

        `b`: blockage of the cell

        `normal`: normal vector of the camber

        `omega`: rotational speed [rad/s] of the cell
        """
        x, y, z = P
        r = np.sqrt(y**2 + z**2)
        theta = np.arctan2(z, y)
        u_cart = u  # cartesian absolute velocity
        u_cyl = ComputeCylindricalVectorFromCartesian(x, y, z, u)

        drag_velocity_cyl = np.array([0, 0, omega*r])
        w_cyl = u_cyl-drag_velocity_cyl
        pitch = 2*np.pi*r/self.config.GetBladesNumber()
        if self.model.lower()=='hall-thollet':
            force = self.ComputeHallTholletForceDensity()
        elif self.model.lower()=='hall':
            force = self.ComputeHallForceDensity(w_cyl, normal, pitch, omega, P)
        else:
            raise ValueError('BFM Model <%s> is not supported' %self.model)

        return force*Vol

    def ComputeHallForceDensity(self, w, normal, pitch, omega, P):
        """
        Compute the force terms (per unit volume) related to the original Hall model based on lift-drag airfoil analogy.

        Parameters
        -------------------------

        `w`: relative velocity vector in cylindrical frame (axial, radial, tangential)

        `normal`: normal of the camber in cylindrical frame (axial, radial, tangential)

        """
        x, y, z = P
        r = np.sqrt(y**2+z**2)
        delta = self.ComputeDeviationAngle(w, normal)
        fn = 0.5 * np.linalg.norm(w)**2 * 2 * np.pi * delta / pitch / np.abs(normal[2])
        fn_versor = self.ComputeInviscidForceDirection(w, normal)

        if omega*fn_versor[2]<0:
            fn_versor *= -1  # the fn must push in the rotation direction
        else:
            pass

        fn_cyl = fn*fn_versor
        fn_cart = ComputeCartesianVectorFromCylindrical(x, y, z, fn_cyl)
        
        source = np.zeros(5)
        source[1] = fn_cart[0]
        source[2] = fn_cart[1]
        source[3] = fn_cart[2]
        source[4] = fn_cyl[2]*omega*r
        return source
    

    def ComputeDeviationAngle(self, w, n):
        """
        Compute the absolute value of the deviation angle between the velocity vector and the camber surface defined by n
        """
        wn = np.dot(w, n)
        delta = np.arcsin(wn/np.linalg.norm(w))
        return np.abs(delta)
    

    def ComputeInviscidForceDirection(self, w, n):
        """
        Compute the direction of the inviscid force component. It must be orthogonal to the relative velocity, and the radial component is found
        using the blade lean angle.

        FOR NOW IS VALID ONLY FOR AXIAL MACHINES WITH ZERO LEAN
        """
        w_dir = w/np.linalg.norm(w)
        fn_dir = np.array([-w_dir[2], 0, w_dir[0]])
        return fn_dir



        
    

    