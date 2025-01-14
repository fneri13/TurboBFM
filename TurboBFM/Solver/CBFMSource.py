import numpy as np
from TurboBFM.Solver.CSolver import CSolver
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.math import ComputeCylindricalVectorFromCartesian, ComputeCartesianVectorFromCylindrical
from TurboBFM.Solver.euler_functions import GetPrimitivesFromConservatives

class CBFMSource():
    """
    BFM source Class, where all the source terms are computed
    """
    def __init__(self, config, solver):
        """
        Initialize the BFM class with config file info
        """
        self.config = config
        self.solver = solver
        self.blockageActive = config.GetBlockageActive()
        self.model = config.GetBFMModel()
    

    def ComputeBlockageSource(self, i, j, k) -> np.ndarray:
        """
        Use the formulation of Magrini (pressure based) to compute the source terms related to the blockage.

        Parameters
        -------------------------

        `i`: i-index of cell element

        `j`: j-index of cell element

        `k`: k-index of cell element
        """
        if self.blockageActive:
            W = GetPrimitivesFromConservatives(self.solver.solution[i,j,k,:])
            rho = W[0]
            u = W[1:-1]
            et = W[-1]
            ht = self.solver.fluid.ComputeTotalEnthalpy_rho_u_et(rho, u, et)
            p = self.solver.fluid.ComputePressure_rho_u_et(rho, u, et)
            
            b = self.solver.mesh.blockage[i,j,k]
            bgrad = self.solver.mesh.blockage_gradient[i,j,k,:]
            Vol = self.solver.mesh.V[i,j,k]

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
    

    def ComputeForceSource(self, i, j, k):
        """
        For a certain BFM model, compute the fource source density terms as a np.ndarray of size 5 (continuity, momentum, tot. energy)
        
        Parameters
        -------------------------

        `i`: i-index of the cell

        `j`: j-index of the cell

        `k`: k-index of the cell
        """
        Vol = self.solver.mesh.V[i,j,k]
        if self.model.lower()=='hall-thollet':
            source_inviscid, source_viscous = self.ComputeHallTholletForceDensity(i, j, k)
        elif self.model.lower()=='hall':
            source_inviscid, source_viscous = self.ComputeHallForceDensity(i, j, k)
        elif self.model.lower()=='none':
            source_inviscid, source_viscous =  np.zeros(5), np.zeros(5)
        else:
            raise ValueError('BFM Model <%s> is not supported' %self.model)

        return source_inviscid*Vol, source_viscous*Vol

    def ComputeHallForceDensity(self, i, j, k):
        """
        Compute the force terms (per unit volume) related to the original Hall model based on lift-drag airfoil analogy.

        Parameters
        -------------------------

        `i`: i-index of the cell

        `j`: j-index of the cell

        `k`: k-index of the cell
        """
        x, y, z = self.solver.mesh.X[i,j,k], self.solver.mesh.Y[i,j,k], self.solver.mesh.Z[i,j,k]
        r = np.sqrt(y**2+z**2)
        theta = np.arctan2(z, y)
        conservative = self.solver.solution[i,j,k,:]  # conservative vector
        primitive = GetPrimitivesFromConservatives(conservative)
        u_cart = primitive[1:-1]  # cartesian absolute velocity
        u_cyl = ComputeCylindricalVectorFromCartesian(x, y, z, u_cart)
        omega = self.solver.mesh.omega[i,j,k]
        drag_velocity_cyl = np.array([0, 0, omega*r])
        w_cyl = u_cyl-drag_velocity_cyl
        pitch = 2*np.pi*r/self.config.GetBladesNumber()
        normal = self.solver.mesh.normal_camber_cyl[i,j,k,:]
        delta = self.ComputeDeviationAngle(w_cyl, normal)
        fn = 0.5 * np.linalg.norm(w_cyl)**2 * 2 * np.pi * delta / pitch / np.abs(normal[2])
        fn_versor = self.ComputeInviscidForceDirection(w_cyl, normal)

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
        return source, np.zeros(5)
    

    def ComputeHallTholletForceDensity(self, i, j, k):
        """
        Compute the force terms (per unit volume) related to the modified Hall model by Thollet based on lift-drag airfoil analogy.

        Parameters
        -------------------------

        `i`: i-index of the cell

        `j`: j-index of the cell

        `k`: k-index of the cell
        """
        x, y, z = self.solver.mesh.X[i,j,k], self.solver.mesh.Y[i,j,k], self.solver.mesh.Z[i,j,k]
        r = np.sqrt(y**2+z**2)
        theta = np.arctan2(z, y)
        conservative = self.solver.solution[i,j,k,:]  
        primitive = GetPrimitivesFromConservatives(conservative)
        u_cart = primitive[1:-1]  
        u_cyl = ComputeCylindricalVectorFromCartesian(x, y, z, u_cart)
        omega = self.solver.mesh.omega[i,j,k]
        drag_velocity_cyl = np.array([0, 0, omega*r])
        w_cyl = u_cyl-drag_velocity_cyl
        pitch = 2*np.pi*r/self.config.GetBladesNumber()
        normal = self.solver.mesh.normal_camber_cyl[i,j,k,:]
        delta = self.ComputeDeviationAngle(w_cyl, normal)
        b = self.solver.mesh.blockage[i,j,k]
        Kmach = self.ComputeCompressibilityCorrection(w_cyl, primitive)
        
        # inviscid component
        fn = Kmach * np.linalg.norm(w_cyl)**2 * np.pi * np.abs(delta) / pitch / np.abs(normal[2]) / b
        fn_versor = self.ComputeInviscidForceDirection(w_cyl, normal)
        if omega*fn_versor[2]<0:
            fn_versor *= -1  # the fn must push in the rotation direction
        else:
            pass
        fn_cyl = fn*fn_versor
        fn_cart = ComputeCartesianVectorFromCylindrical(x, y, z, fn_cyl)

        # viscous component
        rho = primitive[0]
        nu = self.config.GetKinematicViscosity()
        Rex = np.linalg.norm(w_cyl) * self.solver.mesh.stwl[i,j,k] / nu
        Cf = 0.0592*Rex**(-0.2)
        delta0 = delta  # this could be calibrated later, from the deviation of the float at peak efficiency
        fp = np.linalg.norm(w_cyl)**2/(pitch*b*np.abs(normal[2])) * (2*Cf + 2*np.pi*Kmach*(delta-delta0))
        fp_vers_cyl = w_cyl/np.linalg.norm(w_cyl)
        fp_cyl = fp*fp_vers_cyl*(-1)  # opposite to the relative velocity
        fp_cart = ComputeCartesianVectorFromCylindrical(x, y, z, fp_cyl)

        source_inviscid = np.zeros(5)
        source_inviscid[1] = fn_cart[0]
        source_inviscid[2] = fn_cart[1]
        source_inviscid[3] = fn_cart[2]
        source_inviscid[4] = (fn_cyl[2])*omega*r

        source_viscous = np.zeros(5)
        source_viscous[1] = fp_cart[0]
        source_viscous[2] = fp_cart[1]
        source_viscous[3] = fp_cart[2]
        source_viscous[4] = (fp_cyl[2])*omega*r
        return source_inviscid, source_viscous
    

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


    def ComputeCompressibilityCorrection(self, w, primitive):
        """
        Compute the compressibility correction factor for the thollet-hall model.

        Paramters
        --------------------------

        `w`: relative velocity vector

        `primitive`: primitive vector
        """
        wmag = np.linalg.norm(w)
        a = self.solver.fluid.ComputeSoundSpeed_rho_u_et(primitive[0], primitive[1:-1], primitive[-1])
        Mrel = wmag/a
        
        if Mrel==1:
            Mrel=0.999

        if Mrel<1:
            kprime = 1/np.sqrt(1-Mrel**2)
        elif Mrel>1:
            kprime = 2/np.pi/np.sqrt(Mrel**2-1)
        else:
            raise ValueError('M relative out of limitis')
        
        if kprime<=3:
            kmach = kprime
        else:
            kmach = 3
        
        return kmach





        
    

    