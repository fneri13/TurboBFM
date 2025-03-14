import numpy as np
import math
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
    

    def ComputeBlockageSource(self, i, j, k, block_source_type='thollet') -> np.ndarray:
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
            bgrad_cart = ComputeCartesianVectorFromCylindrical(self.solver.mesh.X[i,j,k],
                                                               self.solver.mesh.Y[i,j,k],
                                                               self.solver.mesh.Z[i,j,k],
                                                               bgrad)

            Vol = self.solver.mesh.V[i,j,k]

            if block_source_type.lower() == 'magrini':
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
                
                source = (-1/b*bgrad_cart[0]*F -1/b*bgrad_cart[1]*G + 1/b*Sb)
            
            elif block_source_type.lower()=='thollet':
                common_term = -1/b*rho*np.dot(u, bgrad_cart)
                source = np.zeros(5, dtype=float)
                source[0] = common_term
                source[1] = common_term*u[0]
                source[2] = common_term*u[1]
                source[3] = common_term*u[2]
                source[4] = common_term*ht

            else:
                raise ValueError('Unrecognized blockage source method')
        else:
            source = np.zeros(5, dtype=float)
        
        return source*Vol
    

    def ComputeForceSource(self, i, j, k):
        """
        For a certain BFM model, compute the fource source terms as a np.ndarray of size 5 (continuity, momentum, tot. energy).
        
        Parameters
        -------------------------

        `i`: i-index of the cell

        `j`: j-index of the cell

        `k`: k-index of the cell
        
        Return
        -------------------------

        `source`: np.ndarray of size 5 containing the source terms (the dimension for the momentum equations is [N]). The sources are already multiplied by the volume of the cell
        """
        Vol = self.solver.mesh.V[i,j,k].copy()
        if self.model.lower()=='hall-thollet':
            source_inviscid, source_viscous = self.ComputeHallTholletForceDensity(i, j, k)
        elif self.model.lower()=='hall':
            source_inviscid, source_viscous = self.ComputeHallForceDensity(i, j, k)
        elif self.model.lower()=='none':
            source_inviscid, source_viscous =  np.zeros(5), np.zeros(5)
        elif self.model.lower()=='frozen-forces':
            source_inviscid, source_viscous =  self.ComputeFrozenForcesDensity(i, j, k)
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
        
        Return
        -------------------------

        `source`: np.ndarray of size 5 containing the source terms (the dimension for the momentum equations is [N/m^3])
        """
        x, y, z = self.solver.mesh.X[i,j,k], self.solver.mesh.Y[i,j,k], self.solver.mesh.Z[i,j,k]
        r = np.sqrt(y**2+z**2)
        theta = np.arctan2(z, y)
        conservative = self.solver.solution[i,j,k,:]  # conservative vector
        primitive = GetPrimitivesFromConservatives(conservative)
        rho = primitive[0]
        u_cart = primitive[1:-1]  # cartesian absolute velocity
        u_cyl = ComputeCylindricalVectorFromCartesian(x, y, z, u_cart)
        omega = self.solver.mesh.omega[i,j]
        drag_velocity_cyl = np.array([0, 0, omega*r])
        w_cyl = u_cyl-drag_velocity_cyl
        numberBlades = self.solver.mesh.numberBlades[i,j]
        if numberBlades==0:
            raise ValueError('No blades found in cell %i, %i, %i, where the Hall source term is trying to be computed' %(i,j,k))
        
        pitch = 2*np.pi*r/numberBlades
        n_camber_ax = self.solver.mesh.normal_camber_cyl['Axial'][i,j]
        n_camber_rad = self.solver.mesh.normal_camber_cyl['Radial'][i,j]
        n_camber_tan = self.solver.mesh.normal_camber_cyl['Tangential'][i,j]
        n_vector_cyl = np.array([n_camber_ax, n_camber_rad, n_camber_tan])
        deviationAngle = self.ComputeDeviationAngle(w_cyl, n_vector_cyl)
        fn = np.linalg.norm(w_cyl)**2 * np.pi * deviationAngle / pitch / n_camber_tan
        fn_versor = self.ComputeInviscidForceDirection(w_cyl, n_vector_cyl)

        if omega*fn_versor[2]<0:
            fn_versor *= -1  # the fn must push in the rotation direction
        else:
            pass

        fn_cyl = fn*fn_versor
        fn_cart = ComputeCartesianVectorFromCylindrical(x, y, z, fn_cyl)
        source_inviscid = np.zeros(5)
        source_inviscid[1] = fn_cart[0]
        source_inviscid[2] = fn_cart[1]
        source_inviscid[3] = fn_cart[2]
        source_inviscid[4] = fn_cyl[2]*omega*r
        
        source_viscous = np.zeros(5)
        
        return source_inviscid*rho, source_viscous*rho
    

    def ComputeHallTholletForceDensity(self, i, j, k):
        """
        Compute the force terms (per unit volume) related to the modified Hall model by Thollet based on lift-drag airfoil analogy.

        Parameters
        -------------------------

        `i`: i-index of the cell

        `j`: j-index of the cell

        `k`: k-index of the cell
        
        Return
        -------------------------

        `source`: np.ndarray of size 5 containing the source terms (the dimension for the momentum equations is [N/m^3])
        """
        x, y, z = self.solver.mesh.X[i,j,k], self.solver.mesh.Y[i,j,k], self.solver.mesh.Z[i,j,k]
        r = np.sqrt(y**2+z**2)
        theta = np.arctan2(z, y)
        primitive = GetPrimitivesFromConservatives(self.solver.solution[i,j,k,:])
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
        fp = np.linalg.norm(w_cyl)**2/(pitch*b*np.abs(normal[2])) * (Cf + np.pi*Kmach*(delta-delta0))
        fp_vers_cyl = -w_cyl/np.linalg.norm(w_cyl) # opposite to the relative velocity
        fp_cyl = fp*fp_vers_cyl  
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

        return source_inviscid*rho, source_viscous*rho
    

    def ComputeFrozenForcesDensity(self, i, j, k):
        """
        Compute the force terms (per unit volume) related to the frozen forces extracted from SU2-CFD.

        Parameters
        -------------------------

        `i`: i-index of the cell

        `j`: j-index of the cell

        `k`: k-index of the cell
        
        Return
        -------------------------

        `source`: np.ndarray of size 5 containing the source terms (the dimension for the momentum equations is [N/m^3])
        """
        x, y, z = self.solver.mesh.X[i,j,k], self.solver.mesh.Y[i,j,k], self.solver.mesh.Z[i,j,k]
        r = np.sqrt(y**2+z**2)
        theta = np.arctan2(z, y)
        conservative = self.solver.solution[i,j,k,:]  # conservative vector
        primitive = GetPrimitivesFromConservatives(conservative)
        density = primitive[0]
        u_cart = primitive[1:-1]  # cartesian absolute velocity
        u_cyl = ComputeCylindricalVectorFromCartesian(x, y, z, u_cart)
        omega = self.solver.mesh.omega[i,j,k]
        drag_velocity_cyl = np.array([0, 0, omega*r])
        w_cyl = u_cyl-drag_velocity_cyl
        w_cart = ComputeCartesianVectorFromCylindrical(x, y, z, w_cyl)
        f_loss_versor = -w_cart/np.linalg.norm(w_cart)

        f_ax = self.solver.mesh.force_axial[i,j,k]
        f_rad = self.solver.mesh.force_radial[i,j,k]
        f_tan = self.solver.mesh.force_tangential[i,j,k]

        f_cyl = np.array([f_ax, f_rad, f_tan])
        f_cart = ComputeCartesianVectorFromCylindrical(x, y, z, f_cyl)
        
        fp_cart = np.dot(f_cart, f_loss_versor)*f_loss_versor
        fn_cart = f_cart - fp_cart
        
        fp_cyl = ComputeCylindricalVectorFromCartesian(x, y, z, fp_cart)
        fn_cyl = ComputeCylindricalVectorFromCartesian(x, y, z, fn_cart)
        
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

        return source_inviscid*density, source_viscous*density
    

    def ComputeDeviationAngle(self, w, n):
        """
        Compute the absolute value of the deviation angle between the velocity vector and the camber surface defined by n
        """
        n /= np.linalg.norm(n)
        wn = np.dot(w, -n) # for a compressor the deviation angle is positive when the wn component is opposite to the normal vector pointing in the push direction of the blade
        delta = np.arcsin(wn/np.linalg.norm(w))
        return delta
    

    def ComputeInviscidForceDirection(self, w: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Compute the direction of the inviscid force component. It must be orthogonal to the relative velocity vector, and the radial component is different from zero when only the lean angle is different from zero.
        The relations concering the fn direction are therefore these (3 equations for 3 unknown components):
        1) the magnitude is 1 (is a versor)
        2) the radial component is equal to the radial component of the blade normal versor
        3) The dot product between the force versor and the relative velocity versor must be zero

        The direction is later chosen to be coherent with the rotational speed

        Parameters
        --------------------------------

        `w`: relative velocity vector, in cylindrical coordinates (axial, radial, tangential)

        `n`: blade camber normal vector, in cylindrical coordinates (axial, radial, tangential)

        Return
        --------------------------------

        `fn_dir`: the inviscid force direction in cylindrical coordinates (axial, radial, tangential)
        """
        w_dir = w/np.linalg.norm(w) # this is the relative velocity direction, and defines also the plane where the force must lie. The dot product force*w_dir must always be zero
        n_dir = n/np.linalg.norm(n)
        # fn_dir = np.array([-w_dir[2], 0, w_dir[0]]) # this is valid in those cases defined by zero lean of the blade

        # compute the components 
        fn_dir_r = n[1]  # relation (2)

        # the axial component is found solving a quadratic equation obtained combining relations (1) and (3), which collapse to the easy solution when lean=0
        A = (1+w[0]**2/(w[2]**2+1e-12))
        B = 2*n[1]*w[1]*w[0]/(w[2]**2+1e-12)
        C = n[1]**2 * w[1]**1 / (w[2]**2+1e-12) + n[1]**2 - 1
        sol1 = (-B + np.sqrt(B**2-4*A*C))/2/A
        sol2 = (-B - np.sqrt(B**2-4*A*C))/2/A

        # of the 2 solutions, choose the one having positive value, since the axial component must in general be positive
        if sol1>0:
            fn_dir_ax = sol1
        else:
            fn_dir_ax = sol2
            
        # third component using the last relation
        fn_dir_t = (-n[1]*w[1]-fn_dir_ax*w[0])/w[2]

        # full versor
        fn_dir = np.array([fn_dir_ax, fn_dir_r, fn_dir_t])
        magnitude = np.linalg.norm(fn_dir)
        fn_dir /= magnitude
        
        if any(math.isnan(x) for x in fn_dir):
            print("NaN found during calculation of inviscid force direction. Radial component set to zero.")
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
        else:
            kprime = 2/np.pi/np.sqrt(Mrel**2-1)
        
        if kprime<=3: # clipping at 3
            kmach = kprime
        else:
            kmach = 3
        
        return kmach





        
    

    