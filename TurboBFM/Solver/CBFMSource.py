import numpy as np
import math
from TurboBFM.Solver.CSolver import CSolver
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.math import ComputeCylindricalVectorFromCartesian, ComputeCartesianVectorFromCylindrical
from TurboBFM.Solver.euler_functions import GetPrimitivesFromConservatives
from numpy import sin,cos,tan,arctan2,arccos,arcsin,pi

class CBFMSource():
    """
    BFM source Class, where all the source terms are computed
    """
    def __init__(self, config, solver, iterationCounter):
        """
        Initialize the BFM class with config file info
        """
        self.config = config
        self.solver = solver
        self.iterationCounter = iterationCounter
        self.blockageActive = config.GetBlockageActive()
        self.model = config.GetBFMModel()
        self.deviationAngle = np.zeros_like(self.solver.mesh.X)
    

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
        elif self.model.lower()=='lift-drag':
            source_inviscid, source_viscous =  self.ComputeLiftDragForceDensity(i, j, k)
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
        xNode, yNode, zNode = self.solver.mesh.X[i,j,k], self.solver.mesh.Y[i,j,k], self.solver.mesh.Z[i,j,k]
        radius = np.sqrt(yNode**2+zNode**2)
        conservative = self.solver.solution[i,j,k,:]
        primitive = GetPrimitivesFromConservatives(conservative)
        density = primitive[0]
        velocityCartesian = primitive[1:-1]  # cartesian absolute velocity 
        velocityCylindric = ComputeCylindricalVectorFromCartesian(xNode, yNode, zNode, velocityCartesian)
        omega = self.solver.mesh.omega[i,j]*self.config.getRotationalSpeedRampCoefficient(self.iterationCounter)
        dragVelocityCylindric = np.array([0, 0, omega*radius])
        relativeVelocityCylindric = velocityCylindric-dragVelocityCylindric
        numberBlades = self.solver.mesh.numberBlades[i,j]
        if numberBlades==0:
            raise ValueError('No blades found in cell %i, %i, %i, where the Hall source term is trying to be computed' %(i,j,k))
        pitch = 2*np.pi*radius/numberBlades
        normalCamberAxial = self.solver.mesh.normal_camber_cyl['Axial'][i,j]
        normalCamberRadial = self.solver.mesh.normal_camber_cyl['Radial'][i,j]
        normalCamberTangential = self.solver.mesh.normal_camber_cyl['Tangential'][i,j]
        normalCamberCylindric = np.array([normalCamberAxial, normalCamberRadial, normalCamberTangential])
        deviationAngle = self.ComputeDeviationAngle(relativeVelocityCylindric, normalCamberCylindric)
        self.deviationAngle[i,j,k] = deviationAngle # bookkeep this in memory for later output in vtk file
        
        forceMagnitude = np.linalg.norm(relativeVelocityCylindric)**2 * np.pi * deviationAngle / pitch / np.abs(normalCamberTangential)
        forceVersorCylindric = self.ComputeInviscidForceDirection(relativeVelocityCylindric, normalCamberCylindric)
        forceCylindric = forceMagnitude*forceVersorCylindric
        forceCartesian = ComputeCartesianVectorFromCylindrical(xNode, yNode, zNode, forceCylindric)
        
        sourceInviscid = np.zeros(5)
        sourceInviscid[1] = forceCartesian[0]
        sourceInviscid[2] = forceCartesian[1]
        sourceInviscid[3] = forceCartesian[2]
        sourceInviscid[4] = forceCylindric[2]*omega*radius
    
        sourceViscous = np.zeros(5)
        
        return sourceInviscid*density, sourceViscous*density
    

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
        xNode, yNode, zNode = self.solver.mesh.X[i,j,k], self.solver.mesh.Y[i,j,k], self.solver.mesh.Z[i,j,k]
        radius = np.sqrt(yNode**2+zNode**2)
        conservative = self.solver.solution[i,j,k,:]
        primitive = GetPrimitivesFromConservatives(conservative)
        density = primitive[0]
        velocityCartesian = primitive[1:-1]  # cartesian absolute velocity
        velocityCylindric = ComputeCylindricalVectorFromCartesian(xNode, yNode, zNode, velocityCartesian)
        omega = self.solver.mesh.omega[i,j]*self.config.getRotationalSpeedRampCoefficient(self.iterationCounter)
        dragVelocityCylindric = np.array([0, 0, omega*radius])
        relativeVelocityCylindric = velocityCylindric-dragVelocityCylindric
        numberBlades = self.solver.mesh.numberBlades[i,j]
        if numberBlades==0:
            raise ValueError('No blades found in cell %i, %i, %i, where the Hall source term is trying to be computed' %(i,j,k))
        pitch = 2*np.pi*radius/numberBlades
        normalCamberAxial = self.solver.mesh.normal_camber_cyl['Axial'][i,j]
        normalCamberRadial = self.solver.mesh.normal_camber_cyl['Radial'][i,j]
        normalCamberTangential = self.solver.mesh.normal_camber_cyl['Tangential'][i,j]
        normalCamberCylindric = np.array([normalCamberAxial, normalCamberRadial, normalCamberTangential])
        deviationAngle = self.ComputeDeviationAngle(relativeVelocityCylindric, normalCamberCylindric)
        self.deviationAngle[i,j,k] = deviationAngle # bookkeep this in memory for later output in vtk file
        
        # inviscid component
        blockage = self.solver.mesh.blockage[i,j,k]
        Kmach = self.ComputeCompressibilityCorrection(relativeVelocityCylindric, primitive)
        fn = Kmach * np.linalg.norm(relativeVelocityCylindric)**2 * np.pi * deviationAngle / pitch / np.abs(normalCamberTangential) / blockage
        fn_versor = self.ComputeInviscidForceDirection(relativeVelocityCylindric, normalCamberCylindric)
        fn_cyl = fn*fn_versor
        fn_cart = ComputeCartesianVectorFromCylindrical(xNode, yNode, zNode, fn_cyl)

        # viscous component
        rho = primitive[0]
        nu = self.config.GetKinematicViscosity()
        Rex = np.linalg.norm(relativeVelocityCylindric) * self.solver.mesh.stwl[i,j] / nu
        Cf = 0.0592*Rex**(-0.2)
        delta0 = deviationAngle  # this could be calibrated later, from the deviation of the float at peak efficiency
        fp = np.linalg.norm(relativeVelocityCylindric)**2/(pitch*blockage*np.abs(normalCamberTangential)) * (Cf + np.pi*Kmach*(deviationAngle-delta0)**2)
        fp_vers_cyl = -relativeVelocityCylindric/np.linalg.norm(relativeVelocityCylindric) # opposite to the relative velocity
        fp_cyl = fp*fp_vers_cyl  
        fp_cart = ComputeCartesianVectorFromCylindrical(xNode, yNode, zNode, fp_cyl)

        source_inviscid = np.zeros(5)
        source_inviscid[1] = fn_cart[0]
        source_inviscid[2] = fn_cart[1]
        source_inviscid[3] = fn_cart[2]
        source_inviscid[4] = (fn_cyl[2])*omega*radius

        source_viscous = np.zeros(5)
        source_viscous[1] = fp_cart[0]
        source_viscous[2] = fp_cart[1]
        source_viscous[3] = fp_cart[2]
        source_viscous[4] = (fp_cyl[2])*omega*radius

        return source_inviscid*rho, source_viscous*rho

    
    def ComputeLiftDragForceDensity(self, i, j, k):
        """
        Compute the force terms (per unit volume) related to the lift/drag Thollet model.

        Parameters
        -------------------------

        `i`: i-index of the cell

        `j`: j-index of the cell

        `k`: k-index of the cell
        
        Return
        -------------------------

        `source`: np.ndarray of size 5 containing the source terms (the dimension for the momentum equations is [N/m^3])
        """
        xNode, yNode, zNode = self.solver.mesh.X[i,j,k], self.solver.mesh.Y[i,j,k], self.solver.mesh.Z[i,j,k]
        radius = np.sqrt(yNode**2+zNode**2)
        conservative = self.solver.solution[i,j,k,:]
        primitive = GetPrimitivesFromConservatives(conservative)
        velocityCartesian = primitive[1:-1]  # cartesian absolute velocity
        velocityCylindric = ComputeCylindricalVectorFromCartesian(xNode, yNode, zNode, velocityCartesian)
        omega = self.solver.mesh.omega[i,j]*self.config.getRotationalSpeedRampCoefficient(self.iterationCounter)
        dragVelocityCylindric = np.array([0, 0, omega*radius])
        relativeVelocityCylindric = velocityCylindric-dragVelocityCylindric
        numberBlades = self.solver.mesh.numberBlades[i,j]
        if numberBlades==0:
            raise ValueError('No blades found in cell %i, %i, %i, where the Hall source term is trying to be computed' %(i,j,k))
        normalCamberAxial = self.solver.mesh.normal_camber_cyl['Axial'][i,j]
        normalCamberRadial = self.solver.mesh.normal_camber_cyl['Radial'][i,j]
        normalCamberTangential = self.solver.mesh.normal_camber_cyl['Tangential'][i,j]
        normalCamberCylindric = np.array([normalCamberAxial, normalCamberRadial, normalCamberTangential])
        self.deviationAngle[i,j,k] = self.ComputeDeviationAngle(relativeVelocityCylindric, normalCamberCylindric)
        
        beta_0 = self.solver.mesh.BFcalibrationCoeffs["beta_0"][i,j]
        solidity = self.solver.mesh.BFcalibrationCoeffs["solidity"][i,j]
        h_parameter = self.solver.mesh.BFcalibrationCoeffs["h_parameter"][i,j]
        velMeridionalCylindric = np.array([velocityCylindric[0], velocityCylindric[1], 0])
        
        #distinguish between positive and negative flow angle
        if relativeVelocityCylindric[2]>=0:
            beta_flow = np.arccos(np.dot(relativeVelocityCylindric, velMeridionalCylindric) / (np.linalg.norm(relativeVelocityCylindric) * np.linalg.norm(velMeridionalCylindric)))
        else:
            beta_flow = -np.arccos(np.dot(relativeVelocityCylindric, velMeridionalCylindric) / (np.linalg.norm(relativeVelocityCylindric) * np.linalg.norm(velMeridionalCylindric)))
        
        rotationDirection = self.solver.mesh.machineRotation
        fn_magnitude = self.computeLiftDragInviscidMagnitude(solidity, h_parameter, np.linalg.norm(relativeVelocityCylindric), beta_flow, beta_0, omega, rotationDirection)
        fn_versor = self.ComputeInviscidForceDirection(relativeVelocityCylindric, normalCamberCylindric)
        fn_cyl = fn_magnitude*fn_versor
        fn_cart = ComputeCartesianVectorFromCylindrical(xNode, yNode, zNode, fn_cyl)

        # viscous component
        kp_etaMax = self.solver.mesh.BFcalibrationCoeffs["kp_etaMax"][i,j]
        beta_etaMax = self.solver.mesh.BFcalibrationCoeffs["beta_etaMax"][i,j]
        kp = kp_etaMax + 2*np.pi*solidity*(beta_flow-beta_etaMax)**2
        fp_magnitude = kp*np.linalg.norm(relativeVelocityCylindric)**2/h_parameter
        fp_vers_cyl = -relativeVelocityCylindric/np.linalg.norm(relativeVelocityCylindric) # opposite to the relative velocity
        fp_cyl = fp_magnitude*fp_vers_cyl  
        fp_cart = ComputeCartesianVectorFromCylindrical(xNode, yNode, zNode, fp_cyl)

        source_inviscid = np.zeros(5)
        source_inviscid[1] = fn_cart[0]
        source_inviscid[2] = fn_cart[1]
        source_inviscid[3] = fn_cart[2]
        source_inviscid[4] = (fn_cyl[2])*omega*radius

        source_viscous = np.zeros(5)
        source_viscous[1] = fp_cart[0]
        source_viscous[2] = fp_cart[1]
        source_viscous[3] = fp_cart[2]
        source_viscous[4] = (fp_cyl[2])*omega*radius

        rho = primitive[0]
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
        Compute the value of the deviation angle between the velocity vector and the camber surface defined by n.
        The convention here is that the angle is positive when the relative velocity vector has negative component
        along the direction of the camber normal, since the normal is oriented from SS to PS side of the blade (the push direction).
        """
        n /= np.linalg.norm(n)
        wn = np.dot(w, n)
        delta = -np.arcsin(wn/np.linalg.norm(w)) # positive deviation angle when w points in the oppositve direction as n
        return delta
    

    def ComputeInviscidForceDirection(self, w: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Compute the direction of the inviscid force component. It must be orthogonal to the relative velocity vector, and the radial component is different from zero when only the lean angle is different from zero.
        The relations concering the fn direction are therefore these (3 equations for 3 unknown components):
        1) the magnitude is 1 (is a versor)
        2) the force radial component is equal to the radial component of the blade normal versor
        3) The dot product between the force versor and the relative velocity versor must be zero, since they are orthogonal
        
        Parameters
        --------------------------------

        `w`: relative velocity vector, in cylindrical coordinates (axial, radial, tangential)

        `n`: blade camber normal vector, in cylindrical coordinates (axial, radial, tangential)

        Return
        --------------------------------

        `fn_dir`: the inviscid force direction in cylindrical coordinates (axial, radial, tangential)
        """
        
        w += 1e-6
        w_dir = w/np.linalg.norm(w) 
        n_dir = n/np.linalg.norm(n)
        w_ax = w_dir[0]
        w_rad = w_dir[1]
        w_tan = w_dir[2]
        n_ax = n_dir[0]
        n_rad = n_dir[1]
        n_tan = n_dir[2]
        A = w_tan**2 + w_ax**2
        B = 2 * w_rad * w_ax * n_rad
        C = (w_tan**2 * n_rad**2) + (w_rad**2 * n_rad**2) - w_tan**2 
        deltaEquation = B**2-4*A*C
        
        if deltaEquation < 0:
            print('Delta quadratic equation negative')
        
        fAxial_1 = (-B + np.sqrt(deltaEquation))/2/A
        fAxial_2 = (-B - np.sqrt(deltaEquation))/2/A
        
        def compute_tangential(fax):
            ftan = (-w_ax*fax - w_rad*n_rad)/w_tan
            return ftan
        
        fTangential_1 = compute_tangential(fAxial_1)
        fTangential_2 = compute_tangential(fAxial_2)
        
        fRadial_1 = n_rad
        fRadial_2 = n_rad
        
        fn_versor_1 = np.array([fAxial_1, fRadial_1, fTangential_1])
        fn_versor_2 = np.array([fAxial_2, fRadial_2, fTangential_2])
        
        if np.dot(fn_versor_1, n) > 0:
            fn_dir = fn_versor_1
        else:
            fn_dir = fn_versor_2
        
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


    def computeLiftDragInviscidMagnitude(self, soldity, h_parameter, relVelMag, beta_flow, beta_0, omega, machineRotation):
        """Compute the magnitude of the inviscid component for the L/D model. The magnitude must be positive when the flow is underturned, 
        while negative when overturned with respect to the reference direction beta_0

        Args:
            solditity (float): local solidity
            solditity (float): local h_parameter of the model
            relVelMag (float): local relative velocity magnitude
            beta_flow (float): relative flow angle during simulation [rad]
            beta_0 (float): relative flow angle of the model [rad]
            omega (float): local rotational speed of the cell [rad/s]
            machinRotation (int): rotational direction of the machine (+1, 0, -1) -> (with positive rotor blocks, without any rotor, with counter rotating rotors)
        """
        commonTerm = 2 * np.pi * soldity * (relVelMag**2) / h_parameter
        if np.abs(omega) > 1e-9 : # rotor zone
            if omega < 0:
                return commonTerm * (beta_flow - beta_0)
            else:
                return commonTerm * (beta_0 - beta_flow)
        else: # stator zone
            if machineRotation == 1:
                return commonTerm * (beta_flow - beta_0)
            elif machineRotation == -1:
                return commonTerm * (beta_0 - beta_flow)
            elif machineRotation == 0:
                raise ValueError(" I still didn't figure out how to handle machine with only stator blocks")
            
            
    

    