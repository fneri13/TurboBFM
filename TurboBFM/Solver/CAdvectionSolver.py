import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.euler_functions import GetConservativesFromPrimitives, GetPrimitivesFromConservatives
from TurboBFM.Solver.CScheme_Upwind import CScheme_Upwind
from TurboBFM.Solver.CBoundaryCondition import CBoundaryCondition
from TurboBFM.Solver.CSolver import CSolver
from TurboBFM.Postprocess import styles
from typing import override 


class CAdvectionSolver(CSolver):
    

    def __init__(self, config: CConfig, mesh: CMesh):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        super().__init__(config, mesh)
    

    @override
    def PrintInfoSolver(self):
        """
        Print basic information before running the solver
        """
        u = self.config.GetAdvectionVelocity()

        if self.verbosity>0:
            print('='*25 + ' ADVECTION SOLVER ' + '='*25)
            print('Number of dimensions:                    %i' %(self.nDim))
            print('Advection Velocity [m/s]:                (%.2f, %.2f, %.2f)' %(u[0], u[1], u[2]))
            print('Boundary type at i=0:                    %s' %(self.boundary_types['i']['begin']))
            print('Boundary type at i=ni:                   %s' %(self.boundary_types['i']['end']))
            print('Boundary type at j=0:                    %s' %(self.boundary_types['j']['begin']))
            print('Boundary type at j=nj:                   %s' %(self.boundary_types['j']['end']))
            if self.nDim==3:
                print('Boundary type at k=0:                    %s' %(self.boundary_types['k']['begin']))
                print('Boundary type at k=nk:                   %s' %(self.boundary_types['k']['end']))
            print('Time Integration method:                 %s' %(self.config.GetTimeIntegrationType()))
            print('='*25 + ' END SOLVER INFORMATION ' + '='*25)
            print()


    @override
    def InstantiateFields(self):
        """
        Instantiate basic fields.
        """
        super().InstantiateFields()
        self.solution_names = [r'$\phi$']


    @override
    def ReadBoundaryConditions(self):
        """
        Read the boundary conditions from the input file, and store the information in two dictionnaries
        """
        super().ReadBoundaryConditions()


    @override
    def InitializeSolution(self):
        """
        Initialize the advection initial condition. Sphere of phi=1 around the center of the domain, at a dist max given by radius
        """
        delta_x = np.max(self.mesh.X) - np.min(self.mesh.X)
        delta_y = np.max(self.mesh.Y) - np.min(self.mesh.Y)
        delta_z = np.max(self.mesh.Z) - np.min(self.mesh.Z)

        center = np.array([np.min(self.mesh.X)+delta_x/2,
                           np.min(self.mesh.Y)+delta_y/4,
                           np.min(self.mesh.Z)+delta_z/2])

        radius = (delta_x+delta_y+delta_z)/3/5

        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    distance = np.sqrt((self.mesh.X[i,j,k]-center[0])**2 + 
                                       (self.mesh.Y[i,j,k]-center[1])**2 + 
                                       (self.mesh.Z[i,j,k]-center[2])**2)
                    
                    if distance<radius:
                        self.solution[i,j,k,0] = 1
                    else:
                        self.solution[i,j,k,0] = 0


    @override
    def SpatialIntegration(self, dir, Sol, Res):
        """
        Perform spatial integration loop in a certain direction. 

        Parameters
        -------------------------

        `dir`: i,j or k

        `Sol`: array of the solution

        `Res`: residual arrays of the current time-step that will be updated
        """
        if dir=='i':
            step_mask = np.array([1, 0, 0])
            Surf = self.mesh.Si
        elif dir=='j':
            step_mask = np.array([0, 1, 0])
            Surf = self.mesh.Sj
        else:
            step_mask = np.array([0, 0, 1])
            Surf = self.mesh.Sk
        
        niF, njF, nkF = Surf[:,:,:,0].shape
        
        for iFace in range(niF):
            for jFace in range(njF):
                for kFace in range(nkF):
                    
                    # direction specific parameters
                    if dir=='i':
                        dir_face = iFace
                        stop_face = niF-1  # index of the last volume elements along the specified direction
                    elif dir=='j':
                        dir_face = jFace
                        stop_face = njF-1
                    else:
                        dir_face = kFace
                        stop_face = nkF-1
                    
                    if dir_face==0: 
                        U_r = Sol[iFace, jFace, kFace, :]
                        if self.boundary_types[dir]['begin']=='transparent':
                            U_l = U_r.copy()  
                        elif self.boundary_types[dir]['begin']=='periodic':
                            U_l = Sol[iFace-step_mask[0], jFace-step_mask[1], kFace-step_mask[2]]
                        else:
                            raise ValueError('Unkonwn boundary condition')
                        S = Surf[iFace, jFace, kFace, :]  
                        scheme = CScheme_Upwind(U_l, U_r, S, self.u_advection)
                        flux = scheme.ComputeFlux()
                        area = np.linalg.norm(S)            
                        Res[iFace, jFace, kFace, :] -= flux*area          
                    elif dir_face==stop_face:
                        U_l = Sol[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :]  
                        if self.boundary_types[dir]['end']=='transparent':    
                            U_r = U_l.copy()
                        elif self.boundary_types[dir]['end']=='periodic':
                            if dir=='i':
                                U_r = Sol[0, jFace, kFace, :]  
                            elif dir=='j':
                                U_r = Sol[iFace, 0, kFace, :] 
                            elif dir=='k':
                                U_r = Sol[iFace, jFace, 0, :]  
                        else:
                            raise ValueError('wrong direction')
                        S = Surf[iFace, jFace, kFace, :]                
                        scheme = CScheme_Upwind(U_l, U_r, S, self.u_advection)
                        flux = scheme.ComputeFlux()
                        area = np.linalg.norm(S)
                        Res[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :] += flux*area       
                    else:
                        U_l = Sol[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2],:]
                        U_r = Sol[iFace, jFace, kFace,:]  
                        S = Surf[iFace, jFace, kFace, :]
                        scheme = CScheme_Upwind(U_l, U_r, S, self.u_advection)
                        flux = scheme.ComputeFlux()
                        area = np.linalg.norm(S)
                        Res[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :] += flux*area 
                        Res[iFace, jFace, kFace, :] -= flux*area


    @override
    def PrintInfoResiduals(self, residuals: np.ndarray, it: int, time: float, col_width: int = 14):
        """
        Print the residuals during the simulation.

        Parameters
        -------------------------------

        `residuals`: residuals array [ni,nj,nk,5]

        `it`: iteration step number

        `time`: current physical time (if global time step is active), otherwise just an indication

        `col_width`: singular column width for the print
        """
        res = np.zeros(self.nEq)
        for i in range(self.nEq):
            res[i] = np.linalg.norm(residuals[:,:,:,i].flatten())/len(residuals[:,:,:,i].flatten())
            if res[i]!=0:
                res[i] = np.log10(res[i])
        if it==0:
        # Header
            print("|" + "-" * ((col_width)*7+6) + "|")
            print(f"{'|'}{'Iteration':<{col_width}}{'|'}{'Time[s]':<{col_width}}{'|'}{'rms[Phi]':>{col_width}}{'|'}")
            print("|" + "-" * ((col_width)*7+6) + "|")

        # Data row
        print(
            f"{'|'}{it:<{col_width}}{'|'}{time:<{col_width}.3e}{'|'}{res[0]:>{col_width}.6f}{'|'}"
        )

        # bookkeep the residuals for final plot
        for iEq in range(self.nEq):
            self.residual_history[iEq].append(res[iEq])
        
    
    @override
    def ComputeTimeStepArray(self):
        """
        Compute the time step of the simulation for a certain CFL
        """
        CFL = self.config.GetCFL()
        u = self.config.GetAdvectionVelocity()
        
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    i_edge, j_edge, k_edge = self.mesh.GetElementEdges((i,j,k))
                    i_dir = i_edge/np.linalg.norm(i_edge)
                    j_dir = j_edge/np.linalg.norm(j_edge)
                    k_dir = k_edge/np.linalg.norm(k_edge)
                    
                    dt_i = np.linalg.norm(i_edge) / (np.abs(np.dot(u, i_dir))+1e-16)
                    dt_j = np.linalg.norm(j_edge) / (np.abs(np.dot(u, j_dir))+1e-16)
                    dt_k = np.linalg.norm(k_edge) / (np.abs(np.dot(u, k_dir))+1e-16)

                    if self.nDim==3:
                        self.time_step[i,j,k] = CFL*min(dt_i, dt_j, dt_k)
                    else:
                        self.time_step[i,j,k] = CFL*min(dt_i, dt_j)


