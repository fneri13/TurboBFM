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


    # @override
    # def Solve(self) -> None:
    #     """
    #     Solve the system explicitly in time.
    #     """
    #     nIter = self.config.GetNIterations()
    #     time_physical = 0
    #     start = time.time()
    #     u_advection = self.config.GetAdvectionVelocity()
    #     theta = np.linspace(0, 2*np.pi*2, nIter)

    #     # fig, ax = plt.subplots()
    #     # cbar = None  # Initialize colorbar reference
    #     for it in range(nIter):            
    #         if self.config.GetAdvectionRotation():
    #             u_adv = np.array([np.cos(theta[it])*u_advection[0],
    #                             np.sin(theta[it])*u_advection[0],
    #                             u_advection[2]])
    #         else:
    #             u_adv = u_advection
                
    #         self.ComputeTimeStepArray()
    #         dt = np.min(self.time_step)
            
    #         residuals = np.zeros_like(self.solution)  # defined as flux*surface going out of the cell (i,j,k)
    #         self.CheckConvergence(self.solution, it+1)
            
    #         # i-fluxes
    #         niF, njF, nkF = self.mesh.Si[:, :, :, 0].shape
    #         for iFace in range(niF):
    #             for jFace in range(njF):
    #                 for kFace in range(nkF):
    #                     if iFace==0: 
    #                         U_r = self.solution[iFace, jFace, kFace,:]
    #                         if self.boundary_types['i']['begin']=='transparent':
    #                             U_l = U_r.copy() 
    #                         elif self.boundary_types['i']['begin']=='periodic':
    #                             U_l = self.solution[-1, jFace, kFace,:]
    #                         else:
    #                             raise ValueError('Unknown boundary condition at ni=0')
    #                         S = self.mesh.Si[iFace, jFace, kFace, :]
    #                         scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                         flux = scheme.ComputeFlux()
    #                         area = np.linalg.norm(S)
    #                         residuals[iFace, jFace, kFace, :] -= flux*area          
    #                     elif iFace==niF-1:
    #                         U_l = self.solution[iFace-1, jFace, kFace, :]  
    #                         if self.boundary_types['i']['end']=='transparent':
    #                             U_r = U_r.copy()
    #                         elif self.boundary_types['i']['end']=='periodic':
    #                             U_r = self.solution[0, jFace, kFace, :]
    #                         else:
    #                             raise ValueError('Unknown boundary condition at ni=0')
    #                         S = self.mesh.Si[iFace, jFace, kFace, :]                
    #                         scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                         flux = scheme.ComputeFlux()
    #                         area = np.linalg.norm(S)
    #                         residuals[iFace-1, jFace, kFace, :] += flux*area        
    #                     else:
    #                         U_l = self.solution[iFace-1, jFace, kFace,:]
    #                         U_r = self.solution[iFace, jFace, kFace,:]  
    #                         S = self.mesh.Si[iFace, jFace, kFace, :]
    #                         scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                         flux = scheme.ComputeFlux()
    #                         area = np.linalg.norm(S)
    #                         residuals[iFace-1, jFace, kFace, :] += flux*area 
    #                         residuals[iFace, jFace, kFace, :] -= flux*area
            
    #         # j-fluxes
    #         niF, njF, nkF = self.mesh.Sj[:, :, :, 0].shape
    #         for iFace in range(niF):
    #             for jFace in range(njF):
    #                 for kFace in range(nkF):
    #                     if jFace==0: 
    #                         U_r = self.solution[iFace, jFace, kFace,:]
    #                         if self.boundary_types['j']['begin']=='transparent':
    #                             U_l = U_r.copy() 
    #                         elif self.boundary_types['j']['begin']=='periodic':
    #                             U_l = self.solution[iFace, -1, kFace,:]
    #                         else:
    #                             raise ValueError('Unknown boundary condition at nj=0')
    #                         S = self.mesh.Sj[iFace, jFace, kFace, :]
    #                         scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                         flux = scheme.ComputeFlux()
    #                         area = np.linalg.norm(S)
    #                         residuals[iFace, jFace, kFace, :] -= flux*area          
    #                     elif jFace==njF-1:
    #                         U_l = self.solution[iFace, jFace-1, kFace, :]      
    #                         if self.boundary_types['j']['end']=='transparent':
    #                             U_r = U_l.copy()
    #                         elif self.boundary_types['j']['end']=='periodic':
    #                             U_r = self.solution[iFace, 0, kFace,:]
    #                         else:
    #                             raise ValueError('Unknown boundary condition at j=nj')
    #                         S = self.mesh.Sj[iFace, jFace, kFace, :]                
    #                         scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                         flux = scheme.ComputeFlux()
    #                         area = np.linalg.norm(S)
    #                         residuals[iFace, jFace-1, kFace, :] += flux*area        
    #                     else:
    #                         U_l = self.solution[iFace, jFace-1, kFace,:]
    #                         U_r = self.solution[iFace, jFace, kFace,:]  
    #                         S = self.mesh.Sj[iFace, jFace, kFace, :]
    #                         scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                         flux = scheme.ComputeFlux()
    #                         area = np.linalg.norm(S)
    #                         residuals[iFace, jFace-1, kFace, :] += flux*area 
    #                         residuals[iFace, jFace, kFace, :] -= flux*area
            
    #         # k-fluxes
    #         if self.nDim==3:
    #             niF, njF, nkF = self.mesh.Sk[:, :, :, 0].shape
    #             for iFace in range(niF):
    #                 for jFace in range(njF):
    #                     for kFace in range(nkF):
    #                         if kFace==0: 
    #                             U_r = self.solution[iFace, jFace, kFace,:]
    #                             if self.boundary_types['k']['begin']=='transparent':
    #                                 U_l = U_r.copy() 
    #                             elif self.boundary_types['k']['begin']=='periodic':
    #                                 U_l = self.solution[iFace, jFace, -1,:]
    #                             else:
    #                                 raise ValueError('Unknown boundary condition at nk=0')
    #                             S = self.mesh.Sk[iFace, jFace, kFace, :]
    #                             scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                             flux = scheme.ComputeFlux()
    #                             area = np.linalg.norm(S)
    #                             residuals[iFace, jFace, kFace, :] -= flux*area          
    #                         elif kFace==nkF-1:
    #                             U_l = self.solution[iFace, jFace, kFace-1, :]       
    #                             if self.boundary_types['k']['end']=='transparent':
    #                                 U_r = U_l.copy()
    #                             elif self.boundary_types['k']['end']=='periodic':
    #                                 U_r = self.solution[iFace, jFace, 0,:]
    #                             else:
    #                                 raise ValueError('Unknown boundary condition at k=nk')
    #                             S = self.mesh.Sk[iFace, jFace, kFace, :]                
    #                             scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                             flux = scheme.ComputeFlux()
    #                             area = np.linalg.norm(S)
    #                             residuals[iFace, jFace, kFace-1, :] += flux*area        
    #                         else:
    #                             U_l = self.solution[iFace, jFace, kFace-1,:]
    #                             U_r = self.solution[iFace, jFace, kFace,:]  
    #                             S = self.mesh.Sk[iFace, jFace, kFace, :]
    #                             scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
    #                             flux = scheme.ComputeFlux()
    #                             area = np.linalg.norm(S)
    #                             residuals[iFace, jFace, kFace-1, :] += flux*area 
    #                             residuals[iFace, jFace, kFace, :] -= flux*area
            
    #         self.PrintInfoResiduals(residuals, it, time_physical)
    #         time_physical += dt

    #         for iEq in range(self.nEq):
    #             self.solution[:,:,:,iEq] = self.solution[:,:,:,iEq] - residuals[:,:,:,iEq]*dt/self.mesh.V[:,:,:]  # update the conservative solution

    #         # contour = ax.contourf(self.mesh.X[:, :, 0], self.mesh.Y[:, :, 0], self.solution[:, :, 0, 0], cmap=styles.color_map, vmin=0, vmax=1)
    #         # if cbar:
    #         #     cbar.remove()
    #         # cbar = fig.colorbar(contour, ax=ax)
    #         # # u_quiver = u_adv[0]+np.zeros_like(self.mesh.X[:, :, 0])
    #         # # v_quiver = u_adv[1]+np.zeros_like(self.mesh.X[:, :, 0])
    #         # # ax.quiver(self.mesh.X[:, :, 0], self.mesh.Y[:, :, 0], u_quiver, v_quiver, color='red')
    #         # ax.set_aspect('equal')
    #         # plt.pause(0.001)

    #         self.SaveSolution(it, nIter)
            

    #     end = time.time()
    #     self.PlotResidualsHistory()


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


