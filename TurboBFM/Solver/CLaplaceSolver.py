import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.euler_functions import GetConservativesFromPrimitives, GetPrimitivesFromConservatives
from TurboBFM.Solver.CScheme_Central import CScheme_Central
from TurboBFM.Solver.CBoundaryCondition import CBoundaryCondition
from TurboBFM.Solver.CSolver import CSolver
from TurboBFM.Postprocess import styles
from typing import override 


class CLaplaceSolver(CSolver):
    
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
        if self.verbosity>0:
            print('='*25 + ' LAPLACE SOLVER ' + '='*25)
            print('Number of dimensions:                    %i' %(self.nDim))
            print('Boundary type at i=0:                    %s' %(self.boundary_types['i']['begin']))
            print('Boundary type at i=ni:                   %s' %(self.boundary_types['i']['end']))
            print('Boundary type at j=0:                    %s' %(self.boundary_types['j']['begin']))
            print('Boundary type at j=nj:                   %s' %(self.boundary_types['j']['end']))
            if self.nDim==3:
                print('Boundary type at k=0:                    %s' %(self.boundary_types['k']['begin']))
                print('Boundary type at k=nk:                   %s' %(self.boundary_types['k']['end']))
            print('Boundary value at i=0:                    %.2f' %(self.boundary_values[0]))
            print('Boundary value at i=ni:                   %.2f' %(self.boundary_values[1]))
            print('Boundary value at j=0:                    %.2f' %(self.boundary_values[2]))
            print('Boundary value at j=nj:                   %.2f' %(self.boundary_values[3]))
            if self.nDim==3:
                print('Boundary value at k=0:                    %.2f' %(self.boundary_values[4]))
                print('Boundary value at k=nk:                   %.2f' %(self.boundary_values[5]))
            print('')
            print('='*25 + ' END SOLVER INFORMATION ' + '='*25)
            print()


    @override
    def InstantiateFields(self):
        """
        Instantiate basic fields.
        """
        super().InstantiateFields()
        self.solution_names = [r'$T$']

        # store also the gradient of the solution (point i,j,k, solution variable nEq, and component of the gradient x,y,z)
        self.solution_gradient = np.zeros((self.ni, self.nj, self.nk, self.nEq, 3)) 



    @override
    def ReadBoundaryConditions(self):
        """
        Read the boundary conditions from the input file, and store the information in two dictionnaries
        """
        super().ReadBoundaryConditions()
        self.boundary_values = self.config.GetDirichletValues()
        if self.nDim==3:
            assert len(self.boundary_values)==6, 'You need 6 Dirichlet conditions for a 3D problem'
        else:
            assert len(self.boundary_values)==4, 'You need 4 Dirichlet conditions for a 2D problem'
        

    @override
    def InitializeSolution(self):
        """
        Initialize the advection initial condition. Sphere of phi=1 around the center of the domain, at a dist max given by radius
        """
        avg = np.mean(self.boundary_values)
        self.solution[:,:,:,0] = avg


    @override
    def Solve(self) -> None:
        """
        Solve the system explicitly in time.
        """
        nIter = self.config.GetNIterations()
        time_physical = 0
        start = time.time()

        fig, ax = plt.subplots()
        cbar = None  # Initialize colorbar reference
        for it in range(nIter):            
            self.ComputeTimeStepArray()
            dt = np.min(self.time_step)
            self.ComputeSolutionGradient()
            
            residuals = np.zeros_like(self.solution)  # defined as flux*surface going out of the cell (i,j,k)
            self.CheckConvergence(self.solution, it+1)
            
            # i-fluxes
            niF, njF, nkF = self.mesh.Si[:, :, :, 0].shape
            for iFace in range(niF):
                for jFace in range(njF):
                    for kFace in range(nkF):
                        if iFace==0: 
                            U_r = self.solution_gradient[iFace, jFace, kFace, 0, :]
                            U_l = U_r.copy() 
                            S = self.mesh.Si[iFace, jFace, kFace, :]
                            scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace, jFace, kFace, :] -= flux*area          
                        elif iFace==niF-1:
                            U_l = self.solution_gradient[iFace-1, jFace, kFace, 0, :]
                            U_r = U_l.copy()
                            S = self.mesh.Si[iFace, jFace, kFace, :]
                            scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace-1, jFace, kFace, :] += flux*area        
                        else:
                            U_l = self.solution_gradient[iFace-1, jFace, kFace, 0, :]
                            U_r = self.solution_gradient[iFace, jFace, kFace, 0, :]  
                            S = self.mesh.Si[iFace, jFace, kFace, :]
                            scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace-1, jFace, kFace, :] += flux*area 
                            residuals[iFace, jFace, kFace, :] -= flux*area
            
            # j-fluxes
            niF, njF, nkF = self.mesh.Sj[:, :, :, 0].shape
            for iFace in range(niF):
                for jFace in range(njF):
                    for kFace in range(nkF):
                        if jFace==0: 
                            U_r = self.solution_gradient[iFace, jFace, kFace, 0, :]
                            U_l = U_r.copy() 
                            S = self.mesh.Sj[iFace, jFace, kFace, :]
                            scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace, jFace, kFace, :] -= flux*area          
                        elif jFace==njF-1:
                            U_l = self.solution_gradient[iFace, jFace-1, kFace, 0, :]
                            U_r = U_l.copy()
                            S = self.mesh.Sj[iFace, jFace, kFace, :]
                            scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace, jFace-1, kFace, :] += flux*area        
                        else:
                            U_l = self.solution_gradient[iFace, jFace-1, kFace, 0,:]
                            U_r = self.solution_gradient[iFace, jFace, kFace, 0,:]  
                            S = self.mesh.Sj[iFace, jFace, kFace, :]
                            scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace, jFace-1, kFace, :] += flux*area 
                            residuals[iFace, jFace, kFace, :] -= flux*area
            
            # k-fluxes
            if self.nDim==3:
                niF, njF, nkF = self.mesh.Sk[:, :, :, 0].shape
                for iFace in range(niF):
                    for jFace in range(njF):
                        for kFace in range(nkF):
                            if kFace==0: 
                                U_r = self.solution_gradient[iFace, jFace, kFace, 0, :]
                                U_l = U_r.copy() 
                                S = self.mesh.Sk[iFace, jFace, kFace, :]
                                scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                                flux = scheme.ComputeFlux()
                                area = np.linalg.norm(S)
                                residuals[iFace, jFace, kFace, :] -= flux*area          
                            elif kFace==nkF-1:
                                U_l = self.solution_gradient[iFace, jFace, kFace-1, 0, :]
                                U_r = U_l.copy()
                                S = self.mesh.Sk[iFace, jFace, kFace, :]
                                scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                                flux = scheme.ComputeFlux()
                                area = np.linalg.norm(S)
                                residuals[iFace, jFace, kFace-1, :] += flux*area        
                            else:
                                U_l = self.solution_gradient[iFace, jFace, kFace-1, 0,:]
                                U_r = self.solution_gradient[iFace, jFace, kFace, 0,:]  
                                S = self.mesh.Sk[iFace, jFace, kFace, :]
                                scheme = CScheme_Central(U_l, U_r, S, -self.config.GetLaplaceDiffusivity())
                                flux = scheme.ComputeFlux()
                                area = np.linalg.norm(S)
                                residuals[iFace, jFace, kFace-1, :] += flux*area 
                                residuals[iFace, jFace, kFace, :] -= flux*area
            
            self.PrintInfoResiduals(residuals, it, time_physical)
            time_physical += dt

            for iEq in range(self.nEq):
                self.solution[:,:,:,iEq] = self.solution[:,:,:,iEq] - residuals[:,:,:,iEq]*dt/self.mesh.V[:,:,:]  # update the conservative solution

            contour = ax.contourf(self.mesh.X[:, :, 0], self.mesh.Y[:, :, 0], self.solution[:, :, 0, 0], cmap=styles.color_map)
            if cbar:
                cbar.remove()
            cbar = fig.colorbar(contour, ax=ax)
            # ax.set_aspect('equal')
            plt.pause(0.001)

            self.SaveSolution(it, nIter)
            

        end = time.time()
        self.PlotResidualsHistory()
    
    @override
    def SpatialIntegration(self, sol, res):
        return super().SpatialIntegration(sol, res)


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
            print(f"{'|'}{'Iteration':<{col_width}}{'|'}{'Time[s]':<{col_width}}{'|'}{'rms[T]':>{col_width}}{'|'}")
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
        alpha = self.config.GetLaplaceDiffusivity()
        u = np.array([1, 1, 1])*alpha  # equivalent of diffusivity = 1 ?        
        
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


    @override
    def CheckConvergence(self, array, nIter):
        """
        Check the array of solutions to stop the simulation in case something is wrong.

        Parameters
        ----------------------------

        `array`: array to check for nan, usually conservative solutions

        `nIter`: current iteration number
        """
        if np.isnan(array).any():
            raise ValueError("The simulation diverged. Nan found at iteration %i" %(nIter))
    

    @override
    def InterpolateSolution(self, i, j, k, eq, dir):
        """
        Interpolate the solution between the nodes. At the borders, take the values from the dirichlet bcs.
        """
        if dir=='west':
            if i==0:
                u = self.boundary_values[0]
            else:
                u = (self.solution[i-1,j,k,eq] + self.solution[i,j,k,eq])/2.0

        elif dir=='east':
            if i==self.ni-1:
                u = self.boundary_values[1]
            else:
                u = (self.solution[i+1,j,k,eq] + self.solution[i,j,k,eq])/2.0
        
        elif dir=='south':
            if j==0:
                u = self.boundary_values[2]
            else:
                u = (self.solution[i,j-1,k,eq] + self.solution[i,j,k,eq])/2.0

        elif dir=='north':
            if j==self.nj-1:
                u = self.boundary_values[3]
            else:
                u = (self.solution[i,j+1,k,eq] + self.solution[i,j,k,eq])/2.0
        
        elif dir=='bottom':
            if k==0:
                if self.nDim==3:
                    u = self.boundary_values[4]
                else:
                    u = self.solution[i,j,k,eq]
            else:
                u = (self.solution[i,j,k-1,eq] + self.solution[i,j,k,eq])/2.0

        elif dir=='top':
            if k==self.nk-1:
                if self.nDim==3:
                    u = self.boundary_values[5]
                else:
                    u = self.solution[i,j,k,eq]
            else:
                u = (self.solution[i,j,k+1,eq] + self.solution[i,j,k,eq])/2.0

        else:
            raise ValueError('Unknown direction')
      
        return u