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
                           np.min(self.mesh.Y)+delta_y/2,
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
    def ContoursCheck(self, group: str, perp_direction: str = 'i'):
        """
        Plot the contour of the required group of variables perpendicular to the direction index `perp_direction`, at mid length (for the moment).

        Parameters:
        ------------------------

        `group`: select between primitives or conservatives
        `perp_direction`: select between i,j,k to choose the on which doing the contour
        """
        if group.lower()=='primitives':
            fields = self.ConvertConservativesToPrimitives()
            names = [r'$\rho$', r'$u_x$', r'$u_y$', r'$u_z$', r'$e_t$']
        elif group.lower()=='conservatives':
            fields = self.conservatives
            names = [r'$\rho$', r'$\rho u_x$', r'$\rho u_y$', r'$\rho u_z$', r'$\rho e_t$']

        # function to make contours on different directions
        def contour_template(fields, names, dir_cut, idx_cut):
            fig, ax = plt.subplots(1, len(names), figsize=(20,3))
            for iField in range(len(names)):
                if dir_cut=='i':
                    cnt = ax[iField].contourf(fields[idx_cut,:,:,iField], cmap='jet', levels=20)
                    xlabel = 'K'
                    ylabel = 'J'
                elif dir_cut=='j':
                    cnt = ax[iField].contourf(fields[:,idx_cut,:,iField], cmap='jet', levels=20)
                    xlabel = 'K'
                    ylabel = 'I'
                elif dir_cut=='k':
                    cnt = ax[iField].contourf(fields[:,:,idx_cut,iField], cmap='jet', levels=20)
                    xlabel = 'J'
                    ylabel = 'I'
                plt.colorbar(cnt, ax=ax[iField], orientation='horizontal', pad=0.2)
                ax[iField].set_title(names[iField])
                ax[iField].set_xlabel(xlabel)
                ax[iField].set_ylabel(ylabel)
        
        # call the contour function depending on the chosen direction
        if perp_direction.lower()=='i':
            idx = self.ni//2
            contour_template(fields, names, perp_direction, idx)
        elif perp_direction.lower()=='j':
            idx = self.nj//2
            contour_template(fields, names, perp_direction, idx)
        elif perp_direction.lower()=='k':
            idx = self.nk//2
            contour_template(fields, names, perp_direction, idx)
    

    @override
    def ContoursCheckMeridional(self, group: str):
        """
        Plot the contour of the required group of variables perpendicular to the direction index `perp_direction`, at mid length (for the moment).

        Parameters:
        ------------------------

        `group`: select between primitives or conservatives
        `perp_direction`: select between i,j,k to choose the on which doing the contour
        """
        N_levels = 20
        color_map = 'jet'
        if group.lower()=='primitives':
            fields = self.ConvertConservativesToPrimitives()
            names = [r'$\rho \ \rm{[kg/m^3]}$', r'$u_x \ \rm{[m/s]}$', r'$u_y  \ \rm{[m/s]}$', r'$u_z \ \rm{[m/s]}$', r'$e_t  \ \rm{[J/kg]}$']
        elif group.lower()=='conservatives':
            fields = self.solution
            names = [r'$\rho \ \rm{[kg/m^3]}$', r'$\rho u_x  \ \rm{[kg \cdot m^2/s]}$', r'$\rho u_y  \ \rm{[kg \cdot m^2/s]}$', r'$\rho u_z  \ \rm{[kg \cdot m^2/s]}$', r'$\rho e_t   \ \rm{[J/m^3]}$']

        # function to make contours on different directions
        def contour_template(fields, names, idx_cut):
            fig, ax = plt.subplots(3, 3, figsize=(20,10))
            for iField in range(len(names)):
                X = self.mesh.X[:,:,idx_cut]
                Y = self.mesh.Y[:,:,idx_cut]
                iplot = iField//3
                jplot = iField-3*iplot
                cnt = ax[iplot][jplot].contourf(X, Y, fields[:,:,idx_cut,iField], cmap=color_map, levels=N_levels)
                xlabel = 'x [m]'
                ylabel = 'y [m]'
                plt.colorbar(cnt, ax=ax[iplot][jplot])
                ax[iplot][jplot].set_title(names[iField])
                if jplot==0:
                    ax[iplot][jplot].set_ylabel(ylabel)
                if iplot==2:
                    ax[iplot][jplot].set_xlabel(xlabel)

            for iplot in range(3):
                for jplot in range(3):    
                    ax[iplot][jplot].set_xticks([])
                    ax[iplot][jplot].set_yticks([])
                    if jplot==0:
                        ax[iplot][jplot].set_ylabel(ylabel)
                    if iplot==2:
                        ax[iplot][jplot].set_xlabel(xlabel)

            fields = self.ConvertSolutionToPrimitives(self.solution)
            entropy = np.zeros((self.ni, self.nj))
            for i in range(self.ni):
                for j in range(self.nj):
                    entropy[i,j] = self.fluid.ComputeEntropy_rho_u_et(fields[i,j,idx_cut,0],
                                                                      fields[i,j,idx_cut,1:-1], 
                                                                      fields[i,j,idx_cut,-1])
            cnt = ax[1][2].contourf(self.mesh.X[:,:,idx_cut], self.mesh.Y[:,:,idx_cut], entropy[:,:], cmap=color_map, levels=N_levels)
            plt.colorbar(cnt, ax=ax[1][2])
            ax[1][2].set_title(r'$s \ \rm{[J/kgK]}$')

            pressure = np.zeros((self.ni, self.nj))
            for i in range(self.ni):
                for j in range(self.nj):
                    pressure[i,j] = self.fluid.ComputePressure_rho_u_et(fields[i,j,idx_cut,0],
                                                                        fields[i,j,idx_cut,1:-1], 
                                                                        fields[i,j,idx_cut,-1])
            cnt = ax[2][0].contourf(self.mesh.X[:,:,idx_cut], self.mesh.Y[:,:,idx_cut], pressure[:,:]/1e3, cmap=color_map, levels=N_levels)
            plt.colorbar(cnt, ax=ax[2][0])
            ax[2][0].set_title(r'$p \ \rm{[kPa]}$')

            ht = np.zeros((self.ni, self.nj))
            for i in range(self.ni):
                for j in range(self.nj):
                    ht[i,j] = self.fluid.ComputeTotalEnthalpy_rho_u_et(fields[i,j,idx_cut,0],
                                                                             fields[i,j,idx_cut,1:-1], 
                                                                             fields[i,j,idx_cut,-1])
            cnt = ax[2][1].contourf(self.mesh.X[:,:,idx_cut], self.mesh.Y[:,:,idx_cut], ht[:,:], cmap=color_map, levels=N_levels)
            plt.colorbar(cnt, ax=ax[2][1])
            ax[2][1].set_title(r'$h_t \ \rm{[J/kg]}$')

            temperature = np.zeros((self.ni, self.nj))
            for i in range(self.ni):
                for j in range(self.nj):
                    temperature[i,j] = self.fluid.ComputeStaticTemperature_rho_u_et(fields[i,j,idx_cut,0],
                                                                             fields[i,j,idx_cut,1:-1], 
                                                                             fields[i,j,idx_cut,-1])
            cnt = ax[2][2].contourf(self.mesh.X[:,:,idx_cut], self.mesh.Y[:,:,idx_cut], temperature, cmap=color_map, levels=N_levels)
            plt.colorbar(cnt, ax=ax[2][2])
            ax[2][2].set_title(r'$T \ \rm{[K]}$')

            u = fields[:, :, idx_cut, 1]
            v = fields[:, :, idx_cut, 2]
            w = fields[:, :, idx_cut, 3]
            magnitude = np.sqrt(u**2 + v**2 + w**2)
            plt.figure()
            plt.contourf(self.mesh.X[:,:,idx_cut], self.mesh.Y[:,:,idx_cut], magnitude, cmap=color_map, levels=N_levels)
            plt.colorbar()
            plt.quiver(self.mesh.X[:,:,idx_cut], self.mesh.Y[:,:,idx_cut], u, v)
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.title(r'$|u| \ \rm{[m/s]}$')


        # call the contour function depending on the chosen direction
        idx = self.nk//2
        contour_template(fields, names, idx)
    

    @override
    def ContoursCheckResiduals(self, array: np.ndarray):
        """
        Plot the contour of the residuals.

        Parameters:
        ------------------------

        `array`: array storing the residual values
        """
        N_levels = 20
        color_map = 'jet'
        names = [r'$R(\rho)$', r'$R(\rho u_x)$', r'$R(\rho u_y)$', r'$R(\rho u_z)$', r'$R(\rho e_t)$']
        
        # function to make contours on different directions
        def contour_template(fields, names, idx_cut):
            xlabel = 'x [m]'
            ylabel = 'y [m]'
            fig, ax = plt.subplots(2, 3, figsize=(10,5))
            for iField in range(len(names)):
                X = self.mesh.X[:,:,idx_cut]
                Y = self.mesh.Y[:,:,idx_cut]
                iplot = iField//3
                jplot = iField-3*iplot
                cnt = ax[iplot][jplot].contourf(X, Y, fields[:,:,idx_cut,iField], cmap=color_map, levels=N_levels)
                plt.colorbar(cnt, ax=ax[iplot][jplot])
                ax[iplot][jplot].set_title(names[iField])
                if jplot==0:
                    ax[iplot][jplot].set_ylabel(ylabel)
                if iplot==2:
                    ax[iplot][jplot].set_xlabel(xlabel)
                # ax[iplot][jplot].set_aspect('equal')

            for iplot in range(2):
                for jplot in range(3):    
                    ax[iplot][jplot].set_xticks([])
                    ax[iplot][jplot].set_yticks([])
                    if jplot==0:
                        ax[iplot][jplot].set_ylabel(ylabel)
                    if iplot==2:
                        ax[iplot][jplot].set_xlabel(xlabel)


        # call the contour function depending on the chosen direction
        idx = self.nk//2
        contour_template(array, names, idx)

        
    def ComputeInitFields(self, M: float, T: float, P: float, dir: np.ndarray):
        """
        Compute initialization values from the specified parameters

        Parameters
        -----------------------------

        `M`: Mach number

        `T`: static temperature

        `P`: static pressure

        `dir`: direction vector

        Returns
        -----------------------------

        `rho`: density

        `u`: velocity vector

        `et`: total energy
        """
        gmma = self.config.GetFluidGamma()
        R = self.config.GetFluidRConstant()
        ss = np.sqrt(gmma*R*T)
        u_mag = ss*M
        dir /= np.linalg.norm(dir)
        u = u_mag*dir
        rho = P/R/T
        et = (P / (gmma - 1) / rho) + 0.5*u_mag**2
        return rho, u, et
        

    @override
    def Solve(self) -> None:
        """
        Solve the system explicitly in time.
        """
        nIter = self.config.GetNIterations()
        time_physical = 0
        start = time.time()
        u_advection = self.config.GetAdvectionVelocity()
        theta = np.linspace(0, 2*np.pi*4, nIter)

        fig, ax = plt.subplots()
        cbar = None  # Initialize colorbar reference
        for it in range(nIter):            
            if self.config.GetAdvectionRotation():
                u_adv = np.array([np.cos(theta[it])*u_advection[0],
                                np.sin(theta[it])*u_advection[0],
                                u_advection[2]])
            else:
                u_adv = u_advection
                
            self.ComputeTimeStepArray()
            dt = np.min(self.time_step)
            
            residuals = np.zeros_like(self.solution)  # defined as flux*surface going out of the cell (i,j,k)
            self.CheckConvergence(self.solution, it+1)
            
            # i-fluxes
            niF, njF, nkF = self.mesh.Si[:, :, :, 0].shape
            for iFace in range(niF):
                for jFace in range(njF):
                    for kFace in range(nkF):
                        if iFace==0: 
                            U_r = self.solution[iFace, jFace, kFace,:]
                            if self.boundary_types['i']['begin']=='transparent':
                                U_l = U_r.copy() 
                            elif self.boundary_types['i']['begin']=='periodic':
                                U_l = self.solution[-1, jFace, kFace,:]
                            else:
                                raise ValueError('Unknown boundary condition at ni=0')
                            S = self.mesh.Si[iFace, jFace, kFace, :]
                            scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace, jFace, kFace, :] -= flux*area          
                        elif iFace==niF-1:
                            U_l = self.solution[iFace-1, jFace, kFace, :]  
                            if self.boundary_types['i']['end']=='transparent':
                                U_r = U_r.copy()
                            elif self.boundary_types['i']['end']=='periodic':
                                U_r = self.solution[0, jFace, kFace, :]
                            else:
                                raise ValueError('Unknown boundary condition at ni=0')
                            S = self.mesh.Si[iFace, jFace, kFace, :]                
                            scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace-1, jFace, kFace, :] += flux*area        
                        else:
                            U_l = self.solution[iFace-1, jFace, kFace,:]
                            U_r = self.solution[iFace, jFace, kFace,:]  
                            S = self.mesh.Si[iFace, jFace, kFace, :]
                            scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
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
                            U_r = self.solution[iFace, jFace, kFace,:]
                            if self.boundary_types['j']['begin']=='transparent':
                                U_l = U_r.copy() 
                            elif self.boundary_types['j']['begin']=='periodic':
                                U_l = self.solution[iFace, -1, kFace,:]
                            else:
                                raise ValueError('Unknown boundary condition at nj=0')
                            S = self.mesh.Sj[iFace, jFace, kFace, :]
                            scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace, jFace, kFace, :] -= flux*area          
                        elif jFace==njF-1:
                            U_l = self.solution[iFace, jFace-1, kFace, :]      
                            if self.boundary_types['j']['end']=='transparent':
                                U_r = U_l.copy()
                            elif self.boundary_types['j']['end']=='periodic':
                                U_r = self.solution[iFace, 0, kFace,:]
                            else:
                                raise ValueError('Unknown boundary condition at j=nj')
                            S = self.mesh.Sj[iFace, jFace, kFace, :]                
                            scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
                            flux = scheme.ComputeFlux()
                            area = np.linalg.norm(S)
                            residuals[iFace, jFace-1, kFace, :] += flux*area        
                        else:
                            U_l = self.solution[iFace, jFace-1, kFace,:]
                            U_r = self.solution[iFace, jFace, kFace,:]  
                            S = self.mesh.Sj[iFace, jFace, kFace, :]
                            scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
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
                                U_r = self.solution[iFace, jFace, kFace,:]
                                if self.boundary_types['k']['begin']=='transparent':
                                    U_l = U_r.copy() 
                                elif self.boundary_types['k']['begin']=='periodic':
                                    U_l = self.solution[iFace, jFace, -1,:]
                                else:
                                    raise ValueError('Unknown boundary condition at nk=0')
                                S = self.mesh.Sk[iFace, jFace, kFace, :]
                                scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
                                flux = scheme.ComputeFlux()
                                area = np.linalg.norm(S)
                                residuals[iFace, jFace, kFace, :] -= flux*area          
                            elif kFace==nkF-1:
                                U_l = self.solution[iFace, jFace, kFace-1, :]       
                                if self.boundary_types['k']['end']=='transparent':
                                    U_r = U_l.copy()
                                elif self.boundary_types['k']['end']=='periodic':
                                    U_r = self.solution[iFace, jFace, 0,:]
                                else:
                                    raise ValueError('Unknown boundary condition at k=nk')
                                S = self.mesh.Sk[iFace, jFace, kFace, :]                
                                scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
                                flux = scheme.ComputeFlux()
                                area = np.linalg.norm(S)
                                residuals[iFace, jFace, kFace-1, :] += flux*area        
                            else:
                                U_l = self.solution[iFace, jFace, kFace-1,:]
                                U_r = self.solution[iFace, jFace, kFace,:]  
                                S = self.mesh.Sk[iFace, jFace, kFace, :]
                                scheme = CScheme_Upwind(U_l, U_r, S, u_adv)
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
            # u_quiver = u_adv[0]+np.zeros_like(self.mesh.X[:, :, 0])
            # v_quiver = u_adv[1]+np.zeros_like(self.mesh.X[:, :, 0])
            # ax.quiver(self.mesh.X[:, :, 0], self.mesh.Y[:, :, 0], u_quiver, v_quiver, color='red')
            ax.set_aspect('equal')
            plt.pause(0.001)

        end = time.time()
        self.PlotResidualsHistory()


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

        

    def ComputeJSTFlux(self, Ull: np.ndarray, Ul: np.ndarray, Ur: np.ndarray, Urr: np.ndarray, S: np.ndarray) -> np.ndarray :
        """
        Compute the vector flux between the left and right points defined by their conservative vectors. 
        The surface vector oriented from left to right.
        Formulation taken from `The Origins and Further Development of the Jameson-Schmidt-Turkel (JST) Scheme`, by Jameson.
        
        Parameters
        ---------------------

        `Ull`: conservative vector of the node to the left of the left node

        `Ul`: conservative vector of the node to the left
        
        `Ur`: conservative vector of the node to the right 

        `Urr`: conservative vector of the node to the right of the right node

        `S`: surface vector going from left to right point
        """
        jst = CSchemeJST(self.fluid, Ull, Ul, Ur, Urr, S)
        flux = jst.ComputeFluxJameson()
        
        # check if the two versions give the same results
        # flux2 = jst.ComputeFluxJameson()
        # if np.linalg.norm(flux2-flux)/np.linalg.norm(flux)>1e-1:
        #     print("Blazek: ", flux)
        #     print("Jameson: ", flux2)
        #     print("norm difference: ", np.linalg.norm(flux2-flux)/np.linalg.norm(flux))
        #     raise ValueError('The JST fluxes differ')
        
        return flux
        
    
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
            self.PrintMassFlowPlot()
            self.PrintResidualsPlot()
            plt.show()
            raise ValueError("The simulation diverged. Nan found at iteration %i" %(nIter))
        
        if array[:,:,:,0].any()<=0:
            self.PrintMassFlowPlot()
            self.PrintResidualsPlot()
            plt.show()
            raise ValueError("The simulation diverged. Negative density at iteration %i" %(nIter))


    def ComputeMassFlow(self, direction: str, location: str) -> float:
        """
        Compute the mass flow passing through a boundary defined by direction and location
        
        Parameters
        -------------------------

        `direction`: specify i,j,k

        `location`: specify if `start` or `end` of the direction
        """
        if direction=='i' and location=='start':
            rhoU = self.solution[0,:,:,1:-1]
            S = self.mesh.Si[0,:,:,:]
        elif direction=='i' and location=='end':
            rhoU = self.solution[-1,:,:,1:-1]
            S = self.mesh.Si[-1,:,:,:]
        elif direction=='j' and location=='start':
            rhoU = self.solution[:,0,:,1:-1]
            S = self.mesh.Sj[:,0,:,:]
        elif direction=='j' and location=='end':
            rhoU = self.solution[:,-1,:,1:-1]
            S = self.mesh.Sj[:,-1,:,:]
        elif direction=='k' and location=='start':
            rhoU = self.solution[:,:,0,1:-1]
            S = self.mesh.Sk[:,:,0,:]
        elif direction=='k' and location=='end':
            rhoU = self.solution[:,:,-1,1:-1]
            S = self.mesh.Sk[:,:,-1,:]
        else:
            raise ValueError('Direction and location dont correspond to acceptable values')
        
        mass_flow = 0
        ni, nj = S[:,:,0].shape
        for i in range(ni):
            for j in range(nj):
                area = np.linalg.norm(S[i,j,:])
                normal = S[i,j,:]/area
                rhoU_n = np.dot(rhoU[i,j,:], normal)
                mass_flow += rhoU_n*area
        
        return mass_flow


    def PrintMassFlowPlot(self):
        """
        Print the plot of the mass passing through boundaries
        """
        plt.figure()
        plt.plot(self.mi_in, label=r'$\dot{m}_{i,in}$')
        plt.plot(self.mi_out, label=r'$\dot{m}_{i,out}$')
        plt.plot(self.mj_in, label=r'$\dot{m}_{j,in}$')
        plt.plot(self.mj_out, label=r'$\dot{m}_{j,out}$')
        plt.plot(self.mk_in, label=r'$\dot{m}_{k,in}$')
        plt.plot(self.mk_out, label=r'$\dot{m}_{k,out}$')
        plt.xlabel('iteration [-]')
        plt.ylabel('mass flow [kg/s]')
        plt.legend()
        plt.grid(alpha=0.3)
    

    def ConvertSolutionToPrimitives(self, conservatives) -> np.ndarray:
        """
        Compute primitive variables from conservatives.
        """
        primitives = np.zeros_like(conservatives)
        ni, nj, nk = primitives[:,:,:,0].shape
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    primitives[i,j,k,:] = GetPrimitivesFromConservatives(conservatives[i,j,k,:])
        return primitives
    
    


