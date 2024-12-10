import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.euler_functions import GetConservativesFromPrimitives, GetPrimitivesFromConservatives
from TurboBFM.Solver.CScheme_JST import CSchemeJST
from TurboBFM.Solver.CBoundaryCondition import CBoundaryCondition
from TurboBFM.Solver.CSolver import CSolver
# from TurboBFM.Postprocess import styles
from typing import override 


class CEulerSolver(CSolver):
    
    def __init__(self, config: CConfig, mesh: CMesh):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        super().__init__(config, mesh)

        self.fluidName = self.config.GetFluidName()
        self.fluidGamma = self.config.GetFluidGamma()
        self.fluidModel = self.config.GetFluidModel()
        self.fluidR = self.config.GetFluidRConstant()

        if self.fluidModel.lower()=='ideal':
            self.fluid = FluidIdeal(self.fluidGamma, self.fluidR)
        elif self.fluidModel.lower()=='real':
            raise ValueError('Real Fluid Model not implemented')
        else:
            raise ValueError('Unknown Fluid Model')
    

    @override
    def PrintInfoSolver(self):
        """
        Print basic information before running the solver
        """
        dir = self.config.GetInitDirection()

        if self.verbosity>0:
            print('='*25 + ' SOLVER INFORMATION ' + '='*25)
            print('Number of dimensions:                    %i' %(self.nDim))
            print('Fluid name:                              %s' %(self.fluidName))
            print('Fluid cp/cv ratio [-]:                   %.2f' %(self.fluidGamma))
            print('Fluid model:                             %s' %(self.fluidModel))
            print('Fluid R constant [J/kgK]:                %s' %(self.fluidR))
            print('Initial Mach [-]:                        %.2f' %(self.config.GetInitMach()))
            print('Initial Temperature [K]:                 %.2f' %(self.config.GetInitTemperature()))
            print('Initial Pressure [kPa]:                  %.2f' %(self.config.GetInitPressure()/1e3))
            print('Initial flow direction [-]:              (%.2f, %.2f, %.2f)' %(dir[0], dir[1], dir[2]))
            print('Boundary type at i=0:                    %s' %(self.GetBoundaryCondition('i', 'begin')[0]))
            print('Boundary type at i=ni:                   %s' %(self.GetBoundaryCondition('i', 'end')[0]))
            print('Boundary type at j=0:                    %s' %(self.GetBoundaryCondition('j', 'begin')[0]))
            print('Boundary type at j=nj:                   %s' %(self.GetBoundaryCondition('j', 'end')[0]))
            if self.nDim==3:
                print('Boundary type at k=0:                    %s' %(self.GetBoundaryCondition('k', 'begin')[0]))
                print('Boundary type at k=nk:                   %s' %(self.GetBoundaryCondition('k', 'end')[0]))
            print('Inlet Total Pressure [kPa]:              %.2f' %(self.config.GetInletValue()[0]/1e3))
            print('Inlet Total Temperature [K]:             %.2f' %(self.config.GetInletValue()[1]))
            print('Inlet flow direction [-]:                (%.2f, %.2f, %.2f)' %(self.config.GetInletValue()[2], self.config.GetInletValue()[3], self.config.GetInletValue()[4]))
            print('Outlet static pressure [kPa]:            %.2f' %(self.config.GetOutletValue()/1e3))
            print('='*25 + ' END SOLVER INFORMATION ' + '='*25)
            print()


    @override
    def InstantiateFields(self):
        """
        Instantiate basic fields.
        """
        super().InstantiateFields()
        self.solution_names = [r'$\rho$',
                               r'$\rho u_x$',
                               r'$\rho u_y$',
                               r'$\rho u_z$',
                               r'$\rho e_t$']
        self.mi_in = []                                                 # store the mass flow entering from i
        self.mi_out = []                                                # store the mass flow exiting from i
        self.mj_in = []                                                 # store the mass flow entering from j
        self.mj_out = []                                                # store the mass flow exiting from j
        self.mk_in = []                                                 # store the mass flow entering from k
        self.mk_out = []                                                # store the mass flow exiting from k


    @override
    def ReadBoundaryConditions(self):
        """
        Read the boundary conditions from the input file, and store the information in two dictionnaries
        """
        super().ReadBoundaryConditions()
        
        self.boundary_values = {'i': {},
                               'j': {},
                               'k': {}}
        
        for direction, station in self.boundary_types.items():
            for location, type in station.items():
                if type=='inlet':
                    self.boundary_values[direction][location] = self.config.GetInletValue()
                elif type=='outlet':
                    self.boundary_values[direction][location] = self.config.GetOutletValue()
                elif type=='wall' or type=='empty':
                    self.boundary_values[direction][location] = None
                elif type=='periodic':
                    self.boundary_values[direction][location] = self.config.GetPeriodicValue()
                else:
                    raise ValueError('Unknown type of boundary condition on direction <%s> at location <%s>' %(direction, location))


    @override
    def InitializeSolution(self):
        """
        Given the boundary conditions, initialize the solution as to be associated with them
        """
        M = self.config.GetInitMach()
        T = self.config.GetInitTemperature()
        P = self.config.GetInitPressure()
        dir = self.config.GetInitDirection()
        rho, u , et = self.ComputeInitFields(M, T, P, dir)
        primitives = np.zeros((self.ni, self.nj, self.nk, 5))
        primitives[:,:,:,0] = rho
        primitives[:,:,:,1] = u[0]
        primitives[:,:,:,2] = u[1]
        primitives[:,:,:,3] = u[2]
        primitives[:,:,:,4] = et

        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    self.solution[i,j,k,:] = GetConservativesFromPrimitives(primitives[i,j,k,:])


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
            fields = self.ConvertSolutionToPrimitives(self.solution)
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
           

    def SpatialIntegration(self, dir, Res):
        """
        Perform spatial integration loop in a certain direction. 

        Parameters
        -------------------------

        `dir`: i,j or k

        `res`: residual arrays of the current time-step that will be updated
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
                        bc_type, bc_value = self.GetBoundaryCondition(dir, 'begin')
                        Ub = self.solution[iFace, jFace, kFace, :]   
                        Uint = self.solution[iFace+1*step_mask[0], jFace+1*step_mask[1], kFace+1*step_mask[2], :]     
                        S = -Surf[iFace, jFace, kFace, :]               
                        boundary = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, self.fluid)
                        flux = boundary.ComputeFlux()
                        area = np.linalg.norm(S)
                        Res[iFace, jFace, kFace, :] += flux*area          
                    elif dir_face==stop_face:
                        bc_type, bc_value = self.GetBoundaryCondition(dir, 'end')
                        Ub = self.solution[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :]      
                        Uint = self.solution[iFace-2*step_mask[0], jFace-2*step_mask[1], kFace-2*step_mask[2], :]     
                        S = Surf[iFace, jFace, kFace, :]                
                        boundary = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, self.fluid)
                        flux = boundary.ComputeFlux()
                        area = np.linalg.norm(S)
                        Res[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :] += flux*area       
                    else:
                        U_l = self.solution[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2],:]
                        U_r = self.solution[iFace, jFace, kFace,:]  
                        if dir_face==1:
                            U_ll = U_l
                            U_rr = self.solution[iFace+1*step_mask[0], jFace+1*step_mask[1], kFace+1*step_mask[2],:]
                        elif dir_face==stop_face-1:
                            U_ll = self.solution[iFace-2*step_mask[0], jFace-2*step_mask[1], kFace-2*step_mask[2],:]
                            U_rr = U_r
                        else:
                            U_ll = self.solution[iFace-2*step_mask[0], jFace-2*step_mask[1], kFace-2*step_mask[2],:]
                            U_rr = self.solution[iFace+1*step_mask[0], jFace+1*step_mask[1], kFace+1*step_mask[2],:]
                        S = Surf[iFace, jFace, kFace, :]
                        area = np.linalg.norm(S)
                        flux = self.ComputeJSTFlux(U_ll, U_l, U_r, U_rr, S)
                        Res[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :] += flux*area 
                        Res[iFace, jFace, kFace, :] -= flux*area


    @override
    def Solve(self) -> None:
        """
        Solve the system explicitly in time.
        """
        nIter = self.config.GetNIterations()
        time_physical = 0
        start = time.time()

        for it in range(nIter):            
            self.ComputeTimeStepArray()
            dt = np.min(self.time_step)
            
            self.mi_in.append(self.ComputeMassFlow('i', 'start'))
            self.mi_out.append(self.ComputeMassFlow('i', 'end'))
            self.mj_in.append(self.ComputeMassFlow('j', 'start'))
            self.mj_out.append(self.ComputeMassFlow('j', 'end'))
            self.mk_in.append(self.ComputeMassFlow('k', 'start'))
            self.mk_out.append(self.ComputeMassFlow('k', 'end'))
            
            residuals = np.zeros_like(self.solution)
            self.CheckConvergence(self.solution, it+1)
            
            self.SpatialIntegration('i', residuals)
            self.SpatialIntegration('j', residuals)
            if self.nDim==3:
                self.SpatialIntegration('k', residuals)
            
            self.PrintInfoResiduals(residuals, it, time_physical)
            time_physical += dt
            if self.verbosity==3:
                self.ContoursCheckMeridional('primitives')
                # self.ContoursCheckResiduals(residuals)
                plt.show()

            for iEq in range(5):
                self.solution[:,:,:,iEq] -= residuals[:,:,:,iEq]*dt/self.mesh.V[:,:,:]  # update the conservative solution

        self.ContoursCheckMeridional('conservatives')
        end = time.time()
        self.PrintMassFlowPlot()
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
        res = np.zeros(5)
        for i in range(5):
            res[i] = np.linalg.norm(residuals[:,:,:,i].flatten())/len(residuals[:,:,:,i].flatten())
            if res[i]!=0:
                res[i] = np.log10(res[i])
        if it==0:
        # Header
            print("|" + "-" * ((col_width)*7+6) + "|")
            print(f"{'|'}{'Iteration':<{col_width}}{'|'}{'Time[s]':<{col_width}}{'|'}{'rms[Rho]':>{col_width}}{'|'}{'rms[RhoU]':>{col_width}}{'|'}{'rms[RhoV]':>{col_width}}{'|'}{'rms[RhoW]':>{col_width}}{'|'}{'rms[RhoE]':>{col_width}}{'|'}")
            print("|" + "-" * ((col_width)*7+6) + "|")

        # Data row
        print(
            f"{'|'}{it:<{col_width}}{'|'}{time:<{col_width}.3e}{'|'}{res[0]:>{col_width}.6f}{'|'}{res[1]:>{col_width}.6f}{'|'}{res[2]:>{col_width}.6f}{'|'}{res[3]:>{col_width}.6f}{'|'}{res[4]:>{col_width}.6f}{'|'}"
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
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    i_edge, j_edge, k_edge = self.mesh.GetElementEdges((i,j,k))
                    i_dir = i_edge/np.linalg.norm(i_edge)
                    j_dir = j_edge/np.linalg.norm(j_edge)
                    k_dir = k_edge/np.linalg.norm(k_edge)
                    W = GetPrimitivesFromConservatives(self.solution[i,j,k,:])
                    vel = W[1:-1]
                    rho = W[0]
                    et = W[-1]
                    a = self.fluid.ComputeSoundSpeed_rho_u_et(rho, vel, et)

                    dt_i = np.linalg.norm(i_edge) / (np.abs(np.dot(vel, i_dir))+a)
                    dt_j = np.linalg.norm(j_edge) / (np.abs(np.dot(vel, j_dir))+a)
                    dt_k = np.linalg.norm(k_edge) / (np.abs(np.dot(vel, k_dir))+a)

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
    
    


