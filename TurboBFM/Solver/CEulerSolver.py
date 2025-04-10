import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.euler_functions import GetConservativesFromPrimitives, GetPrimitivesFromConservatives
from TurboBFM.Solver.CScheme_JST import CScheme_JST
from TurboBFM.Solver.CScheme_Roe import CScheme_Roe
from TurboBFM.Solver.CBoundaryCondition import CBoundaryCondition
from TurboBFM.Solver.CSolver import CSolver
from TurboBFM.Solver.CBFMSource import CBFMSource
from TurboBFM.Postprocess import styles
from TurboBFM.Solver.math import ComputeCartesianVectorFromCylindrical, ComputeCylindricalVectorFromCartesian, IntegrateScalarFlux, IntegrateVectorFlux
from scipy.integrate import odeint
from scipy.interpolate import interp1d, griddata
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
            raise ValueError('Real Fluid Model not implemented yet')
        else:
            raise ValueError('Unknown Fluid Model')
        
        self.radial_pressure_profile = None
    

    @override
    def PrintInfoSolver(self):
        """
        Print basic information before running the solver
        """
        dir = self.config.GetInitDirection()

        if self.verbosity>0:
            print('='*25 + ' SOLVER INFORMATION ' + '='*25)
            print('Number of dimensions:                    %i' %(self.nDim))
            print('Mesh topology:                           %s' %(self.config.GetTopology()))
            print('Fluid name:                              %s' %(self.fluidName))
            print('Fluid Gamma [-]:                         %.2f' %(self.fluidGamma))
            print('Fluid model:                             %s' %(self.fluidModel))
            print('Fluid R constant [J/kgK]:                %s' %(self.fluidR))
            print('Initial Mach [-]:                        %.2f' %(self.config.GetInitMach()))
            print('Initial Temperature [K]:                 %.2f' %(self.config.GetInitTemperature()))
            print('Initial Pressure [kPa]:                  %.2f' %(self.config.GetInitPressure()/1e3))
            print('Initial flow direction [-]:              %s' %(dir))
            print('Boundary type at i=0:                    %s' %(self.GetBoundaryCondition('i', 'begin', 1e6)[0]))
            print('Boundary type at i=ni:                   %s' %(self.GetBoundaryCondition('i', 'end', 1e6)[0]))
            print('Boundary type at j=0:                    %s' %(self.GetBoundaryCondition('j', 'begin', 1e6)[0]))
            print('Boundary type at j=nj:                   %s' %(self.GetBoundaryCondition('j', 'end', 1e6)[0]))
            if self.nDim==3 or (self.nDim==2 and self.config.GetTopology()=='axisymmetric'):
                print('Boundary type at k=0:                    %s' %(self.GetBoundaryCondition('k', 'begin', 1e6)[0]))
                print('Boundary type at k=nk:                   %s' %(self.GetBoundaryCondition('k', 'end', 1e6)[0]))
            print('Inlet Total Pressure [kPa]:              %.2f' %(self.config.GetInletValue()[0]/1e3))
            print('Inlet Total Temperature [K]:             %.2f' %(self.config.GetInletValue()[1]))
            print('Inlet flow direction [-]:                (%.2f, %.2f, %.2f)' %(self.config.GetInletValue()[2], self.config.GetInletValue()[3], self.config.GetInletValue()[4]))
            print('Outlet static pressure [kPa]:            %.2f' %(self.config.GetOutletValue()/1e3))
            print('Time Integration method:                 %s' %(self.config.GetTimeIntegrationType()))
            print('CFL used:                                %.2f' %(self.config.GetCFL()))
            if self.config.GetTimeStepGlobal():
                print('Delta time method:                       global')
            else:
                print('Delta time method:                       local')
            print('Convection scheme:                       %s' %(self.config.GetConvectionScheme()))
            print('Total number of iterations:              %s' %(self.config.GetNIterations()))
            if self.config.GetSaveUnsteady():
                print('Solution saved every:                    %i intervals' %(self.config.GetSaveUnsteadyInterval()))
            if self.config.GetRestartSolution():
                print('Solution restarted from file:            %s' %(self.config.GetRestartSolutionFilepath()))
                print('Solution restarted from iteration:       %i' %(self.restart_iterations))
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

        if self.config.GetTurboOutput():
            self.beta_tt = []                                               # store the total pressure ratio for turbo cases
            self.tau_tt = []                                                # store the total temperature ratio for turbo cases
            self.eta_tt = []                                                # store the total efficiency ratio for turbo cases
            self.m_turbo = []                                               # store the mass flow rate of the machine


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
                if type=='inlet' or type=='inlet_ss':
                    self.boundary_values[direction][location] = self.config.GetInletValue()
                    self.inlet_bc_type = self.config.GetInletBCType()
                elif type=='outlet' or type=='outlet_ss' or type=='outlet_re' or type=='outlet_mf':
                    self.boundary_values[direction][location] = self.config.GetOutletValue()
                elif type=='wall' or type=='empty' or type=='wedge':
                    self.boundary_values[direction][location] = None
                elif type=='periodic':
                    self.boundary_values[direction][location] = self.config.GetPeriodicValue()
                else:
                    raise ValueError('Unknown type of boundary condition on direction <%s> at location <%s>' %(direction, location))


    @override
    def InitializeSolution(self):
        """
        Given the boundary conditions, initialize the solution to be associated with them
        """
        if self.config.GetRestartSolution()==False:
            M = self.config.GetInitMach()
            T = self.config.GetInitTemperature()
            P = self.config.GetInitPressure()
            dir = self.config.GetInitDirection()
            rho, uMagnitude , et = self.ComputeInitFields(M, T, P)
            
            primitives = np.zeros((self.ni, self.nj, self.nk, 5))
            primitives[:,:,:,0] = rho
            
            if isinstance(dir, np.ndarray):
                initDirection = dir/np.linalg.norm(dir)
                primitives[:,:,:,1] = uMagnitude*initDirection[0]
                primitives[:,:,:,2] = uMagnitude*initDirection[1]
                primitives[:,:,:,3] = uMagnitude*initDirection[2]
            elif dir=='adaptive':
                # the flow direction follows the i-direction of the grid
                initDirection = self.mesh.getGridDirection('i')
                primitives[:,:,:,1] = uMagnitude*initDirection[:,:,:,0]
                primitives[:,:,:,2] = uMagnitude*initDirection[:,:,:,1]
                primitives[:,:,:,3] = uMagnitude*initDirection[:,:,:,2]
            
            primitives[:,:,:,4] = et

            for i in range(self.ni):
                for j in range(self.nj):
                    for k in range(self.nk):
                        self.solution[i,j,k,:] = GetConservativesFromPrimitives(primitives[i,j,k,:])
        
        elif self.config.GetRestartSolution()==True:
            filepath = self.config.GetRestartSolutionFilepath()
            with open(filepath, 'rb') as file:
                restartData = pickle.load(file)
            
            shapeRestartFile = restartData['U'].shape
            shapeCurrentFile = self.solution.shape
            
            # initialize the fields with the restart file
            if shapeCurrentFile==shapeRestartFile:
                self.solution = restartData['U']
            else:
                print()
                print('WARNING: The restart file has different dimensions than the current mesh. Nearest Neighbor interpolation is used.')
                print()
                
                xPoints = restartData['X']
                yPoints = restartData['Y']
                zPoints = restartData['Z']
                points = (xPoints.flatten(), yPoints.flatten(), zPoints.flatten())
                
                xEval = self.mesh.X.flatten()
                yEval = self.mesh.Y.flatten()
                zEval = self.mesh.Z.flatten()
                
                for i in range(5):
                    values = (restartData['U'][:,:,:,i]).flatten()
                    interpValues = griddata(points, values, (xEval, yEval, zEval), method='nearest')
                    self.solution[:,:,:,i] = np.reshape(interpValues, (self.ni, self.nj, self.nk))    
            
            # update the iteration number
            its = filepath.split('.pik')
            self.restart_iterations = int(its[0][-6:])

        else:
            raise ValueError('Check the restart files')



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
                cnt = ax[iplot][jplot].contourf(X, Y, fields[:,:,idx_cut,iField], cmap=styles.color_map, levels=styles.N_levels)
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

        
    def ComputeInitFields(self, M: float, T: float, P: float):
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
        rho = P/R/T
        et = (P / (gmma - 1) / rho) + 0.5*u_mag**2
        return rho, u_mag, et
           
    @override
    def SpatialIntegration(self, dir, Sol, Res, iterationCounter):
        """
        Perform spatial integration loop in a certain direction. 

        Parameters
        -------------------------

        `dir`: i,j or k

        `Sol`: array of the solution to use

        `Res`: residual arrays of the current time-step that will be updated
        
        `iterationCounter`: int, needed to get ramp coefficients
        """
        if dir=='i':
            step_mask = np.array([1, 0, 0])
            Surf = self.mesh.Si
            midpS = self.mesh.CGi
        elif dir=='j':
            step_mask = np.array([0, 1, 0])
            Surf = self.mesh.Sj
            midpS = self.mesh.CGj
        else:
            step_mask = np.array([0, 0, 1])
            Surf = self.mesh.Sk
            midpS = self.mesh.CGk
        
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
                        bc_type, bc_value = self.GetBoundaryCondition(dir, 'begin', iterationCounter)
                        Ub = Sol[iFace, jFace, kFace, :]   
                        S = -Surf[iFace, jFace, kFace, :]
                        CG = midpS[iFace, jFace, kFace, :]       
                        boundary = CBoundaryCondition(bc_type, bc_value, Ub, S, CG, 
                                                      self.radial_pressure_profile, jFace,
                                                      self.fluid, self.mesh.boundary_areas[dir]['begin'], self.inlet_bc_type)
                        flux = boundary.ComputeFlux()
                        area = np.linalg.norm(S)
                        Res[iFace, jFace, kFace, :] += flux*area          
                    elif dir_face==stop_face:
                        bc_type, bc_value = self.GetBoundaryCondition(dir, 'end', iterationCounter)
                        Ub = Sol[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :]      
                        S = Surf[iFace, jFace, kFace, :]  
                        CG = midpS[iFace, jFace, kFace, :]                   
                        boundary = CBoundaryCondition(bc_type, bc_value, Ub, S, CG, 
                                                      self.radial_pressure_profile, jFace-1*step_mask[1],
                                                      self.fluid, self.mesh.boundary_areas[dir]['end'], self.inlet_bc_type)
                        flux = boundary.ComputeFlux()
                        area = np.linalg.norm(S)
                        Res[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :] += flux*area       
                    else:
                        U_l = Sol[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2],:]
                        U_r = Sol[iFace, jFace, kFace,:]  
                        if dir_face==1:
                            U_ll = U_l
                            U_rr = Sol[iFace+1*step_mask[0], jFace+1*step_mask[1], kFace+1*step_mask[2],:]
                        elif dir_face==stop_face-1:
                            U_ll = Sol[iFace-2*step_mask[0], jFace-2*step_mask[1], kFace-2*step_mask[2],:]
                            U_rr = U_r
                        else:
                            U_ll = Sol[iFace-2*step_mask[0], jFace-2*step_mask[1], kFace-2*step_mask[2],:]
                            U_rr = Sol[iFace+1*step_mask[0], jFace+1*step_mask[1], kFace+1*step_mask[2],:]
                        S = Surf[iFace, jFace, kFace, :]
                        area = np.linalg.norm(S)
                        flux = self.ComputeFlux(U_ll, U_l, U_r, U_rr, S)
                        Res[iFace-1*step_mask[0], jFace-1*step_mask[1], kFace-1*step_mask[2], :] += flux*area 
                        Res[iFace, jFace, kFace, :] -= flux*area
    

    def CorrectBoundaryVelocities(self, sol):
        """
        For all nodes on wall boundaries, correct the velocities to be tangential to the boundaries.
        """
        dirs = ['i', 'j', 'k']
        locs = ['begin', 'end']
        
        for dir in dirs:
            for loc in locs:
                bc_type, bc_value = self.GetBoundaryCondition(dir, loc)
                if bc_type=='wall':
                    sol = self.CorrectWallVelocities(dir, loc, sol)
        
        return sol
    

    def CorrectWallVelocities(self, dir, loc, sol):
        """
        Correct the velocities on a boundary to be tangential. 

        Parameters
        -------------------------

        `dir`: i,j,k

        `loc`: begin or end

        `sol`: solution array to correct
        """
        
        def project_vel(cons, S):
            prim = GetPrimitivesFromConservatives(cons)
            vel = prim[1:-1]
            S_dir = S/np.linalg.norm(S)
            vel_new = vel-np.dot(vel,S_dir)*S_dir
            prim_new = prim.copy()
            prim_new[1:-1] = vel_new
            prim_new[-1] = prim[-1] - 0.5*np.linalg.norm(vel)**2 + 0.5*np.linalg.norm(vel_new)**2
            cons_new = GetConservativesFromPrimitives(prim_new)
            return cons_new
        
        if dir=='i' and loc=='begin':
            U = sol[0,:,:,:]
            S = -self.mesh.Si[0,:,:,:]
            nj,nk = U[0,:,:,0].shape
            for j in range(nj):
                for k in range(nk):
                    sol[0,j,k,:] = project_vel(U[j,k,:], S[j,k,:])
        elif dir=='i' and loc=='end':
            U = sol[-1,:,:,:]
            S = self.mesh.Si[-1,:,:,:]
            nj,nk = U[-1,:,:,0].shape
            for j in range(nj):
                for k in range(nk):
                    sol[-1,j,k,:] = project_vel(U[j,k,:], S[j,k,:])
        elif dir=='j' and loc=='begin':
            U = sol[:,0,:,:]
            S = -self.mesh.Sj[:,0,:,:]
            ni,nk = U[:,:,0].shape
            for i in range(ni):
                for k in range(nk):
                    sol[i,0,k,:] = project_vel(U[i,k,:], S[i,k,:])
        elif dir=='j' and loc=='end':
            U = sol[:,-1,:,:]
            S = self.mesh.Sj[:,-1,:,:]
            ni,nk = U[:,:,0].shape
            for i in range(ni):
                for k in range(nk):
                    sol[i,-1,k,:] = project_vel(U[i,k,:], S[i,k,:])
        elif dir=='k' and loc=='begin':
            U = sol[:,:,0,:]
            S = -self.mesh.Sj[:,:,0,:]
            ni,nj = U[:,:,0].shape
            for i in range(ni):
                for j in range(nj):
                    sol[i,j,0,:] = project_vel(U[i,j,:], S[i,j,:])
        elif dir=='k' and loc=='end':
            U = sol[:,:,-1,:]
            S = self.mesh.Sj[:,:,-1,:]
            ni,nj = U[:,:,0].shape
            for i in range(ni):
                for j in range(nj):
                    sol[i,j,-1,:] = project_vel(U[i,j,:], S[i,j,:])
        else:
            raise ValueError("Unknow combination of location and direction")
        
        return sol
        
        
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
        if it==0 or self.residualFlag is True:
            # Header
            print("|" + "-" * ((col_width)*7+6) + "|")
            print(f"{'|'}{'Iteration':<{col_width}}{'|'}{'Time[s]':<{col_width}}{'|'}{'rms[Rho]':>{col_width}}{'|'}{'rms[RhoU]':>{col_width}}{'|'}{'rms[RhoV]':>{col_width}}{'|'}{'rms[RhoW]':>{col_width}}{'|'}{'rms[RhoE]':>{col_width}}{'|'}")
            print("|" + "-" * ((col_width)*7+6) + "|")
            self.residualFlag = False # reset the flag to not print the header every time

        # Data row
        print(
            f"{'|'}{it:<{col_width}}{'|'}{time:<{col_width}.3e}{'|'}{res[0]:>{col_width}.6f}{'|'}{res[1]:>{col_width}.6f}{'|'}{res[2]:>{col_width}.6f}{'|'}{res[3]:>{col_width}.6f}{'|'}{res[4]:>{col_width}.6f}{'|'}"
        )

        # bookkeep the residuals for final plot
        for iEq in range(self.nEq):
            self.residual_history[iEq].append(res[iEq])
    

    def ComputeFlux(self, Ull: np.ndarray, Ul: np.ndarray, Ur: np.ndarray, Urr: np.ndarray, S: np.ndarray) -> np.ndarray :
        """
        Compute the vector flux between the left and right points defined by their conservative vectors. 
        The surface vector oriented from left to right.
        
        Parameters
        ---------------------

        `Ull`: conservative vector of the node to the left of the left node

        `Ul`: conservative vector of the node to the left
        
        `Ur`: conservative vector of the node to the right 

        `Urr`: conservative vector of the node to the right of the right node

        `S`: surface vector going from left to right point
        """
        if self.conv_scheme.lower()=='jst':
            jst = CScheme_JST(self.fluid, Ull, Ul, Ur, Urr, S)
            flux = jst.ComputeFluxJameson()
        elif self.conv_scheme.lower()=='roe':
            roe = CScheme_Roe(Ul, Ur, S, self.fluid)
            flux = roe.ComputeFlux()
        
        return flux
        
    
    @override
    def ComputeTimeStepArray(self, sol):
        """
        Compute the time step of the simulation for a certain CFL and conservative solution

        Parameters
        -------------------------------

        `sol`: conservative solution array
        """
        CFL = self.config.GetCFL()
        timestep = np.zeros((self.ni, self.nj, self.nk))
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
                        timestep[i,j,k] = CFL*min(dt_i, dt_j, dt_k)
                    else:
                        timestep[i,j,k] = CFL*min(dt_i, dt_j)
        if self.config.GetTimeStepGlobal():
            return np.min(timestep) # the minimum of all
        else:
            return timestep  # local per elelement, lose of time coherency



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
        
        if array[:,:,:,0].any()<=0:
            raise ValueError("The simulation diverged. Negative density at iteration %i" %(nIter))


    def ComputeMassFlow(self, direction: str, location: str) -> float:
        """
        Compute the mass flow passing through a boundary defined by direction and location. Just keep in mind that if the simulation is axisymmetric, 
        the mass flows computed on the k-boundaries are not correct, since the velocity vectors don't lie on those boundaries, and the mass flow would therefore
        be wrongly computed. In any case their behavior should be similar, given the projection vector of those boundaries.
        
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
        
        mass_flow = IntegrateVectorFlux(rhoU, S)
        return mass_flow
    

    def ComputeTurboOutput(self) -> float:
        """
        Compute the total pressure ratio for a turbomachinery case
        """
        direction = self.config.GetTurboDirection()
        if direction=='i':
            sol_in = self.solution[0,:,:,:]
            S_in = self.mesh.Si[0,:,:,:]
            sol_out = self.solution[-1,:,:,:]
            S_out = self.mesh.Si[-1,:,:,:]
        else:
            raise ValueError('At the moment, the turbomachine must be directed along i on the streamwise direction')
        
        pt_in, Tt_in, m_in = self.ComputeTurboMassFlowAverage(sol_in, S_in)
        pt_out, Tt_out, m_out = self.ComputeTurboMassFlowAverage(sol_out, S_out)

        PRtt = pt_out/pt_in
        TRtt = Tt_out/Tt_in
        ETAtt = (PRtt**((self.fluid.gmma-1)/self.fluid.gmma)-1)/(TRtt-1)
        mflow = 0.5*(m_in+m_out)
        
        return PRtt, TRtt, ETAtt, mflow
    

    def ComputeTurboMassFlowAverage(self, U, S):
        """
        From the conservative soluton and surface array over which is defined, compute the mass flow averaged values ot total pressure and temperature, 
        needed for turbomachinery performance analysis
        """
        ni, nj = U[:,:,0].shape
        assert U[0,0,:].shape[0] == self.nEq, 'The conservative vector array must have 5 solutions'
        assert U[:,:,0].shape[0] == S[:,:,0].shape[0], 'The arrays of surface vector and conservative solution must be defined on the same number of cells'
        
        rhoU = np.zeros((ni,nj,3))
        rhoU_pt = np.zeros((ni,nj,3))
        rhoU_Tt = np.zeros((ni,nj,3))
        for i in range(ni):
            for j in range(nj):
                rhoU[i,j,:] = U[i,j,1:-1]
                w = GetPrimitivesFromConservatives(U[i,j,:])
                pt = self.fluid.ComputeTotalPressure_rho_u_et(w[0], w[1:-1], w[-1])
                Tt = self.fluid.ComputeTotalTemperature_rho_u_et(w[0], w[1:-1], w[-1])

                rhoU_pt[i,j,:] = rhoU[i,j,:]*pt
                rhoU_Tt[i,j,:] = rhoU[i,j,:]*Tt

        mflow = IntegrateVectorFlux(rhoU, S)
        pt_avg = IntegrateVectorFlux(rhoU_pt, S)/mflow
        Tt_avg = IntegrateVectorFlux(rhoU_Tt, S)/mflow

        return pt_avg, Tt_avg, mflow


        

                
        




    def PrintMassFlowPlot(self):
        """
        Print the plot of the mass passing through boundaries
        """
        plt.figure()
        plt.plot(self.mi_in, label=r'$\dot{m}_{i,IN}$')
        plt.plot(self.mi_out, label=r'$\dot{m}_{i,OUT}$')
        plt.plot(self.mj_in, label=r'$\dot{m}_{j,IN}$')
        plt.plot(self.mj_out, label=r'$\dot{m}_{j,OUT}$')
        plt.plot(self.mk_in, label=r'$\dot{m}_{k,IN}$')
        plt.plot(self.mk_out, label=r'$\dot{m}_{k,OUT}$')
        plt.xlabel('iteration [-]')
        plt.ylabel('mass flow [kg/s]')
        plt.legend()
        plt.grid(alpha=styles.grid_opacity)
    

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
    
    @override
    def ComputeSourceTerm(self, sol, iterationCounter):
        """
        Compute the source term for a certain solution
        """
        if self.config.IsBFM():
            source = self.ComputeBFMSource(sol, iterationCounter)
        else:
            source = np.zeros_like(sol)
        return source
    
    def ComputeBFMSource(self, sol, iterationCounter):
        """
        Compute the blockage source terms for every cell element, depending on the actual solution `sol`
        """
        self.bfm = CBFMSource(self.config, self, iterationCounter)
        blockage_source = np.zeros_like(sol)
        self.body_force_source_inviscid = np.zeros_like(sol)
        self.body_force_source_viscous = np.zeros_like(sol)
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    
                    if self.config.GetBlockageActive():
                        bgrad = self.mesh.blockage_gradient[i,j,k,:] # blockage source terms computed only if the blockage gradient is different from zero                    
                        if np.linalg.norm(bgrad)<1e-9:
                            blockage_source[i,j,k,:] = np.zeros(5, dtype=float)
                        else: 
                            blockage_source[i,j,k,:] = self.bfm.ComputeBlockageSource(i, j, k)
                    
                    if self.config.GetBFMModel().lower()!='none':
                        if self.mesh.bladeIsPresent[i,j]>0.5:
                            self.body_force_source_inviscid[i,j,k,:], self.body_force_source_viscous[i,j,k,:] = self.bfm.ComputeForceSource(i,j,k)  
                        else:
                            self.body_force_source_inviscid[i,j,k,:] = np.zeros(5, dtype=float)
                            self.body_force_source_viscous[i,j,k,:] = np.zeros(5, dtype=float)

        return blockage_source+self.body_force_source_inviscid+self.body_force_source_viscous
    

    def ComputeRadialEquilibriumPressureProfile(self, sol, dir, loc, p0):
        """
        For a given solution, compute the static pressure profile along the radius that corresponds to it.
        The integrated function is: dp/dr = rho*utheta**2/r, valid for simple flows under inviscid and zero radial 
        velocity assumptions.

        Parameters
        --------------------------------

        `sol`: conservative vector array storing the solution

        `dir`: direction of the boundary where the profile will be computed (i,j,k)

        `loc`: location along the direction (begin or end)

        `p0`: specified static pressure at hub (at minimum radius)
        """
        if dir=='i' and loc=='begin':
            U = np.sum(sol[0,:,:,:], axis=1)/sol[0,:,:,:].shape[1] # circum average of the cons vector
            y = self.Y[0,:,0]
            z = self.Z[0,:,0]
        elif dir=='i' and loc=='end':
            U = np.sum(sol[-1,:,:,:], axis=1)/sol[-1,:,:,:].shape[1] # circum average of the cons vector
            Y = self.mesh.Y[-1,:,0]
            Z = self.mesh.Z[-1,:,0]
        else:
            raise ValueError('The radial quilibrium outlet is supported only if the outlet boundary is along the i-axis')
        
        nspan, nEq = U.shape
        rho = np.zeros(nspan)
        utheta = np.zeros(nspan)
        r = np.zeros(nspan)

        for ispan in range(nspan):
            W = GetPrimitivesFromConservatives(U[ispan,:])
            rho[ispan] = W[0]
            ux = W[1]
            uy = W[2]
            uz = W[3]
            et = W[4]
            y = Y[ispan]
            z = Z[ispan]
            r[ispan] = np.sqrt(y**2 + z**2)
            theta = np.arctan2(z, y)
            utheta[ispan] = -uy*np.sin(theta)+uz*np.cos(theta)
        
        # Interpolate rho and utheta
        rho_interp = interp1d(r, rho, kind='linear', fill_value="extrapolate")
        utheta_interp = interp1d(r, utheta, kind='linear', fill_value="extrapolate")

        # Define the radial equilibrium function
        def radial_equilibrium(p, radius, rho_interp, utheta_interp):
            density = rho_interp(radius)
            vel_theta = utheta_interp(radius)
            dpdr = density * vel_theta**2 / radius
            return dpdr

        # Call odeint and pass the interpolated functions as args
        pressure_profile = odeint(radial_equilibrium, p0, r, args=(rho_interp, utheta_interp))

        # Flatten the solution (odeint output is 2D)
        pressure_profile = pressure_profile.flatten()

        # Plot the results
        # plt.plot(r, pressure_profile, label="Pressure Profile")
        # plt.xlabel("Radius")
        # plt.ylabel("Pressure")
        # plt.legend()
        # plt.show()

        return pressure_profile
    

    def GetPressureSolution(self, solution):
        """
        Return the pressure array correspoding to solution
        """
        p = np.zeros((self.ni, self.nj, self.nk))
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    U = GetPrimitivesFromConservatives(solution[i,j,k,:])
                    p[i,j,k] = self.fluid.ComputePressure_rho_u_et(U[0], U[1:-1], U[-1])
        return p
    
    def GetTotalPressureSolution(self, solution):
        """
        Return the total pressure array correspoding to solution
        """
        pressure = self.GetPressureSolution(solution)
        mach = self.GetMachSolution(solution)
        totalPressure = self.fluid.ComputeTotalPressure_p_M(pressure, mach)
        return totalPressure
    
    def GetTotalTemperatureSolution(self, solution):
        """
        Return the total temp array correspoding to solution
        """
        temperature = self.GetTemperatureSolution(solution)
        mach = self.GetMachSolution(solution)
        totalTemp = self.fluid.ComputeTotalTemperature_T_M(temperature, mach)
        return totalTemp

    def GetEntropySolution(self, solution):
        pressure = self.GetPressureSolution(solution)
        density = solution[:,:,:,0]
        entropy = self.fluid.ComputeEntropy_p_rho(pressure, density)
        return entropy

    def GetTemperatureSolution(self, solution):
        """
        Return the temperature array correspoding to solution
        """
        T = np.zeros((self.ni, self.nj, self.nk))
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    U = GetPrimitivesFromConservatives(solution[i,j,k,:])
                    T[i,j,k] = self.fluid.ComputeStaticTemperature_rho_u_et(U[0], U[1:-1], U[-1])
        return T
    

    def GetRelativeVelocitySolution(self, solution):
        """
        Return the relative velocity vector for every cell
        """
        w = np.zeros((self.ni,self.nj,self.nk,3))
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    U = GetPrimitivesFromConservatives(solution[i,j,k,:])
                    vel = U[1:-1]
                    x = self.mesh.X[i,j,k]
                    y = self.mesh.Y[i,j,k]
                    z = self.mesh.Z[i,j,k]
                    r = np.sqrt(y**2+z**2)
                    drag_speed = self.mesh.omega[i,j]*r
                    drag_vel_cyl = np.array([0, 0, drag_speed])
                    drag_vel_cart = ComputeCartesianVectorFromCylindrical(x, y, z, drag_vel_cyl)
                    w[i,j,k,:] = vel-drag_vel_cart
        return w
    

    def GetMachSolution(self, solution):
        """
        Return the Mach array corresponding to solution
        """
        M = np.zeros((self.ni, self.nj, self.nk))
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    U = GetPrimitivesFromConservatives(solution[i,j,k,:])
                    M[i,j,k] = self.fluid.ComputeMachNumber_rho_u_et(U[0], U[1:-1], U[-1])
        return M
    

    def GetRelativeMachSolution(self, solution, relative_velocity):
        """
        Use the previously computed relative velocity and the solution to compute the relative mach number
        """
        Mrel = np.zeros((self.ni, self.nj, self.nk))
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    U = GetPrimitivesFromConservatives(solution[i,j,k,:])
                    Mrel[i,j,k] = self.fluid.ComputeMachNumber_rho_u_et(U[0], relative_velocity[i,j,k,:], U[-1])
        return Mrel










    
    


