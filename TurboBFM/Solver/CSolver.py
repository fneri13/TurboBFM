import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.euler_functions import EulerFluxFromConservatives, GetConservativesFromPrimitives, GetPrimitivesFromConservatives
from TurboBFM.Solver.CScheme_JST import CSchemeJST
from TurboBFM.Solver.CBoundaryCondition import CBoundaryCondition


class CSolver():
    
    def __init__(self, config, mesh):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        self.config = config
        self.mesh = mesh
        self.verbosity = self.config.GetVerbosity()
        self.fluidName = self.config.GetFluidName()
        self.fluidGamma = self.config.GetFluidGamma()
        self.fluidModel = self.config.GetFluidModel()
        self.fluidR = self.config.GetFluidRConstant()
        
        # the internal (physical) points indexes differ from the ghost ones
        # these are the number of elements in the geometry including the ghost points
        self.ni = mesh.ni
        self.nj = mesh.nj
        self.nk = mesh.nk
        
        if self.fluidModel.lower()=='ideal':
            self.fluid = FluidIdeal(self.fluidGamma, self.fluidR)
        elif self.fluidModel.lower()=='real':
            raise ValueError('Real Fluid Model not implemented')
        else:
            raise ValueError('Unknown Fluid Model')
        
        self.InstantiateFields()
        self.ReadBoundaryConditions()
        self.InitializeSolution()
    
    def InstantiateFields(self):
        """
        Instantiate basic fields.
        """
        self.conservatives = np.zeros((self.ni, self.nj, self.nk, 5))
        self.time_step = np.zeros((self.ni, self.nj, self.nk))

        
    def ReadBoundaryConditions(self):
        """
        Read the boundary conditions from the input file, and store the information in two dictionnaries
        """
        self.boundary_types = {'i': {},
                               'j': {},
                               'k': {}}
        self.boundary_types['i'] = {'begin': self.config.GetBoundaryTypeI()[0],
                                    'end': self.config.GetBoundaryTypeI()[1]}
        self.boundary_types['j'] = {'begin': self.config.GetBoundaryTypeJ()[0],
                                    'end': self.config.GetBoundaryTypeJ()[1]}
        self.boundary_types['k'] = {'begin': self.config.GetBoundaryTypeK()[0],
                                    'end': self.config.GetBoundaryTypeK()[1]}
        
        self.boundary_values = {'i': {},
                               'j': {},
                               'k': {}}
        
        for direction, station in self.boundary_types.items():
            for location, type in station.items():
                if type=='inlet':
                    self.boundary_values[direction][location] = self.config.GetInletValue()
                elif type=='outlet':
                    self.boundary_values[direction][location] = self.config.GetOutletValue()
                elif type=='wall':
                    self.boundary_values[direction][location] = None
                elif type=='periodic':
                    self.boundary_values[direction][location] = self.config.GetPeriodicValue()
                else:
                    raise ValueError('Unknown type of boundary condition on direction <%s> at location <%s>' %(direction, location))
    
    def GetBoundaryCondition(self, direction: str, location: str):
        """
        Get the boundary condition type and value for a specified direction and location

        Parameters
        ------------------------

        `direction`: choose string between i, j, k

        `location`: choose between begin or end
        """
        bc_type = self.boundary_types[direction][location]
        bc_value = self.boundary_values[direction][location]
        return bc_type, bc_value
            

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
                    self.conservatives[i,j,k,:] = GetConservativesFromPrimitives(primitives[i,j,k,:])

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
    

    def ConvertConservativesToPrimitives(self) -> np.ndarray:
        """
        Compute primitive variables from conservatives.
        """
        primitives = np.zeros_like(self.conservatives)
        ni, nj, nk = primitives.shape
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    primitives[i,j,k,:] = GetPrimitivesFromConservatives(self.conservatives[i,j,k,:])
        return primitives


    def ContoursCheckMeridional(self, group: str):
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
        def contour_template(fields, names, idx_cut):
            fig, ax = plt.subplots(2, 3, figsize=(16,9))
            for iField in range(len(names)):
                X = self.mesh.X[:,:,idx_cut]
                Y = self.mesh.Y[:,:,idx_cut]
                iplot = iField//3
                jplot = iField-3*iplot
                cnt = ax[iplot][jplot].contourf(X, Y, fields[:,:,idx_cut,iField], cmap='jet', levels=20)
                xlabel = 'x [m]'
                ylabel = 'y [m]'
                plt.colorbar(cnt, ax=ax[iplot][jplot])
                ax[iplot][jplot].set_title(names[iField])
                ax[iplot][jplot].set_xlabel(xlabel)
                ax[iplot][jplot].set_ylabel(ylabel)
        
        # call the contour function depending on the chosen direction
        idx = self.nk//2
        contour_template(fields, names, idx)
            
    
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
        

    def Solve(self, nIter : int = 100) -> None:
        """
        Solve the system explicitly in time.
        """
        # self.ImposeBoundaryConditions()  # before calculating the flux, the conditions on the boundary must be known and used
        start = time.time()
        self.ComputeTimeStep()
        dt = np.min(self.time_step)

        for it in range(nIter):
            print('Iteration %i of %i' %(it+1, nIter))
            
            sol_old = self.conservatives.copy()
            self.CheckConservativeVariables(sol_old, it+1)
            if self.verbosity==3:
                self.CheckConservativeVariables(sol_old, it+1)
                self.ContoursCheckMeridional('conservatives')
                # self.ContoursCheck('conservatives', 'j')
                plt.show()
            
            # i-fluxes
            niF, njF, nkF = self.mesh.Si[:, :, :, 0].shape
            for iFace in range(niF):
                for jFace in range(njF):
                    for kFace in range(nkF):
                        if iFace==0: # flux coming from boundary istart
                            bc_type, bc_value = self.GetBoundaryCondition('i', 'begin')
                            Ub = sol_old[iFace, jFace, kFace, :]        # conservative vector on the boundary
                            Uint = sol_old[iFace+1, jFace, kFace, :]    # conservative vector internal
                            S = -self.mesh.Si[iFace, jFace, kFace, :]   # the normal is oriented towards the boundary
                            boundary = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, self.fluid)
                            flux = boundary.ComputeFlux()
                            area = np.linalg.norm(S)
                            self.conservatives[iFace, jFace, kFace, :] -= flux*area*dt/self.mesh.V[iFace, jFace, kFace]
                        elif iFace==niF-1:
                            bc_type, bc_value = self.GetBoundaryCondition('i', 'end')
                            Ub = sol_old[iFace-1, jFace, kFace, :]      # conservative vector on the boundary
                            Uint = sol_old[iFace-2, jFace, kFace, :]    # conservative vector internal
                            S = self.mesh.Si[iFace, jFace, kFace, :]    # the normal is oriented towards the boundary
                            boundary = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, self.fluid)
                            flux = boundary.ComputeFlux()
                            area = np.linalg.norm(S)
                            self.conservatives[iFace-1, jFace, kFace, :] -= flux*area*dt/self.mesh.V[iFace-1, jFace, kFace]
                        else:
                            U_l = sol_old[iFace-1, jFace, kFace,:]
                            U_r = sol_old[iFace, jFace, kFace,:]
                            try:
                                U_ll = sol_old[iFace-2, jFace, kFace,:]
                            except:
                                U_ll = U_l - (U_r-U_l)
                            try:
                                U_rr = sol_old[iFace+1, jFace, kFace,:]
                            except:
                                U_rr = U_r + (U_r-U_l)
                            
                            S, CG = self.mesh.GetSurfaceData(iFace-1, jFace, kFace, 'east', 'all')  # surface oriented from left to right
                            area = np.linalg.norm(S)
                            flux = self.ComputeJSTFlux(U_ll, U_l, U_r, U_rr, S)
                            self.conservatives[iFace-1, jFace, kFace, :] -= flux*area*dt/self.mesh.V[iFace-1, jFace, kFace]          
                            self.conservatives[iFace, jFace, kFace, :] += flux*area*dt/self.mesh.V[iFace, jFace, kFace]
            
            # j-fluxes
            niF, njF, nkF = self.mesh.Sj[:, :, :, 0].shape
            for iFace in range(niF):
                for jFace in range(njF):
                    for kFace in range(nkF):
                        if jFace==0:
                            bc_type, bc_value = self.GetBoundaryCondition('j', 'begin')
                            Ub = sol_old[iFace, jFace, kFace, :]        # conservative vector on the boundary
                            Uint = sol_old[iFace, jFace+1, kFace, :]    # conservative vector internal
                            S = -self.mesh.Sj[iFace, jFace, kFace, :]   # surface oriented towards the wall      
                            boundary = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, self.fluid)
                            flux = boundary.ComputeFlux()
                            area = np.linalg.norm(S)
                            self.conservatives[iFace, jFace, kFace, :] -= flux*area*dt/self.mesh.V[iFace, jFace, kFace]
                        elif jFace==njF-1:
                            bc_type, bc_value = self.GetBoundaryCondition('j', 'end')
                            Ub = sol_old[iFace, jFace-1, kFace, :]      # conservative vector on the boundary
                            Uint = sol_old[iFace, jFace-2, kFace, :]    # conservative vector internal
                            S = self.mesh.Sj[iFace, jFace, kFace, :]    # surface oriented towards the wall  
                            boundary = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, self.fluid)
                            flux = boundary.ComputeFlux()
                            area = np.linalg.norm(S)
                            self.conservatives[iFace, jFace-1, kFace, :] -= flux*area*dt/self.mesh.V[iFace, jFace-1, kFace]
                        else:
                            U_l = sol_old[iFace, jFace-1, kFace,:]
                            U_r = sol_old[iFace, jFace, kFace,:]
                            try:
                                U_ll = sol_old[iFace, jFace-2, kFace,:]
                            except:
                                U_ll = U_l - (U_r-U_l)
                            try:
                                U_rr = sol_old[iFace, jFace+1, kFace,:]
                            except:
                                U_rr = U_r + (U_r-U_l)
                            
                            S, CG = self.mesh.GetSurfaceData(iFace, jFace-1, kFace, 'north', 'all')  # surface oriented from left to right
                            area = np.linalg.norm(S)
                            flux = self.ComputeJSTFlux(U_ll, U_l, U_r, U_rr, S)
                            self.conservatives[iFace, jFace-1, kFace, :] -= flux*area*dt/self.mesh.V[iFace, jFace-1, kFace]          
                            self.conservatives[iFace, jFace, kFace, :] += flux*area*dt/self.mesh.V[iFace, jFace, kFace]
            
            # k-fluxes
            niF, njF, nkF = self.mesh.Sk[:, :, :, 0].shape
            for iFace in range(niF):
                for jFace in range(njF):
                    for kFace in range(nkF):
                        if kFace==0:
                            bc_type, bc_value = self.GetBoundaryCondition('k', 'begin')
                            Ub = sol_old[iFace, jFace, kFace, :]        # conservative vector on the boundary
                            Uint = sol_old[iFace, jFace, kFace+1, :]    # conservative vector internal
                            S = -self.mesh.Sk[iFace, jFace, kFace, :]   # surface oriented towards the wall      
                            boundary = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, self.fluid)
                            flux = boundary.ComputeFlux()
                            area = np.linalg.norm(S)
                            self.conservatives[iFace, jFace, kFace, :] -= flux*area*dt/self.mesh.V[iFace, jFace, kFace]
                        elif kFace==nkF-1:
                            bc_type, bc_value = self.GetBoundaryCondition('k', 'end')
                            Ub = sol_old[iFace, jFace, kFace-1, :]      # conservative vector on the boundary
                            Uint = sol_old[iFace, jFace, kFace-2, :]    # conservative vector internal
                            S = self.mesh.Sk[iFace, jFace, kFace-1, :]  # surface oriented towards the wall  
                            boundary = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, self.fluid)
                            flux = boundary.ComputeFlux()
                            area = np.linalg.norm(S)
                            self.conservatives[iFace, jFace, kFace-1, :] -= flux*area*dt/self.mesh.V[iFace, jFace, kFace-1]
                        else:
                            U_l = sol_old[iFace, jFace, kFace-1,:]
                            U_r = sol_old[iFace, jFace, kFace,:]
                            try:
                                U_ll = sol_old[iFace, jFace, kFace-2,:]
                            except:
                                U_ll = U_l - (U_r-U_l)
                            try:
                                U_rr = sol_old[iFace, jFace, kFace+1,:]
                            except:
                                U_rr = U_r + (U_r-U_l)
                            
                            S, CG = self.mesh.GetSurfaceData(iFace, jFace, kFace-1, 'top', 'all')  # surface oriented from left to right
                            area = np.linalg.norm(S)
                            flux = self.ComputeJSTFlux(U_ll, U_l, U_r, U_rr, S)
                            self.conservatives[iFace, jFace, kFace-1, :] -= flux*area*dt/self.mesh.V[iFace, jFace, kFace-1]          
                            self.conservatives[iFace, jFace, kFace, :] += flux*area*dt/self.mesh.V[iFace, jFace, kFace]


        end = time.time()
        print()
        print('For a (%i,%i,%i) grid, with %i internal faces, %i explicit iterations are computed every second' %(self.ni, self.nj, self.nk, (niF*njF*nkF*3), nIter/(end-start)))

        
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
        flux = jst.ComputeFluxBlazek()
        return flux
        
    
    def ComputeTimeStep(self):
        """
        Compute the time step of the simulation for a certain CFL
        """
        CFL = self.config.GetCFL()
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    i_edge, j_edge, k_edge = self.mesh.GetElementEdges((i,j,k))
                    W = GetPrimitivesFromConservatives(self.conservatives[i,j,k,:])
                    vel = W[1:-1]
                    rho = W[0]
                    et = W[-1]
                    a = self.fluid.ComputeSoundSpeed_rho_u_et(rho, vel, et)

                    dt_i = np.linalg.norm(i_edge) / (np.abs(np.dot(vel, i_edge))+a)
                    dt_j = np.linalg.norm(j_edge) / (np.abs(np.dot(vel, j_edge))+a)
                    dt_k = np.linalg.norm(k_edge) / (np.abs(np.dot(vel, k_edge))+a)

                    self.time_step[i,j,k] = min(dt_i, dt_j, dt_k)


    def CheckConservativeVariables(self, array, nIter):
        """
        Check the array of solutions to stop the simulation in case something is wrong
        """
        if np.isnan(array).any():
            raise ValueError("The simulation diverged. Nan found at iteration %i" %(nIter))
        
        if array[:,:,:,0].any()<=0:
            raise ValueError("The simulation diverged. Negative density at iteration %i" %(nIter))