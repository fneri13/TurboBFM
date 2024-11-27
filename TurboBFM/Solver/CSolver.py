import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.euler_functions import EulerFluxFromConservatives, GetConservativesFromPrimitives, GetPrimitivesFromConservatives
from TurboBFM.Solver.CScheme_JST import CSchemeJST


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
        
        # the internal (physical) points indexes differ from the ghost ones
        # these are the number of elements in the geometry including the ghost points
        self.ni = mesh.ni
        self.nj = mesh.nj
        self.nk = mesh.nk
        
        if self.fluidModel.lower()=='ideal':
            self.fluid = FluidIdeal(self.fluidGamma)
        elif self.fluidModel.lower()=='real':
            raise ValueError('Real Fluid Model not implemented')
        else:
            raise ValueError('Unknown Fluid Model')
        
        self.InstantiateFields()
        self.ReadBoundaryConditions()
        self.InitializeSolution()
    
    def InstantiateFields(self):
        self.conservatives = np.zeros((self.ni, self.nj, self.nk, 5))
        self.primitives = np.zeros((self.ni, self.nj, self.nk, 5))
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
        self.primitives[:,:,:,0] = rho
        self.primitives[:,:,:,1] = u[0]
        self.primitives[:,:,:,2] = u[1]
        self.primitives[:,:,:,3] = u[2]
        self.primitives[:,:,:,4] = et

        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    self.conservatives[i,j,k,:] = GetConservativesFromPrimitives(self.primitives[i,j,k,:])

    def ContoursCheck(self, group: str, perp_direction: str = 'i'):
        """
        Plot the contour of the required group of variables perpendicular to the direction index `perp_direction`, at mid length (for the moment).

        Parameters:
        ------------------------

        `group`: select between primitives or conservatives
        `perp_direction`: select between i,j,k to choose the on which doing the contour
        """
        if group.lower()=='primitives':
            fields = self.primitives
            names = [r'$\rho$', r'$u_x$', r'$u_y$', r'$u_z$', r'$e_t$']
        elif group.lower()=='conservatives':
            fields = self.conservatives
            names = [r'$\rho$', r'$\rho u_x$', r'$\rho u_y$', r'$\rho u_z$', r'$\rho e_t$']

        # function to make contours on different directions
        def contour_template(fields, names, dir_cut, idx_cut):
            fig, ax = plt.subplots(1, len(names), figsize=(20,3))
            for iField in range(len(names)):
                if dir_cut=='i':
                    cnt = ax[iField].contourf(fields[idx_cut,:,:,iField])
                    xlabel = 'K'
                    ylabel = 'J'
                elif dir_cut=='j':
                    cnt = ax[iField].contourf(fields[:,idx_cut,:,iField])
                    xlabel = 'K'
                    ylabel = 'I'
                elif dir_cut=='k':
                    cnt = ax[iField].contourf(fields[:,:,idx_cut,iField])
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
            

    
    def ComputeInitFields(self, M, T, P, dir):
        gmma = self.config.GetFluidGamma()
        R = self.config.GetFluidRConstant()
        ss = np.sqrt(gmma*R*T)
        u_mag = ss*M
        u = np.zeros(3)
        u[dir] = u_mag
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
            
            # i-flux, for internal points
            niF, njF, nkF = self.mesh.Si[:, :, :, 0].shape
            for iFace in range(niF):
                for jFace in range(njF):
                    for kFace in range(nkF):
                        if iFace==0: # flux coming from boundary istart
                            pass
                        elif iFace==niF-1: # flux coming from boundary iend
                            pass
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
        flux = jst.ComputeFlux()
        return flux
    

    def ImposeBoundaryConditions(self):
        """
        For every boundary of the domain, read the boundary conditions and store that data on the respective nodes
        """
        directions = ['i', 'j', 'k']
        locations = ['begin', 'end']

        for direction in directions:
            for location in locations:
                bc_type, bc_value = self.GetBoundaryCondition(direction, location)
                new_patch = self.ComputeNewPatchValues(direction, location, bc_type, bc_value)


    def ComputeNewPatchValues(self, direction: str, location: str, bc_type: str, bc_value):
        """
        Compute the new conservative variables values for all those nodes belonging to the specified patch, for a strong imposition of BCs

        Parameters:

        --------------------------------------

        `direction`: i, j or k
        
        `location`: begin or end
        
        `bc_type`: boundary condition type (inlet, outlet, wall, periodic)
        
        `bc_value`: boundary condition value (e.g. for inlet type, total pressure, total temperature and flow direction)

        """
        pass
        
    
    def ComputeTimeStep(self):
        """
        Compute the time step of the simulation for a certain CFL
        """
        CFL = self.config.GetCFL()
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    i_edge, j_edge, k_edge = self.mesh.GetElementEdges((i,j,k))
                    vel = self.primitives[i,j,k,1:-1]
                    rho = self.primitives[i,j,k,0]
                    et = self.primitives[i,j,k,-1]
                    a = self.fluid.ComputeSoundSpeed_rho_u_et(rho, vel, et)

                    dt_i = np.linalg.norm(i_edge) / (np.abs(np.dot(vel, i_edge))+a)
                    dt_j = np.linalg.norm(j_edge) / (np.abs(np.dot(vel, j_edge))+a)
                    dt_k = np.linalg.norm(k_edge) / (np.abs(np.dot(vel, k_edge))+a)

                    self.time_step[i,j,k] = min(dt_i, dt_j, dt_k)

    def CheckConservativeVariables(self, array, nIter):
        if np.isnan(array).any():
            raise ValueError("The simulation diverged. Nan found at iteration %i" %(nIter))
        
        if array[:,:,:,0].any()<=0:
            raise ValueError("The simulation diverged. Negative density at iteration %i" %(nIter))