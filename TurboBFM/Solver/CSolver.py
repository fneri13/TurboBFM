import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.CBoundaryCondition import CBoundaryCondition
from TurboBFM.Postprocess import styles
from TurboBFM.Solver.math import GreenGaussGradient
from TurboBFM.Postprocess.functions import contour_template
from abc import ABC, abstractmethod
from pyevtk.hl import gridToVTK
import os


# Abstract class. Abstract methods need to be overridden in the child classes
class CSolver(ABC):
    """
    Virtual class used as a base for other solvers.
    """
    
    def __init__(self, config: CConfig, mesh: CMesh):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        self.config = config
        self.mesh = mesh
        self.nDim = mesh.nDim
        self.verbosity = self.config.GetVerbosity()
        self.residualFlag = True
        
        # save a local copy of elements number
        self.ni = mesh.ni
        self.nj = mesh.nj
        self.nk = mesh.nk

        self.kindSolver = self.config.GetKindSolver()
        if self.kindSolver=='Euler':
            self.nEq = 5
            self.conv_scheme = self.config.GetConvectionScheme()
        elif self.kindSolver=='Advection':
            self.nEq = 1
        elif self.kindSolver=='Laplace':
            self.nEq = 1
        else:
            raise ValueError('Unknown kind of solver. Specify <Euler>, <Advection> or <Laplace>')
        
        
    @abstractmethod
    def PrintInfoSolver(self):
        """
        Print basic information before running the solver
        """
    

    def InstantiateFields(self):
        """
        Instantiate basic fields.
        """

        self.solution = np.zeros((self.ni, self.nj, self.nk, self.nEq))     # store all the solution variables for all points
        self.time_step = np.zeros((self.ni, self.nj, self.nk))              # store the CFL related time-step for each point at current step
        self.residual_history = np.empty(self.nEq, dtype=object)            # store the residual history of the simulation
        for i in range(self.nEq):
            self.residual_history[i] = []

        
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
        
    
    def GetBoundaryCondition(self, direction: str, location: str, iterationCounter):
        """
        Get the boundary condition type and value for a specified direction and location

        Parameters
        ------------------------

        `direction`: choose string between i, j, k

        `location`: choose between begin or end
        """
        bc_type = self.boundary_types[direction][location]
        bc_value = self.boundary_values[direction][location]
        
        if bc_type == 'outlet':
            rampCoeff = self.config.getOutletPressureRampCoefficient(iterationCounter)
            pressureInit = self.config.GetInitPressure()
            bc_value = pressureInit + rampCoeff*(bc_value-pressureInit)
        return bc_type, bc_value
            

    @abstractmethod
    def InitializeSolution(self):
        """
        Given the boundary conditions, initialize the solution associated with them
        """


    def Solve(self) -> None:
        """
        Solve the system explicitly in time.
        """
        nIter = self.config.GetNIterations()
        time_physical = 0
        kind_solver = self.config.GetKindSolver()

        if self.config.IsBFM():
            print('========================= BFM INFORMATION =========================')
            print('The BFM mode is active')
            print('Blockage active: %s' %self.config.GetBlockageActive())
            print('BFM model: %s' %self.config.GetBFMModel())
            print('Source Term Ramp Iterations: %i' %self.config.GetSourceRampIterations())
            print('======================= END BFM INFORMATION =======================\n')
            
            BFModel = self.config.GetBFMModel().lower()
            
            if self.config.GetBlockageActive():
                self.mesh.AddBlockageGrid()
                self.mesh.blockage_gradient = self.ComputeGradient(self.mesh.blockage)
                if self.config.GetVerbosity()>2:
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.blockage[:,:,0], r'$b$ [-]')
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.blockage_gradient[:,:,0,0], r'$\partial_x b$ [1/m]')
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.blockage_gradient[:,:,0,1], r'$\partial_y b$ [1/m]')
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.blockage_gradient[:,:,0,2], r'$\partial_z b$ [1/m]')

            if BFModel == 'hall-thollet' or BFModel.lower() == 'hall':
                self.mesh.AddRPMGrid()
                self.mesh.AddCamberNormalGrid()
                self.mesh.AddBladeIsPresentGrid()
                self.mesh.AddNumberBladesGrid()

                if self.config.GetVerbosity()>2:
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.omega, r'$\Omega$ [rad/s]')
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.normal_camber_cyl['Axial'], r'$n_{ax}$ [-]')
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.normal_camber_cyl['Radial'], r'$n_{r}$ [-]')
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.normal_camber_cyl['Tangential'], r'$n_{\theta}$ [-]')
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.bladeIsPresent, r'Blade Presence')
                    contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.numberBlades, r'Blades Number')
     
                if BFModel=='hall-thollet':
                    self.mesh.AddStreamwiseLengthGrid()
                    if self.config.GetVerbosity()>2: 
                        contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.stwl, r'$s_{stwl}$ [-]')
                
                if BFModel=='frozen-forces':
                    raise ValueError('Warning, check how the things are implemented, dont duplicate data unnecessarily')
                    self.mesh.AddBodyForcesGrids()
                    if self.config.GetVerbosity()>2:
                        contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.force_axial[:,:,0], r'$f_{ax}$ [-]')
                        contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.force_radial[:,:,0], r'$f_{r}$ [-]')
                        contour_template(self.mesh.X[:,:,0], self.mesh.Y[:,:,0], self.mesh.force_tangential[:,:,0], r'$f_{\theta}$ [-]')

        it_ramp = 0
        for it in range(nIter): 
            it_ramp += 1

            if self.config.GetRestartSolution(): # if the solution was restarted update the it number
                it += (self.restart_iterations+1)
            sol_old = self.solution.copy()
            dt = self.ComputeTimeStepArray(sol_old) 
            
            # compute mass flows for euler solver
            if kind_solver=='Euler':
                self.mi_in.append(self.ComputeMassFlow('i', 'start'))
                self.mi_out.append(self.ComputeMassFlow('i', 'end'))
                self.mj_in.append(self.ComputeMassFlow('j', 'start'))
                self.mj_out.append(self.ComputeMassFlow('j', 'end'))
                self.mk_in.append(self.ComputeMassFlow('k', 'start'))
                self.mk_out.append(self.ComputeMassFlow('k', 'end'))
            
            if self.config.GetTurboOutput():
                beta_tt, tau_tt, eta_tt, mflow = self.ComputeTurboOutput()
                self.beta_tt.append(beta_tt)
                self.tau_tt.append(tau_tt)
                self.eta_tt.append(eta_tt)
                self.m_turbo.append(mflow)

            
            if kind_solver=='Advection':
                self.u_advection = self.config.GetAdvectionVelocity()
            
            # Time-integration coefficient list
            rk_coeff = self.config.GetRungeKuttaCoeffs()
            for iStep in range(len(rk_coeff)):      
                alpha = rk_coeff[iStep]
                fluxResiduals = self.ComputeFluxResiduals(sol_old, it)
                sourceTerms = self.ComputeSourceTerm(sol_old)
                sourceTerms *= self.GetSourceRampCoefficient(it_ramp)

                sol_new = self.solution.copy() 
                for iEq in range(self.nEq):
                    sol_new[:,:,:,iEq] -= alpha*dt/self.mesh.V[:,:,:] * fluxResiduals[:,:,:,iEq]
                    sol_new[:,:,:,iEq] += alpha*dt/self.mesh.V[:,:,:] * sourceTerms[:,:,:,iEq]
                sol_old = sol_new.copy() 
            
            self.solution = sol_new.copy()
            self.PrintInfoResiduals(-fluxResiduals+sourceTerms, it, time_physical)
            
            if self.config.GetTimeStepGlobal():
                time_physical += dt
            else:
                time_physical += np.min(dt)
            
            self.CheckConvergence(self.solution, it+1) # proceed only if nans are not found
            self.SaveSolution(it, nIter)

    def GetSourceRampCoefficient(self, itRamp):
        totalRampIterations = self.config.GetSourceRampIterations()
        if totalRampIterations < 1:
            return 1
        
        coeff = itRamp/totalRampIterations
        if coeff>=1:
            coeff = 1
        return coeff
    

    def ComputeFluxResiduals(self, sol, iterationCounter):
        """
        For a given flow solution, compute the advection fluxes
        """
        # before computing the residuals, perform some preprocessing on the boundary conditions
        dirs = ['i', 'j', 'k']
        locs = ['begin', 'end']

        for dir in dirs:
            for loc in locs:
                bc_type, bc_values = self.GetBoundaryCondition(dir, loc, iterationCounter)
                if bc_type=='outlet_re':
                    self.radial_pressure_profile = self.ComputeRadialEquilibriumPressureProfile(sol, dir, loc, bc_values)

        fluxResidual = np.zeros_like(sol)
        self.SpatialIntegration('i', sol, fluxResidual, iterationCounter)
        self.SpatialIntegration('j', sol, fluxResidual, iterationCounter)
        if self.nDim==3 or (self.nDim==2 and self.config.GetTopology()=='axisymmetric'):
            self.SpatialIntegration('k', sol, fluxResidual, iterationCounter)

        return fluxResidual

    
    @abstractmethod
    def SpatialIntegration(self, sol, res):
        """
        Every solver will specify the fluxes evaluation based on `sol` array, and store the results in the `res` array
        """
        

    @abstractmethod
    def ComputeTimeStepArray(self):
        """
        Compute the time step of the simulation for a certain CFL
        """


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
    

    def PlotResidualsHistory(self):
        """
        Plot the residuals
        """
        def shift_to_zero(array):
            return array-array[0]
        
        plt.figure()
        for iEq in range(self.nEq):
            residual = np.array(self.residual_history[iEq])
            name = self.solution_names[iEq]
            plt.plot((residual), label=name)

        plt.xlabel('iteration [-]')
        plt.ylabel('residuals drop [-]')
        plt.legend()
        plt.grid(alpha=styles.grid_opacity)
        


    @abstractmethod
    def ComputeTimeStepArray(self):
        """
        Abstract method
        """


    @abstractmethod
    def PrintInfoResiduals(self, residuals: np.ndarray, it: int, time: float, col_width: int = 14):
        """
        Abstract method
        """
    

    def SaveSolution(self, it, nIter, col_width: int = 14):
        """
        Check if a solution must be saved, and in case do it.

        Parameters
        --------------------------------

        `it`: current time step 

        `nIter`: number of time steps
        """
        save = self.config.GetSaveUnsteady()
        if save:
            interval = self.config.GetSaveUnsteadyInterval()
            if ((save and it%interval==0) or (save and it==nIter-1)) and it>0:
                file_name = self.config.GetSolutionName()
                file_name += '_%06i' %(it)
                
                print("|" + "-" * ((col_width)*7+6) + "|")
                print('|RESULTS OUTPUT:')

                #PICKLE OUTPUT
                if self.config.SavePIK():
                    results = {'X': self.mesh.X,
                            'Y': self.mesh.Y,
                            'Z': self.mesh.Z,
                            'U': self.solution,
                            'Res': self.residual_history}
                
                    if self.kindSolver.lower()=='euler':
                        results['MassFlow'] = (self.mi_in, self.mi_out, self.mj_in, self.mj_out, self.mk_in, self.mk_out)
                    
                    if self.config.GetBlockageActive():
                        results['Blockage'] = self.mesh.blockage
                        results['Blockage_Gradient'] = self.mesh.blockage_gradient
                    
                    if self.config.GetTurboOutput():
                        results['PRtt'] = self.beta_tt
                        results['TRtt'] = self.tau_tt
                        results['ETAtt'] = self.eta_tt
                        results['MassFlowTurbo'] = self.m_turbo
                    
                    os.makedirs('Results', exist_ok=True)
                    with open('Results/%s.pik' %file_name, 'wb') as file:
                        pickle.dump(results, file)
                        print('|Saved %s.pik in Results folder' %file_name)
                
                # VTK OUTPUT
                if self.config.SaveVTK():
                    os.makedirs('Results_VTK', exist_ok=True)
                    output_filename = file_name

                    pointsData = {} # main dictionary storing all the fields

                    pointsData["Density [kg/m³]"] = np.ascontiguousarray(self.solution[:,:,:,0])
                    pointsData["Velocity [m/s]"] = (np.ascontiguousarray(self.solution[:,:,:,1]/self.solution[:,:,:,0]),
                                                    np.ascontiguousarray(self.solution[:,:,:,2]/self.solution[:,:,:,0]),
                                                    np.ascontiguousarray(self.solution[:,:,:,3]/self.solution[:,:,:,0]))
                    pointsData["Mach [-]"] = np.ascontiguousarray(self.GetMachSolution(self.solution))
                    pointsData["Pressure [Pa]"] = np.ascontiguousarray(self.GetPressureSolution(self.solution))
                    pointsData["Total Pressure [Pa]"] = np.ascontiguousarray(self.GetTotalPressureSolution(self.solution))
                    pointsData["Temperature [K]"] = np.ascontiguousarray(self.GetTemperatureSolution(self.solution))
                    pointsData["Total Temperature [K]"] = np.ascontiguousarray(self.GetTotalTemperatureSolution(self.solution))
                    pointsData[" Entropy [J/kgK]"] = np.ascontiguousarray(self.GetEntropySolution(self.solution))
                    
                    if self.config.IsBFM() and self.config.GetTurboOutput():
                        w = self.GetRelativeVelocitySolution(self.solution)
                        pointsData['Relative Velocity [m/s]'] = (np.ascontiguousarray(w[:,:,:,0]),
                                                                 np.ascontiguousarray(w[:,:,:,1]),
                                                                 np.ascontiguousarray(w[:,:,:,2]))
                        
                        pointsData["Relative Mach [-]"] = np.ascontiguousarray(self.GetRelativeMachSolution(self.solution, w))

                        pointsData["Inviscid Body Force [N/m³]"] = (np.ascontiguousarray(self.body_force_source_inviscid[:,:,:,1])/self.mesh.V,
                                                                    np.ascontiguousarray(self.body_force_source_inviscid[:,:,:,2])/self.mesh.V,
                                                                    np.ascontiguousarray(self.body_force_source_inviscid[:,:,:,3])/self.mesh.V)
                        
                        pointsData["Viscous Body Force [N/m³]"] = (np.ascontiguousarray(self.body_force_source_viscous[:,:,:,1])/self.mesh.V,
                                                                   np.ascontiguousarray(self.body_force_source_viscous[:,:,:,2])/self.mesh.V,
                                                                   np.ascontiguousarray(self.body_force_source_viscous[:,:,:,3])/self.mesh.V)
                        
                        try: # if the blade camber is available, include it in the output
                            pointsData["Camber Normal [-]"] = (np.ascontiguousarray(self.mesh.normalCamberCartesian[:,:,:,0]),
                                                               np.ascontiguousarray(self.mesh.normalCamberCartesian[:,:,:,1]),
                                                               np.ascontiguousarray(self.mesh.normalCamberCartesian[:,:,:,2]))
                        except:
                            pass
                        
                        try:
                            pointsData["Deviation Angle [deg]"] = np.ascontiguousarray(self.bfm.deviationAngle*180/np.pi)
                        except:
                            pass
                    
                    gridToVTK('Results_VTK/' + output_filename, 
                              np.ascontiguousarray(self.mesh.X), # x coords
                              np.ascontiguousarray(self.mesh.Y), # y coords
                              np.ascontiguousarray(self.mesh.Z), # z coords
                              pointData=pointsData)              # flow fields
                    
                    print('|Saved %s.vts in Results_VTK folder' %output_filename)
                    print("|" + "-" * ((col_width)*7+6) + "|")
                    

    def ComputeGradient(self, phi):
        """
        Compute the gradient of the scalar solution phi using Green-Gauss theorem
        """
        # check that the dimensions coincide with the grid
        assert phi.shape[0]==self.ni
        assert phi.shape[1]==self.nj
        assert phi.shape[2]==self.nk

        gradient = np.zeros((self.ni, self.nj, self.nk, 3))
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    Sw, CGw = self.mesh.GetSurfaceData(i, j, k, 'west', 'all')
                    Se, CGe = self.mesh.GetSurfaceData(i, j, k, 'east', 'all')
                    Sn, CGn = self.mesh.GetSurfaceData(i, j, k, 'north', 'all')
                    Ss, CGs = self.mesh.GetSurfaceData(i, j, k, 'south', 'all')
                    Sb, CGb = self.mesh.GetSurfaceData(i, j, k, 'bottom', 'all')
                    St, CGt = self.mesh.GetSurfaceData(i, j, k, 'top', 'all')
                    Uw = self.InterpolateScalar(phi, i, j, k, CGw, 'west')
                    Ue = self.InterpolateScalar(phi, i, j, k, CGe, 'east')
                    Un = self.InterpolateScalar(phi, i, j, k, CGn, 'north')
                    Us = self.InterpolateScalar(phi, i, j, k, CGs, 'south')
                    Ut = self.InterpolateScalar(phi, i, j, k, CGt, 'top')
                    Ub = self.InterpolateScalar(phi, i, j, k, CGb, 'bottom')
                    gradient[i,j,k,:] = GreenGaussGradient((Sw, Se, Sn, Ss, St, Sb),
                                                            (Uw, Ue, Un, Us, Ut, Ub),
                                                            self.mesh.V[i,j,k])
        return gradient
    


    def InterpolateScalar(self, field, i, j, k, CG, dir):
        """
        Interpolate the field between the nodes. Zero order, at the extremes, the extreme values is taken.
        """
        try:
            p1 = np.array([self.mesh.X[i,j,k], self.mesh.Y[i,j,k], self.mesh.Z[i,j,k]])
            u1 = field[i,j,k]
            if dir=='west':
                p2 = np.array([self.mesh.X[i-1,j,k], self.mesh.Y[i-1,j,k], self.mesh.Z[i-1,j,k]])
                u2 = field[i-1,j,k]
            elif dir=='east':
                p2 = np.array([self.mesh.X[i+1,j,k], self.mesh.Y[i+1,j,k], self.mesh.Z[i+1,j,k]])
                u2 = field[i+1,j,k]
            elif dir=='north':
                p2 = np.array([self.mesh.X[i,j+1,k], self.mesh.Y[i,j+1,k], self.mesh.Z[i,j+1,k]])
                u2 = field[i,j+1,k]
            elif dir=='south':
                p2 = np.array([self.mesh.X[i,j-1,k], self.mesh.Y[i,j-1,k], self.mesh.Z[i,j-1,k]])
                u2 = field[i,j-1,k]
            elif dir=='top':
                p2 = np.array([self.mesh.X[i,j,k+1], self.mesh.Y[i,j,k+1], self.mesh.Z[i,j,k+1]])
                u2 = field[i,j,k+1]
            elif dir=='bottom':
                p2 = np.array([self.mesh.X[i,j,k-1], self.mesh.Y[i,j,k-1], self.mesh.Z[i,j,k-1]])
                u2 = field[i,j,k-1]
            d1 = np.linalg.norm(CG-p1)
            d2 = np.linalg.norm(CG-p2)
            u_intp = u1+(u2-u1)*d1/(d1+d2)
        except:
            u_intp = field[i,j,k]
        return u_intp
    

    
    




        
