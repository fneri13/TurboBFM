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
from abc import ABC, abstractmethod


# Abstract class. Abstract methods need to be overridden in the child classes
class CSolver(ABC):
    
    def __init__(self, config: CConfig, mesh: CMesh):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        self.config = config
        self.mesh = mesh
        self.nDim = mesh.nDim
        self.verbosity = self.config.GetVerbosity()
        
        # save a local copy of elements number
        self.ni = mesh.ni
        self.nj = mesh.nj
        self.nk = mesh.nk

        self.kindSolver = self.config.GetKindSolver()
        if self.kindSolver=='Euler':
            self.nEq = 5
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
            
    @abstractmethod
    def InitializeSolution(self):
        """
        Given the boundary conditions, initialize the solution associated with them
        """


    @abstractmethod
    def Solve(self) -> None:
        """
        Solve the system explicitly in time.
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
            plt.plot(shift_to_zero(residual), label=name)

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
    

    def SaveSolution(self, it, nIter):
        """
        Check if a solution must be saved, and in case do it.

        Parameters
        --------------------------------

        `it`: current time step 

        `nIter`: number of time steps
        """
        save = self.config.GetSaveUnsteady()
        interval = self.config.GetSaveUnsteadyInterval()
        if (save and it%interval==0) or (save and it==nIter-1) or (save and it==0):
            file_name = self.config.GetSolutionName()
            file_name += '_%03i.pik' %(it)

            if self.nDim==3:
                results = {'X': self.mesh.X,
                           'Y': self.mesh.Y,
                           'Z': self.mesh.Z,
                           'U': self.solution}
            elif self.nDim==2:
                results = {'X': self.mesh.X,
                           'Y': self.mesh.Y,
                           'U': self.solution}
            with open('Results/%s' %file_name, 'wb') as file:
                pickle.dump(results, file)
    

    def ComputeSolutionGradient(self):
        """
        Compute the gradient of the solution using Green-Gauss theorem
        """
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    Sw, CGw = self.mesh.GetSurfaceData(i, j, k, 'west', 'all')
                    Se, CGe = self.mesh.GetSurfaceData(i, j, k, 'east', 'all')
                    Sn, CGn = self.mesh.GetSurfaceData(i, j, k, 'north', 'all')
                    Ss, CGs = self.mesh.GetSurfaceData(i, j, k, 'south', 'all')
                    Sb, CGb = self.mesh.GetSurfaceData(i, j, k, 'bottom', 'all')
                    St, CGt = self.mesh.GetSurfaceData(i, j, k, 'top', 'all')
                    for eq in range(self.nEq):
                        Uw = self.InterpolateSolution(i, j, k, eq, 'west')
                        Ue = self.InterpolateSolution(i, j, k, eq, 'east')
                        Un = self.InterpolateSolution(i, j, k, eq, 'north')
                        Us = self.InterpolateSolution(i, j, k, eq, 'south')
                        Ut = self.InterpolateSolution(i, j, k, eq, 'top')
                        Ub = self.InterpolateSolution(i, j, k, eq, 'bottom')
                        self.solution_gradient[i,j,k,eq,:] = GreenGaussGradient((Sw, Se, Sn, Ss, St, Sb),
                                                                                (Uw, Ue, Un, Us, Ut, Ub),
                                                                                self.mesh.V[i,j,k])
    


    def InterpolateSolution(self, i, j, k, eq, dir):
        """
        Interpolate the solution between the nodes. Zero order, at the extremes, the extreme values is taken.
        """
        try:
            if dir=='west':
                u = (self.solution[i-1,j,k,eq] + self.solution[i,j,k,eq])/2.0
            elif dir=='east':
                u = (self.solution[i+1,j,k,eq] + self.solution[i,j,k,eq])/2.0
            elif dir=='north':
                u = (self.solution[i,j+1,k,eq] + self.solution[i,j,k,eq])/2.0
            elif dir=='south':
                u = (self.solution[i,j-1,k,eq] + self.solution[i,j,k,eq])/2.0
            elif dir=='top':
                u = (self.solution[i,j,k+1,eq] + self.solution[i,j,k,eq])/2.0
            elif dir=='bottom':
                u = (self.solution[i,j,k-1,eq] + self.solution[i,j,k,eq])/2.0
        except:
            u = self.solution[i,j,k,eq]
        
        return u

        
