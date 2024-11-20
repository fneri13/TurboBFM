import numpy as np
import matplotlib.pyplot as plt
from .CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal, FluidReal

class CSolver():
    
    def __init__(self, config, mesh, verbosity=0):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        self.config = config
        self.verbosity = self.config.GetVerbosity()
        self.fluidName = self.config.GetFluidName()
        self.fluidGamma = self.config.GetFluidGamma()
        self.fluidModel = self.config.GetFluidModel()
        
        # the internal (physical) points indexes differ from the ghost ones
        self.ni = mesh.ni-2
        self.nj = mesh.nj-2
        self.nk = mesh.nk-2
        self.conservatives = {'U1': np.zeros((self.ni, self.nj, self.nk)),
                              'U2': np.zeros((self.ni, self.nj, self.nk)),
                              'U3': np.zeros((self.ni, self.nj, self.nk)),
                              'U4': np.zeros((self.ni, self.nj, self.nk)),
                              'U5': np.zeros((self.ni, self.nj, self.nk))}
        
        if self.fluidModel.lower()=='ideal':
            self.fluid = FluidIdeal(self.fluidGamma)
        elif self.fluidModel.lower()=='real':
            self.fluid = FluidReal(self.fluidName)
        else:
            raise ValueError('Unknown Fluid Model')
        

        

        

        

                    

