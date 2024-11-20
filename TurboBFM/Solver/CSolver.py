import numpy as np
import matplotlib.pyplot as plt
from .CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal, FluidReal

class CSolver():
    
    def __init__(self, mesh, *fluid_props, verbosity=0):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        self.verbosity = verbosity
        
        # the internal (physical) points indexes differ from the ghost ones
        self.ni = mesh.ni-2
        self.nj = mesh.nj-2
        self.nk = mesh.nk-2
        self.conservatives = {'U1': np.zeros((self.ni, self.nj, self.nk)),
                              'U2': np.zeros((self.ni, self.nj, self.nk)),
                              'U3': np.zeros((self.ni, self.nj, self.nk)),
                              'U4': np.zeros((self.ni, self.nj, self.nk)),
                              'U5': np.zeros((self.ni, self.nj, self.nk))}
        
        self.fluid_name = fluid_props[0]
        if fluid_props[1].lower()=='ideal':
            assert(len(fluid_props)>=3)
            self.fluid_model = 'ideal'
            self.gmma = fluid_props[2]
            self.fluid = FluidIdeal(self.gmma)
        elif fluid_props[1].lower()=='real':
            assert(len(fluid_props)>=2)
            self.fluid_model = 'real'
            self.fluid = FluidReal(self.fluid_name)
        

        

        

        

                    

