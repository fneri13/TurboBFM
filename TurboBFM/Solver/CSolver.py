import numpy as np
import matplotlib.pyplot as plt
import time
from .CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal, FluidReal

class CSolver():
    
    def __init__(self, config, mesh, verbosity=0):
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
        self.ni = mesh.ni
        self.nj = mesh.nj
        self.nk = mesh.nk
        self.conservatives = np.zeros((self.ni, self.nj, self.nk, 5))
        
        if self.fluidModel.lower()=='ideal':
            self.fluid = FluidIdeal(self.fluidGamma)
        elif self.fluidModel.lower()=='real':
            self.fluid = FluidReal(self.fluidName)
        else:
            raise ValueError('Unknown Fluid Model')
        
        self.boundary_markers = {'i': self.config.GetMarkersI(),
                                 'j': self.config.GetMarkersJ(),
                                 'k': self.config.GetMarkersK()}
        
    

    def Solve(self, nIter = 1000):
        # number of faces
        niF = self.mesh.Si.shape[0]
        njF = self.mesh.Si.shape[1]
        nkF = self.mesh.Si.shape[2]

        flux = np.zeros(5) # vectorial flux for every element

        start = time.time()
        for it in range(nIter):
            print('Iteration %i of %i' %(it+1, nIter))
            for i in range(niF):
                for j in range(njF):
                    for k in range(nkF):
                        flux = np.zeros(5) # compute the flux
                        # S = self.mesh.Si[i,j,k,:]  # get the surface connecting (i,j,k) to (i+1,j,k)
                        self.conservatives[i,j,k,:] -= flux  # sum on one side and remove from the other
                        self.conservatives[i+1,j,k,:] += flux

                        flux = np.zeros(5)
                        # S = self.mesh.Sj[i,j,k,:]
                        self.conservatives[i,j,k,:] -= flux
                        self.conservatives[i,j+1,k,:] += flux

                        flux = np.zeros(5)
                        # S = self.mesh.Sk[i,j,k,:]
                        self.conservatives[i,j,k,:] -= flux
                        self.conservatives[i,j,k+1,:] += flux


        end = time.time()
        print()
        print('For a (%i,%i,%i) grid, with %i internal faces, %i explicit iterations are computed every second' %(self.ni, self.nj, self.nk, (niF*njF*nkF*3), nIter/(end-start)))


        
        

        

        

        

                    

