import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from .CMesh import CMesh
from TurboBFM.Solver.CFluid import FluidIdeal, FluidReal
from TurboBFM.Solver.CConfig import Config
from TurboBFM.Solver.euler_functions import *


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
        # these are the number of elements in the geometry including the ghost points
        self.ni = mesh.ni
        self.nj = mesh.nj
        self.nk = mesh.nk
        
        if self.fluidModel.lower()=='ideal':
            self.fluid = FluidIdeal(self.fluidGamma)
        elif self.fluidModel.lower()=='real':
            self.fluid = FluidReal(self.fluidName)
        else:
            raise ValueError('Unknown Fluid Model')
        
        self.InstantiateFields()
        self.ReadBoundaryConditions()
        self.InitializeSolution()
    
    def InstantiateFields(self):
        self.conservatives = np.zeros((self.ni, self.nj, self.nk, 5))
        self.primitives = np.zeros((self.ni, self.nj, self.nk, 5))

        
    def ReadBoundaryConditions(self):
        self.boundary_types =  {'i': self.config.GetBoundaryTypeI(),
                                'j': self.config.GetBoundaryTypeJ(),
                                'k': self.config.GetBoundaryTypeK()}
        
        for keys, values in self.boundary_types.items():
            if 'inlet' in values:
                self.inlet_value = self.config.GetInletValue()
            if 'outlet' in values:
                self.outlet_value = self.config.GetOutletValue()
            

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
                    self.conservatives[i,j,k,:] = GetConservativesFromPrimitives(self.primitives[i,j,k,:], self.fluid)

    def ContoursCheck(self, group, perp_direction = 'k'):
        """
        Plot the contour of the required group of variables perpendicular to the direction index `perp_direction`, at mid length (for the moment).
        """
        if group.lower()=='primitives':
            fields = self.primitives
            names = [r'$\rho$', r'$u_x$', r'$u_y$', r'$u_z$', r'$e_t$']
        elif group.lower()=='conservatives':
            fields = self.conservatives
            names = [r'$\rho$', r'$\rho u_x$', r'$\rho u_y$', r'$\rho u_z$', r'$\rho e_t$']

        if perp_direction.lower()=='i':
            idx = self.ni//2
            for iField in range(len(names)):
                plt.figure()
                plt.contourf(fields[idx,:,:,iField])
                plt.colorbar()
                plt.title(names[iPrim])
                plt.xlabel('K')
                plt.ylabel('J')
        elif perp_direction.lower()=='j':
            idx = self.nj//2
            for iField in range(len(names)):
                plt.figure()
                plt.contourf(self.primitives[:,idx,:,iField])
                plt.colorbar()
                plt.title(names[iField])
                plt.xlabel('K')
                plt.ylabel('I')
        elif perp_direction.lower()=='k':
            idx = self.nk//2
            for iField in range(len(names)):
                plt.figure()
                plt.contourf(self.primitives[:,:,idx,iField])
                plt.colorbar()
                plt.title(names[iField])
                plt.xlabel('J')
                plt.ylabel('I')
            

    
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
        

    def Solve(self, nIter = 100):
        # number of faces, where the 0 is the interface between the ghost point and the first internal point
        niF = self.mesh.Si.shape[0]
        njF = self.mesh.Si.shape[1]
        nkF = self.mesh.Si.shape[2]

        start = time.time()
        for it in range(nIter):
            print('Iteration %i of %i' %(it+1, nIter))
            for iFace in range(niF):
                for jFace in range(njF):
                    for kFace in range(nkF):
                        
                        # i direction surface, connecting points (i,j,k) to (i+1,j,k)
                        U_a = self.GetElementConservatives(iFace, jFace, kFace)
                        U_b = self.GetElementConservatives(iFace+1, jFace, kFace)
                        S = self.mesh.Si[iFace, jFace, kFace, :]
                        flux = self.ComputeFlux(U_a, U_b, S)
                        self.conservatives[iFace, jFace, kFace, :] -= flux          # a positive flux leaves the first element
                        self.conservatives[iFace+1, jFace, kFace, :] += flux        # and enters in the second

                        # j direction surface, connecting points (i,j,k) to (i,j+1,k)
                        U_a = self.GetElementConservatives(iFace, jFace, kFace)
                        U_b = self.GetElementConservatives(iFace, jFace+1, kFace)
                        S = self.mesh.Sj[iFace, jFace, kFace, :]
                        flux = self.ComputeFlux(U_a, U_b, S)
                        self.conservatives[iFace, jFace, kFace, :] -= flux
                        self.conservatives[iFace, jFace+1, kFace, :] += flux

                        # k direction surface, connecting points (i,j,k) to (i,j,k+1)
                        U_a = self.GetElementConservatives(iFace, jFace, kFace)
                        U_b = self.GetElementConservatives(iFace, jFace, kFace+1)
                        S = self.mesh.Sk[iFace, jFace, kFace, :]
                        flux = self.ComputeFlux(U_a, U_b, S)
                        self.conservatives[iFace, jFace, kFace, :] -= flux
                        self.conservatives[iFace, jFace, kFace+1, :] += flux

                        pass



        end = time.time()
        print()
        print('For a (%i,%i,%i) grid, with %i internal faces, %i explicit iterations are computed every second' %(self.ni, self.nj, self.nk, (niF*njF*nkF*3), nIter/(end-start)))


        
        
    def GetElementPrimitives(self, i, j, k):
        return self.primitives[i,j,k,:]
    
    def GetElementConservatives(self, i, j, k):
        return self.conservatives[i,j,k,:]
        

        
    def ComputeFlux(self, U1, U2, S):
        """
        Compute the vector flux between the two elements defined by their conservative vectors, and the surface vector oriented from 1 to 2.
        """
        f1 = EulerFluxFromConservatives(U1, S, self.fluid)
        f2 = EulerFluxFromConservatives(U2, S, self.fluid)
        f = 0.5*(f1+f2)
        return f
        

                    

