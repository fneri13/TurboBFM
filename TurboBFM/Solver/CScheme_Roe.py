import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from numpy import sqrt
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.euler_functions import *
from TurboBFM.Solver.math import GetProjectedVector, GetTangentialVector
from numpy.linalg import norm


class CScheme_Roe:
    def __init__(self, U_l: np.ndarray, U_r: np.ndarray, S: np.ndarray, fluid: np.ndarray) -> None:
        """
        Roe scheme numerics for ideal gas. Parameters are left and right values of density, velocity and pressure, and the fluid object.
        Formulation based on x-split Riemann Solver in the book by Toro.
        The flux is calculate in the normal direciton to the surface, therefore the velocities are decomposed in a normal
        component and a tangential component. The normal flux is then reconverted at the end of the routing, retransforming 
        to the original x,y,z reference frame. 
        """
        self.U_l = U_l
        self.U_r = U_r
        Wl = GetPrimitivesFromConservatives(U_l)
        Wr = GetPrimitivesFromConservatives(U_r)
        self.S_dir = S/np.linalg.norm(S)
        self.fluid = fluid
        self.rhoL = Wl[0]
        self.rhoR = Wr[0]
        self.uL = Wl[1:-1]
        self.uR = Wr[1:-1]
        self.pL = fluid.ComputePressure_rho_u_et(Wl[0], Wl[1:-1], Wl[-1])
        self.pR = fluid.ComputePressure_rho_u_et(Wr[0], Wr[1:-1], Wr[-1])
        self.gmma = fluid.gmma
        self.htL = fluid.ComputeTotalEnthalpy_rho_u_et(Wl[0], Wl[1:-1], Wl[-1])
        self.htR = fluid.ComputeTotalEnthalpy_rho_u_et(Wr[0], Wr[1:-1], Wr[-1])
        self.aL = self.fluid.ComputeSoundSpeed_p_rho(self.pL, self.rhoL)
        self.aR = self.fluid.ComputeSoundSpeed_p_rho(self.pR, self.rhoR)

        self.uL_n = GetProjectedVector(self.uL, self.S_dir)
        self.uL_t = GetTangentialVector(self.uL, self.S_dir)

        self.uR_n = GetProjectedVector(self.uR, self.S_dir)
        self.uR_t = GetTangentialVector(self.uR, self.S_dir)


    def RoeAVG(self, fL: float, fR: float) -> float:
        """
        Roe Averaging Operator
        """
        favg = (sqrt(self.rhoL)*fL + sqrt(self.rhoR)*fR)/(sqrt(self.rhoL)+ sqrt(self.rhoR))
        return favg

    
    def ComputeAveragedVariables(self) -> None:
        """
        Compute the Roe averaged variables for the 1D Euler equations.
        u is considered the normal component of the velocities, v the tangential, and w is zero.
        Keep in mind this during the later reconstruction.
        """
        self.rhoAVG = sqrt(self.rhoL*self.rhoR)
        self.uAVG = self.RoeAVG(norm(self.uL_n), norm(self.uR_n))
        self.vAVG = self.RoeAVG(norm(self.uL_t), norm(self.uR_t))
        self.wAVG = 0
        self.hAVG = self.RoeAVG(self.htL, self.htR)
        self.aAVG = sqrt((self.gmma-1)*(self.hAVG-0.5*(self.uAVG**2 + self.vAVG**2 + self.wAVG**2)))
    

    def ComputeAveragedEigenvalues(self):
        """
        Compute eigenvalues of the averaged Jacobian
        """
        self.lambda_vec = np.array([self.uAVG-self.aAVG, 
                                    self.uAVG, 
                                    self.uAVG,
                                    self.uAVG,
                                    self.uAVG+self.aAVG])
    

    def ComputeAveragedEigenvectors(self):
        """
        Compute eigenvector matrix of the averaged flux Jacobian.
        """
        self.eigenvector_mat = np.zeros((5, 5))
        
        self.eigenvector_mat[0, 0] = 1
        self.eigenvector_mat[1, 0] = self.uAVG-self.aAVG
        self.eigenvector_mat[2, 0] = self.vAVG
        self.eigenvector_mat[3, 0] = self.wAVG
        self.eigenvector_mat[4, 0] = self.hAVG-self.uAVG*self.aAVG

        self.eigenvector_mat[0, 1] = 1
        self.eigenvector_mat[1, 1] = self.uAVG
        self.eigenvector_mat[2, 1] = self.vAVG
        self.eigenvector_mat[3, 1] = self.wAVG
        self.eigenvector_mat[4, 1] = 0.5*(self.uAVG**2 + self.vAVG**2 + self.wAVG**2)

        self.eigenvector_mat[0, 2] = 0
        self.eigenvector_mat[1, 2] = 0
        self.eigenvector_mat[2, 2] = 1
        self.eigenvector_mat[3, 2] = 0
        self.eigenvector_mat[4, 2] = self.vAVG

        self.eigenvector_mat[0, 3] = 0
        self.eigenvector_mat[1, 3] = 0
        self.eigenvector_mat[2, 3] = 0
        self.eigenvector_mat[3, 3] = 1
        self.eigenvector_mat[4, 3] = self.wAVG

        self.eigenvector_mat[0, 4] = 1
        self.eigenvector_mat[1, 4] = self.uAVG+self.aAVG
        self.eigenvector_mat[2, 4] = self.vAVG
        self.eigenvector_mat[3, 4] = self.wAVG
        self.eigenvector_mat[4, 4] = self.hAVG+self.uAVG*self.aAVG
    

    def ComputeWaveStrengths(self):
        """
        Characteristic jumps due to initial conditions. Remember that the reference frame is now the one oriented along
        the normal of the face
        """
        # left conservative vector in new ref frame
        self.U_left = self.U_l.copy()
        self.U_left[1:-1] = GetProjectedVector(self.U_l[1:-1], self.S_dir)

        # right conservative vector in new ref frame
        self.U_right = self.U_r.copy()
        self.U_right[1:-1] = GetProjectedVector(self.U_r[1:-1], self.S_dir)

        Deltas = self.U_right-self.U_left  # initial jumps
        Delta_bar = Deltas[4]-(Deltas[2]-self.vAVG*Deltas[0])*self.vAVG-(Deltas[3]-self.wAVG*Deltas[0])*self.wAVG

        # wave strengths
        self.alphas = np.zeros(5)
        self.alphas[2] = Deltas[2]-self.vAVG*Deltas[0]
        self.alphas[3] = Deltas[3]-self.wAVG*Deltas[0]
        self.alphas[1] = (self.gmma-1)/(self.aAVG**2) * (Deltas[0]*(self.hAVG-self.uAVG**2)+self.uAVG*Deltas[1]-Delta_bar)
        self.alphas[0] = 1/2/self.aAVG*(Deltas[0]*(self.uAVG+self.aAVG)-Deltas[1]-self.aAVG*self.alphas[1])
        self.alphas[4] = Deltas[0] - (self.alphas[0]+self.alphas[1])
        

    def ComputeFlux(self, entropy_fix=True):
        """
        Compute the Roe flux. The flux is computed for 1D problems.
        """
        self.ComputeAveragedVariables()
        self.ComputeAveragedEigenvalues()
        self.ComputeAveragedEigenvectors()
        self.ComputeWaveStrengths()


        fluxL = EulerFluxFromConservatives(self.U_l, self.S_dir, self.fluid)
        fluxR = EulerFluxFromConservatives(self.U_r, self.S_dir, self.fluid)
        fluxRoe = 0.5*(fluxL+fluxR)

        absEig = np.abs(self.lambda_vec)

        # # compute the entropy fixed abs eigenvalues
        # absEig = np.zeros(3)
        # if entropy_fix==False:
        #     absEig = np.abs(self.lambda_vec)
        # else:
        #     ## Harten-Hymann entropy fix
        #     self.ComputeLeftRightEigenvalues()
        #     for k in range(3):
        #         tmp = np.array([0, self.lambda_vec[k]-self.lambda_vecL[k], self.lambda_vec[k]-self.lambda_vecR[k]])
        #         delta = np.max(tmp)
        #         if np.abs(self.lambda_vec[k])<delta:
        #             absEig[k] = delta
        #         else:
                    # absEig[k] = np.abs(self.lambda_vec[k])

        for iDim in range(5):
            for jVec in range(5):
                fluxRoe[iDim] -= 0.5*self.alphas[jVec]*absEig[jVec]*self.eigenvector_mat[iDim, jVec]
        
        # now we probably need to reorient the components of the flux in the x,y,z frame. 
        

        return fluxRoe
        