from TurboBFM.Solver.CFluid import FluidIdeal
import numpy as np

def GetPrimitivesFromConservatives(conservative, fluid):
        """
        Compute primitive variables vector from conservative vector

        Parameters
        -----------

       Conservative vector defined by: 

        `u1`: density

        `u2`: density*velocity_x

        `u3`: density*velocity_y

        `u4`: density*velocity_z

        `u5`: density*total energy

        `fluid`: fluid object, ideal or real

        Returns
        -----------

        Primitive vector defined by:

        `rho`: density

        `ux`: velocity_x

        `uy`: velocity_y

        `uz`: velocity_z

        `e`: total energy

        """
        rho = conservative[0]
        ux = conservative[1]/conservative[0]
        uy = conservative[2]/conservative[0]
        uz = conservative[3]/conservative[0]
        et = conservative[4]/conservative[0]
        return np.array([rho, ux, uy, uz, et])


def GetConservativesFromPrimitives(primitive, fluid):
        """
        Compute conservative variables from primitive vector

        Parameters
        -----------
        Primitive vector defined by:

        `rho`: density

        `ux`: velocity_x

        `uy`: velocity_y

        `uz`: velocity_z

        `et`: total energy

        `fluid`: fluid object, ideal or real

        Returns
        -----------
        Conservative vector defined by: 

        `u1`: density

        `u2`: density*velocity_x

        `u3`: density*velocity_y

        `u4`: density*velocity_z

        `u5`: density*total energy

        """
        u1 = primitive[0]
        u2 = primitive[0]*primitive[1]
        u3 = primitive[0]*primitive[2]
        u4 = primitive[0]*primitive[3]
        u5 = primitive[0]*primitive[4]
        return np.array([u1, u2, u3, u4, u5])


def EulerFluxFromConservatives(cons, surf, fluid):
        """
        Compute Euler flux vector from conservative variables, passing through a surface defined by vector S. 

        Returns
        --------
        `flux`: flux vector
        """
        prim = GetPrimitivesFromConservatives(cons, fluid)
        normal = surf/np.linalg.norm(surf)
        vel = prim[1:-1]
        vel_n = np.dot(vel, normal)
        p = prim[0]*prim[4]*(fluid.gmma-1) - 0.5*prim[0]*np.linalg.norm(vel)
        
        flux = np.zeros(5)
        flux[0] = prim[0]*vel_n
        flux[1] = prim[0]*vel_n*prim[1] + p*normal[0]
        flux[2] = prim[0]*vel_n*prim[2] + p*normal[1]
        flux[3] = prim[0]*vel_n*prim[3] + p*normal[2]
        flux[4] = vel_n*(prim[0]*prim[4]+p)

        return flux