from TurboBFM.Solver.CFluid import FluidIdeal
import numpy as np

def GetPrimitivesFromConservatives(conservative: np.ndarray) -> np.ndarray:
        """
        Compute primitive variables vector from conservative vector

        Parameters
        -----------

        `conservative`: conservative variable vector (rho, rho ux, rho uy, rho uz, rho et)

        Returns
        -----------

        `primitive`: primitive variables vector (rho, ux, uy, uz, et)

        """
        rho = conservative[0]
        ux = conservative[1]/conservative[0]
        uy = conservative[2]/conservative[0]
        uz = conservative[3]/conservative[0]
        et = conservative[4]/conservative[0]
        return np.array([rho, ux, uy, uz, et])


def GetConservativesFromPrimitives(primitive: np.ndarray) -> np.ndarray:
        """
        Compute conservative variables from primitive vector

        Parameters
        -----------

        `primitive`: primitive variables vector (rho, ux, uy, uz, et)


        Returns
        -----------

        `conservative`: conservative variable vector (rho, rho ux, rho uy, rho uz, rho et)

        """
        u1 = primitive[0]
        u2 = primitive[0]*primitive[1]
        u3 = primitive[0]*primitive[2]
        u4 = primitive[0]*primitive[3]
        u5 = primitive[0]*primitive[4]
        return np.array([u1, u2, u3, u4, u5])


def EulerFluxFromConservatives(cons: np.ndarray, surf: np.ndarray, fluid: FluidIdeal) -> np.ndarray:
        """
        Compute Euler flux vector from conservative variables, passing through a surface defined by vector S. Ideal gas version
        for the moment

        Parameters
        -----------
        Primitive vector defined by:

        `cons`: conservative variable vector (rho, rho ux, rho uy, rho uz, rho et)

        `surf`: surface vector (nx, ny, nz)

        `fluid`: fluid object

        Returns
        --------
        `flux`: flux vector
        """
        prim = GetPrimitivesFromConservatives(cons, fluid)
        normal = surf/np.linalg.norm(surf)
        vel = prim[1:-1]
        vel_n = np.dot(vel, normal)
        p = prim[0]*prim[4]*(fluid.gmma-1) - 0.5*prim[0]*np.linalg.norm(vel)**2
        
        flux = np.zeros(5)
        flux[0] = prim[0]*vel_n
        flux[1] = prim[0]*vel_n*prim[1] + p*normal[0]
        flux[2] = prim[0]*vel_n*prim[2] + p*normal[1]
        flux[3] = prim[0]*vel_n*prim[3] + p*normal[2]
        flux[4] = vel_n*(prim[0]*prim[4]+p)

        return flux