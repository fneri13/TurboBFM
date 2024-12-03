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
        Compute Euler flux (density) vector from conservative variables, passing through a surface defined by vector S. Ideal gas version
        for the moment. The flux is NOT already integrated on the surface [quantity/second]. It is a flux density [quantity/second/surface]

        Parameters
        -----------
        Primitive vector defined by:

        `cons`: conservative variable vector (rho, rho ux, rho uy, rho uz, rho et)

        `surf`: surface vector (nx, ny, nz)

        `gamma`: cp/cv ratio

        Returns
        --------
        `flux`: flux vector
        """
        prim = GetPrimitivesFromConservatives(cons)
        area = np.linalg.norm(surf)
        normal = surf/area
        vel = prim[1:-1]
        vel_n = np.dot(vel, normal)
        rho = prim[0]
        et = prim[4]
        p = fluid.ComputePressure_rho_u_et(rho, vel, et)
        ht = et+p/rho
        
        flux = np.zeros(5, dtype=float)
        flux[0] = rho*vel_n
        flux[1] = rho*vel_n*vel[0] + p*normal[0]
        flux[2] = rho*vel_n*vel[1] + p*normal[1]
        flux[3] = rho*vel_n*vel[2] + p*normal[2]
        flux[4] = rho*vel_n*ht

        return flux