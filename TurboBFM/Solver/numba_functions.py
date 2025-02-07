import numba
import numpy as np

def fake_spatial_integration(residual):
    ni, nj, nk, neq = residual.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for n in range(neq):
                    a = np.linspace(0, 1, neq)
                    b = a**2+3-a/2
                    residual[i,j,k,n] = 0

@numba.njit
def numba_fake_spatial_integration(residual):
    ni, nj, nk, neq = residual.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for n in range(neq):
                    a = np.linspace(0, 1, neq)
                    b = a**2+3-a/2
                    residual[i,j,k,n] = 0
