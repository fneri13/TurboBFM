import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

class CSurface():
    
    def __init__(self, xNodes, yNodes, zNodes, xDual, yDual, zDual):
        
        ni_nodes, nj_nodes, nk_nodes = xNodes.shape
        ni_dual, nj_dual, nk_dual = xDual.shape

        if (ni_dual!=ni_nodes+1 or nj_dual!=nj_nodes+1 or nk_dual!=nk_nodes+1):
            raise ValueError('Error in the arrays dimensions')
        
        # the number of surfaces (edges) in one direction is equivalent to the number of primary grid points in that direction minus 1
        ni = ni_nodes-1
        nj = nj_nodes-1
        nk = nk_nodes-1

        # the fourth direction contains the 3 components x,y,z of the surface normals
        self.surface = np.zeros((ni, nj, nk, 3))

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    pass

