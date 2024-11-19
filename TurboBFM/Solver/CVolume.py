import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D 
from .CGrid import CGrid

class CVolume():
    
    def __init__(self, geometry):
        
        ni, nj, nk = geometry.X.shape

        # the dual grid nodes are primary grid nodes + 1 in every direction
        ni += 1
        nj += 1
        nk += 1

        # vertices of the dual grid
        xv = np.zeros((ni, nj, nk))
        yv = np.zeros((ni, nj, nk))
        zv = np.zeros((ni, nj, nk))

        # internal points
        for i in range(1, ni-1):
            for j in range(1, nj-1):
                for k in range(1, nk-1):
                    xv[i, j, k] = (geometry.X[i-1, j, k] + geometry.X[i, j-1, k] + geometry.X[i, j, k-1] + geometry.X[i-1, j-1, k] + \
                                  geometry.X[i-1, j, k-1] + geometry.X[i, j-1, k-1] + geometry.X[i-1, j-1, k-1] + geometry.X[i, j, k] )/8.0
                    yv[i, j, k] = (geometry.Y[i-1, j, k] + geometry.Y[i, j-1, k] + geometry.Y[i, j, k-1] + geometry.Y[i-1, j-1, k] + \
                                  geometry.Y[i-1, j, k-1] + geometry.Y[i, j-1, k-1] + geometry.Y[i-1, j-1, k-1] + geometry.Y[i, j, k] )/8.0
                    zv[i, j, k] = (geometry.Z[i-1, j, k] + geometry.Z[i, j-1, k] + geometry.Z[i, j, k-1] + geometry.Z[i-1, j-1, k] + \
                                  geometry.Z[i-1, j, k-1] + geometry.Z[i, j-1, k-1] + geometry.Z[i-1, j-1, k-1] + geometry.Z[i, j, k] )/8.0
        
        # fix the corners
        def fix_corners(arr1, arr2):
            arr1[0,0,0] = arr2[0,0,0]
            arr1[0,0,-1] = arr2[0,0,-1]
            arr1[0,-1,0] = arr2[0,-1,0]
            arr1[-1,0,0] = arr2[-1,0,0]
            arr1[0,-1,-1] = arr2[0,-1,-1]
            arr1[-1,0,-1] = arr2[-1,0,-1]
            arr1[-1,-1,0] = arr2[-1,-1,0]
            arr1[-1,-1,-1] = arr2[-1,-1,-1]
            return arr1
        
        xv = fix_corners(xv, geometry.X)
        yv = fix_corners(yv, geometry.Y)
        zv = fix_corners(zv, geometry.Z)

        # fix the edges
        def fix_edges(arr1, arr2):
            # i oriented edges
            arr1[1:-1,0,0] = (arr2[0:-1,0,0]+arr2[1:,0,0])/2.0
            arr1[1:-1,-1,0] = (arr2[0:-1,-1,0]+arr2[1:,-1,0])/2.0
            arr1[1:-1,0,-1] = (arr2[0:-1,0,-1]+arr2[1:,0,-1])/2.0
            arr1[1:-1,-1,-1] = (arr2[0:-1,-1,-1]+arr2[1:,-1,-1])/2.0

            # j oriented edges
            arr1[0,1:-1,0] = (arr2[0,0:-1,0]+arr2[0,1:,0])/2.0
            arr1[-1,1:-1,0] = (arr2[-1,0:-1,0]+arr2[-1,1:,0])/2.0
            arr1[0,1:-1,-1] = (arr2[0,0:-1,-1]+arr2[0,1:,-1])/2.0
            arr1[-1,1:-1,-1] = (arr2[-1,0:-1,-1]+arr2[-1,1:,-1])/2.0

            # k oriented edges
            arr1[0,0,1:-1] = (arr2[0,0,0:-1]+arr2[0,0,1:])/2.0
            arr1[-1,0,1:-1] = (arr2[-1,0,0:-1]+arr2[-1,0,1:])/2.0
            arr1[0,-1,1:-1] = (arr2[0,-1,0:-1]+arr2[0,-1,1:])/2.0
            arr1[-1,-1,1:-1] = (arr2[-1,-1,0:-1]+arr2[-1,-1,1:])/2.0

            return arr1
        
        xv = fix_edges(xv, geometry.X)
        yv = fix_edges(yv, geometry.Y)
        zv = fix_edges(zv, geometry.Z)

        # fix the boundaries
        def fix_boundaries(arr1, arr2):
            # i faces
            arr1[0,1:-1,1:-1] = (arr2[0,0:-1,0:-1] + arr2[0,1:,0:-1] + arr2[0,0:-1,1:] + arr2[0,1:,1:])/4.0
            arr1[-1,1:-1,1:-1] = (arr2[-1,0:-1,0:-1] + arr2[-1,1:,0:-1] + arr2[-1,0:-1,1:] + arr2[-1,1:,1:])/4.0

            # j faces
            arr1[1:-1,0,1:-1] = (arr2[0:-1,0,0:-1] + arr2[1:,0,0:-1] + arr2[0:-1,0,1:] + arr2[1:,0,1:])/4.0
            arr1[1:-1,-1,1:-1] = (arr2[0:-1,-1,0:-1] + arr2[1:,-1,0:-1] + arr2[0:-1,-1,1:] + arr2[1:,-1,1:])/4.0

            # k faces
            arr1[1:-1,1:-1,0] = (arr2[0:-1,0:-1,0] + arr2[1:,0:-1,0] + arr2[0:-1,1:,0] + arr2[1:,1:,0])/4.0
            arr1[1:-1,1:-1,-1] = (arr2[0:-1,0:-1,-1] + arr2[1:,0:-1,-1] + arr2[0:-1,1:,-1] + arr2[1:,1:,-1])/4.0

            return arr1
        
        xv = fix_boundaries(xv, geometry.X)
        yv = fix_boundaries(yv, geometry.Y)
        zv = fix_boundaries(zv, geometry.Z)


        # plt.figure()
        # plt.scatter(geometry.Z[:,:,0], geometry.X[:,:,0], marker='o', label='primary grid')
        # plt.scatter(zv[:,:,0], xv[:,:,0], marker='x', label='secondary grid')
        # plt.legend()
        # ax = plt.gca()
        # ax.set_aspect('equal', adjustable='box')

        # plt.figure()
        # plt.scatter(geometry.X[5,:,:], geometry.Y[5,:,:], marker='o', label='primary grid')
        # plt.scatter(xv[5,:,:], yv[5,:,:], marker='x', label='secondary grid')
        # plt.legend()
        # ax = plt.gca()
        # ax.set_aspect('equal', adjustable='box')

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(geometry.X, geometry.Y, geometry.Z, marker='o', label='primary grid')
        # ax.scatter(xv, yv, zv, marker='x', label='secondary grid')
        # ax.set_xlabel('Z')
        # ax.set_ylabel('X')
        # ax.set_zlabel('Y')
        # ax.set_aspect('equal', adjustable='box')

        # plt.show()

        def compute_volume(x1, x2, x3, x4, 
                           y1, y2, y3, y4, 
                           z1, z2, z3, z4):
            
            a = np.array([x2-x1, y2-y1, z2-z1])
            b = np.array([x3-x1, y3-y1, z3-z1])
            c = np.array([x4-x1, y4-y1, z4-z1])

            volume = np.linalg.norm(a@(np.cross(b,c)))
            return volume

        ni, nj, nk = geometry.X.shape
        self.volume = np.zeros((ni,nj,nk))
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    self.volume[i,j,k] = compute_volume(xv[i,j,k], xv[i+1,j,k], xv[i,j+1,k], xv[i,j,k+1],
                                                        yv[i,j,k], yv[i+1,j,k], yv[i,j+1,k], yv[i,j,k+1],
                                                        zv[i,j,k], zv[i+1,j,k], zv[i,j+1,k], zv[i,j,k+1])
        

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(geometry.X, geometry.Y, geometry.Z, c=self.volume)
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        

                    

