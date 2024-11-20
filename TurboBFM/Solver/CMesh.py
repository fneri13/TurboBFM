import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D 
from .CGrid import CGrid

class CMesh():
    
    def __init__(self, geometry):
        """
        Starting from the grid points coordinates, generate the array of Volumes associated to the points, and the array of surfaces associated to the interfaces between the points (Dual grid formulation).
        If the grid has (ni,nj,nk) structured points, it has (ni,nj,nk) volumes and (ni-1, nj-1, nk-1)*3 internal surfaces (Si the surface connect point (i,j,k) and (i+1,j,k), and analogusly for Sj and Sk).
        """
        
        self.ni, self.nj, self.nk = geometry.X.shape
        self.X, self.Y, self.Z = geometry.X, geometry.Y, geometry.Z

        # the dual grid nodes are primary grid nodes + 1 in every direction
        # vertices of the dual grid
        xv = np.zeros((self.ni+1, self.nj+1, self.nk+1))
        yv = np.zeros((self.ni+1, self.nj+1, self.nk+1))
        zv = np.zeros((self.ni+1, self.nj+1, self.nk+1))

        # fix the internal points
        def fix_internals(arr1, arr2):
            arr1[1:-1, 1:-1, 1:-1] = (arr2[0:-1,0:-1,0:-1] + arr2[1:,0:-1,0:-1] + arr2[0:-1,1:,0:-1] + arr2[0:-1,0:-1,1:] + 
                                      arr2[1:,1:,0:-1] + arr2[1:,0:-1,1:] + arr2[0:-1,1:,1:] + arr2[1:,1:,1:])/8.0
            return arr1
        
        xv = fix_internals(xv, geometry.X)
        yv = fix_internals(yv, geometry.Y)
        zv = fix_internals(zv, geometry.Z)
        
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

        self.V = np.zeros((self.ni,self.nj,self.nk))
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    self.V[i,j,k] = compute_volume(xv[i,j,k], xv[i+1,j,k], xv[i,j+1,k], xv[i,j,k+1],
                                                   yv[i,j,k], yv[i+1,j,k], yv[i,j+1,k], yv[i,j,k+1],
                                                   zv[i,j,k], zv[i+1,j,k], zv[i,j+1,k], zv[i,j,k+1])
        
        

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(geometry.X[1:-1], geometry.Y[1:-1], geometry.Z[1:-1], c=self.V[1:-1])
        # ax.set_xlabel('Z')
        # ax.set_ylabel('X')
        # ax.set_zlabel('Y')
        # ax.set_aspect('equal', adjustable='box')
        # plt.show()


        def compute_surface_vector(x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4):
            """
            Compute the surface vector of the ordered quadrilater identified from the coords of the 4 vertices
            """
            # triangulate the face
            a1 = np.array([x2-x1, y2-y1, z2-z1])
            b1 = np.array([x3-x2, y3-y2, z3-z2])
            a2 = np.array([x3-x1, y3-y1, z3-z1])
            b2 = np.array([x4-x3, y4-y3, z4-z3])
            Str = 0.5*(np.cross(a1, b1) + np.cross(a2, b2))
            return Str
        
        self.Si = np.zeros((self.ni-1, self.nj-1, self.nk-1, 3))
        self.Sj = np.zeros((self.ni-1, self.nj-1, self.nk-1, 3))
        self.Sk = np.zeros((self.ni-1, self.nj-1, self.nk-1, 3))
        for i in range(self.ni-1):
            for j in range(self.nj-1):
                for k in range(self.nk-1):
                    self.Si[i,j,k,:] = compute_surface_vector(xv[i+1,j,k], xv[i+1, j+1, k], xv[i+1, j+1, k+1], xv[i+1, j, k+1], 
                                                              yv[i+1,j,k], yv[i+1, j+1, k], yv[i+1, j+1, k+1], yv[i+1, j, k+1],
                                                              zv[i+1,j,k], zv[i+1, j+1, k], zv[i+1, j+1, k+1], zv[i+1, j, k+1])
                    
                    self.Sj[i,j,k,:] = compute_surface_vector(xv[i+1,j+1,k], xv[i, j+1, k], xv[i, j+1, k+1], xv[i+1, j+1, k+1], 
                                                              yv[i+1,j+1,k], yv[i, j+1, k], yv[i, j+1, k+1], yv[i+1, j+1, k+1], 
                                                              zv[i+1,j+1,k], zv[i, j+1, k], zv[i, j+1, k+1], zv[i+1, j+1, k+1])
                    
                    self.Sk[i,j,k,:] = compute_surface_vector(xv[i+1,j+1,k+1], xv[i, j+1, k+1], xv[i, j, k+1], xv[i+1, j, k+1],
                                                              yv[i+1,j+1,k+1], yv[i, j+1, k+1], yv[i, j, k+1], yv[i+1, j, k+1],
                                                              zv[i+1,j+1,k+1], zv[i, j+1, k+1], zv[i, j, k+1], zv[i+1, j, k+1])
        
        print('='*20 + 'SURFACE VECTORS INFORMATION' + '='*20)
        for i in range(self.ni-1):
            for j in range(self.nj-1):
                for k in range(self.nk-1):
                    print('For point (%i,%i,%i):' %(i,j,k))
                    print('                         Si=[%.2e,%.2e,%.2e]' %(self.Si[i,j,k,0],self.Si[i,j,k,1],self.Si[i,j,k,2]))
                    print('                         Sj=[%.2e,%.2e,%.2e]' %(self.Sj[i,j,k,0],self.Sj[i,j,k,1],self.Sj[i,j,k,2]))
                    print('                         Sk=[%.2e,%.2e,%.2e]' %(self.Sk[i,j,k,0],self.Sk[i,j,k,1],self.Sk[i,j,k,2]))
                    print()
        print('='*20 + 'END SURFACE VECTORS INFORMATION' + '='*20)



        

        

                    

