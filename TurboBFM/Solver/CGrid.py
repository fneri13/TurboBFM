import numpy as np
import pyvista as pv

class CGrid():
    
    def __init__(self, x, y, z, verbosity=0):
        self.verbosity = verbosity
        self.X = x
        self.Y = y
        self.Z = z

        self.nz = x.shape[0]
        self.nr = x.shape[1]
        self.ntheta = x.shape[2]

        self.AddGhostPoints()

    def VisualizeMesh(self):
        # Create a 3D scatter plot
        mesh = pv.StructuredGrid(self.X, self.Y, self.Z)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
        plotter.show_axes()
        plotter.show()

    def AddGhostPoints(self):
        """
        Add the ghost points to the geometry for BC implementation
        """

        def add_ghost_coords(arr):
            """
            given a coord array, enlarge it with linear extrapolation
            """
            ni, nj, nk = arr.shape
            new_arr = np.zeros((ni+2,nj+2,nk+2))

            new_arr[1:-1, 1:-1, 1:-1] = arr

            new_arr[0,:,:] = new_arr[1,:,:] - (new_arr[2,:,:]-new_arr[1,:,:])
            new_arr[:,0,:] = new_arr[:,1,:] - (new_arr[:,2,:]-new_arr[:,1,:])
            new_arr[:,:,0] = new_arr[:,:,1] - (new_arr[:,:,2]-new_arr[:,:,1])
            
            new_arr[-1,:,:] = new_arr[-2,:,:] + (new_arr[-2,:,:]-new_arr[-3,:,:])
            new_arr[:,-1,:] = new_arr[:,-2,:] + (new_arr[:,-2,:]-new_arr[:,-3,:])
            new_arr[:,:,-1] = new_arr[:,:,-2] + (new_arr[:,:,-2]-new_arr[:,:,-3])

            return new_arr
        
        self.X = add_ghost_coords(self.X)
        self.Y = add_ghost_coords(self.Y)
        self.Z = add_ghost_coords(self.Z)

        if self.verbosity == 2:
            ni, nj, nk = self.X.shape
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        print('For point (%i,%i,%i):' %(i,j,k))
                        print('                         (x,y,z)=[%.2e,%.2e,%.2e]' %(self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k]))
                        print()


