import unittest
import numpy as np
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import CConfig

# test dimensions
LX = 5
LY = 3

x = np.linspace(0 , LX, LX+1)
y = np.linspace(0, LY, LY+1)
X, Y = np.meshgrid(x,y, indexing='ij')
grid = {'X': X, 'Y': Y}
config = CConfig('input.ini')
mesh = CMesh(config, grid)
ni,nj,nk = mesh.X.shape

class TestMesh(unittest.TestCase):
    
    def test_surface(self):
        """
        Check the surface component magnitudes
        """
        for i in range(1,ni-1):
            for j in range(1,nj-1):
                for k in range(1):
                    Si = mesh.Si[i,j,k,:]
                    Sj = mesh.Sj[i,j,k,:]
                    Sk = mesh.Sk[i,j,k,:]

                    self.assertEqual(Si[0],  1)
                    self.assertEqual(Si[1],  0)
                    self.assertEqual(Si[2],  0)

                    self.assertEqual(Sj[0],  0)
                    self.assertEqual(Sj[1],  1)
                    self.assertEqual(Sj[2],  0)

                    self.assertEqual(Sk[0],  0)
                    self.assertEqual(Sk[1],  0)
                    self.assertEqual(Sk[2],  1)
        
        # some surface from the boundaries
        Si = mesh.Si[0,1,0,:]
        Sj = mesh.Sj[1,0,0,:]
        Sk = mesh.Sk[1,1,0,:]
        self.assertEqual(Si[0],  1)
        self.assertEqual(Sj[1],  1)
        self.assertEqual(Sk[2],  1)
    


    def test_volume(self):
        """
        Check the volume magnitudes
        """
        for i in range(1,ni-1):
            for j in range(1,nj-1):
                for k in range(1):
                    V = mesh.V[i,j,k]
                    self.assertEqual(V,  1)
    


    def test_get_surface_data(self):
        """
        Check the method to get surface data for a cell
        """
        for i in range(1,ni-1):
            for j in range(1,nj-1):
                for k in range(0):
                    directions = ['east', 'west', 'north', 'south', 'top', 'bottom']
                    solutions = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
                                 np.array([0, 1, 0]), np.array([0, -1, 0]),
                                 np.array([0, 0, 1]), np.array([0, 0, -1])]
    
                    for idir in range(len(directions)):
                        S = mesh.GetSurfaceData(i, j, k, directions[idir], 'surface')
                        for idim in range(3):
                           self.assertEqual(S[idim],  solutions[idir][idim]) 
    


    def test_fundamental_entities_number(self):
        """
        Check the number of fundamental entities
        """
        n_elems = mesh.n_elements
        n_faces = mesh.Si[:,:,:,0].size + mesh.Sj[:,:,:,0].size + mesh.Sk[:,:,:,0].size
        ref_elems = ni*nj*nk
        ref_faces = (ni+1)*nj*nk + (nj+1)*ni*nk + (nk+1)*ni*nj
        self.assertEqual(n_elems, ni*nj*nk)
        self.assertEqual(n_faces, ref_faces)
    
    def test_no_zero_surface(self):
        """
        Check that there are no zero area surfaces
        """
        def check_surface_area(S):
            ni, nj, nk = S[:,:,:,0].shape
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        area = np.linalg.norm(S[i,j,k,:])
                        self.assertGreater(area, 1e-12)
        check_surface_area(mesh.Si)
        check_surface_area(mesh.Sj)
        check_surface_area(mesh.Sk)
    
    def test_no_zero_volume(self):
        """
        Check that there are no zero volume elements
        """
        ni, nj, nk = mesh.V.shape
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    volume = np.linalg.norm(mesh.V[i,j,k])
                    self.assertGreater(volume, 1e-12)
    
    def test_cg_singular_locations(self):   
        """
        Check that all surface midpoints are unique
        """
        def check_unique_points(CG):
            ni,nj,nk,nd = CG.shape
            CG =np.reshape(CG, (ni*nj*nk,nd))
            unique_points = np.unique(CG, axis=0)
            self.assertEqual(len(CG), len(unique_points))
        
        check_unique_points(mesh.CGi)
        check_unique_points(mesh.CGj)
        check_unique_points(mesh.CGk)
        

        
if __name__ == '__main__':
    unittest.main()


