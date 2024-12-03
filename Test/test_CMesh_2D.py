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
        for i in range(1,ni-1):
            for j in range(1,nj-1):
                for k in range(1):
                    V = mesh.V[i,j,k]
                    self.assertEqual(V,  1)
    


    def test_get_surface_data(self):
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
        n_elems = mesh.n_elements
        n_faces = mesh.Si[:,:,:,0].size + mesh.Sj[:,:,:,0].size + mesh.Sk[:,:,:,0].size
        ref_elems = ni*nj*nk
        ref_faces = (ni+1)*nj*nk + (nj+1)*ni*nk + (nk+1)*ni*nj
        self.assertEqual(n_elems, ni*nj*nk)
        self.assertEqual(n_faces, ref_faces)
        

        
if __name__ == '__main__':
    unittest.main()


