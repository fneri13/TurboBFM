import unittest
import numpy as np
from TurboBFM.Solver.CElement import CGrid
from TurboBFM.Solver.CMesh import CMesh

x = np.linspace(0 , 10, 11)
y = np.linspace(0, 5 , 6)
z = np.linspace(0, 3, 4)
X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
geometry = CGrid(X, Y, Z)
mesh = CMesh(geometry)
ni,nj,nk = geometry.X.shape

class TestGrid(unittest.TestCase):
    def test_surface(self):
        for i in range(1,ni-1):
            for j in range(1,nj-1):
                for k in range(1,nk-1):
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
        Si = mesh.Si[0,1,1,:]
        Sj = mesh.Sj[1,0,1,:]
        Sk = mesh.Sk[1,1,0,:]
        self.assertEqual(Si[0],  1)
        self.assertEqual(Sj[1],  1)
        self.assertEqual(Sk[2],  1)
    
    def test_volume(self):
        for i in range(1,ni-1):
            for j in range(1,nj-1):
                for k in range(1,nk-1):
                    V = mesh.V[i,j,k]
                    self.assertEqual(V,  1)
        

        
if __name__ == '__main__':
    unittest.main()


