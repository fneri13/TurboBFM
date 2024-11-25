import unittest
import numpy as np
from TurboBFM.Solver.CElement import CGrid



class TestGrid(unittest.TestCase):
    def test_physical_points(self):
        x = np.linspace(0 , 10, 11)
        y = np.linspace(0, 5 , 6)
        z = np.linspace(0, 3, 4)
        X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
        geometry = CGrid(X, Y, Z)
        X, Y, Z = geometry.X, geometry.Y, geometry.Z

        self.assertEqual(X[1,1,1], 0)
        self.assertEqual(Y[1,1,1], 0)
        self.assertEqual(Z[1,1,1], 0)

        self.assertEqual(X[-2,-2,-2], 10)
        self.assertEqual(Y[-2,-2,-2], 5)
        self.assertEqual(Z[-2,-2,-2], 3)

        self.assertEqual(X[1,2,3], 0)
        self.assertEqual(Y[1,2,3], 1)
        self.assertEqual(Z[1,2,3], 2)
    
    
    def test_ghost_points(self):
        x = np.linspace(0 , 10, 11)
        y = np.linspace(0, 5 , 6)
        z = np.linspace(0, 3, 4)
        X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
        geometry = CGrid(X, Y, Z)
        X, Y, Z = geometry.X, geometry.Y, geometry.Z

        self.assertEqual(X[0,0,0], -1)
        self.assertEqual(Y[0,0,0], -1)
        self.assertEqual(Z[0,0,0], -1)

        self.assertEqual(X[-1,-1,-1], 11)
        self.assertEqual(Y[-1,-1,-1], 6)
        self.assertEqual(Z[-1,-1,-1], 4)

        self.assertEqual(X[1,0,-1], 0)
        self.assertEqual(Y[1,0,-1], -1)
        self.assertEqual(Z[1,0,-1], 4)
    


        
if __name__ == '__main__':
    unittest.main()


