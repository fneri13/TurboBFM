import unittest
import numpy as np
from TurboBFM.Solver.euler_functions import *

arr0 = np.array([1, 1, 0, 0, 1])
arr1 = np.array([2, 2, 3, 4, 5])
arr2 = np.array([2, 0, 0, 0, 1])
gmma = 1.4

class TestGrid(unittest.TestCase):
    def test_GetPrimitivesFromConservatives(self):
        
        res = GetPrimitivesFromConservatives(arr1)
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], 3/2)
        self.assertEqual(res[3], 2)
        self.assertEqual(res[4], 5/2)

        res = GetPrimitivesFromConservatives(arr2)
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 1/2)
    

    def test_GetConservativesFromPrimitives(self):
        
        res = GetConservativesFromPrimitives(arr1)
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1], 4)
        self.assertEqual(res[2], 6)
        self.assertEqual(res[3], 8)
        self.assertEqual(res[4], 10)

        res = GetConservativesFromPrimitives(arr2)
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 2)
    



if __name__ == '__main__':
    unittest.main()


