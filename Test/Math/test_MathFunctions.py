import numpy as np
import unittest
from TurboBFM.Solver.math import *

# test coords
x = np.array([1, 1, 0, 0, 1])
y = np.array([1, 1, 0, 0, 0])
z = np.array([0, 0, 1, 1, 1])

# test cartesian vectors
vectorCartesian =           [np.array([1, 1, 0]),
                            np.array([0, 0, 1]),
                            np.array([1, 0, 0]),
                            np.array([0, 0, 1]),
                            np.array([0, 1, 0])]

# test Cylindrical vectors
vectorCylindrical =         [np.array([1, 1, 0]),
                            np.array([0, 0, 1]),
                            np.array([1, 0, 0]),
                            np.array([0, 1, 0]),
                            np.array([0, 0, -1])]

class TestFluid(unittest.TestCase):
    def test_ComputeCylindricalVectorFromCartesian(self):
        input = vectorCartesian.copy()
        outputExpected = vectorCylindrical.copy()
        
        for iVector in range(len(vectorCartesian)):
            output = ComputeCylindricalVectorFromCartesian(x[iVector], y[iVector], z[iVector], input[iVector])
            for iDim in range(3):
                self.assertAlmostEqual(output[iDim], outputExpected[iVector][iDim])
    
    def test_ComputeCartesianVectorFromCylindrical(self):
        input = vectorCylindrical.copy()
        outputExpected = vectorCartesian.copy()
        
        for iVector in range(len(vectorCartesian)):
            output = ComputeCartesianVectorFromCylindrical(x[iVector], y[iVector], z[iVector], input[iVector])
            for iDim in range(3):
                self.assertAlmostEqual(output[iDim], outputExpected[iVector][iDim])
           
    
if __name__ == '__main__':
    unittest.main()