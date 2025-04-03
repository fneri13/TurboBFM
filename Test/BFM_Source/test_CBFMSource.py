import numpy as np
import pickle
import unittest
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.CEulerSolver import CEulerSolver, CBFMSource


config = CConfig('input.ini')
with open(config.GetGridFilepath(), 'rb') as file:
    grid = pickle.load(file)
mesh = CMesh(config, grid)
solver = CEulerSolver(config, mesh)

class TestFluid(unittest.TestCase):
    def test_LiftDragInviscidForceMagnitude_Rotor_Positive(self):
        bfmSource = CBFMSource(config, solver, 0)
        
        betaFlow =          np.array([40,       -40,        -10]) * np.pi/180
        beta0 =             np.array([25,       -25,        -20]) * np.pi/180
        omega =             np.array([-100,     100,        -100])
        machineRotation =   np.array([0,        0,          0])
        
        for i in range(len(betaFlow)):
            forceMagnitude = bfmSource.computeLiftDragInviscidMagnitude(1, 1, 1, betaFlow[i], beta0[i], omega[i], machineRotation[i])
            self.assertGreater(forceMagnitude, 0)
    
    def test_LiftDragInviscidForceMagnitude_Rotor_Negative(self):
        bfmSource = CBFMSource(config, solver, 0)
        
        betaFlow =          np.array([40,       -40]) * np.pi/180
        beta0 =             np.array([25,       -25]) * np.pi/180
        omega =             np.array([+100,     -100])
        machineRotation =   np.array([0,        0])
        
        for i in range(len(betaFlow)):
            forceMagnitude = bfmSource.computeLiftDragInviscidMagnitude(1, 1, 1, betaFlow[i], beta0[i], omega[i], machineRotation[i])
            self.assertLess(forceMagnitude, 0)
    
    def test_LiftDragInviscidForceMagnitude_Stator_Positive(self):
        bfmSource = CBFMSource(config, solver, 0)
        
        betaFlow =          np.array([+10,  -10,    50,     -5]) * np.pi/180
        beta0 =             np.array([+20,  -5,     30,     -25]) * np.pi/180
        omega =             np.array([0,    0,      0,      0])
        machineRotation =   np.array([-1,   -1,     1,      1])
        
        for i in range(len(betaFlow)):
            forceMagnitude = bfmSource.computeLiftDragInviscidMagnitude(1, 1, 1, betaFlow[i], beta0[i], omega[i], machineRotation[i])
            self.assertGreater(forceMagnitude, 0)
       
    def test_LiftDragInviscidForceMagnitude_Stator_Negative(self):
        bfmSource = CBFMSource(config, solver, 0)
        
        betaFlow =          np.array([+10,  -10,    50,     -5]) * np.pi/180
        beta0 =             np.array([+20,  -5,     30,     -25]) * np.pi/180
        omega =             np.array([0,    0,      0,      0])
        machineRotation =   np.array([1,   1,     -1,      -1])
        
        for i in range(len(betaFlow)):
            forceMagnitude = bfmSource.computeLiftDragInviscidMagnitude(1, 1, 1, betaFlow[i], beta0[i], omega[i], machineRotation[i])
            self.assertLess(forceMagnitude, 0)

        
if __name__ == '__main__':
    unittest.main()