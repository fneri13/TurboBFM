import unittest
import numpy as np
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CConfig import CConfig

# test values
gmma = 1.5
fluid = FluidIdeal(gmma)

class TestFluid(unittest.TestCase):
    def test_ComputeStaticEnergy_p_rho(self):
        p = 2
        rho = 1
        e = fluid.ComputeStaticEnergy_p_rho(p, rho)
        self.assertEqual(e, 4)
    
    def test_ComputePressure_rho_e(self):
        rho = 1
        e = 1
        p = fluid.ComputePressure_rho_e(rho, e)
        self.assertEqual(p, 0.5)
    
    def test_ComputeSoundSpeed_p_rho(self):
        p = 1
        rho = 1
        a = fluid.ComputeSoundSpeed_p_rho(p, rho)
        self.assertEqual(a, np.sqrt(1.5))
    
    def test_ComputeStaticEnergy_u_et(self):
        u = np.array([1, 1, 1])
        et = 2
        e = fluid.ComputeStaticEnergy_u_et(u, et)
        self.assertEqual(e, et-0.5*np.linalg.norm(u)**2)

        u = 1
        et = 2
        e = fluid.ComputeStaticEnergy_u_et(u, et)
        self.assertEqual(e, et-0.5*np.linalg.norm(u)**2)
    
    def test_ComputeSoundSpeed_rho_u_et(self):
        rho = 1
        u = 1
        et = 1
        a = fluid.ComputeSoundSpeed_rho_u_et(rho, u, et)
        self.assertGreater(a, 0)
    
    def test_ComputePressure_rho_u_et(self):
        rho = 1
        u = 1
        et = 1
        p = fluid.ComputePressure_rho_u_et(rho, u, et)
        self.assertGreater(p, 0)


        
        

        
if __name__ == '__main__':
    unittest.main()

