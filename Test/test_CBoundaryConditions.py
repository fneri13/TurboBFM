import unittest
import numpy as np
from TurboBFM.Solver.euler_functions import *
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CBoundaryCondition import CBoundaryCondition

gmma = 1.4
R = 287.05
fluid = FluidIdeal(gmma, R)


class TestBoundaryConditions(unittest.TestCase):
    def test_WallBoundary(self):
        """
        Verify the correct implementation and direction. A flux is positive when entering the wall and leaving the domain
        """
        bc_type = 'wall'
        bc_value = None
        Ub = np.array([1.5, 100, 5, 0, 200e3])
        Uint = np.array([1.5, 100, 7, 0, 200e3])
        
        # surface at positive y
        S = np.array([0,1.5,0]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        for i in range(5):
            if i!=2:
                self.assertEqual(flux[i], 0)
            else:
                self.assertGreater(flux[i], 0)
        
        # surface at negative y
        S = np.array([0,-1,0]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        for i in range(5):
            if i!=2:
                self.assertEqual(flux[i], 0)
            else:
                self.assertLess(flux[i], 0)
        
        # surface at positive z
        S = np.array([0,0,1]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        for i in range(5):
            if i!=3:
                self.assertEqual(flux[i], 0)
            else:
                self.assertGreater(flux[i], 0)
        
        # surface at negative z
        S = np.array([0,0,-1]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        for i in range(5):
            if i!=3:
                self.assertEqual(flux[i], 0)
            else:
                self.assertLess(flux[i], 0)
        
        # surface at negative x
        S = np.array([-0.5,0,0]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        for i in range(5):
            if i!=1:
                self.assertEqual(flux[i], 0)
            else:
                self.assertLess(flux[i], 0)
        
        # surface at positive x
        S = np.array([2,0,0]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        for i in range(5):
            if i!=1:
                self.assertEqual(flux[i], 0)
            else:
                self.assertGreater(flux[i], 0)
        
        #inclined surface
        S = np.array([1,1,1]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        self.assertEqual(flux[0], 0)
        self.assertEqual(flux[-1], 0)
        for i in range(1, 3):
            self.assertEqual(flux[i], flux[i+1])  # same components also on the fluxes
        for i in range(1, 4):
            self.assertGreater(flux[i], 0)

        #inclined surface
        S = np.array([-4,-2,-1]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        self.assertEqual(flux[0], 0)
        self.assertEqual(flux[-1], 0)
        for i in range(1, 3):
            self.assertEqual(flux[i], 2*flux[i+1])
        for i in range(1, 4):
            self.assertLess(flux[i], 0)

            


        



if __name__ == '__main__':
    unittest.main()


