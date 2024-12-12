import unittest
import numpy as np
from TurboBFM.Solver.euler_functions import *
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CBoundaryCondition import CBoundaryCondition

gmma = 1.4
R = 287.05
fluid = FluidIdeal(gmma, R)


class TestBoundaryConditions(unittest.TestCase):
    def test_WallBoundaryDirection(self):
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
    
    def test_wallBoundaryIntensity(self):
        bc_type = 'wall'
        bc_value = None
        Ub = np.array([1.5, 100, 5, 0, 200e3])
        Uint = np.array([1.5, 100, 7, 0, 200e3])
        Wb = GetPrimitivesFromConservatives(Ub)
        p_wall = fluid.ComputePressure_rho_u_et(Wb[0], Wb[1:-1], Wb[-1])
        Wint = GetPrimitivesFromConservatives(Uint)

        S = np.array([-0.7,0,0]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        self.assertEqual(flux[0], 0)
        self.assertEqual(flux[1], -p_wall)
        self.assertEqual(flux[2], 0)
        self.assertEqual(flux[3], 0)
        self.assertEqual(flux[4], 0)

        S = np.array([0.7,0,0]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        self.assertEqual(flux[0], 0)
        self.assertEqual(flux[1], p_wall)
        self.assertEqual(flux[2], 0)
        self.assertEqual(flux[3], 0)
        self.assertEqual(flux[4], 0)

        S = np.array([0,-0.3,0]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        self.assertEqual(flux[0], 0)
        self.assertEqual(flux[1], 0)
        self.assertEqual(flux[2], -p_wall)
        self.assertEqual(flux[3], 0)
        self.assertEqual(flux[4], 0)

        S = np.array([0,0.3,0]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        self.assertEqual(flux[0], 0)
        self.assertEqual(flux[1], 0)
        self.assertEqual(flux[2], p_wall)
        self.assertEqual(flux[3], 0)
        self.assertEqual(flux[4], 0)

        S = np.array([0,0,-0.1]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        self.assertEqual(flux[0], 0)
        self.assertEqual(flux[1], 0)
        self.assertEqual(flux[2], 0)
        self.assertEqual(flux[3], -p_wall)
        self.assertEqual(flux[4], 0)

        S = np.array([0,0,0.1]) 
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()
        self.assertEqual(flux[0], 0)
        self.assertEqual(flux[1], 0)
        self.assertEqual(flux[2], 0)
        self.assertEqual(flux[3], p_wall)
        self.assertEqual(flux[4], 0)
    

    def test_BCFlux_Inlet_direction(self):
        bc_type = 'inlet'
        bc_value = np.array([100e3, 300, 0.5, 0, 0])
        Ub = np.array([2, 100, 0, 0, 200e3])
        Uint = np.array([2, 100, 0, 0, 200e3])
        Wb = GetPrimitivesFromConservatives(Ub)
        Wint = GetPrimitivesFromConservatives(Uint)

        S = np.array([-0.5, 0, 0])
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()

        S *= 10
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux2 = bc.ComputeFlux()
        for i in range(5):
            self.assertEqual(flux[i], flux2[i])
    

    def test_BCFlux_Outlet_direction(self):
        bc_type = 'outlet'
        bc_value = 100e3
        Ub = np.array([2, 100, 0, 0, 200e3])
        Uint = np.array([2, 100, 0, 0, 200e3])
        Wb = GetPrimitivesFromConservatives(Ub)
        Wint = GetPrimitivesFromConservatives(Uint)


        S = np.array([-0.5, 0, 0])
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux = bc.ComputeFlux()

        S *= 10
        bc = CBoundaryCondition(bc_type, bc_value, Ub, Uint, S, fluid)
        flux2 = bc.ComputeFlux()
        for i in range(5):
            self.assertEqual(flux[i], flux2[i])




            


        



if __name__ == '__main__':
    unittest.main()


