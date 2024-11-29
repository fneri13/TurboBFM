import unittest
import numpy as np
from TurboBFM.Solver.euler_functions import *
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.CScheme_JST import CSchemeJST

gmma = 1.4
R = 287.14
fluid = FluidIdeal(gmma, R)


class TestJST(unittest.TestCase):
    def test_Compute_fluxes(self):
        """
        Compare the fluxes computed with the two different formulations and verify they don't differ for more than 1%
        """
        limit = 1
        cons = np.array([1, 100, 120, 15, 300e3])
        S = np.array([1, 1, 1])
        
        
        jst = CSchemeJST(fluid, cons*1.05, cons*1.05, cons*1.1, cons*1, S)
        flux1 = jst.ComputeFluxJameson()
        flux2 = jst.ComputeFluxBlazek()
        diff = np.abs(flux1-flux2)/flux1*100
        for i in range(5):
            self.assertGreater(limit, diff[i])
        

        jst = CSchemeJST(fluid, cons*1.1, cons*1.25, cons*1.1, cons*1, S)
        flux1 = jst.ComputeFluxJameson()
        flux2 = jst.ComputeFluxBlazek()
        diff = np.abs(flux1-flux2)/flux1*100
        for i in range(5):
            self.assertGreater(limit, diff[i])
        

        jst = CSchemeJST(fluid, cons*1.1, cons*1.2, cons*1.3, cons*1.4, S)
        flux1 = jst.ComputeFluxJameson()
        flux2 = jst.ComputeFluxBlazek()
        diff = np.abs(flux1-flux2)/flux1*100
        for i in range(5):
            self.assertGreater(limit, diff[i])
    
    def test_Flux_Directions(self):
        cons = np.array([1, 100, 0, 0, 300e3])
        S = np.array([1, 0, 0])
        jst = CSchemeJST(fluid, cons*1.05, cons*1.05, cons*1.1, cons*1, S)
        flux = jst.ComputeFluxJameson()
        for i in range(2, 4):
            self.assertEqual(flux[i], 0)
        
        cons = np.array([1, 0, 1, 0, 300e3])
        S = np.array([0, 1, 0])
        jst = CSchemeJST(fluid, cons*1.05, cons*1.05, cons*1.1, cons*1, S)
        flux = jst.ComputeFluxJameson()
        for i in range(1, 4, 2):
            self.assertEqual(flux[i], 0)
        
        cons = np.array([1, 0, 0, 1, 300e3])
        S = np.array([0, 0, 1])
        jst = CSchemeJST(fluid, cons*1.05, cons*1.05, cons*1.1, cons*1, S)
        flux = jst.ComputeFluxJameson()
        for i in range(1, 3):
            self.assertEqual(flux[i], 0)
    
    def test_Flux_Multiplicity(self):
        """
        Check that if the surface is enlarged, the flux density still remains the same
        """
        alpha = 5
        cons = np.array([1, 100, 0, 0, 300e3])
        S1 = np.array([1, 2, 3])
        S2 = alpha*S1
        jst1 = CSchemeJST(fluid, cons*1.05, cons*1.05, cons*1.1, cons*1, S1)
        jst2 = CSchemeJST(fluid, cons*1.05, cons*1.05, cons*1.1, cons*1, S2)
        flux1 = jst1.ComputeFluxJameson()
        flux2 = jst2.ComputeFluxJameson()
        flux3 = jst1.ComputeFluxBlazek()
        flux4 = jst2.ComputeFluxBlazek()
        for i in range(5):
            self.assertAlmostEqual(flux1[i], flux2[i])
            self.assertAlmostEqual(flux3[i], flux4[i])
    


        
        

        
    



if __name__ == '__main__':
    unittest.main()


