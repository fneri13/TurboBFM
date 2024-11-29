import numpy as np
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.euler_functions import GetPrimitivesFromConservatives, EulerFluxFromConservatives, GetConservativesFromPrimitives
from scipy.optimize import fsolve


class CBoundaryCondition():
    """
    Class for the evaluation of fluxes due to boundary conditions (Weak form implementation).
    """
    def __init__(self, bc_type: str, bc_value: any, Ub: np.ndarray, Uint: np.ndarray, S: np.ndarray, fluid: FluidIdeal):
        """
        The left to right orientation follows the orientation of the normal

        Parameters
        --------------------------

        `bc_type`: string stating the type of bc (inlet, outlet, wall, etc..)

        `bc_value`: values related to the particular bc_type

        `Ub`: conservative vector at the boundary

        `Uint`: conservative vector on the first internal point

        `S`: surface vector (from internal domain towards boundary)

        `fluid`: fluid object
        """
        self.bc_type = bc_type
        self.bc_value = bc_value
        self.Ub = Ub                # point on the boundary
        self.Uint = Uint            # point inside the domain
        self.Uout = Ub-(Uint-Ub)    # extrapolated point outside of the domain (ghost cell value)
        self.S = S                  
        self.fluid = fluid
        self.S_dir = self.S/np.linalg.norm(S)
        self.Wb = GetPrimitivesFromConservatives(self.Ub)
        self.Wint = GetPrimitivesFromConservatives(self.Uint)
        self.Wout = GetPrimitivesFromConservatives(self.Uout)
    

    def ComputeFlux(self):
        """
        Choose among the specific boundary condition type and calculate the associated flux
        """
        if self.bc_type=='wall':
            flux = self.ComputeBCFlux_Wall()
        elif self.bc_type=='inlet':
            flux = self.ComputeBCFlux_Inlet()
        elif self.bc_type=='outlet':
            flux = self.ComputeBCFlux_Outlet()
        else:
            raise ValueError('Boundary condition <%s> not recognized or not available' %(self.bc_type))
        
        return flux


    def ComputeBCFlux_Wall(self):
        """
        Compute the flux coming from a wall due to tangential velocity condition
        """
        p = self.fluid.ComputePressure_rho_u_et(self.Ub[0], self.Ub[1:-1], self.Ub[-1])
        flux = np.array([0, p*self.S_dir[0], p*self.S_dir[1], p*self.S_dir[2], 0])
        return flux


    def ComputeBCFlux_Inlet(self):
        """
        Assumption of normal inflow for the moment. Formulation taken from 'Formulation and Implementation 
        of Inflow/Outflow Boundary Conditions to Simulate Propulsive Effects', Rodriguez et al.
        """
        a_int = self.fluid.ComputeSoundSpeed_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        ht_int = self.fluid.ComputeTotalEnthalpy_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        u_int = self.Wint[1:-1]
        S_dir_int = -self.S_dir # take a the normal towards interior of the domain
        Jm = -np.dot(u_int, S_dir_int)+2*a_int/(self.fluid.gmma-1)

        def a_b_function(a_b):
            f = ht_int-a_b**2/(self.fluid.gmma-1)+0.5*(Jm-2*a_b/(self.fluid.gmma-1))**2
            return f
        
        a_b = fsolve(a_b_function, a_int)
        a_b = np.max(a_b)
        un_b = 2*a_b/(self.fluid.gmma-1)-Jm
        M_b = un_b/a_b
        p_b = self.fluid.ComputeStaticPressure_pt_M(self.bc_value[0], M_b)
        T_b = self.fluid.ComputeStaticTemperature_Tt_M(self.bc_value[1], M_b)
        rho_b = self.fluid.ComputeDensity_p_T(p_b, T_b)
        e_b = self.fluid.ComputeStaticEnergy_p_rho(p_b, rho_b)
        et_b = e_b+0.5*un_b**2
        W_b = np.array([rho_b, un_b*S_dir_int[0], un_b*S_dir_int[1], un_b*S_dir_int[2], et_b])
        U_b = GetConservativesFromPrimitives(W_b)
        flux = EulerFluxFromConservatives(U_b, self.S, self.fluid)
        return flux


    def ComputeBCFlux_Outlet(self):
        """
        Assumption of normal outflow for the moment. Formulation taken from 'Formulation and Implementation 
        of Inflow/Outflow Boundary Conditions to Simulate Propulsive Effects', Rodriguez et al.
        """
        S_dir_int = -self.S_dir
        s_b = self.fluid.ComputeEntropy_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        u_int = self.Wint[1:-1]
        un_int = np.dot(u_int, S_dir_int)
        a_int = self.fluid.ComputeSoundSpeed_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        Jm = -un_int + 2*a_int/(self.fluid.gmma-1)
        p_b = self.bc_value
        a_b = np.sqrt(self.fluid.gmma*p_b**((self.fluid.gmma-1)/self.fluid.gmma)*s_b**(1/self.fluid.gmma))
        Jp = Jm-4*a_b/(self.fluid.gmma-1)
        rho_b = (a_b**2/self.fluid.gmma/s_b)**(1/(self.fluid.gmma-1))
        un_b = -0.5*(Jm+Jp)

        utan_int = u_int-un_int*S_dir_int
        u_b = un_b*S_dir_int + utan_int
        et_b = self.fluid.ComputeStaticEnergy_p_rho(p_b, rho_b) + 0.5*np.linalg.norm(u_b)**2
        W_b = np.array([rho_b, u_b[0], u_b[1], u_b[2], et_b])
        U_b = GetConservativesFromPrimitives(W_b)
        flux = EulerFluxFromConservatives(U_b, self.S, self.fluid)
        
        return flux



    