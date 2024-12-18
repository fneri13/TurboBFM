import numpy as np
from TurboBFM.Solver.CFluid import FluidIdeal
from TurboBFM.Solver.euler_functions import GetPrimitivesFromConservatives, EulerFluxFromConservatives, GetConservativesFromPrimitives
from TurboBFM.Solver.math import rotate_xyz_vector_along_x
from scipy.optimize import fsolve


class CBoundaryCondition():
    """
    Class for the evaluation of fluxes due to boundary conditions (Weak form implementation).
    """
    def __init__(self, bc_type: str, bc_value: any, Uint: np.ndarray, S: np.ndarray, CG: np.ndarray, fluid: FluidIdeal, tot_area: float = None, inlet_bc_type: str = 'PT'):
        """
        The left to right orientation follows the orientation of the normal

        Parameters
        --------------------------

        `bc_type`: string stating the type of bc (inlet, outlet, wall, etc..)

        `bc_value`: values related to the particular bc_type

        `Uint`: conservative vector on the first internal point

        `S`: surface vector (from internal cell towards boundary)

        `fluid`: fluid object
        """
        self.bc_type = bc_type
        if bc_type == 'inlet':
            self.inlet_bc_type = inlet_bc_type
        else:
            self.inlet_bc_type = None
        self.bc_value = bc_value
        self.Uint = Uint               
        self.S = S          
        self.CG = CG        
        self.fluid = fluid
        self.S_dir = self.S/np.linalg.norm(S)
        self.Wint = GetPrimitivesFromConservatives(self.Uint)
        if self.inlet_bc_type=='MT':
            assert self.tot_area!=None, 'For Mass-Total temperature inlet BC you need to specify the total area of the boundary'
        self.tot_area = tot_area
    

    def ComputeFlux(self):
        """
        Choose among the specific boundary condition type and calculate the associated flux
        """
        state = self.GetFluidState(self.Wint)

        if self.bc_type=='wall':
            flux = self.ComputeBCFlux_Wall()
        elif self.bc_type=='inlet':
            if self.inlet_bc_type=='PT':
                flux = self.ComputeBCFlux_Inlet_PT() # total pressure and total temperature
            elif self.inlet_bc_type=='MT':
                flux = self.ComputeBCFlux_Inlet_MT() # mass flow rate and total temperature
            else:
                raise ValueError('Unknown inlet bc type')
        elif self.bc_type=='inlet_ss':
            flux = self.ComputeBCFlux_Inlet_Supersonic()
        elif self.bc_type=='outlet':
            flux = self.ComputeBCFlux_Outlet2()
        elif self.bc_type=='outlet_ss':
            flux = self.ComputeBCFlux_Outlet_Supersonic()
        elif self.bc_type=='wedge':
            flux = self.ComputeBCFlux_Wedge()
        else:
            raise ValueError('Boundary condition <%s> not recognized or not available' %(self.bc_type))
        return flux


    def GetFluidState(self, W):
        """
        Return the state of the fluid: subsonic or supersonic
        """
        a = self.fluid.ComputeSoundSpeed_rho_u_et(W[0], W[1:-1], W[-1])
        umag = np.linalg.norm(W[1:-1])
        if umag>=a:
            return 'supersonic'
        else:
            return 'subsonic'



    def ComputeBCFlux_Wall(self):
        """
        Compute the flux coming from a wall due to tangential velocity condition
        """
        p = self.fluid.ComputePressure_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        flux = np.array([0, p*self.S_dir[0], p*self.S_dir[1], p*self.S_dir[2], 0])

        return flux


    def ComputeBCFlux_Inlet_PT(self):
        """
        Assumption of normal inflow for the moment. Formulation taken from 'Formulation and Implementation 
        of Inflow/Outflow Boundary Conditions to Simulate Propulsive Effects', Rodriguez et al.
        Note: _b and _int refer to boundary and internal points.
        """
        gmma = self.fluid.gmma
        a_int = self.fluid.ComputeSoundSpeed_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        ht_int = self.fluid.ComputeTotalEnthalpy_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        u_int = self.Wint[1:-1]
        S_dir_int = -self.S_dir 
        Jm = -np.linalg.norm(u_int)+2*a_int/(self.fluid.gmma-1)

        # solve the quadratic equation (16) alpha*x**2+beta*x+zeta=0
        alpha = 1/(gmma-1)+2/(gmma-1)**2
        beta = -2*Jm/(gmma-1)
        zeta = 0.5*Jm**2-ht_int
        a_b = np.maximum((-beta+np.sqrt(beta**2-4*alpha*zeta))/2/alpha,
                               (-beta-np.sqrt(beta**2-4*alpha*zeta))/2/alpha)

        un_b = 2*a_b/(gmma-1)-Jm

        dir = np.array([self.bc_value[2], self.bc_value[3], self.bc_value[4]])
        dir /= np.linalg.norm(dir)
        umag_b = un_b/(np.dot(dir, S_dir_int))
        u_b_vec = umag_b*dir

        M_b = umag_b/a_b
        p_b = self.fluid.ComputeStaticPressure_pt_M(self.bc_value[0], M_b)
        T_b = self.fluid.ComputeStaticTemperature_Tt_M(self.bc_value[1], M_b)
        rho_b = self.fluid.ComputeDensity_p_T(p_b, T_b)
        e_b = self.fluid.ComputeStaticEnergy_p_rho(p_b, rho_b)
        u_b = umag_b*dir
        et_b = e_b+0.5*np.linalg.norm(u_b)**2
        W_b = np.array([rho_b, u_b[0], u_b[1], u_b[2], et_b])
        U_b = GetConservativesFromPrimitives(W_b)
        flux = EulerFluxFromConservatives(U_b, self.S_dir, self.fluid)
        return flux
    

    def ComputeBCFlux_Inlet_MT(self):
        """
        Assumption of normal inflow for the moment. Formulation taken from 'Formulation and Implementation 
        of Inflow/Outflow Boundary Conditions to Simulate Propulsive Effects', Rodriguez et al. Inlet mass flow rate and total temperature specified.
        Note: _b and _int refer to boundary and internal points.
        """
        mdot = self.bc_value[0]
        Tt = self.bc_value[1]
        gmma = self.fluid.gmma

        rho_b = self.Wint[0]
        un_b = mdot/self.tot_area/rho_b
        T_b = Tt - (gmma-1)/gmma/self.fluid.R*un_b**2

        S_dir_int = -self.S_dir
        u_b = un_b*S_dir_int
        p_b = rho_b*self.fluid.R*T_b
        e_b = self.fluid.ComputeStaticEnergy_p_rho(p_b, rho_b)
        et_b = e_b+0.5*un_b**2
        W_b = np.array([rho_b, u_b[0], u_b[1], u_b[2], et_b])
        U_b = GetConservativesFromPrimitives(W_b)
        flux = EulerFluxFromConservatives(U_b, self.S_dir, self.fluid)
        return flux
    

    def ComputeBCFlux_Inlet2(self) -> np.ndarray:
        """
        Assumption of normal inflow for the moment. Formulation taken from NASA report.
        """
        gmma = self.fluid.gmma
        ht_i = self.fluid.ComputeTotalEnthalpy_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        a_int = self.fluid.ComputeSoundSpeed_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        Rplus = -np.linalg.norm(self.Wint[1:-1])-2*a_int/(self.fluid.gmma-1)

        #quadratic solution
        a = 1+2/(gmma-1)
        b = 2*Rplus
        c = (gmma-1)/2*(Rplus**2-2*ht_i)
        a_b = np.array([-b/2/a+np.sqrt(b**2-4*a*c)/2/a,
                        -b/2/a-np.sqrt(b**2-4*a*c)/2/a])
        a_b = np.max(a_b)
        umag_b = -2*a_b/(gmma-1)-Rplus
        M_b = umag_b/a_b
        pt_b = self.bc_value[0]
        Tt_b = self.bc_value[1]
        p_b = self.fluid.ComputeStaticPressure_pt_M(pt_b, M_b)
        T_b = self.fluid.ComputeStaticTemperature_Tt_M(Tt_b, M_b)
        rho_b = p_b/self.fluid.R/T_b
        e_b = self.fluid.ComputeStaticEnergy_p_rho(p_b, rho_b)
        et_b = e_b+0.5*umag_b**2
        W_b = np.array([p_b/self.fluid.R/T_b, -umag_b*self.S_dir[0], -umag_b*self.S_dir[1], -umag_b*self.S_dir[2], et_b])
        U_b = GetConservativesFromPrimitives(W_b)
        flux = EulerFluxFromConservatives(U_b, self.S, self.fluid)
        return flux


    def ComputeBCFlux_Outlet(self) -> np.ndarray:
        """
        Assumption of normal outflow for the moment. Formulation taken from 'Formulation and Implementation 
        of Inflow/Outflow Boundary Conditions to Simulate Propulsive Effects', Rodriguez et al.
        Note: _b and _int refer to boundary and internal points.

        WARNING: this outlet BC seems like not working. Probably there are errors in the source article
        """
        S_dir_out = +self.S_dir     # unit normal directed outwards of the domain
        rho_int = self.Wint[0]
        s_b = self.fluid.ComputeEntropy_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        u_int = self.Wint[1:-1]
        un_int = np.dot(u_int, S_dir_out)
        a_int = self.fluid.ComputeSoundSpeed_rho_u_et(self.Wint[0], self.Wint[1:-1], self.Wint[-1])
        Jm = -un_int + 2*a_int/(self.fluid.gmma-1)  # Riemann invariant directed outwards
        p_b = self.bc_value
        a_b = np.sqrt(self.fluid.gmma*p_b**((self.fluid.gmma-1)/self.fluid.gmma)*s_b**(1/self.fluid.gmma))
        Jp = Jm-4*a_b/(self.fluid.gmma-1)  # Riemann invariant directed inwards
        rho_b = (a_b**2/self.fluid.gmma/s_b)**(1/(self.fluid.gmma-1))
        un_b = -0.5*(Jm+Jp)

        utan_int = u_int-un_int*S_dir_out
        u_b = un_b*S_dir_out + utan_int
        et_b = self.fluid.ComputeStaticEnergy_p_rho(p_b, rho_b) + 0.5*np.linalg.norm(u_b)**2
        W_b = np.array([rho_b, u_b[0], u_b[1], u_b[2], et_b])
        U_b = GetConservativesFromPrimitives(W_b)
        flux = EulerFluxFromConservatives(U_b, self.S_dir, self.fluid)  # positive flux is directed outwards
        return flux
    

    def ComputeBCFlux_Outlet2(self) -> np.ndarray:
        """
        Formulation taken from 'Inflow/Outflow Boundary Conditions with Application to FUN3D' by Carlson.
        """
        rho_int = self.Wint[0]
        u_int = self.Wint[1:-1]
        et_int = self.Wint[-1]
        p_int = self.fluid.ComputePressure_rho_u_et(rho_int, u_int, et_int)

        p_b = self.bc_value
        rho_b = p_b*rho_int/p_int
        u_b = u_int
        e_b = self.fluid.ComputeStaticEnergy_p_rho(p_b, rho_b)
        et_b = e_b + 0.5*np.linalg.norm(u_b)**2
        Wb = np.array([rho_b, u_b[0], u_b[1], u_b[2], et_b])
        Ub = GetConservativesFromPrimitives(Wb)
        flux = EulerFluxFromConservatives(Ub, self.S_dir, self.fluid)  # positive flux is directed outwards
        return flux
    
    
    def ComputeBCFlux_Inlet_Supersonic(self) -> np.ndarray:
        """
        Supersonic inlet flux. The state is taken from the specified BCs
        """
        p = self.bc_value[0]
        T = self.bc_value[1]
        vel = self.bc_value[2:]

        rho = self.fluid.ComputeDensity_p_T(p, T)
        et = self.fluid.ComputeStaticEnergy_p_rho(p, rho) + 0.5*np.linalg.norm(vel)**2

        W = np.array([rho, vel[0], vel[1], vel[2], et])
        U = GetConservativesFromPrimitives(W)

        flux = EulerFluxFromConservatives(U, self.S_dir, self.fluid)
        return flux


    def ComputeBCFlux_Outlet_Supersonic(self) -> np.ndarray:
        """
        Flux compute directly with internal flow values
        """
        flux = EulerFluxFromConservatives(self.Uint, self.S, self.fluid)
        return flux


    def ComputeBCFlux_Wedge(self) -> np.ndarray:
        """
        Compute the boundary flux for axisymmetric simulations at wedge boundaries.
        Be careful, that is assumed that element where Ub is stored is at theta=0.
        """
        # coordinates of the surface midpoint
        ys = self.CG[1]
        zs = self.CG[2]
        rs = np.sqrt(ys**2+zs**2)
        thetas = np.arctan2(zs, ys)

        u_int = self.Wint[1:-1]
        u_s = rotate_xyz_vector_along_x(u_int, thetas)

        Ws = self.Wint.copy()
        Ws[1:-1]  = u_s
        Us = GetConservativesFromPrimitives(Ws)
        flux = EulerFluxFromConservatives(Us, self.S_dir, self.fluid)
        return flux
            
        




        





    