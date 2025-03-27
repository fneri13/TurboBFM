import configparser
import ast
import numpy as np
import os


class CConfig:
    def __init__(self, config_file='input.ini'):
        """
        Class used to retrieve input data across the solver
        """
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file)

        cwd = os.getcwd()
        print()
        print('Configuration file path: %s' % os.path.join(cwd, config_file))
        print()

    def get_config_value(self, section, option, default=None):
        """
        Helper method to retrieve a configuration value with a default fallback.
        """
        try:
            return self.config_parser.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def print_config(self):
        """
        Print the entire configuration.
        """
        for section in self.config_parser.sections():
            print(f"[{section}]")
            for option, value in self.config_parser.items(section):
                print(f"{option} = {value}")
            print()

    def create_attributes(self):
        """
        Dynamically create attributes from the configuration.
        """
        for section in self.config_parser.sections():
            for option, value in self.config_parser.items(section):
                setattr(self, option, value)







    def GetGridFilepath(self) -> str:
        return str(self.config_parser.get('CFD', 'GRID_FILE'))
    

    def GetVerbosity(self) -> int:
        return int(self.config_parser.get('DEBUG', 'VERBOSITY_LEVEL'))
    

    def GetFluidName(self) -> str:
        return str(self.config_parser.get('CFD', 'FLUID_NAME'))
    

    def GetFluidGamma(self) -> float:
        return float(self.config_parser.get('CFD', 'FLUID_GAMMA'))
        

    def GetFluidRConstant(self) -> float:
        return float(self.config_parser.get('CFD', 'FLUID_R_CONSTANT'))


    def GetFluidModel(self) -> str:
        return str(self.config_parser.get('CFD', 'FLUID_MODEL'))


    def GetKindSolver(self) -> str:
        return str(self.config_parser.get('CFD', 'KIND_SOLVER'))
    

    def GetBoundaryTypeI(self) -> list:
        mark = self.config_parser.get('CFD', 'BOUNDARY_TYPE_I')
        assert(len(mark.split())==2)
        mark = mark.split(',')
        markers = (mark[0].strip(), mark[1].strip())
        return markers
    

    def GetBoundaryTypeJ(self) -> list:
        mark = self.config_parser.get('CFD', 'BOUNDARY_TYPE_J')
        assert(len(mark.split())==2)
        mark = mark.split(',')
        markers = (mark[0].strip(), mark[1].strip())
        return markers
    

    def GetBoundaryTypeK(self) -> list:
        mark = self.config_parser.get('CFD', 'BOUNDARY_TYPE_K')
        assert(len(mark.split())==2)
        mark = mark.split(',')
        markers = (mark[0].strip(), mark[1].strip())
        return markers
    

    def GetInletValue(self) -> list:
        inlet = self.config_parser.get('CFD', 'INLET_VALUE')
        inlet = [float(x.strip()) for x in inlet.split(',')]
        return inlet
    

    def GetPeriodicValue(self) -> list:
        inlet = self.config_parser.get('CFD', 'PERIODIC_VALUE')
        inlet = [float(x.strip()) for x in inlet.split(',')]
        return inlet
    

    def GetAdvectionVelocity(self) -> list:
        u = self.config_parser.get('CFD', 'ADVECTION_VELOCITY')
        u = [float(x.strip()) for x in u.split(',')]
        return u
    

    def GetOutletValue(self) -> float:
        return float(self.config_parser.get('CFD', 'OUTLET_VALUE'))
    

    def GetInitMach(self) -> float:
        return float(self.config_parser.get('CFD', 'INIT_MACH_NUMBER'))


    def GetInitTemperature(self) -> float:
        return float(self.config_parser.get('CFD', 'INIT_TEMPERATURE'))
    

    def GetInitPressure(self) -> float:
        return float(self.config_parser.get('CFD', 'INIT_PRESSURE'))
    

    def GetInitDirection(self) -> np.ndarray:
        try:
            dir = self.config_parser.get('CFD', 'INIT_DIRECTION')
            dir = [float(x.strip()) for x in dir.split(',')]
            return np.array(dir)
        except:
            return 'adaptive'
    

    def GetCFL(self) -> float:
        return float(self.config_parser.get('CFD', 'CFL'))
    

    def GetLaplaceDiffusivity(self) -> float:
        return float(self.config_parser.get('CFD', 'LAPLACE_DIFFUSIVITY'))
    

    def GetAdvectionRotation(self) -> bool:
        rot = str(self.config_parser.get('CFD', 'ADVECTION_ROTATION')).lower()
        if rot=='yes':
            return True
        else:
            return False


    def GetNIterations(self) -> int:
        return int(self.config_parser.get('CFD', 'N_ITERATIONS'))
    

    def GetSaveUnsteady(self) -> bool:
        res = str(self.config_parser.get('CFD', 'SAVE_UNSTEADY'))
        if res.lower()=='yes':
            return True
        else:
            return False
    

    def GetSaveUnsteadyInterval(self) -> int:
        return int(self.config_parser.get('CFD', 'SAVE_UNSTEADY_INTERVAL'))
    

    def GetSolutionName(self) -> str:
        return str(self.config_parser.get('CFD', 'SOLUTION_NAME'))
    

    def GetDirichletValues(self) -> np.ndarray:
        dir = self.config_parser.get('CFD', 'DIRICHLET_VALUES')
        dir = [float(x.strip()) for x in dir.split(',')]
        return np.array(dir)
    

    def GetTimeIntegrationType(self)  -> str:
        """
        Return the time integration integration type. Following Simon PhD Thesis and Essers (2003) suggestions:
        `rk`=0 -> RK 4-th order with good dissipation properties at boundaries
        `rk`=1 -> RK 3-rd order with good transient damping properties
        """
        a = int(self.config_parser.get('CFD', 'TIME_INTEGRATION_TYPE'))
        if a==0:
            return 'RK41A'
        elif a==1:
            return 'RK31B'
        else:
            raise ValueError('Time integration type in input file can be 0 or 1 or 2')
    

    def GetRungeKuttaCoeffs(self):
        """
        Return the coefficients in a list of list. Every element specifies the coefficients for each step
        Example:

        w1 = w0 - coeffs[0][0]*dt*R0
        
        w2 = w0 - coeffs[1][0]*dt*R0 - coeffs[1][1]*dt*R1
        
        w3 = w0 - coeffs[2][0]*dt*R0 - coeffs[2][1]*dt*R1 - coeffs[2][2]*dt*R2
        
        w4 = w0 - coeffs[3][0]*dt*R0 - coeffs[3][1]*dt*R1 - coeffs[3][2]*dt*R2 - - coeffs[3][3]*dt*R3
        """
        rk = self.GetTimeIntegrationType()
        if rk=='RK41A':
            coeffs = [1/4, 5/14, 14/25, 1]
        elif rk=='RK31B':
            coeffs = [8/17, 17/20, 1]
        else:
            raise ValueError('Unknown type of Runge-Kutta integration type.')
        return coeffs
    
    
    def GetTimeStepGlobal(self) -> bool:
        method = self.config_parser.get('CFD', 'TIME_STEP_METHOD')
        if method.lower()=='global':
            return True
        elif method.lower()=='local':
            return False
        else:
            raise ValueError('Unknown time step method')
    
    
    def GetSourceRampIterations(self) -> float:
        try:
            return float(self.config_parser.get('CFD', 'SOURCE_RAMP_ITERATIONS'))
        except:
            return 0 # default
    
    
    def GetOutletPressureRampIterations(self) -> float:
        try:
            return float(self.config_parser.get('CFD', 'OUTLET_PRESSURE_RAMP_ITERATIONS'))
        except:
            return 0 # default
    
    def getOutletPressureRampCoefficient(self, iterationCounter) -> float:
        iterationsMax = self.GetOutletPressureRampIterations()
        coeff = iterationCounter/iterationsMax
        if coeff>1:
            coeff = 1
        return coeff
    
    
    def GetRotationalSpeedRampIterations(self) -> float:
        try:
            return float(self.config_parser.get('CFD', 'ROTATIONAL_SPEED_RAMP_ITERATIONS'))
        except:
            return 0 # default
    

    def GetTimeStepLocal(self) -> bool:
        glob = self.GetTimeStepGlobal()
        if glob:
            return False
        else:
            True
    

    def GetRestartSolution(self) -> bool:
        try:
            res = str(self.config_parser.get('CFD', 'RESTART_SOLUTION'))
            if res.lower()=='yes':
                return True
            else:
                return False
        except:
            return False # false by default
    
    def GetTopology(self) -> str:
        try:
            res = str(self.config_parser.get('CFD', 'TOPOLOGY'))
            if res.lower()=='axisymmetric':
                return 'axisymmetric'
            else:
                return 'cartesian'
        except:
            return 'cartesian' # default value
    
    def GetRestartSolutionFilepath(self) -> str:
        return str(self.config_parser.get('CFD', 'RESTART_SOLUTION_FILEPATH'))
    
    
    def GetInletBCType(self) -> str:
        return str(self.config_parser.get('CFD', 'INLET_BC_TYPE'))
    

    def GetConvectionScheme(self) -> str:
        return str(self.config_parser.get('CFD', 'CONVECTION_SCHEME'))
    

    def IsBFM(self) -> bool:
        try:
            bfm = str(self.config_parser.get('CFD', 'BFM_ACTIVE'))
            if bfm.lower()=='yes':
                return True
            else:
                return False
        except:
            return False
    

    def GetBlockageActive(self) -> bool:
        try:
            block = str(self.config_parser.get('CFD', 'BLOCKAGE_ACTIVE'))
            if block.lower()=='yes':
                return True
            else:
                return False
        except:
            return True
    
    
    def GetBFMModel(self) -> str:
        try:
            model = str(self.config_parser.get('CFD', 'BFM_MODEL'))
            return model
        except:
            return 'Hall' # default
    

    def GetBlockageFilePath(self) -> str:
        return str(self.config_parser.get('CFD', 'BLOCKAGE_FILE_PATH'))
    

    def GetRotationAxis(self) -> str:
        vec = self.config_parser.get('CFD', 'ROTATION_AXIS')
        vec = [float(x.strip()) for x in vec.split(',')]
        vec = np.array(vec)
        vec /= np.linalg.norm(vec)
        
        # check that the components are 0 or 1, not skewed axis for the moment
        if vec[0]==1:
            axis = 'x'
        elif vec[0]==-1:
            axis = '-x'
        elif vec[1]==1:
            axis = 'y'
        elif vec[1]==-1:
            axis = '-y'
        elif vec[2]==1:
            axis = 'z'
        elif vec[2]==-1:
            axis = '-z'
        else:
            raise ValueError('The rotation axis cannot be skewed')
        
        return axis
    


    def SaveVTK(self) -> bool:
        """
        If not specified, False by default
        """
        try:
            vtk =  str(self.config_parser.get('CFD', 'SAVE_VTK'))
            if vtk.lower()=='yes':
                return True
            else:
                return False
        except:
            return False
    

    def SavePIK(self) -> bool:
        """
        If not specified, True by default
        """
        try:
            vtk =  str(self.config_parser.get('CFD', 'SAVE_PIK'))
            if vtk.lower()=='yes':
                return True
            else:
                return False
        except:
            return True


    def GetKinematicViscosity(self) -> float:
        try:
            nu =  float(self.config_parser.get('CFD', 'FLUID_KINEMATIC_VISCOSITY'))
        except:
            nu = 1.48e-5 # default value for air
        return nu
    

    def GetTurboOutput(self) -> bool:
        """
        Stores the data related to turbo performances. False by default
        """
        try:
            turbo =  str(self.config_parser.get('CFD', 'TURBO_OUTPUT'))
            if turbo.lower()=='yes':
                return True
            else:
                return False
        except:
            return False
    

    def GetTurboDirection(self) -> bool:
        """
        Stores the data related to turbo performances. False by default
        """
        try:
            dir =  str(self.config_parser.get('CFD', 'TURBO_DIRECTION'))
            return dir
        except:
            return 'i'  # default turbomachinery direction evolution
        
    
