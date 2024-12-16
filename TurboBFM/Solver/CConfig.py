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







    def GetGridFilepath(self):
        return str(self.config_parser.get('CFD', 'GRID_FILE'))
    

    def GetVerbosity(self):
        return int(self.config_parser.get('DEBUG', 'VERBOSITY_LEVEL'))
    

    def GetFluidName(self):
        return str(self.config_parser.get('CFD', 'FLUID_NAME'))
    

    def GetFluidGamma(self):
        try:
            return float(self.config_parser.get('CFD', 'FLUID_GAMMA'))
        except: 
            return 0.0
    

    def GetFluidRConstant(self):
        return float(self.config_parser.get('CFD', 'FLUID_R_CONSTANT'))


    def GetFluidModel(self):
        return str(self.config_parser.get('CFD', 'FLUID_MODEL'))


    def GetKindSolver(self):
        return str(self.config_parser.get('CFD', 'KIND_SOLVER'))
    

    def GetBoundaryTypeI(self):
        mark = self.config_parser.get('CFD', 'BOUNDARY_TYPE_I')
        assert(len(mark.split())==2)
        mark = mark.split(',')
        markers = (mark[0].strip(), mark[1].strip())
        return markers
    

    def GetBoundaryTypeJ(self):
        mark = self.config_parser.get('CFD', 'BOUNDARY_TYPE_J')
        assert(len(mark.split())==2)
        mark = mark.split(',')
        markers = (mark[0].strip(), mark[1].strip())
        return markers
    

    def GetBoundaryTypeK(self):
        mark = self.config_parser.get('CFD', 'BOUNDARY_TYPE_K')
        assert(len(mark.split())==2)
        mark = mark.split(',')
        markers = (mark[0].strip(), mark[1].strip())
        return markers
    

    def GetInletValue(self):
        inlet = self.config_parser.get('CFD', 'INLET_VALUE')
        inlet = [float(x.strip()) for x in inlet.split(',')]
        return inlet
    

    def GetPeriodicValue(self):
        inlet = self.config_parser.get('CFD', 'PERIODIC_VALUE')
        inlet = [float(x.strip()) for x in inlet.split(',')]
        return inlet
    

    def GetAdvectionVelocity(self):
        u = self.config_parser.get('CFD', 'ADVECTION_VELOCITY')
        u = [float(x.strip()) for x in u.split(',')]
        return u
    

    def GetOutletValue(self):
        return float(self.config_parser.get('CFD', 'OUTLET_VALUE'))
    

    def GetInitMach(self):
        return float(self.config_parser.get('CFD', 'INIT_MACH_NUMBER'))


    def GetInitTemperature(self):
        return float(self.config_parser.get('CFD', 'INIT_TEMPERATURE'))
    

    def GetInitPressure(self):
        return float(self.config_parser.get('CFD', 'INIT_PRESSURE'))
    

    def GetInitDirection(self):
        dir = self.config_parser.get('CFD', 'INIT_DIRECTION')
        dir = [float(x.strip()) for x in dir.split(',')]
        return np.array(dir)
    

    def GetCFL(self):
        return float(self.config_parser.get('CFD', 'CFL'))
    

    def GetLaplaceDiffusivity(self):
        return float(self.config_parser.get('CFD', 'LAPLACE_DIFFUSIVITY'))
    

    def GetAdvectionRotation(self):
        rot = str(self.config_parser.get('CFD', 'ADVECTION_ROTATION')).lower()
        if rot=='yes':
            return True
        else:
            return False


    def GetNIterations(self):
        return int(self.config_parser.get('CFD', 'N_ITERATIONS'))
    

    def GetSaveUnsteady(self):
        res = str(self.config_parser.get('CFD', 'SAVE_UNSTEADY'))
        if res.lower()=='yes':
            return True
        else:
            return False
    

    def GetSaveUnsteadyInterval(self):
        return int(self.config_parser.get('CFD', 'SAVE_UNSTEADY_INTERVAL'))
    

    def GetSolutionName(self):
        return str(self.config_parser.get('CFD', 'SOLUTION_NAME'))
    

    def GetDirichletValues(self):
        dir = self.config_parser.get('CFD', 'DIRICHLET_VALUES')
        dir = [float(x.strip()) for x in dir.split(',')]
        return np.array(dir)
    

    def GetTimeIntegrationType(self):
        """
        Return the time integration integration type. Following Anderson advices:
        `rk`=0 -> forward euler
        `rk`=1 -> rk4 time accurate
        `rk`=2 -> low memory consumption, to use only for steady problems
        """
        a = int(self.config_parser.get('CFD', 'TIME_INTEGRATION_TYPE'))
        if a==0:
            return 'FORWARD_EULER'
        elif a==1:
            return 'RK4_TIME_ACCURATE'
        elif a==2:
            return 'RK4_LOW_MEMORY'
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
        if rk=='RK4_TIME_ACCURATE':
            coeffs = [[0.5], 
                      [0, 0.5],  
                      [0, 0, 1], 
                      [1/6, 2/6, 2/6, 1/6]]
        elif rk=='RK4_LOW_MEMORY':
            coeffs = [[0.25], 
                      [0, 1/3],  
                      [0, 0, 1/2], 
                      [0, 0, 0, 1]]
        elif rk=='FORWARD_EULER':
            coeffs = [[1]]
        return coeffs
    
    
    def GetTimeStepGlobal(self):
        method = self.config_parser.get('CFD', 'TIME_STEP_METHOD')
        if method.lower()=='global':
            return True
        elif method.lower()=='local':
            return False
        else:
            raise ValueError('Unknown time step method')
    

    def GetTimeStepLocal(self):
        glob = self.GetTimeStepGlobal()
        if glob:
            return False
        else:
            True
    
    
    def GetInletBCType(self):
        return str(self.config_parser.get('CFD', 'INLET_BC_TYPE'))
    

    def GetConvectionScheme(self):
        return str(self.config_parser.get('CFD', 'CONVECTION_SCHEME'))
    

    def IsBFM(self):
        try:
            bfm = str(self.config_parser.get('CFD', 'BFM_ACTIVE'))
            if bfm.lower()=='yes':
                return True
        except:
            return False
    
    def GetBlockageFilePath(self):
        return str(self.config_parser.get('CFD', 'BLOCKAGE_FILE_PATH'))
    

    def GetRotationAxis(self):
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

    
        
    
