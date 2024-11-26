import configparser
import ast
import numpy as np
import os


class CConfig:
    def __init__(self, config_file='input.ini'):
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file)

        cwd = os.getcwd()
        print('Configuration file path: %s' % os.path.join(cwd, config_file))

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
        mark = self.config_parser.get('CFD', 'BOUNDARY_TYPE_J')
        assert(len(mark.split())==2)
        mark = mark.split(',')
        markers = (mark[0].strip(), mark[1].strip())
        return markers
    
    def GetInletValue(self):
        inlet = self.config_parser.get('CFD', 'INLET_VALUE')
        inlet = [float(x.strip()) for x in inlet.split(',')]
        return inlet
    
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
        if dir.lower() == 'x':
            return 0
        elif dir.lower() == 'y':
            return 1
        elif dir.lower() == 'z':
            return 2 
        else:
            raise ValueError('INIT_DIRECTION not recognized. Select between x,y and z')
    