import configparser
import ast
import numpy as np
import os


class Config:
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
    
    def GetFluidModel(self):
        return str(self.config_parser.get('CFD', 'FLUID_MODEL'))