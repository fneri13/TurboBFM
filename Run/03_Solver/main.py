import numpy as np
import pickle
import pyvista as pv
from TurboBFM.Solver.CGrid import CGrid
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CSolver import CSolver
from TurboBFM.Solver.CConfig import Config

with open('../02_ControlVolumes/Mesh/Mesh_52_12_07.pik', 'rb') as file:
    mesh = pickle.load(file)
config = Config('input.ini')


solver = CSolver(config, mesh)