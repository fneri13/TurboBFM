import numpy as np
import pickle
import pyvista as pv
from TurboBFM.Solver.CGrid import CGrid
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CSolver import CSolver

with open('../02_ControlVolumes/Mesh/Mesh_52_12_07.pik', 'rb') as file:
    mesh = pickle.load(file)

FLUID_NAME = 'air'
FLUID_MODEL = 'ideal'
FLUID_GAMMA = 1.4

solver = CSolver(mesh, FLUID_NAME, FLUID_MODEL, FLUID_GAMMA)