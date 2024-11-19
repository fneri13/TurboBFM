import numpy as np
import pickle
import pyvista as pv
from TurboBFM.Solver.CGrid import CGrid
from TurboBFM.Solver.CVolume import CVolume

with open('Mesh/Grid_50_10_05.pik', 'rb') as file:
    grid = pickle.load(file)

geometry = CGrid(grid['X'], grid['Y'], grid['Z'])
geometry.AddGhostPoints()
elements = CVolume(geometry)

