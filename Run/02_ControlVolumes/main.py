import numpy as np
import pickle
import pyvista as pv
from TurboBFM.Solver.CGrid import CGrid
from TurboBFM.Solver.CMesh import CMesh

with open('Mesh/Grid_50_10_05.pik', 'rb') as file:
    grid = pickle.load(file)

# x = np.linspace(0 , 10, 11)
# y = np.linspace(0, 5 , 6)
# z = np.linspace(0, 3, 4)

# X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

geometry = CGrid(grid['X'], grid['Y'], grid['Z'])
geometry.AddGhostPoints()
elements = CMesh(geometry)

