import numpy as np
import pickle
import pyvista as pv
import matplotlib.pyplot as plt
from TurboBFM.Solver.CGrid import CGrid
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CSolver import CSolver
from TurboBFM.Solver.CConfig import Config


config = Config('input.ini')
with open(config.GetGridFilepath(), 'rb') as file:
    grid = pickle.load(file)

geometry = CGrid(grid['X'], grid['Y'], grid['Z'])
mesh = CMesh(geometry)

solver = CSolver(config, mesh)
solver.InitializeSolution()

solver.Solve()
solver.ContoursCheck('primitives')
solver.ContoursCheck('conservatives')

plt.show()