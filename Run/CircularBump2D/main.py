import numpy as np
import pickle
import matplotlib.pyplot as plt
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.CEulerSolver import CEulerSolver
from TurboBFM.Solver.CAdvectionSolver import CAdvectionSolver
from TurboBFM.Solver.CLaplaceSolver import CLaplaceSolver

print("\n" + "=" * 80)
print(" " * 25 + "🚀  Welcome to TurboBFM 🚀")
print(" " * 22 + "CFD tool for Turbomachinery BFM") 
print("=" * 80)

config = CConfig('input.ini')
with open(config.GetGridFilepath(), 'rb') as file:
    grid = pickle.load(file)

mesh = CMesh(config, grid)
kind_solver = config.GetKindSolver()
if kind_solver=='Euler':
    solver = CEulerSolver(config, mesh)
elif kind_solver=='Advection':
    solver = CAdvectionSolver(config, mesh)
elif kind_solver=='Laplace':
    solver = CLaplaceSolver(config, mesh)
solver.InstantiateFields()
solver.ReadBoundaryConditions()
solver.InitializeSolution()
solver.PrintInfoSolver()
solver.Solve()


plt.show() 