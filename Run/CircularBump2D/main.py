import numpy as np
import pickle
import matplotlib.pyplot as plt
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.CEulerSolver import CEulerSolver

print("\n" + "=" * 80)
print(" " * 25 + "🚀  Welcome to TurboBFM 🚀")
print(" " * 22 + "CFD tool for Turbomachinery BFM") 
print("=" * 80)

config = CConfig('input.ini')
with open(config.GetGridFilepath(), 'rb') as file:
    grid = pickle.load(file)

mesh = CMesh(config, grid)
solver = CEulerSolver(config, mesh)
solver.Solve()


plt.show() 