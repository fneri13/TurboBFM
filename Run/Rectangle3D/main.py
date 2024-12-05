import pickle
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.CSolver import CSolver

print("\n" + "=" * 80)
print(" " * 25 + "🚀  Welcome to TurboBFM 🚀")
print(" " * 22 + "CFD tool for Turbomachinery BFM")
print("=" * 80)

config = CConfig('input.ini')
with open(config.GetGridFilepath(), 'rb') as file:
    grid = pickle.load(file)

mesh = CMesh(config, grid)
solver = CSolver(config, mesh)
solver.Solve()