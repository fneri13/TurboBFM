import pickle
import matplotlib.pyplot as plt
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CSolver import CSolver
from TurboBFM.Solver.CConfig import CConfig


with open('../02_ControlVolumes/Mesh/Mesh_100_20_05.pik', 'rb') as file:
    mesh = pickle.load(file)
config = CConfig('input.ini')
solver = CSolver(config, mesh)
solver.Solve()


plt.show()