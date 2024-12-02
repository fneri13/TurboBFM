import numpy as np
import pickle
import matplotlib.pyplot as plt
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import CConfig

config = CConfig('input.ini')
with open(config.GetGridFilepath(), 'rb') as file:
    grid = pickle.load(file)

mesh = CMesh(config, grid)
mesh.PlotMeshQuality()
# mesh.PlotElement((100, 15, 0))

with open('Mesh/Mesh_%02i_%02i_%02i.pik' %(mesh.ni, mesh.nj, mesh.nk), 'wb') as file:
    pickle.dump(mesh, file)

plt.show()