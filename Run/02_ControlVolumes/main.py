import numpy as np
import pickle
import pyvista as pv
import matplotlib.pyplot as plt
from TurboBFM.Solver.CGrid import CGrid
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import Config

config = Config('input.ini')
with open(config.GetGridFilepath(), 'rb') as file:
    grid = pickle.load(file)

elements = CMesh(grid)
elements.ComputeMeshQuality()
elements.PlotMeshQuality()
plt.show()

with open('Mesh/Mesh_%02i_%02i_%02i.pik' %(elements.ni, elements.nj, elements.nk), 'wb') as file:
    pickle.dump(elements, file)