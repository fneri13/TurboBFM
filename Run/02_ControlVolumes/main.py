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



x = np.linspace(0 , 10, 11)
y = np.linspace(0, 5 , 6)
z = np.linspace(0, 3, 4)
X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
geometry = CGrid(X, Y, Z)
elements = CMesh(geometry)



# geometry = CGrid(grid['X'], grid['Y'], grid['Z'])
# elements = CMesh(geometry)
elements.ComputeMeshQuality()
elements.PlotMeshQuality()
plt.show()

with open('Mesh/Mesh_%02i_%02i_%02i.pik' %(elements.ni, elements.nj, elements.nk), 'wb') as file:
    pickle.dump(elements, file)