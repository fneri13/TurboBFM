import numpy as np
import pickle
import pyvista as pv
from TurboBFM.Solver.CGrid import CGrid
from TurboBFM.Solver.CMesh import CMesh

with open('../01_GridGeneration/Mesh/Grid_50_10_05.pik', 'rb') as file:
    grid = pickle.load(file)

geometry = CGrid(grid['X'], grid['Y'], grid['Z'])
geometry.AddGhostPoints()
elements = CMesh(geometry)

with open('Mesh/Mesh_%02i_%02i_%02i.pik' %(elements.ni, elements.nj, elements.nk), 'wb') as file:
    pickle.dump(elements, file)