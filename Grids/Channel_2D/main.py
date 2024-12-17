import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pickle
from TurboBFM.Preprocess.grid_generation import transfinite_grid_generation




"""
Generate a 2D rectangular geometry, that will be used as verification
"""
OUTPUT_FOLDER = 'Grid'
LX = 3
R_IN = 1
R_OUT = 2
NX = 48
NY = 16

x = np.linspace(0, LX, NX)
y = np.linspace(R_IN, R_OUT, NY)



X, Y = np.meshgrid(x, y, indexing='ij')


# Create a 3D scatter plots
mesh = pv.StructuredGrid(X, Y, np.zeros_like(X))
plotter = pv.Plotter()
plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
plotter.show_axes()


grid = {'X': X, 'Y': Y}
# Create output directory
if os.path.exists(OUTPUT_FOLDER):
    print('Output Folder already present')
else:
    os.mkdir(OUTPUT_FOLDER)
with open(OUTPUT_FOLDER + '/grid_%02i_%02i.pik' %(NX, NY), 'wb') as file:
    pickle.dump(grid, file)


plotter.show()
plt.show()







