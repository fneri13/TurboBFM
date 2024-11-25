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
NX = 20
NY = 10
NZ = 2

x = np.linspace(0, NX-1, NX)
y = np.linspace(0, NY-1, NY)
if NZ>1:
    z = np.linspace(0, NZ-1, NZ)
else:
    z = np.array([0])

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


# Create a 3D scatter plots
mesh = pv.StructuredGrid(X, Y, Z)
plotter = pv.Plotter()
plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
plotter.show_axes()


grid = {'X': X, 'Y': Y, 'Z': Z}
# Create output directory
if os.path.exists(OUTPUT_FOLDER):
    print('Output Folder already present')
else:
    os.mkdir(OUTPUT_FOLDER)
with open(OUTPUT_FOLDER + '/grid_%02i_%02i_%02i.pik' %(NX, NY, NZ), 'wb') as file:
    pickle.dump(grid, file)


plotter.show()
plt.show()







