import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pickle

with open('mesh_123_40_02_0.403-deg.pickle', 'rb') as file:
    coords = pickle.load(file)
OUTPUT_FOLDER = 'Grid'

X, Y, Z = coords['x'], coords['y'], coords['z']
NZ, NR, NTHETA = X.shape
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
with open(OUTPUT_FOLDER + '/grid_%02i_%02i_%02i.pik' %(NZ, NR, NTHETA), 'wb') as file:
    pickle.dump(grid, file)


plotter.show()
plt.show()







