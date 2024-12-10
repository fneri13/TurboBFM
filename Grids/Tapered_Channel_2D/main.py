import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pickle
from TurboBFM.Preprocess.grid_generation import transfinite_grid_generation

# Create output directory
outfolder = 'Mesh'
if os.path.exists(outfolder):
    print('Output Folder already present')
else:
    os.mkdir(outfolder)



"""
READ INPUT
"""
variables = {}
with open('input.txt', 'r') as file:
    for line in file:
        name, value = line.strip().split('=')
        variables[name.strip()] = float(value.strip())
L = variables['LENGTH']
H_IN = variables['H_IN']
H_OUT = variables['H_OUT']
NZ = int(variables['NZ'])
NY = int(variables['NY'])
NX = int(variables['NX'])
SPAN = variables['SPAN']




"""
GENERATE MESH
"""
hubZ = np.linspace(0, L, NZ)
hubR = np.linspace(0, (H_IN - H_OUT)/2, NZ)
shroudZ = hubZ.copy()
shroudR = np.linspace(H_IN, H_OUT, NZ)

inletZ = np.zeros(NY)
inletR = np.linspace(0, H_IN, NY)
outletZ = np.zeros(NY)+L
outletR = np.linspace((H_IN-H_OUT)/2, (H_IN-H_OUT)/2 + H_OUT, NY)

zGrid, yGrid = transfinite_grid_generation(np.vstack((inletZ, inletR)), 
                                           np.vstack((hubZ, hubR)), 
                                           np.vstack((outletZ, outletR)), 
                                           np.vstack((shroudZ, shroudR)))

X = np.zeros((NZ, NY, NX))
Y = np.zeros((NZ, NY, NX))
Z = np.zeros((NZ, NY, NX))
for i in range(NX):
    X[:,:,i] = zGrid
    Y[:,:,i] = yGrid
    Z[:,:,i] = i*SPAN/(NX-1)


# Create a 3D scatter plot
mesh = pv.StructuredGrid(X, Y, Z)
plotter = pv.Plotter()
plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
plotter.show_axes()


grid = {'X': X, 'Y': Y, 'Z': Z}
with open(outfolder + '/Channel2D_%02i_%02i_%02i.pik' %(NZ, NY, NX), 'wb') as file:
    pickle.dump(grid, file)


plotter.show()
plt.show()







