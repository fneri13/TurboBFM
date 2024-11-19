import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pickle
from TurboBFM.Preprocess.grid_generation import transfinite_grid_generation
from TurboBFM.Preprocess.grid_generation import compute_three_dimensional_mesh

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
R1 = variables['INNER_RADIUS']
R2 = variables['OUTER_RADIUS']
NZ = int(variables['NZ'])
NR = int(variables['NR'])
NTHETA = int(variables['NTHETA'])
THETA_MAX = variables['THETA_MAX']




"""
GENERATE MESH
"""
hubZ = np.linspace(0, L, NZ)
hubR = np.zeros_like(hubZ) + R1
shroudZ = hubZ.copy()
shroudR = np.zeros_like(hubZ) + R2

inletZ = np.zeros(NR)
inletR = np.linspace(R1, R2, NR)
outletZ = np.zeros(NR)+L
outletR = inletR.copy()

zGrid, rGrid = transfinite_grid_generation(np.vstack((inletZ, inletR)), 
                                           np.vstack((hubZ, hubR)), 
                                           np.vstack((outletZ, outletR)), 
                                           np.vstack((shroudZ, shroudR)))

X,Y,Z = compute_three_dimensional_mesh(zGrid, rGrid, THETA_MAX, NTHETA)


# Create a 3D scatter plot
mesh = pv.StructuredGrid(X, Y, Z)
plotter = pv.Plotter()
plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
plotter.show_axes()


grid = {'X': X, 'Y': Y, 'Z': Z}
with open(outfolder + '/Grid_%02i_%02i_%02i.pik' %(NZ, NR, NTHETA), 'wb') as file:
    pickle.dump(grid, file)


plotter.show()
plt.show()







