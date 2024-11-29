import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pickle
from TurboBFM.Preprocess.grid_generation import transfinite_grid_generation
from TurboBFM.Preprocess.grid_generation import eriksson_stretching_function_initial




"""
Generate a 2D rectangular geometry, that will be used as verification
"""
OUTPUT_FOLDER = 'Grid'
R = 0.1
R_OUT = 5*R
NR = 50
NTHETA = 50
THETA_MAX = np.pi
SPAN = 0.02
NZ = 5

ir = eriksson_stretching_function_initial(np.linspace(0,1,NR), 3)
r = np.zeros(NR)
for i in range(NR):
    r[i] = R+ir[i]*(R_OUT-R)
theta = np.linspace(0, THETA_MAX, NTHETA)
z = np.linspace(0, SPAN, NZ)

X, Y, Z = np.zeros((NR, NTHETA, NZ)), np.zeros((NR, NTHETA, NZ)), np.zeros((NR, NTHETA, NZ))
for i in range(NR):
    for j in range(NTHETA):
        for k in range(NZ):
            X[i,j,k] = r[i]*np.cos(theta[j])
            Y[i,j,k] = r[i]*np.sin(theta[j])
            Z[i,j,k] = z[k]


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
with open(OUTPUT_FOLDER + '/grid_%02i_%02i_%02i.pik' %(NR, NTHETA, NZ), 'wb') as file:
    pickle.dump(grid, file)


plotter.show()
plt.show()







