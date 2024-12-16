import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pickle
from scipy.optimize import fsolve
from TurboBFM.Preprocess.grid_generation import transfinite_grid_generation


"""
Test case taken from section 11.5.2 of Numerical Computation of Internal and External Flows: 
The Fundamentals of Computational Fluid Dynamics (Second Edition) by Charles Hirsch.
"""

OUTPUT_FOLDER = 'Grid'
R = 0.2                 # radius of the cylinder
NX = 64                 # total number of points along x
NY = 32                 # total number of points along y
H = 10*R                # semi-height of the channel (symmetric condition in the middle)
L = 40*R                # length of the channel
STREAMWISE_COEFF = [3, 1, 3]    # stretch in x
SPANWISE_COEFF = [3, 3, 3]      # stretch in y





NX_cyl = NX//3

x1 = np.linspace(0, L/2-R, NX_cyl)
y1 = np.zeros_like(x1)

theta = np.linspace(-np.pi, 0, NX_cyl)
x2 = L/2 + R*np.cos(theta)
y2 = R*np.abs(np.sin(theta))

x3 = np.linspace(L/2+R, L, NX_cyl)
y3 = np.zeros_like(x3)

x_wall = np.concatenate((x1, x2[1:], x3[1:]))
y_wall = np.concatenate((y1, y2[1:], y3[1:]))
NX = len(x_wall)


x_inlet = np.zeros(NY)
y_inlet = np.linspace(0, H, NY)

x_outlet = np.zeros(NY)+L
y_outlet = np.linspace(0, H, NY)

x_up = x_wall.copy()
y_up = np.zeros_like(y_wall)+H

stretch_stream = ['right', 'both', 'left']
stretch_span = ['bottom', 'bottom', 'bottom']
xgrid, ygrid = transfinite_grid_generation(np.vstack((x_inlet, y_inlet)), 
                                            np.vstack((x_wall, y_wall)), 
                                            np.vstack((x_outlet, y_outlet)), 
                                            np.vstack((x_up, y_up)),
                                            stretch_type_stream='both', stretch_type_span='bottom',
                                            streamwise_coeff=1, spanwise_coeff=6)

plt.show()

# # aseemble a single block
# X = np.concatenate((Xmulti[0], Xmulti[1][1:,:], Xmulti[2][1:,:]), axis=0)
# Y = np.concatenate((Ymulti[0], Ymulti[1][1:,:], Ymulti[2][1:,:]), axis=0)

# NX, NY = X.shape


# # Create a 3D scatter plots
# mesh = pv.StructuredGrid(X, Y, np.zeros_like(X))
# plotter = pv.Plotter()
# plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
# plotter.show_axes()


# grid = {'X': X, 'Y': Y}
# # Create output directory
# if os.path.exists(OUTPUT_FOLDER):
#     print('Output Folder already present')
# else:
#     os.mkdir(OUTPUT_FOLDER)
# with open(OUTPUT_FOLDER + '/grid_%02i_%02i.pik' %(NX, NY), 'wb') as file:
#     pickle.dump(grid, file)


plotter.show()
plt.show()







