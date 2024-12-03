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
L = 1
NX = 32
NY = 16
SPAN = L/32

def func(alpha):
    y = np.cos(alpha)/2/np.sin(alpha) + 0.1-1/2/np.sin(alpha)
    return y
alpha = fsolve(func, 1)[0]
r_bump = 1/2/np.sin(alpha)

NX_bump = NX//3

x1 = np.linspace(0, L, NX_bump)
y1 = np.zeros_like(x1)

theta = np.linspace(0, 2*alpha, NX_bump)
x2 = 1.5*L + r_bump*np.cos(np.pi/2+alpha-theta)
y2 = -(r_bump-0.1*L)+r_bump*np.sin(np.pi/2+alpha-theta)

x3 = np.linspace(2*L, 3*L, NX_bump)
y3 = np.zeros_like(x3)

x_wall = np.concatenate((x1, x2[1:-1], x3))
y_wall = np.concatenate((y1, y2[1:-1], y3))
NX = len(x_wall)


x_outlet = np.zeros(NY)+3*L
y_outlet = np.zeros(NY)

x_up = np.linspace(0, 3*L, NX)
y_up = np.zeros_like(x_up)+L

x_inlet = np.zeros(NY)
y_inlet = np.linspace(0, L, NY)


xgrid, ygrid = transfinite_grid_generation(np.vstack((x_inlet, y_inlet)), 
                                           np.vstack((x_wall, y_wall)), 
                                           np.vstack((x_outlet, y_outlet)), 
                                           np.vstack((x_up, y_up)))


# Create a 3D scatter plots
mesh = pv.StructuredGrid(xgrid, ygrid, np.zeros_like(xgrid))
plotter = pv.Plotter()
plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
plotter.show_axes()


grid = {'X': xgrid, 'Y': ygrid}
# Create output directory
if os.path.exists(OUTPUT_FOLDER):
    print('Output Folder already present')
else:
    os.mkdir(OUTPUT_FOLDER)
with open(OUTPUT_FOLDER + '/grid_%02i_%02i.pik' %(NX, NY), 'wb') as file:
    pickle.dump(grid, file)


plotter.show()
plt.show()







