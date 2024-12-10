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
STREAMWISE_COEFF = 2
SPANWISE_COEFF = 2





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

# three blocks
x_wall = [x1, x2, x3]
y_wall = [y1, y2, y3]

x_inlet = [np.zeros(NY),
           np.zeros(NY)+x1[-1],
           np.zeros(NY)+x2[-1]]
y_inlet = [np.linspace(0, L, NY),
           np.linspace(0, L, NY),
           np.linspace(0, L, NY)]

x_outlet = [np.zeros(NY)+x1[-1],
           np.zeros(NY)+x2[-1],
           np.zeros(NY)+x3[-1]]
y_outlet = [np.linspace(0, L, NY),
           np.linspace(0, L, NY),
           np.linspace(0, L, NY)]

x_up = [np.linspace(0, L, NX_bump),
        np.linspace(L, 2*L, NX_bump),
        np.linspace(2*L, 3*L, NX_bump)]
y_up = [np.zeros(NX_bump)+L,
        np.zeros(NX_bump)+L,
        np.zeros(NX_bump)+L]

Xmulti, Ymulti = [], []
stretch_stream = ['right', 'both', 'left']
stretch_span = ['bottom', 'bottom', 'bottom']
for i in range(3):
    xgrid, ygrid = transfinite_grid_generation(np.vstack((x_inlet[i], y_inlet[i])), 
                                               np.vstack((x_wall[i], y_wall[i])), 
                                               np.vstack((x_outlet[i], y_outlet[i])), 
                                               np.vstack((x_up[i], y_up[i])),
                                               stretch_type_stream=stretch_stream[i], stretch_type_span=stretch_span[i],
                                               streamwise_coeff=STREAMWISE_COEFF, spanwise_coeff=SPANWISE_COEFF)
    Xmulti.append(xgrid)
    Ymulti.append(ygrid)
    plt.show()

# aseemble a single block
X = np.concatenate((Xmulti[0], Xmulti[1][1:,:], Xmulti[2][1:,:]), axis=0)
Y = np.concatenate((Ymulti[0], Ymulti[1][1:,:], Ymulti[2][1:,:]), axis=0)

NX, NY = X.shape


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







