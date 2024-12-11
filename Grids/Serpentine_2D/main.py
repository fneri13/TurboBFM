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
NX = 60
NY = 20
L = 1
H = 0.3
N_WAVES = 1

x = np.linspace(0, L, NX)
y = L/5*np.sin(N_WAVES*2*np.pi*x/L)

xgrid, ygrid = np.zeros((NX, NY)), np.zeros((NX, NY))

for i in range(NX):
    for j in range(NY):
        xgrid[i,j] = x[i]
        ygrid[i,j] = y[i] + j*H/(NY-1)


grid = {'X': xgrid, 'Y': ygrid}
# Create output directory
if os.path.exists(OUTPUT_FOLDER):
    print('Output Folder already present')
else:
    os.mkdir(OUTPUT_FOLDER)
with open(OUTPUT_FOLDER + '/grid_%02i_%02i.pik' %(NX, NY), 'wb') as file:
    pickle.dump(grid, file)

plt.show()







