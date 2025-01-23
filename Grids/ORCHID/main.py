import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pickle
from scipy.optimize import fsolve
from TurboBFM.Preprocess.grid_generation import transfinite_grid_generation
from TurboBFM.Preprocess.su2_mesh_generator import generate_SU2mesh

NX = 500
NY = 100
NTOT = NX*NY

data = np.loadtxt('Nominal_geometry.txt')
x = np.array([0])
y = np.array([0.012698519])
x_wall = np.concatenate((x, data[:,0]))
y_wall = np.concatenate((y, data[:,1]))

x_in = np.zeros(NY)+x[0]
y_in = np.linspace(0, y_wall[0], NY)

x_bottom = x_wall.copy()
y_bottom = np.zeros_like(x_bottom)

x_out = x_in = np.zeros(NY)+x[-1]
y_out = np.linspace(0, y_wall[-1], NY)

plt.figure()
plt.plot(x, y)


xgrid, ygrid = transfinite_grid_generation(np.vstack((x_in, y_in)), 
                                            np.vstack((x_bottom, y_bottom)), 
                                            np.vstack((x_out, y_out)), 
                                            np.vstack((x_wall, y_wall)),
                                            stretch_type_stream='both', stretch_type_span='both',
                                            streamwise_coeff=1, spanwise_coeff=1.5, 
                                            nx=NX, ny=NY)

for i in range(NX):
    xgrid[i,1:] = xgrid[i,0]

plt.figure()
for i in range(NX):
    plt.plot(xgrid[i, :], ygrid[i, :], 'k', lw=0.5)
for j in range(NY):
    plt.plot(xgrid[:, j], ygrid[:, j], 'k', lw=0.5)
ax = plt.gca()
ax.set_aspect('equal')

generate_SU2mesh(xgrid, ygrid, kind_elem=9, kind_bound=3, filename='ORCHID_%i_%i_%i' %(NX,NY,NTOT))


plt.show()







