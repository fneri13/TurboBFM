import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from TurboBFM.Solver.CMesh import CMesh
from TurboBFM.Solver.CConfig import CConfig
from TurboBFM.Solver.CEulerSolver import CEulerSolver
from TurboBFM.Solver.CAdvectionSolver import CAdvectionSolver

sol_dir = 'Results'
pik_files = [file for file in os.listdir(sol_dir) if file.endswith('.pik')]
pik_files = sorted(pik_files)

sol_dir2 = '../Advection2D_Translating_RK/Results'
pik_files2 = [file for file in os.listdir(sol_dir2) if file.endswith('.pik')]
pik_files2 = sorted(pik_files2)


fig, ax = plt.subplots(1, 2, figsize=(13,5))
cbar1 = None
cbar2 = None

for pik in pik_files:
    with open(sol_dir + '/' + pik, 'rb') as file:
        sol = pickle.load(file)
    
    with open(sol_dir2 + '/' + pik, 'rb') as file2:
        sol2 = pickle.load(file2)

    # Remove colorbars before the next iteration
    if cbar1: cbar1.remove()
    if cbar2: cbar2.remove()
    
    # Plot first contour
    contour = ax[0].contourf(sol['X'][:, :, 0], sol['Y'][:, :, 0], sol['U'][:, :, 0, 0], vmin=0, vmax=1, cmap='plasma', levels=10)
    cbar1 = plt.colorbar(contour, ax=ax[0])
    
    # Plot second contour
    contour2 = ax[1].contourf(sol2['X'][:, :, 0], sol2['Y'][:, :, 0], sol2['U'][:, :, 0, 0], vmin=0, vmax=1, cmap='plasma', levels=10)
    cbar2 = plt.colorbar(contour2, ax=ax[1])

    # Set titles
    ax[0].set_title('Forward Euler')
    ax[1].set_title('Runge Kutta 4')

    # Pause to show the current plots
    plt.pause(0.05)

plt.show()


    


