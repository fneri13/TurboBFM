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


fig, ax = plt.subplots()
cbar = None  

for pik in pik_files:
    with open(sol_dir + '/' + pik, 'rb') as file:
        sol = pickle.load(file)
    
    contour = ax.contourf(sol['X'][:, :, 0], sol['Y'][:, :, 0], sol['U'][:, :, 0, 0], vmin=0, vmax=1, cmap='plasma')
    if cbar:
        cbar.remove()
    cbar = fig.colorbar(contour, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.pause(0.05)

plt.show()
    


