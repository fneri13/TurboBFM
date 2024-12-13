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
    
    rho = sol['U'][:,:,0,0]
    ux = sol['U'][:,:,0,1]/rho
    uy = sol['U'][:,:,0,2]/rho
    u = np.sqrt(ux**2+uy**2)
    
    contour = ax.contourf(sol['X'][:, :, 0], sol['Y'][:, :, 0], ux, cmap='coolwarm', levels=20)
    if cbar:
        cbar.remove()
    cbar = fig.colorbar(contour, ax=ax)
    # ax.quiver(sol['X'][:, :, 0], sol['Y'][:, :, 0],
    #           ux, uy)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.pause(0.25)
    
plt.show()

