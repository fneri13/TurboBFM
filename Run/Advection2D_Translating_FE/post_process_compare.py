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

sol_dir2 = '../Advection2D_Translating_RK_timeAcc/Results'
pik_files2 = [file for file in os.listdir(sol_dir2) if file.endswith('.pik')]
pik_files2 = sorted(pik_files2)

sol_dir3 = '../Advection2D_Translating_RK_lowMem/Results'
pik_files3 = [file for file in os.listdir(sol_dir3) if file.endswith('.pik')]
pik_files3 = sorted(pik_files3)


fig, ax = plt.subplots(1, 3, figsize=(18,5))
cbar1 = None
cbar2 = None
cbar3 = None
iso_contour = None
iso_contour2 = None
iso_contour3 = None
white_cnt = False

cnt_levels = [0.5]

for pik in pik_files:
    with open(sol_dir + '/' + pik, 'rb') as file:
        sol = pickle.load(file)
    
    with open(sol_dir2 + '/' + pik, 'rb') as file2:
        sol2 = pickle.load(file2)
    
    with open(sol_dir3 + '/' + pik, 'rb') as file3:
        sol3 = pickle.load(file3)

    # Remove colorbars before the next iteration
    if cbar1: cbar1.remove()
    if cbar2: cbar2.remove()
    if cbar3: cbar3.remove()
    if iso_contour: iso_contour.remove()
    if iso_contour2: iso_contour2.remove()
    if iso_contour3: iso_contour3.remove()
    
    # Plot first contour
    contour = ax[0].contourf(sol['X'][:, :, 0], sol['Y'][:, :, 0], sol['U'][:, :, 0, 0], vmin=0, vmax=1, cmap='plasma')
    cbar1 = plt.colorbar(contour, ax=ax[0])
    # Add isocontour levels in white
    if white_cnt: 
        iso_contour = ax[0].contour(sol['X'][:, :, 0], sol['Y'][:, :, 0], sol['U'][:, :, 0, 0], colors='white', levels=cnt_levels)
        ax[0].clabel(iso_contour, inline=True, fontsize=8)

    # Plot second contour
    contour2 = ax[1].contourf(sol2['X'][:, :, 0], sol2['Y'][:, :, 0], sol2['U'][:, :, 0, 0], vmin=0, vmax=1, cmap='plasma')
    cbar2 = plt.colorbar(contour2, ax=ax[1])
    # Add isocontour levels in white
    if white_cnt: 
        iso_contour2 = ax[1].contour(sol2['X'][:, :, 0], sol2['Y'][:, :, 0], sol2['U'][:, :, 0, 0], colors='white', levels=cnt_levels)
        ax[1].clabel(iso_contour2, inline=True, fontsize=8)

    # Plot third contour
    contour3 = ax[2].contourf(sol3['X'][:, :, 0], sol3['Y'][:, :, 0], sol3['U'][:, :, 0, 0], vmin=0, vmax=1, cmap='plasma')
    cbar3 = plt.colorbar(contour3, ax=ax[2])
    # Add isocontour levels in white
    if white_cnt: 
        iso_contour3 = ax[2].contour(sol3['X'][:, :, 0], sol3['Y'][:, :, 0], sol3['U'][:, :, 0, 0], colors='white', levels=cnt_levels)
        ax[2].clabel(iso_contour3, inline=True, fontsize=8)

    # Set titles
    ax[0].set_title('FE')
    ax[1].set_title('RK4')
    ax[2].set_title('RK4-LM')

    for axx in ax:
        axx.set_xticks([])
        axx.set_yticks([])

    # Pause to show the current plots
    plt.pause(0.05)

plt.show()


    


