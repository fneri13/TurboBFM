import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from TurboBFM.Postprocess.CPostProcess import CPostProcess

sol_dir = 'Results'
pik_files = [sol_dir + '/' + file for file in os.listdir(sol_dir) if file.endswith('.pik')]
pik_files = sorted(pik_files)
 
proc = CPostProcess(pik_files[-1])
proc.PlotResiduals(save_filename='Residuals', dim=3)
proc.PlotMassFlow(save_filename='MassFlow', dim=3)

proc.Contour2D('rho', save_filename='Contour')
proc.Contour2D('Mach', save_filename='Contour')
proc.Contour2D('p', save_filename='Contour')
proc.Contour2D('ur', save_filename='Contour')
proc.Contour2D('ut', save_filename='Contour')
proc.Contour2D('ua', save_filename='Contour')
proc.Plot1D('p', 'j', 0, save_filename='Plot')
proc.Plot1D('Mach', 'j', 0, save_filename='Plot')

plt.show()

