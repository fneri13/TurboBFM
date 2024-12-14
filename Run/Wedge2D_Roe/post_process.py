import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from TurboBFM.Postprocess.CPostProcess import CPostProcess

sol_dir = 'Results'
pik_files = [sol_dir + '/' + file for file in os.listdir(sol_dir) if file.endswith('.pik')]
pik_files = sorted(pik_files)
 
proc = CPostProcess(pik_files[-1])
proc.PlotResiduals(drop=False, save_filename='Residuals')

proc.Contour2D('rho', save_filename='Contour')
proc.Contour2D('ux', save_filename='Contour')
proc.Contour2D('uy', save_filename='Contour')

proc.Contour2D('Mach', save_filename='Contour')
proc.Contour2D('p', save_filename='Contour')
    
    
plt.show()

