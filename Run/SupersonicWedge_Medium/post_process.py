import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from TurboBFM.Postprocess.CPostProcess import CPostProcess

sol_dir = 'Results'
pik_files = [sol_dir + '/' + file for file in os.listdir(sol_dir) if file.endswith('.pik')]
pik_files = sorted(pik_files)
 
proc = CPostProcess(pik_files[-1])
proc.PlotResiduals(drop=True, save_filename='Residuals', dim=2)
proc.Contour2D('Mach', save_filename='Contour')
proc.Contour2D('ux', save_filename='Contour')
proc.Contour2D('uy', save_filename='Contour')
proc.Contour2D('p', save_filename='Contour')

proc.Plot1D('p', 'j', 0, 0)
    
    
plt.show()
