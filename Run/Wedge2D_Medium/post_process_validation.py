import numpy as np
import csv
import pickle
import os
import matplotlib.pyplot as plt
from TurboBFM.Postprocess.CPostProcess import CPostProcess

### MY RESULTS
sol_dir = 'Results'
pik_files = [sol_dir + '/' + file for file in os.listdir(sol_dir) if file.endswith('.pik')]
pik_files = sorted(pik_files)
proc = CPostProcess(pik_files[-1])
proc.Contour2D('Mach')
proc.Contour2D('p')
proc.PlotResiduals()

# SU2 RESULTS
def read_csv_to_dict(filename):
    data_dict = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the first row as variable names
        
        # Initialize the dictionary with empty lists for each variable
        data_dict = {header: [] for header in headers}
        
        # Read the rest of the rows and store the data
        for row in reader:
            for header, value in zip(headers, row):
                data_dict[header].append(float(value))  # Convert to float
                
    # Convert lists to numpy arrays for better numerical operations
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])
    
    return data_dict

filename = 'su2_63_48.csv'  # Replace with your CSV file path
data = read_csv_to_dict(filename)

x = data['Points_0']
pressure = data['Pressure']/1e3
rho = data['Density']
ux = data['Velocity_0']
uy = data['Velocity_1']
T = data['Temperature']
mach = data['Mach']

# PLOT VALIDATION
proc.Plot1D('p', 'j', 0, save_filename='Plot', ref_points=(x, pressure, 'SU2'))
proc.Plot1D('Mach', 'j', 0, save_filename='Plot', ref_points=(x, mach, 'SU2'))
proc.Plot1D('rho', 'j', 0, save_filename='Plot', ref_points=(x, rho, 'SU2'))
proc.Plot1D('ux', 'j', 0, save_filename='Plot', ref_points=(x, ux, 'SU2'))
proc.Plot1D('uy', 'j', 0, save_filename='Plot', ref_points=(x, uy, 'SU2'))
proc.Plot1D('T', 'j', 0, save_filename='Plot', ref_points=(x, T, 'SU2'))
    
plt.show()

