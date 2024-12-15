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

filename = 'su2_samemesh.csv'  # Replace with your CSV file path
data = read_csv_to_dict(filename)

x = data['Points_0']
pressure = data['Pressure']/1e3
cp = data['Pressure_Coefficient']
cp = (cp - cp.min())/ (cp.max() - cp.min())
mach = data['Mach']

# PLOT VALIDATION
proc.Plot1D('p', 'j', 0, save_filename='Plot', ref_points=(x, pressure, 'SU2'))
proc.Plot1D('Mach', 'j', 0, save_filename='Plot', ref_points=(x, mach, 'SU2'))
proc.Plot1D('cp', 'j', 0, save_filename='Plot', ref_points=(x, cp, 'SU2'))
    
plt.show()

