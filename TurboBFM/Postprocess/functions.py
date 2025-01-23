import numpy as np
import matplotlib.pyplot as plt
from TurboBFM.Postprocess.styles import N_levels, color_map

def contour_template(z, r, f, name, vmin=None, vmax=None, save_filename=None):
        """
        Template function to create contours.

        Parameters
        -----------------------------------

        `z`: 2d array of x coordinates
        
        `r`: 2d array of y coordinates

        `f`: 2d array of function values

        `name`: string name of the plot title

        `vmin`: minimum value to truncate the color range

        `vmax`: max value to truncate the color range

        """
        if vmin == None:
            minval = np.min(f)
        else:
            minval = vmin
        if vmax == None:
            maxval = np.max(f)
        else:
            maxval = vmax
        
        if minval==maxval:
             maxval += 1e-16
        levels = np.linspace(minval, maxval, N_levels)
        fig, ax = plt.subplots()
        contour = ax.contourf(z, r, f, levels=levels, cmap=color_map, vmin = minval, vmax = maxval)
        cbar = fig.colorbar(contour)
        contour = ax.contour(z, r, f, levels=levels, colors='black', vmin = minval, vmax = maxval, linewidths=0.1)
        plt.title(name)
        ax.set_aspect('equal', adjustable='box')
        if save_filename is not None:
             plt.savefig(save_filename + '.pdf', bbox_inches='tight')