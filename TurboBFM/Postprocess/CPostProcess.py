import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from TurboBFM.Postprocess import styles

class CPostProcess():
    
    def __init__(self, pik_path):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        with open(pik_path, 'rb') as file:
            self.data = pickle.load(file)
        
        self.pictures_folder = 'Pictures'
        os.makedirs(self.pictures_folder, exist_ok=True)
    

    def PlotResiduals(self, drop=True, save_filename=None):
        """
        Plot the residuals. If drop=True shift to zero value at first iteration
        """

        def shift(y):
            return y-y[0]

        names = [r'$R(\rho)$',
                 r'$R(\rho u_x)$',
                 r'$R(\rho u_y)$',
                 r'$R(\rho u_z)$',
                 r'$R(\rho e_t)$']
        plt.figure()

        for i in range(len(self.data['Res'])):
            if drop:
                plt.plot(shift(self.data['Res'][i]), label=names[i])
            else:
                plt.plot(self.data['Res'][i], label=names[i])
        plt.grid(alpha = styles.grid_opacity)
        plt.xlabel('Iterations')
        if drop:
            plt.ylabel('Residuals Drop')
        else:
            plt.ylabel('Residuals')
        plt.legend()
        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '.pdf', bbox_inches='tight')
    

    def Contour2D(self, field_name, idx_k=0, cbar = 'h', save_filename=None):
        """
        Plot the residuals. Specify the field name you want to draw the contour
        """
        if field_name.lower()=='rho':
            name = 'Density'
            label = r'$\rho \ \rm{[kg/m^3]}$'
            field = self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='ux':
            name = 'VelocityX'
            label = r'$u_x \ \rm{[m/s]}$'
            field = self.data['U'][:,:,idx_k,1]/self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='uy':
            name = 'VelocityY'
            label = r'$u_y \ \rm{[m/s]}$'
            field = self.data['U'][:,:,idx_k,2]/self.data['U'][:,:,idx_k,0]
        
        X = self.data['X'][:,:,idx_k]
        Y = self.data['Y'][:,:,idx_k]

        plt.figure()
        contour = plt.contourf(X, Y, field, levels=styles.N_levels, cmap=styles.color_map, vmin=field.min(), vmax=field.max())
        plt.xticks([])
        plt.yticks([])
        plt.title(label)
        
        if cbar == 'h':
            cbar = plt.colorbar(contour, orientation='horizontal', pad=0.025)
        else:
            cbar = plt.colorbar(contour)
        
        ax = plt.gca()  # Get the current axes
        ax.set_aspect('equal')  # Set the aspect ratio to equal
    
        cbar.set_ticks([field.min(), (field.min()+field.max())/2, field.max()])
        cbar.ax.set_xticklabels([f"{field.min():.3f}", f"{(field.min()+field.max())/2:.3f}", f"{field.max():.3f}"])  # Format as needed

        # Set custom formatting for the ticks
        # cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '_' + name + '.pdf', bbox_inches='tight')

        
        