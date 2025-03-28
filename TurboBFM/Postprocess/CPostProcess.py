import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from TurboBFM.Postprocess import styles
from TurboBFM.Solver.CFluid import FluidIdeal

class CPostProcess():
    
    def __init__(self, pik_path):
        """
        Instantiate the Euler Solver, by using the information contained in the mesh object (Points, Volumes, and Surfaces)
        """
        with open(pik_path, 'rb') as file:
            self.data = pickle.load(file)
        
        self.pictures_folder = 'Pictures'
        os.makedirs(self.pictures_folder, exist_ok=True)

        self.fluid = FluidIdeal(1.4, 287.052874) # ideal gas functions
    

    def PlotResiduals(self, drop=True, save_filename=None, dim=2):
        """
        Plot the residuals. If drop=True shift to zero value at first iteration. If dim=2 it doesn't plot the residual for the z-momentum
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
            if i==3 and dim==2:
                pass
            else:
                if drop:
                    plt.plot(shift(self.data['Res'][i]), label=names[i])
                else:
                    plt.plot(self.data['Res'][i], label=names[i])
        plt.grid(alpha = styles.grid_opacity)
        plt.xlabel('Iterations [-]')
        if drop:
            plt.ylabel('Residuals Drop [-]')
        else:
            plt.ylabel('Residuals [-]')
        plt.legend()
        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '.pdf', bbox_inches='tight')


    def PlotMassFlow(self, save_filename=None, dim=2, topology='axisymmetric'):
        """
        Plot the residuals. If drop=True shift to zero value at first iteration. If dim=2 it doesn't plot the residual for the z-momentum
        """

        names = [r'$\dot{m}_{I,IN}$',
                 r'$\dot{m}_{I,OUT}$',
                 r'$\dot{m}_{J,IN}$',
                 r'$\dot{m}_{J,OUT}$',
                 r'$\dot{m}_{K,IN}$',
                 r'$\dot{m}_{K,OUT}$']
        
        markers = ['-o', '-s', '->', '-<', '-^', '-o']
        
        if topology == 'axisymmetric':
            rangeLoop = 4
        else:
            rangeLoop = len(names)
        
        plt.figure()
        for i in range(rangeLoop):
            plt.plot(self.data['MassFlow'][i], markers[i], mfc='none', label=names[i])
        plt.grid(alpha = styles.grid_opacity)
        plt.xlabel('Iterations [-]')
        plt.ylabel(r'Mass Flow $[kg/s]$')
        plt.legend()
        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '.pdf', bbox_inches='tight')
    

    def PlotTurboPerformance(self, save_filename=None, axisymmetric=True):
        """
        Plot the turbo performance in the iterations
        """
        plt.figure()
        plt.plot(self.data['PRtt'])
        plt.grid(alpha = styles.grid_opacity)
        plt.xlabel('Iterations [-]')
        plt.ylabel(r'$\beta_{tt}$ [-]')
        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '_beta_tt.pdf', bbox_inches='tight')
        
        plt.figure()
        plt.plot(self.data['ETAtt'])
        plt.grid(alpha = styles.grid_opacity)
        plt.xlabel('Iterations [-]')
        plt.ylabel(r'$\eta_{tt}$ [-]')
        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '_eta_tt.pdf', bbox_inches='tight')
        
        mflow = np.array(self.data['MassFlowTurbo'])
        if axisymmetric: mflow*=360  # total mass flow rate for axissymetric simulations
        plt.figure()
        plt.plot(mflow)
        plt.grid(alpha = styles.grid_opacity)
        plt.xlabel('Iterations [-]')
        plt.ylabel(r'$\dot{m}$ [kg/s]')
        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '_mflow.pdf', bbox_inches='tight')

    

    def Contour2D(self, field_name, idx_k=0, cbar = 'h', save_filename=None, quiver_plot=False):
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
        elif field_name.lower()=='ur':
            name = 'VelocityRadial'
            label = r'$u_r \ \rm{[m/s]}$'
            field = self.ComputeRadialVelocity(self.data['U'][:,:,idx_k,:], self.data['Y'][:,:,idx_k], self.data['Z'][:,:,idx_k])
        elif field_name.lower()=='ut':
            name = 'VelocityTangential'
            label = r'$u_{\theta} \ \rm{[m/s]}$'
            field = self.ComputeTangentialVelocity(self.data['U'][:,:,idx_k,:], self.data['Y'][:,:,idx_k], self.data['Z'][:,:,idx_k])
        elif field_name.lower()=='ua':
            name = 'VelocityAxial'
            label = r'$u_{ax} \ \rm{[m/s]}$'
            field = self.data['U'][:,:,idx_k,1]/self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='uy':
            name = 'VelocityY'
            label = r'$u_y \ \rm{[m/s]}$'
            field = self.data['U'][:,:,idx_k,2]/self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='uz':
            name = 'VelocityZ'
            label = r'$u_z \ \rm{[m/s]}$'
            field = self.data['U'][:,:,idx_k,3]/self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='mach':
            name = 'Mach'
            label = r'$M \ \rm{[-]}$'
            field = self.ComputeMachNumber(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='p':
            name = 'Pressure'
            label = r'$p \ \rm{[kPa]}$'
            field = self.ComputePressure(self.data['U'][:,:,idx_k,:])/1e3
        elif field_name.lower()=='pt':
            name = 'TotalPressure'
            label = r'$p_t \ \rm{[kPa]}$'
            field = self.ComputeTotalPressure(self.data['U'][:,:,idx_k,:])/1e3
        elif field_name.lower()=='tt':
            name = 'TotalTemperature'
            label = r'$T_t \ \rm{[K]}$'
            field = self.ComputeTotalTemperature(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='t':
            name = 'Temperature'
            label = r'$T \ \rm{[K]}$'
            field = self.ComputeTemperature(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='betatt':
            name = 'PRtt'
            label = r'$\beta_{tt} \ \rm{[-]}$'
            field = self.ComputeBetaTT(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='trtt':
            name = 'TRtt'
            label = r'$TR_{tt} \ \rm{[-]}$'
            field = self.ComputeTrTT(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='s':
            name = 'Entropy'
            label = r'$s \ \rm{[kJ/kgK]}$'
            field = self.ComputeEntropy(self.data['U'][:,:,idx_k,:])/1e3
        else:
            raise ValueError('Unknown field to plot')
        
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

        if quiver_plot:
            ux = self.data['U'][:,:,idx_k,1]/self.data['U'][:,:,idx_k,0]
            uy = self.data['U'][:,:,idx_k,2]/self.data['U'][:,:,idx_k,0]
            ax.quiver(X, Y, ux, uy)

    
        cbar.set_ticks([field.min(), (field.min()+field.max())/2, field.max()])
        cbar.ax.set_xticklabels([f"{field.min():.3f}", f"{(field.min()+field.max())/2:.3f}", f"{field.max():.3f}"])  # Format as needed

        contour = plt.contour(X, Y, field, levels=styles.N_levels, colors='black', vmin = field.min(), vmax = field.max(), linewidths=0.1)

        # Set custom formatting for the ticks
        # cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '_' + name + '.pdf', bbox_inches='tight')

    
    def ComputeMachNumber(self, data):
        rho = data[:,:,0]
        ux = data[:,:,1]/data[:,:,0]
        uy = data[:,:,2]/data[:,:,0]
        uz = data[:,:,3]/data[:,:,0]
        umag = np.sqrt(ux**2+uy**2+uz**2)
        et = data[:,:,-1]/data[:,:,0]
        M = self.fluid.ComputeMachNumber_rho_umag_et(rho, umag, et)
        return M
    
    def ComputePressure(self, data):
        rho = data[:,:,0]
        ux = data[:,:,1]/data[:,:,0]
        uy = data[:,:,2]/data[:,:,0]
        uz = data[:,:,3]/data[:,:,0]
        umag = np.sqrt(ux**2+uy**2+uz**2)
        et = data[:,:,-1]/data[:,:,0]
        p = self.fluid.ComputePressure_rho_u_et(rho, umag, et)
        return p


    def ComputeTotalPressure(self, data):
        p = self.ComputePressure(data)
        M = self.ComputeMachNumber(data)
        pt = p*(1+(self.fluid.gmma-1)/2*M**2)**(self.fluid.gmma/(self.fluid.gmma-1))
        return pt


    def ComputeBetaTT(self, data):
        pt = self.ComputeTotalPressure(data)
        betaTT = np.zeros_like(pt)
        for i in range(pt.shape[0]):
            betaTT[i,:] = pt[i,:]/pt[0,:]
        return betaTT
    

    def ComputeEtaTT(self, data):
        betaTT = self.ComputeBetaTT(data)
        TrTT = self.ComputeTrTT(data)
        etaTT = (betaTT**((self.fluid.gmma-1)/self.fluid.gmma)-1)/(TrTT-1+1e-12)
        return etaTT
    

    def ComputeTrTT(self, data):
        Tt = self.ComputeTotalTemperature(data)
        TrTT = np.zeros_like(Tt)
        for i in range(Tt.shape[0]):
            TrTT[i,:] = Tt[i,:]/Tt[0,:]
        return TrTT
    

    def ComputeTemperature(self, data):
        rho = data[:,:,0]
        ux = data[:,:,1]/data[:,:,0]
        uy = data[:,:,2]/data[:,:,0]
        uz = data[:,:,3]/data[:,:,0]
        umag = np.sqrt(ux**2+uy**2+uz**2)
        et = data[:,:,-1]/data[:,:,0]
        p = self.fluid.ComputePressure_rho_u_et(rho, umag, et)
        T = p/rho/self.fluid.R
        return T
    

    def ComputeTotalTemperature(self, data):
        T = self.ComputeTemperature(data)
        M = self.ComputeMachNumber(data)
        Tt = T*(1+(self.fluid.gmma-1)/2*M**2)
        return Tt
    
    
    def ComputePressureCoefficient(self, data):
        rho = data[:,:,0]
        ux = data[:,:,1]/data[:,:,0]
        uy = data[:,:,2]/data[:,:,0]
        uz = data[:,:,3]/data[:,:,0]
        umag = np.sqrt(ux**2+uy**2+uz**2)
        et = data[:,:,-1]/data[:,:,0]
        p = self.fluid.ComputePressure_rho_u_et(rho, umag, et)
        
        p_ref = p[0,0]
        rho_ref = rho[0,0]
        u_ref = umag[0,0]

        cp = (p-p_ref)/(rho_ref*u_ref**2)
        return cp
    
    def ComputeEntropy(self, data):
        rho = data[:,:,0]
        ux = data[:,:,1]/data[:,:,0]
        uy = data[:,:,2]/data[:,:,0]
        uz = data[:,:,3]/data[:,:,0]
        umag = np.sqrt(ux**2+uy**2+uz**2)
        et = data[:,:,-1]
        M = self.fluid.ComputeEntropy_rho_u_et(rho, umag, et)
        return M
    

    def Plot1D(self, field_name, bound_dir, bound_loc, idx_k=0, save_filename=None, ref_points=None):
        if field_name.lower()=='p':
            name = 'Pressure1D'
            label = r'$p \ \rm{[kPa]}$'
            field2D = self.ComputePressure(self.data['U'][:,:,idx_k,:])
            field2D /= 1e3
        elif field_name.lower()=='pt':
            name = 'TotalPressure1D'
            label = r'$p_t \ \rm{[kPa]}$'
            field2D = self.ComputeTotalPressure(self.data['U'][:,:,idx_k,:])
            field2D /= 1e3
        elif field_name.lower()=='betatt':
            name = 'PRtt_1D'
            label = r'$\beta_{tt} \ \rm{[-]}$'
            field2D = self.ComputeBetaTT(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='trtt':
            name = 'TRtt_1D'
            label = r'$TR_{tt} \ \rm{[-]}$'
            field2D = self.ComputeTrTT(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='etatt':
            name = 'ETAtt_1D'
            label = r'$\eta_{tt} \ \rm{[-]}$'
            field2D = self.ComputeEtaTT(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='cp':
            name = 'Cp1D'
            label = r'$\bar{C}_p \ \rm{[-]}$'
            field2D = self.ComputePressureCoefficient(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='mach':
            name = 'Mach1D'
            label = r'$M \ \rm{[-]}$'
            field2D = self.ComputeMachNumber(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='rho':
            name = 'Density'
            label = r'$\rho \ \rm{[kg/m^3]}$'
            field2D = self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='ux':
            name = 'VelocityX'
            label = r'$u_x \ \rm{[m/s]}$'
            field2D = self.data['U'][:,:,idx_k,1]/self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='uy':
            name = 'VelocityY'
            label = r'$u_y \ \rm{[m/s]}$'
            field2D = self.data['U'][:,:,idx_k,2]/self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='uz':
            name = 'VelocityZ'
            label = r'$u_z \ \rm{[m/s]}$'
            field2D = self.data['U'][:,:,idx_k,3]/self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='et':
            name = 'TotalEnergy'
            label = r'$e_t \ \rm{[J/kg]}$'
            field2D = self.data['U'][:,:,idx_k,4]/self.data['U'][:,:,idx_k,0]
        elif field_name.lower()=='t':
            name = 'Temperature'
            label = r'$T \ \rm{[K]}$'
            field2D  =self.ComputeTemperature(self.data['U'][:,:,idx_k,:])
        elif field_name.lower()=='tt':
            name = 'TotalTemperature'
            label = r'$T_t \ \rm{[K]}$'
            field2D  =self.ComputeTotalTemperature(self.data['U'][:,:,idx_k,:])
        else:
            raise ValueError('Unknown field to plot')
        
        if bound_dir=='i':
            field = field2D[bound_loc, :]
            x = self.data['Y'][bound_loc, :]
            xlabel = (r'$y \ \rm{[m]}$')
        elif bound_dir=='j':
            field = field2D[:, bound_loc]
            x = self.data['X'][:, bound_loc]
            xlabel = (r'$x \ \rm{[m]}$')
        
        plt.figure()
        plt.plot(x, field, '-o', ms=5, mfc='none')
        plt.grid(alpha=styles.grid_opacity)
        plt.xlabel(xlabel)
        plt.ylabel(label)

        if ref_points is not None:
            plt.plot(ref_points[0], ref_points[1], 's', ms=5, mfc='none', label=ref_points[2])
            plt.legend()

        if save_filename is not None:
            plt.savefig(self.pictures_folder + '/' + save_filename + '_' + name + '.pdf', bbox_inches='tight')
    

    # def Plot1D_yAVG(self, field_name, idx_k=0, save_filename=None, ref_points=None, xlim=None):
    #     """
    #     Plot the y-avg solution along x, at index idx_k of a certain field.

    #     WARNING: this average is correct only when the grid is uniform along the span, otherwise is not correct.
    #     """
    #     def compute_y_avg(f2D):
    #         favg = np.sum(f2D, axis=1)/f2D.shape[1]
    #         return favg

    #     if field_name.lower()=='p':
    #         name = 'Pressure_yAVG'
    #         label = r'$p \ \rm{[kPa]}$'
    #         field2D = self.ComputePressure(self.data['U'][:,:,idx_k,:])
    #         field2D /= 1e3
    #     elif field_name.lower()=='rho':
    #         name = 'Density_yAVG'
    #         label = r'$\rho \ \rm{[kg/m^3]}$'
    #         field2D = self.data['U'][:,:,idx_k,0]
    #     elif field_name.lower()=='mach':
    #         name = 'Mach1D_yAVG'
    #         label = r'$M \ \rm{[-]}$'
    #         field2D = self.ComputeMachNumber(self.data['U'][:,:,idx_k,:])
    #     elif field_name.lower()=='momentumx':
    #         name = 'MomentumX_yAVG'
    #         label = r'$\rho u_x \ \rm{[m/s]}$'
    #         field2D = self.data['U'][:,:,idx_k,1]
    #     elif field_name.lower()=='t':
    #         name = 'Temperature_yAVG'
    #         label = r'$T \ \rm{[K]}$'
    #         field2D  =self.ComputeTemperature(self.data['U'][:,:,idx_k,:])
    #     else:
    #         raise ValueError('Unknown field to plot')
        
    #     x_avg = compute_y_avg(self.data['X'][:,:,idx_k])
    #     field_AVG = compute_y_avg(field2D)
    #     xlabel = (r'$x \ \rm{[m]}$')
        
    #     plt.figure()
    #     plt.plot(x_avg, field_AVG, '-o', ms=5, mfc='none')
    #     plt.grid(alpha=styles.grid_opacity)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(label)
    #     if xlim is not None:
    #         plt.xlim(xlim)

    #     if save_filename is not None:
    #         plt.savefig(self.pictures_folder + '/' + save_filename + '_' + name + '.pdf', bbox_inches='tight')
    

    def Save_1D_yAVG(self, save_filename=None, idx_k=0):
        """
        Save the pickle of the y-avg solution along x, at index idx_k of a certain field.
        """
        def compute_y_avg_axisymm(f2D):
            """
            Area average
            """
            ni,nj = f2D.shape
            favg = np.zeros(ni)
            for i in range(ni):
                f1D = f2D[i,:]
                r1D = self.data['Y'][i,:,idx_k]
                favg[i] = np.sum(f1D*r1D)/np.sum(r1D)
            # favg = np.sum(f2D, axis=1)/f2D.shape[1]
            return favg

        field2D = self.ComputePressure(self.data['U'][:,:,idx_k,:])
        p_AVG = compute_y_avg_axisymm(field2D)

        field2D = self.data['U'][:,:,idx_k,0]
        rho_AVG = compute_y_avg_axisymm(field2D)

        field2D = self.ComputeMachNumber(self.data['U'][:,:,idx_k,:])
        mach_AVG = compute_y_avg_axisymm(field2D) 

        field2D  =self.ComputeTemperature(self.data['U'][:,:,idx_k,:])
        T_AVG = compute_y_avg_axisymm(field2D) 

        x_avg = compute_y_avg_axisymm(self.data['X'][:,:,idx_k])
        field_AVG = compute_y_avg_axisymm(field2D)


        pik = {'Density': rho_AVG,
               'Mach': mach_AVG,
               'Pressure': p_AVG,
               'Temperature': T_AVG,
               'Points_2': x_avg}

        with open(save_filename+'.pik', 'wb') as file:
            pickle.dump(pik, file)



    def ComputeRadialVelocity(self, data, y, z):
        """
        This is needed for 3D simulations. For 2D axisymmetric, check that ur is equivalent to uy
        """
        uy = data[:,:,2]/data[:,:,0]
        uz = data[:,:,3]/data[:,:,0]
        theta = np.arctan2(z, y)
        ur = uy*np.cos(theta)-uz*np.sin(theta)
        return ur
    
    def ComputeTangentialVelocity(self, data, y, z):
        """
        This is needed for 3D simulations. For 2D axisymmetric, check that ur is equivalent to uy
        """
        uy = data[:,:,2]/data[:,:,0]
        uz = data[:,:,3]/data[:,:,0]
        theta = np.arctan2(z, y)
        ut = uy*np.sin(theta)+uz*np.cos(theta)
        return ut
    

    def PrintDeltaMassFlows(self, threshold=1E-02):
        def compute_delta(m1, m2):
            if np.abs(m1)<threshold and np.abs(m2)<threshold:
                return 0 # no flow in this direction
            else:
                return np.abs(m1-m2)/np.mean(m1+m2)*100 # averaged error
        
        mi_in = self.data['MassFlow'][0][-1]
        mi_out = self.data['MassFlow'][1][-1]

        mj_in = self.data['MassFlow'][2][-1]
        mj_out = self.data['MassFlow'][3][-1]

        mk_in = self.data['MassFlow'][4][-1]
        mk_out = self.data['MassFlow'][5][-1]

        print(r'Delta mass flows along i [percent]:     %.3f' %(compute_delta(mi_in, mi_out)))
        print(r'Delta mass flows along j [percent]:     %.3f' %(compute_delta(mj_in, mj_out)))
        print(r'Delta mass flows along k [percent]:     %.3f' %(compute_delta(mk_in, mk_out)))


    def PrintTurboPerformance(self, axisymmetric=True, save_csv=True):
        mflow = self.data['MassFlowTurbo'][-1]
        if axisymmetric: mflow*=360
        prTT = self.data['PRtt'][-1]
        etaTT = self.data['ETAtt'][-1]
        trTT = self.data['TRtt'][-1]
        
        print()
        print('TURBOMACHINERY PERFORMANCE')
        print(r'Mass Flow [kg/s]:                       %.3f' %(mflow))
        print(r'Tot-to-Tot Pressure ratio [-]:          %.3f' %(prTT))
        print(r'Tot-to-Tot Temperature ratio [-]:       %.3f' %(trTT))
        print(r'Tot-to-Tot Efficiency [-]:              %.3f' %(etaTT))

        # if save_csv:
        #     with open('performance.csv', 'w') as file:
        #         file.write('MassFlow [kg/s],Betatt [-], Etatt[-]\n')
        #         file.write('%6.3f,%6.3f,%6.3f' %(mflow, prTT, etaTT))

