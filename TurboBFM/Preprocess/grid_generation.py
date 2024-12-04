import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def transfinite_grid_generation(c_left: np.ndarray, c_bottom: np.ndarray, c_right: np.ndarray, c_top: np.ndarray, 
                                stretch_type_stream='both', stretch_type_span='both', 
                                streamwise_coeff=1, spanwise_coeff=1, 
                                nx=None, ny=None):
    """
    Method used to generate the grid with transfinite grid interpolation method.

    Parameters
    ----------------------------------------
    `c_left`: left border points (x, y)
    
    `c_bottom`: bottom border points (x, y)
    
    `c_right`: right border points (x, y)
    
    `c_top`: top border points (x, y)
    
    `stretch_type_stream`: left, both, or right to impose the stretching functions in the streamwise direction

    `stretch_type_span`: bottom, both, or up to impose the stretching functions in the spanwise direction
    
    `streamwise_coeff`: coefficient of stretching function along streamwise direction
    
    `spanwise_coeff`: coefficient of stretching function along spanwise direction
    
    `nx`: number of points in streamwise direction (if None, default is used)
    
    `ny`: number of points in spanwise direction (if None, default is used)
    """
    if nx is None:
        nx = c_bottom.shape[1]
    if ny is None:
        ny = c_left.shape[1]

    t_streamwise = np.linspace(0, 1, nx)
    t_spanwise = np.linspace(0, 1, ny)

    splinex_bottom = CubicSpline(t_streamwise, c_bottom[0, :])
    spliney_bottom = CubicSpline(t_streamwise, c_bottom[1, :])

    splinex_top = CubicSpline(t_streamwise, c_top[0, :])
    spliney_top = CubicSpline(t_streamwise, c_top[1, :])

    splinex_left = CubicSpline(t_spanwise, c_left[0, :])
    spliney_left = CubicSpline(t_spanwise, c_left[1, :])

    splinex_right = CubicSpline(t_spanwise, c_right[0, :])
    spliney_right = CubicSpline(t_spanwise, c_right[1, :])

    xi = np.linspace(0, 1, nx)
    eta = np.linspace(0, 1, ny)

    # stretching functions applied to the computational cordinates. if the coefficients are equal to 1 and 1, no stretching
    # (this is needed because eriksson with a value of 1 is different from no stretching)
    if streamwise_coeff != 1:
        if stretch_type_stream.lower() == 'right':
            xi = eriksson_stretching_function_final(xi, streamwise_coeff)
        elif stretch_type_stream.lower() == 'both':
            xi = eriksson_stretching_function_both(xi, streamwise_coeff)
        elif stretch_type_stream.lower() == 'left':
            xi = eriksson_stretching_function_initial(xi, streamwise_coeff)
        else:
            raise ValueError('Unrecognized block topology')

    if spanwise_coeff != 1:
        if stretch_type_span.lower() == 'up':
            eta = eriksson_stretching_function_final(eta, streamwise_coeff)
        elif stretch_type_span.lower() == 'both':
            eta = eriksson_stretching_function_both(eta, streamwise_coeff)
        elif stretch_type_span.lower() == 'bottom':
            eta = eriksson_stretching_function_initial(eta, streamwise_coeff)
        else:
            raise ValueError('Unrecognized block topology')

    XI, ETA = np.meshgrid(xi, eta, indexing='ij')
    X, Y = XI * 0, ETA * 0

    # TRANSFINITE INTERPOLATION
    for i in range(nx):
        for j in range(ny):
            X[i, j] = (1 - XI[i, j]) * splinex_left(ETA[i, j]) + XI[i, j] * splinex_right(ETA[i, j]) + (
                    1 - ETA[i, j]) * splinex_bottom(XI[i, j]) + ETA[i, j] * splinex_top(XI[i, j]) - (1 - XI[i, j]) * (
                              1 - ETA[i, j]) * splinex_left(0) - (1 - XI[i, j]) * ETA[i, j] * splinex_left(1) - (1 - ETA[i, j]) * \
                      XI[i, j] * splinex_right(0) - XI[i, j] * ETA[i, j] * splinex_right(1)

            Y[i, j] = (1 - XI[i, j]) * spliney_left(ETA[i, j]) + XI[i, j] * spliney_right(ETA[i, j]) + (
                    1 - ETA[i, j]) * spliney_bottom(XI[i, j]) + ETA[i, j] * spliney_top(XI[i, j]) - (1 - XI[i, j]) * (
                              1 - ETA[i, j]) * spliney_left(0) - (1 - XI[i, j]) * ETA[i, j] * spliney_left(1) - (1 - ETA[i, j]) * \
                      XI[i, j] * spliney_right(0) - XI[i, j] * ETA[i, j] * spliney_right(1)

    plt.figure()
    for i in range(nx):
        plt.plot(X[i, :], Y[i, :], 'k', lw=0.5)
    for j in range(ny):
        plt.plot(X[:, j], Y[:, j], 'k', lw=0.5)
    plt.gca().set_aspect('equal', adjustable='box')


    return X, Y



def eriksson_stretching_function_initial(x, alpha):
    """
    equation 4.93 Farrashkhalvat Book. Gives clustering at the initial part of the computational domain
    """
    f = (np.exp(alpha * x) - 1) / (np.exp(alpha) - 1)
    # plt.figure()
    # plt.plot(x, f, '-o', label=r'$f/f_{max}$')
    # plt.xlabel(r'$x$')
    # plt.xlabel(r'$f$')
    # plt.grid(alpha=grid_opacity)
    return f


def eriksson_stretching_function_final(x, alpha):
    """
    equation 4.95 Farrashkhalvat Book. Gives clustering at the final part of the computational domain
    """
    f = (np.exp(alpha) - np.exp(alpha * (1 - x))) / (np.exp(alpha) - 1)
    # plt.figure()
    # plt.plot(x, f, '-o', label=r'$f/f_{max}$')
    # plt.xlabel(r'$x$')
    # plt.xlabel(r'$f$')
    # plt.grid(alpha=grid_opacity)
    return f


def eriksson_stretching_function_both(x, alpha):
    """
    equation 4.97 Farrashkhalvat Book. Gives clustering at the initial and final part of the computational domain
    """
    x_midpoint = 0.5
    f = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] <= x_midpoint:
            f[i] = x_midpoint * (np.exp(alpha * x[i] / x_midpoint) - 1) / (np.exp(alpha) - 1)
        else:
            f[i] = 1 - (1 - x_midpoint) * (np.exp(alpha * (1 - x[i]) / (1 - x_midpoint)) - 1) / (np.exp(alpha) - 1)

    # plt.figure()
    # plt.plot(x, f, '-o', label=r'$f/f_{max}$')
    # plt.xlabel(r'$x$')
    # plt.xlabel(r'$f$')
    # plt.grid(alpha=grid_opacity)
    return f


def compute_three_dimensional_mesh(Z, R, theta_max, nodes_number):
        """
        Compute the structured three-dimensional mesh X,Y,Z as 3D arrays, starting from 2D Z,R grids.
        :param conserve_AR: if True, tries to conserve the AR choosing the correct delta-theta between each tang. node
        :param theta_max: [deg] angle of the mesh sector.
        :param nodes_number: number of nodes from zero to theta_max
        :param dimensional: if True, reconverts the coordinates to original dimensions in [m]
        """
        NZ = Z.shape[0]
        NR = Z.shape[1]

        theta = np.linspace(0, theta_max * np.pi / 180, nodes_number)

        X_mesh = np.zeros((NZ, NR, nodes_number))
        Y_mesh = np.zeros((NZ, NR, nodes_number))
        Z_mesh = np.zeros((NZ, NR, nodes_number))
        for i in range(NZ):
            for j in range(NR):
                for k in range(nodes_number):
                    X_mesh[i, j, k] = R[i, j] * np.cos(theta[k])
                    Y_mesh[i, j, k] = R[i, j] * np.sin(theta[k])
                    Z_mesh[i, j, k] = Z[i, j]
        
        return X_mesh, Y_mesh, Z_mesh
