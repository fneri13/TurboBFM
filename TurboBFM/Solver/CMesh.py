import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D 
from TurboBFM.Solver.CConfig import CConfig

class CMesh():
    
    def __init__(self, config: CConfig, coords: dict):
        """
        Starting from the grid points coordinates, generate the array of Volumes associated to the points, and the array of surfaces associated to the interfaces between the points (Dual grid formulation).
        If the grid has (ni,nj,nk) structured points, it has (ni,nj,nk) elements.
        The Mesh for 2D geometries is handled like openfoam (one cell in the k-direction, with thickness 1 for correct evaluation of volumes and surfaces)

        Parameters
        -------------------------

        `config`: CConfig object of the simulation

        `coords`: dictionnary with the coordinates

        """
        self.config = config
        self.verbosity = self.config.GetVerbosity()
        self.X = coords['X']
        self.Y = coords['Y']
        if len(coords['X'].shape)==3:
            self.nDim = 3
            self.X, self.Y, self.Z = coords['X'], coords['Y'], coords['Z']
            self.ni, self.nj, self.nk = self.X.shape
        elif len(coords['X'].shape)==2:
            self.nDim = 2
            self.ni, self.nj = coords['X'].shape
            self.nk = 1
            self.X = np.zeros((self.ni, self.nj, self.nk))
            self.Y = np.zeros((self.ni, self.nj, self.nk))
            self.Z = np.zeros((self.ni, self.nj, self.nk))
            self.X[:,:,0] = coords['X']
            self.Y[:,:,0] = coords['Y']
        else:
            raise ValueError('The coordinates arrays must have dimension 2 or 3')

        self.ni_dual, self.nj_dual, self.nk_dual = self.ni+1, self.nj+1, self.nk+1      # these are the number of dual grid points
        self.n_elements = self.ni*self.nj*self.nk                                       # number of finite volumes (excluding ghost elements)
        
        print('Computing Dual Grid..')
        if self.nDim==2:
            self.ComputeDualGrid2D()
        else:
            self.ComputeDualGrid3D()

        self.ComputeInterfaces()

        self.ComputeVolumes()

        self.ComputeMeshQuality()

        self.PrintMeshInfo()

    def ComputeInterfaces(self):
        """
        Build the interfaces arrays.
        The ordering respect the following ideas:
        - Si[i,j,k,:] are the three components of the normal surface interfacing the primary grid point [i,j,k] with [i+1,j,k].
        - Sj[i,j,k,:] are the three components of the normal surface interfacing the primary grid point [i,j,k] with [i,j+1,k].
        - Sk[i,j,k,:] are the three components of the normal surface interfacing the primary grid point [i,j,k] with [i,j,k+1].
        For every point [i,j,k] only three surfaces are needed, since the other three come from the following points on the respective directions.
        The centers of gravity of every interface (CGi, CGj, CGk) follows the same indexing.
        """
        print('Computing Interfaces..')

        def compute_surface_vector_and_cg(x1: float, x2: float, x3: float, x4: float, 
                                          y1: float, y2: float, y3: float, y4: float,
                                          z1: float, z2: float, z3: float, z4: float):
            """
            Compute the surface vector of the ordered quadrilater identified from the coords of the 4 vertices. The surface direction corresponds to the
            right-hand rule for the points ordered as 1->2->3->4.
            """
            # triangulate the face
            a1 = np.array([x2-x1, y2-y1, z2-z1])
            b1 = np.array([x3-x2, y3-y2, z3-z2])
            a2 = np.array([x3-x1, y3-y1, z3-z1])
            b2 = np.array([x4-x3, y4-y3, z4-z3])
            S1 = np.cross(a1, b1)
            S2 = np.cross(a2, b2)
            Str = 0.5*(S1 + S2)

            cg1 = (np.array([x1+x2+x3, y1+y2+y3, z1+z2+z3]))/3.0
            cg2 = (np.array([x1+x3+x4, y1+y3+y4, z1+z3+z4]))/3.0
            CG = (cg1*np.linalg.norm(S1)+cg2*np.linalg.norm(S2))/(np.linalg.norm(S1)+np.linalg.norm(S2))

            return Str, CG
        
        """
        Si has ni dual components in the i direction, while only nj-1 and nk-1 dual components in the j,k directions. Analogous thing for Sj and Sk.
        """
        self.Si = np.zeros((self.ni_dual, self.nj_dual-1, self.nk_dual-1, 3))            # surface vector connecting point (i,j,k) to (i+1,j,k)
        self.Sj = np.zeros((self.ni_dual-1, self.nj_dual, self.nk_dual-1, 3))            # surface vector connecting point (i,j,k) to (i,j+1,k)
        self.Sk = np.zeros((self.ni_dual-1, self.nj_dual-1, self.nk_dual, 3))            # surface vector connecting point (i,j,k) to (i,j,k+1)
        self.CGi = np.zeros((self.ni_dual, self.nj_dual-1, self.nk_dual-1, 3))           # center of face vector connecting point (i,j,k) to (i+1,j,k)
        self.CGj = np.zeros((self.ni_dual-1, self.nj_dual, self.nk_dual-1, 3))           # center of face vector connecting point (i,j,k) to (i,j+1,k)
        self.CGk = np.zeros((self.ni_dual-1, self.nj_dual-1, self.nk_dual, 3))           # center of face vector connecting point (i,j,k) to (i,j,k+1)
        
        print('Computing i-interfaces')
        for i in range(self.ni_dual):
            for j in range(self.nj_dual-1):
                for k in range(self.nk_dual-1):
                    self.Si[i,j,k,:], self.CGi[i,j,k,:] = compute_surface_vector_and_cg(self.xv[i,j,k], self.xv[i, j+1, k], self.xv[i, j+1, k+1], self.xv[i, j, k+1], 
                                                                                        self.yv[i,j,k], self.yv[i, j+1, k], self.yv[i, j+1, k+1], self.yv[i, j, k+1], 
                                                                                        self.zv[i,j,k], self.zv[i, j+1, k], self.zv[i, j+1, k+1], self.zv[i, j, k+1])
        
        print('Computing j-interfaces')
        for i in range(self.ni_dual-1):
            for j in range(self.nj_dual):
                for k in range(self.nk_dual-1):
                    self.Sj[i,j,k,:], self.CGj[i,j,k,:] = compute_surface_vector_and_cg(self.xv[i,j,k], self.xv[i, j, k+1], self.xv[i+1, j, k+1], self.xv[i+1, j, k], 
                                                                                        self.yv[i,j,k], self.yv[i, j, k+1], self.yv[i+1, j, k+1], self.yv[i+1, j, k], 
                                                                                        self.zv[i,j,k], self.zv[i, j, k+1], self.zv[i+1, j, k+1], self.zv[i+1, j, k])
        
        print('Computing k-interfaces') 
        for i in range(self.ni_dual-1):
            for j in range(self.nj_dual-1):
                for k in range(self.nk_dual):
                    self.Sk[i,j,k,:], self.CGk[i,j,k,:] = compute_surface_vector_and_cg(self.xv[i,j,k], self.xv[i+1, j, k], self.xv[i+1, j+1, k], self.xv[i, j+1, k], 
                                                                                        self.yv[i,j,k], self.yv[i+1, j, k], self.yv[i+1, j+1, k], self.yv[i, j+1, k], 
                                                                                        self.zv[i,j,k], self.zv[i+1, j, k], self.zv[i+1, j+1, k], self.zv[i, j+1, k])

        
        # Plot of the ij plane at k=0
        if self.verbosity==3:
            plt.figure()
            def plot_grid_lines(xgrid, ygrid, color, label):
                ni, nj = xgrid.shape[0], xgrid.shape[1]
                for i in range(ni):
                    if i==0:
                        plt.plot(xgrid[i,:,0], ygrid[i,:,0], '%s' %(color), lw=0.75, mfc='none', label=label)
                    else:
                        plt.plot(xgrid[i,:,0], ygrid[i,:,0], '%s' %(color), lw=0.75, mfc='none')
                for j in range(nj):
                    plt.plot(xgrid[:,j,0], ygrid[:,j,0], '%s' %(color), lw=0.75, mfc='none')
                return None
            
            plot_grid_lines(self.X, self.Y, 'ko', label='Primary')
            plot_grid_lines(self.xv, self.yv, '--r^', label='Dual')
            for i in range(self.ni_dual):
                for j in range(self.nj_dual-1):
                    plt.quiver(self.CGi[i,j,0,0], self.CGi[i,j,0,1], self.Si[i,j,0,0], self.Si[i,j,0,1])
            for i in range(self.ni_dual-1):
                for j in range(self.nj_dual):
                    plt.quiver(self.CGj[i,j,0,0], self.CGj[i,j,0,1], self.Sj[i,j,0,0], self.Sj[i,j,0,1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title('k=%i plane' %(0))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()


    def ComputeVolumes(self):
        """
        Compute the volumes for every element, using Green Gauss Theorem. 
        """
        print('Computing Volumes')
        def compute_volume(S, CG, iDir):
            """
            For the 6 surfaces enclosing an element, compute the volume using green gauss theorem. iDir stands for the direction used (0,1,2) for (x,y,z)
            """
            assert(len(S)==len(CG))
            vol = 0
            for iFace in range(len(S)):
                vol += CG[iFace][iDir]*S[iFace][iDir]
            return vol

        self.V = np.zeros((self.ni,self.nj,self.nk)) # the ghost point volumes will be zero for the moment
        for i in range(0,self.ni):
            for j in range(0,self.nj):
                for k in range(0,self.nk):
                    # assemble the tuple of Surfaces enclosing the element, being careful to set the facing outside
                    S = (-self.Si[i,j,k,:], -self.Sj[i,j,k,:], -self.Sk[i,j,k,:], self.Si[i+1,j,k,:], self.Sj[i,j+1,k,:], self.Sk[i,j,k+1,:])
                    CG = (self.CGi[i,j,k,:], self.CGj[i,j,k,:], self.CGk[i,j,k,:], self.CGi[i+1,j,k,:], self.CGj[i,j+1,k,:], self.CGk[i,j,k+1,:])
                    self.V[i,j,k] = compute_volume(S, CG, 0)
    
    def PrintMeshInfo(self):
        """
        Print relevant information for the mesh
        """
        if self.verbosity==3:
            print('='*20 + ' ELEMENTS INFORMATION ' + '='*20)
            for i in range(self.ni):
                for j in range(self.nj):
                    for k in range(self.nk):
                        print('For point (%i,%i,%i):' %(i,j,k))
                        print('                         Si=[%.2e,%.2e,%.2e]' %(self.Si[i,j,k,0],self.Si[i,j,k,1],self.Si[i,j,k,2]))
                        print('                         CGi=[%.2e,%.2e,%.2e]' %(self.CGi[i,j,k,0],self.CGi[i,j,k,1],self.CGi[i,j,k,2]))
                        print('                         Sj=[%.2e,%.2e,%.2e]' %(self.Sj[i,j,k,0],self.Sj[i,j,k,1],self.Sj[i,j,k,2]))
                        print('                         CGj=[%.2e,%.2e,%.2e]' %(self.CGj[i,j,k,0],self.CGj[i,j,k,1],self.CGj[i,j,k,2]))
                        print('                         Sk=[%.2e,%.2e,%.2e]' %(self.Sk[i,j,k,0],self.Sk[i,j,k,1],self.Sk[i,j,k,2]))
                        print('                         CGk=[%.2e,%.2e,%.2e]' %(self.CGk[i,j,k,0],self.CGk[i,j,k,1],self.CGk[i,j,k,2]))
                        print('                         Vol=%.4e' %(self.V[i,j,k]))
                        print()
            print('='*20 + ' END ELEMENTS INFORMATION ' + '='*20)
        
        if self.nDim==3:
            total_intf = self.ni_dual*(self.nj_dual-1)*(self.nk_dual-1) + self.nj_dual*(self.ni_dual-1)*(self.nk_dual-1) + self.nk_dual*(self.ni_dual-1)*(self.nj_dual-1)
        else:
            total_intf = self.ni_dual*(self.nj_dual-1) + self.nj_dual*(self.ni_dual-1)

        if self.verbosity>0:
            print('='*25 + ' MESH INFORMATION ' + '='*25)
            print('Number of elements:                  %i' %(self.n_elements))
            print('Number of total interfaces:          %i' %(total_intf))
            print('Number of internal interfaces:       %i' %(len(self.orthogonality)))
            print('Aspect Ratio:')
            print('                                     min: %.12f' %(np.min(self.aspect_ratio)))
            print('                                     max: %.12f' %(np.max(self.aspect_ratio)))
            print('                                     average: %.12f' %(np.mean(self.aspect_ratio)))
            print('                                     std: %.12f' %(np.std(self.aspect_ratio)))
            print('Skewness:')
            print('                                     min: %.12f' %(np.min(self.skewness)))
            print('                                     max: %.12f' %(np.max(self.skewness)))
            print('                                     average: %.12f' %(np.mean(self.skewness)))
            print('                                     std: %.12f' %(np.std(self.skewness)))
            print('Orthogonality [deg]:')
            print('                                     min: %.12f' %(np.min(self.orthogonality)*180/np.pi))
            print('                                     max: %.12f' %(np.max(self.orthogonality)*180/np.pi))
            print('                                     average: %.12f' %(np.mean(self.orthogonality)*180/np.pi))
            print('                                     std: %.12f' %(np.std(self.orthogonality)*180/np.pi))
            print('='*25 + ' END MESH INFORMATION ' + '='*25)



    def ComputeDualGrid2D(self):
        """
        Compute the dual grid points. The openfoam modeling is applied, where the thickness is equivalent to 1 for every cell. Therefore
        the k=0 plane is located at z=0, and the k=1 plane is located at z=1.
        """
        # Compute the vertices of the dual grid
        # (the dual grid nodes are equal to the primary grid nodes + 1 in every direction)
        xv = np.zeros((self.ni_dual, self.nj_dual, self.nk_dual))
        yv = np.zeros((self.ni_dual, self.nj_dual, self.nk_dual))
        zv = np.zeros((self.ni_dual, self.nj_dual, self.nk_dual))

        # fix the internal points
        def fix_internals(arr1 : np.ndarray, arr2 : np.ndarray) -> np.ndarray:
            """
            The internal dual points are found as baricenter of 8 surrounding points
            """
            arr1[1:-1, 1:-1] = (arr2[0:-1,0:-1] + arr2[1:,0:-1] + arr2[0:-1,1:] + arr2[1:,1:])/4.0
            return arr1
        xv[:,:,0] = fix_internals(xv[:,:,0], self.X[:,:,0])
        yv[:,:,0] = fix_internals(yv[:,:,0], self.Y[:,:,0])

        # fix the corners
        def fix_corners(arr1 : np.ndarray, arr2 : np.ndarray) -> np.ndarray:
            """
            The corners of the dual grid coincide with the corners of the primary grid
            """
            arr1[0,0] = arr2[0,0]
            arr1[0,-1] = arr2[0,-1]
            arr1[-1,-1] = arr2[-1,-1]
            arr1[-1,0] = arr2[-1,0]
            return arr1
        xv[:,:,0] = fix_corners(xv[:,:,0], self.X[:,:,0])
        yv[:,:,0] = fix_corners(yv[:,:,0], self.Y[:,:,0])

        # fix the edges
        def fix_edges(arr1 : np.ndarray, arr2 : np.ndarray) -> np.ndarray:
            """
            The dual grid nodes on the edges are found halfway between two successive primary grid nodes on that edge
            """
            # i oriented edges
            arr1[1:-1,0] = (arr2[0:-1,0]+arr2[1:,0])/2.0
            arr1[1:-1,-1] = (arr2[0:-1,-1]+arr2[1:,-1])/2.0
            # j oriented edges
            arr1[0,1:-1] = (arr2[0,0:-1]+arr2[0,1:])/2.0
            arr1[-1,1:-1] = (arr2[-1,0:-1]+arr2[-1,1:])/2.0
            return arr1
        xv[:,:,0] = fix_edges(xv[:,:,0], self.X[:,:,0])
        yv[:,:,0] = fix_edges(yv[:,:,0], self.Y[:,:,0])

        # fix also the second plane in k-direction
        xv[:,:,1] = xv[:,:,0]       # same x-coordinates
        yv[:,:,1] = yv[:,:,0]       # same y-coordinates
        zv[:,:,1] = zv[:,:,0]+1     # thickness value of 1 everywhere 

        # mantain a copy of the vertices, to compute also the quality
        self.xv, self.yv, self.zv = xv, yv, zv



    def ComputeDualGrid3D(self):
        """
        Compute the dual grid points.
        """
        # Compute the vertices of the dual grid
        # (the dual grid nodes are equal to the primary grid nodes + 1 in every direction)
        xv = np.zeros((self.ni_dual, self.nj_dual, self.nk_dual))
        yv = np.zeros((self.ni_dual, self.nj_dual, self.nk_dual))
        zv = np.zeros((self.ni_dual, self.nj_dual, self.nk_dual))

        # fix the internal points
        def fix_internals(arr1 : np.ndarray, arr2 : np.ndarray) -> np.ndarray:
            """
            The internal dual points are found as baricenter of 8 surrounding points
            """
            arr1[1:-1, 1:-1, 1:-1] = (arr2[0:-1,0:-1,0:-1] + arr2[1:,0:-1,0:-1] + arr2[0:-1,1:,0:-1] + arr2[0:-1,0:-1,1:] + 
                                      arr2[1:,1:,0:-1] + arr2[1:,0:-1,1:] + arr2[0:-1,1:,1:] + arr2[1:,1:,1:])/8.0
            return arr1
        xv = fix_internals(xv, self.X)
        yv = fix_internals(yv, self.Y)
        zv = fix_internals(zv, self.Z)
        

        # fix the corners
        def fix_corners(arr1 : np.ndarray, arr2 : np.ndarray) -> np.ndarray:
            """
            The corners of the dual grid coincide with the corners of the primary grid
            """
            arr1[0,0,0] = arr2[0,0,0]
            arr1[0,0,-1] = arr2[0,0,-1]
            arr1[0,-1,0] = arr2[0,-1,0]
            arr1[-1,0,0] = arr2[-1,0,0]
            arr1[0,-1,-1] = arr2[0,-1,-1]
            arr1[-1,0,-1] = arr2[-1,0,-1]
            arr1[-1,-1,0] = arr2[-1,-1,0]
            arr1[-1,-1,-1] = arr2[-1,-1,-1]
            return arr1
        xv = fix_corners(xv, self.X)
        yv = fix_corners(yv, self.Y)
        zv = fix_corners(zv, self.Z)


        # fix the edges
        def fix_edges(arr1 : np.ndarray, arr2 : np.ndarray) -> np.ndarray:
            """
            The dual grid nodes on the edges are found halfway between two successive primary grid nodes on that edge
            """
            # i oriented edges
            arr1[1:-1,0,0] = (arr2[0:-1,0,0]+arr2[1:,0,0])/2.0
            arr1[1:-1,-1,0] = (arr2[0:-1,-1,0]+arr2[1:,-1,0])/2.0
            arr1[1:-1,0,-1] = (arr2[0:-1,0,-1]+arr2[1:,0,-1])/2.0
            arr1[1:-1,-1,-1] = (arr2[0:-1,-1,-1]+arr2[1:,-1,-1])/2.0

            # j oriented edges
            arr1[0,1:-1,0] = (arr2[0,0:-1,0]+arr2[0,1:,0])/2.0
            arr1[-1,1:-1,0] = (arr2[-1,0:-1,0]+arr2[-1,1:,0])/2.0
            arr1[0,1:-1,-1] = (arr2[0,0:-1,-1]+arr2[0,1:,-1])/2.0
            arr1[-1,1:-1,-1] = (arr2[-1,0:-1,-1]+arr2[-1,1:,-1])/2.0

            # k oriented edges
            arr1[0,0,1:-1] = (arr2[0,0,0:-1]+arr2[0,0,1:])/2.0
            arr1[-1,0,1:-1] = (arr2[-1,0,0:-1]+arr2[-1,0,1:])/2.0
            arr1[0,-1,1:-1] = (arr2[0,-1,0:-1]+arr2[0,-1,1:])/2.0
            arr1[-1,-1,1:-1] = (arr2[-1,-1,0:-1]+arr2[-1,-1,1:])/2.0

            return arr1
        xv = fix_edges(xv, self.X)
        yv = fix_edges(yv, self.Y)
        zv = fix_edges(zv, self.Z)


        # fix the boundaries
        def fix_boundaries(arr1 : np.ndarray, arr2 : np.ndarray) -> np.ndarray:
            """
            The dual grid nodes on the internal part of the boundaries are found as the baricenter of 4 surrounding points of the primary grid
            """
            # i faces
            arr1[0,1:-1,1:-1] = (arr2[0,0:-1,0:-1] + arr2[0,1:,0:-1] + arr2[0,0:-1,1:] + arr2[0,1:,1:])/4.0
            arr1[-1,1:-1,1:-1] = (arr2[-1,0:-1,0:-1] + arr2[-1,1:,0:-1] + arr2[-1,0:-1,1:] + arr2[-1,1:,1:])/4.0

            # j faces
            arr1[1:-1,0,1:-1] = (arr2[0:-1,0,0:-1] + arr2[1:,0,0:-1] + arr2[0:-1,0,1:] + arr2[1:,0,1:])/4.0
            arr1[1:-1,-1,1:-1] = (arr2[0:-1,-1,0:-1] + arr2[1:,-1,0:-1] + arr2[0:-1,-1,1:] + arr2[1:,-1,1:])/4.0

            # k faces
            arr1[1:-1,1:-1,0] = (arr2[0:-1,0:-1,0] + arr2[1:,0:-1,0] + arr2[0:-1,1:,0] + arr2[1:,1:,0])/4.0
            arr1[1:-1,1:-1,-1] = (arr2[0:-1,0:-1,-1] + arr2[1:,0:-1,-1] + arr2[0:-1,1:,-1] + arr2[1:,1:,-1])/4.0

            return arr1
        xv = fix_boundaries(xv, self.X)
        yv = fix_boundaries(yv, self.Y)
        zv = fix_boundaries(zv, self.Z)

        # mantain a copy of the vertices, to compute also the quality
        self.xv, self.yv, self.zv = xv, yv, zv



    def ComputeMeshQuality(self):
        """
        Given the geometry information stored, compute some quality metrics. 
        """
        print('Computing Mesh Quality')
        self.ComputeAspectRatio()
        self.ComputeSkewnessOrthogonality()


    def ComputeAspectRatio(self):
        """
        Compute aspect ratio defined as longest to shortes cell edge.
        """
        ni, nj, nk = self.X.shape
        self.aspect_ratio = []

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    pt_0 = np.array([self.xv[i,j,k], self.yv[i,j,k], self.zv[i,j,k]])
                    pt_i = np.array([self.xv[i+1,j,k], self.yv[i+1,j,k], self.zv[i+1,j,k]])
                    pt_j = np.array([self.xv[i,j+1,k], self.yv[i,j+1,k], self.zv[i,j+1,k]])
                    pt_k = np.array([self.xv[i,j,k+1], self.yv[i,j,k+1], self.zv[i,j,k+1]])
                    i_edge = np.linalg.norm(pt_i-pt_0)
                    j_edge = np.linalg.norm(pt_j-pt_0)
                    k_edge = np.linalg.norm(pt_k-pt_0)
                    if self.nDim==3:
                        ar = max(i_edge, j_edge, k_edge)/min(i_edge, j_edge, k_edge)
                    else:
                        ar = max(i_edge, j_edge)/min(i_edge, j_edge)
                    self.aspect_ratio.append(ar)
        self.aspect_ratio = np.array(self.aspect_ratio)
    
    def ComputeSkewnessOrthogonality(self):
        """
        Compute the mesh skewness, defined as the distance between the real center face and the midpoint connecting the two cells, normalized
        by the distance between the two elements center. Compute also the orthogonality, defined as the angle between the connecting line and the face normal.
        """
        self.skewness = []
        self.orthogonality = []

        def compute_angle(v1, v2):
            """
            Slightly modified angle calculation to account for those cases where the first vector has zero magnitude (when the node c)
            """
            v1_dir = v1 / np.linalg.norm(v1)
            v2_dir = v2 / np.linalg.norm(v2)
            return np.arccos(np.dot(v1_dir, v2_dir))

        # internal interfaces along i
        ni, nj, nk = self.Si[:,:,:,0].shape
        for i in range(1,ni-1):
            for j in range(nj):
                for k in range(nk):
                    pt_0 = np.array([self.X[i-1,j,k], self.Y[i-1,j,k], self.Z[i-1,j,k]])
                    pt_i = np.array([self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k]])
                    midpoint = pt_0 + 0.5*(pt_i-pt_0)   
                    CG = self.CGi[i,j,k,:]             
                    l1 = np.linalg.norm(midpoint-CG)
                    l2 = np.linalg.norm(pt_i-pt_0)
                    self.skewness.append(l1/l2)
                    S = self.Si[i,j,k,:]
                    l2 = pt_i-pt_0 
                    angle = compute_angle(l2, S)
                    self.orthogonality.append(angle)
        
        # internal interfaces along j
        ni, nj, nk = self.Sj[:,:,:,0].shape
        for i in range(ni):
            for j in range(1,nj-1):
                for k in range(nk):
                    pt_0 = np.array([self.X[i,j-1,k], self.Y[i,j-1,k], self.Z[i,j-1,k]])
                    pt_j = np.array([self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k]])
                    midpoint = pt_0 + 0.5*(pt_j-pt_0)   
                    CG = self.CGj[i,j,k,:]             
                    l1 = np.linalg.norm(midpoint-CG)
                    l2 = np.linalg.norm(pt_j-pt_0)
                    self.skewness.append(l1/l2)
                    S = self.Sj[i,j,k,:]
                    l2 = pt_j-pt_0 
                    angle = compute_angle(l2, S)
                    self.orthogonality.append(angle)
        
        # internal interfaces along k
        if self.nDim==3:
            ni, nj, nk = self.Sk[:,:,:,0].shape
            for i in range(ni):
                for j in range(nj):
                    for k in range(1, nk-1):
                        pt_0 = np.array([self.X[i,j,k-1], self.Y[i,j,k-1], self.Z[i,j,k-1]])
                        pt_k = np.array([self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k]])
                        midpoint = pt_0 + 0.5*(pt_k-pt_0)   
                        CG = self.CGk[i,j,k,:]             
                        l1 = np.linalg.norm(midpoint-CG)
                        l2 = np.linalg.norm(pt_k-pt_0)
                        self.skewness.append(l1/l2)
                        S = self.Sk[i,j,k,:]
                        l2 = pt_k-pt_0 
                        angle = compute_angle(l2, S)
                        self.orthogonality.append(angle)
        else:
            pass
        
        self.skewness = np.array(self.skewness)
        self.orthogonality = np.array(self.orthogonality)


    def GetSurfaceData(self, i: int, j: int, k: int, direction: str, data: str) -> np.ndarray:
        """
        Get the surface vector that connects the primary node with another node in the specified direction.

        Parameters
        -------------------

        `i`: element index along first direction
        
        `j`: element index along second direction
        
        `k`: element index along third direction

        `direction`: string defining the direction (i,j,k) -> (east or west, south or north, down or up)

        `data`: string specifying if you want the midpoint coordinates, or the surface vector, or all
        """

        if direction=='west':
            S = -self.Si[i,j,k,:]
            CG = self.CGi[i,j,k,:]
        
        elif direction=='east':
            S =  self.Si[i+1,j,k,:]
            CG = self.CGi[i+1,j,k,:]
        
        elif direction=='south':
            S = -self.Sj[i,j,k,:]
            CG = self.CGj[i,j,k,:]
        
        elif direction=='north':
            S =  self.Sj[i,j+1,k,:]
            CG = self.CGj[i,j+1,k,:]
        
        elif direction=='bottom':
            S = -self.Sk[i,j,k,:]
            CG = self.CGk[i,j,k,:]
        
        elif direction=='top':
            S =  self.Sk[i,j,k+1,:]
            CG = self.CGk[i,j,k+1,:]
        
        else:
            raise ValueError('Direction not recognized')

        if data=='surface':
            return S
        elif data=='midpoint':
            return CG
        elif data=='all':
            return S, CG
        else:
            raise ValueError('Data requested not recognized. Choose between surface, midpoint, or all')
    

    def PlotMeshQuality(self):
        """
        Plot the histograms of aspect ratio, skewness and orthogonality
        """
        fig, ax = plt.subplots(1, 3, figsize=(16,9))

        ax[0].hist(self.aspect_ratio, edgecolor='black', color='C0', align='left')
        ax[0].set_title('AR of %i Elements' %(self.n_elements))
        ax[0].set_xlabel('Aspect Ratio [-]')
        ax[0].set_ylabel('N')  

        ax[1].hist(self.skewness, edgecolor='black', color='C1', align='left')
        ax[1].set_title('Skewness of %i Faces' %(len(self.skewness)))
        ax[1].set_xlabel('Skewness [-]')
        ax[1].set_ylabel('N')

        ax[2].hist(self.orthogonality*180/np.pi, edgecolor='black', color='C2', align='left')
        ax[2].set_title('Orthogonality of %i Faces' %(len(self.orthogonality)))
        ax[2].set_xlabel('Orthogonality [deg]')
        ax[2].set_ylabel('N')
    
    def VisualizeMesh(self):
        # Create a 3D scatter plot
        mesh = pv.StructuredGrid(self.X, self.Y, self.Z)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
        plotter.show_axes()
        plotter.show()

    def GetDualNeighbours(self, idx: tuple) -> np.ndarray:
        """
        Return the coordinates of the dual grid points surrounding an element identified by `idx`
        """
        i, j, k = idx[0], idx[1], idx[2]
        
        x = np.array([self.xv[i,j,k], self.xv[i+1, j, k], self.xv[i+1, j+1, k], self.xv[i, j+1, k], 
                      self.xv[i, j+1, k+1], self.xv[i,j,k+1], self.xv[i+1,j,k+1], self.xv[i+1,j+1,k+1]])
        
        y = np.array([self.yv[i,j,k], self.yv[i+1, j, k], self.yv[i+1, j+1, k], self.yv[i, j+1, k], 
                      self.yv[i, j+1, k+1], self.yv[i,j,k+1], self.yv[i+1,j,k+1], self.yv[i+1,j+1,k+1]])
        
        z = np.array([self.zv[i,j,k], self.zv[i+1, j, k], self.zv[i+1, j+1, k], self.zv[i, j+1, k], 
                      self.zv[i, j+1, k+1], self.zv[i,j,k+1], self.zv[i+1,j,k+1], self.zv[i+1,j+1,k+1]])
        
        return np.vstack((x, y, z))
    
    def GetElementEdges(self, idx: tuple) -> tuple:
        """
        Return the length of the three edges of an element
        """
        i, j, k = idx[0], idx[1], idx[2]
        pt_0 = np.array([self.xv[i,j,k], self.yv[i,j,k], self.zv[i,j,k]])
        pt_i = np.array([self.xv[i+1,j,k], self.yv[i+1,j,k], self.zv[i+1,j,k]])
        pt_j = np.array([self.xv[i,j+1,k], self.yv[i,j+1,k], self.zv[i,j+1,k]])
        pt_k = np.array([self.xv[i,j,k+1], self.yv[i,j,k+1], self.zv[i,j,k+1]])
        i_edge = pt_i-pt_0
        j_edge = pt_j-pt_0
        k_edge = pt_k-pt_0
        return i_edge, j_edge, k_edge
        

    def PlotElement(self, idx: tuple):
        """
        Plot an element corresponding to its indexes tuple `idx`
        """
        i, j, k = idx[0], idx[1], idx[2]
        dual_neighbors = self.GetDualNeighbours(idx)
        ref_length = np.sqrt((dual_neighbors[0,0]-dual_neighbors[0,-1])**2 + 
                             (dual_neighbors[1,0]-dual_neighbors[1,-1])**2 +
                             (dual_neighbors[2,0]-dual_neighbors[2,-1])**2)
        ref_length /= 4
        
        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k], c='red', label='Cell Center')
        ax.plot(dual_neighbors[0,:], dual_neighbors[1,:], dual_neighbors[2,:], '-s', label='Dual Points')

        # Label axes
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        

        self.directions = ['north', 'south', 'east', 'west', 'bottom', 'top']
        for dir in self.directions:
            S, CG = self.GetSurfaceData(i, j, k, dir, 'all')
            S /= np.linalg.norm(S)
            ax.quiver(CG[0], CG[1], CG[2], S[0], S[1], S[2], color='black', length=ref_length)
        ax.set_title('Element (%i, %i, %i)' %(i,j,k))
        ax.set_aspect('equal')
        ax.legend()
                    





        

                    

