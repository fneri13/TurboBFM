import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D 
from TurboBFM.Solver.CConfig import Config

class CMesh():
    
    def __init__(self, config, coords, verbosity=2):
        """
        Starting from the grid points coordinates, generate the array of Volumes associated to the points, and the array of surfaces associated to the interfaces between the points (Dual grid formulation).
        If the grid has (ni,nj,nk) structured points, it has (ni,nj,nk) elements and (ni+1, nj+1, nk+1)*3 internal surfaces (Si the surface connect point (i,j,k) and (i+1,j,k), and analogusly for Sj and Sk).
        """
        self.config = config
        self.verbosity = verbosity
        self.ni, self.nj, self.nk = coords['X'].shape                                   # these are the number of primary nodes
        self.ni_dual, self.nj_dual, self.nk_dual = self.ni+1, self.nj+1, self.nk+1      # these are the number of dual grid points
        self.ni_faces, self.nj_faces, self.nk_faces = self.ni+1, self.nj+1, self.nk+1   # these are the number of interfaces along single directions


        # compute the number of relevant quantities for later use
        self.n_elements = self.ni*self.nj*self.nk                                       # number of finite volumes (excluding ghost elements)
        self.n_faces = (self.ni_faces)*(self.nj_faces)*(self.nk_faces)*3                # total number of internal faces, where fluxes need to be computed
        self.X, self.Y, self.Z = coords['X'], coords['Y'], coords['Z']                  # store a local copy of the primary grid coordinate

        self.ComputeDualGrid()
        self.ComputeInterfaces()
        self.ComputeVolumes()

    def ComputeInterfaces(self):
        """
        Build the interfaces structure, based on an array of CEdge objects
        """

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
        Surface Computations here. The ordering respect the following ideas:
        - Si[i,j,k,:] are the three components of the normal surface interfacing the primary grid point [i,j,k] with [i+1,j,k].
        - Sj[i,j,k,:] are the three components of the normal surface interfacing the primary grid point [i,j,k] with [i,j+1,k].
        - Sk[i,j,k,:] are the three components of the normal surface interfacing the primary grid point [i,j,k] with [i,j,k+1].
        For every point [i,j,k] only three surfaces are needed, since the other three come from the following points on the respective directions.
        
        Interfaces between two ghost points are not needed, therefore the ordering respects this idea: 
        - Si[0,0,0,:] is the i-interface connecting the ghost P[0,1,1] to physical point P[1,1,1].
        - Sj[0,0,0,:] is the j-interface connecting the ghost P[1,0,1] to physical point P[1,1,1].
        - Sk[0,0,0,:] is the k-interface connecting the ghost P[1,1,0] to physical point P[1,1,1].

        The centers of gravity of every interface (CGi, CGj, CGk) follows the same indexing
        """
        self.Si = np.zeros((self.ni_faces, self.nj_faces-1, self.nk_faces-1, 3))            # surface vector connecting point (i,j,k) to (i+1,j,k)
        self.Sj = np.zeros((self.ni_faces-1, self.nj_faces, self.nk_faces-1, 3))            # surface vector connecting point (i,j,k) to (i,j+1,k)
        self.Sk = np.zeros((self.ni_faces-1, self.nj_faces-1, self.nk_faces, 3))            # surface vector connecting point (i,j,k) to (i,j,k+1)
        self.CGi = np.zeros((self.ni_faces, self.nj_faces-1, self.nk_faces-1, 3))           # center of face vector connecting point (i,j,k) to (i+1,j,k)
        self.CGj = np.zeros((self.ni_faces-1, self.nj_faces, self.nk_faces-1, 3))           # center of face vector connecting point (i,j,k) to (i,j+1,k)
        self.CGk = np.zeros((self.ni_faces-1, self.nj_faces-1, self.nk_faces, 3))           # center of face vector connecting point (i,j,k) to (i,j,k+1)
        
        for i in range(self.ni_faces):
            for j in range(self.nj_faces-1):
                for k in range(self.nk_faces-1):
                    self.Si[i,j,k,:], self.CGi[i,j,k,:] = compute_surface_vector_and_cg(self.xv[i,j,k], self.xv[i, j+1, k], self.xv[i, j+1, k+1], self.xv[i, j, k+1], 
                                                                                        self.yv[i,j,k], self.yv[i, j+1, k], self.yv[i, j+1, k+1], self.yv[i, j, k+1], 
                                                                                        self.zv[i,j,k], self.zv[i, j+1, k], self.zv[i, j+1, k+1], self.zv[i, j, k+1])
        
        for i in range(self.ni_faces-1):
            for j in range(self.nj_faces):
                for k in range(self.nk_faces-1):
                    self.Sj[i,j,k,:], self.CGj[i,j,k,:] = compute_surface_vector_and_cg(self.xv[i,j,k], self.xv[i, j, k+1], self.xv[i+1, j, k+1], self.xv[i+1, j, k], 
                                                                                        self.yv[i,j,k], self.yv[i, j, k+1], self.yv[i+1, j, k+1], self.yv[i+1, j, k], 
                                                                                        self.zv[i,j,k], self.zv[i, j, k+1], self.zv[i+1, j, k+1], self.zv[i+1, j, k])
            
        for i in range(self.ni_faces-1):
            for j in range(self.nj_faces-1):
                for k in range(self.nk_faces):
                    self.Sk[i,j,k,:], self.CGk[i,j,k,:] = compute_surface_vector_and_cg(self.xv[i,j,k], self.xv[i+1, j, k], self.xv[i+1, j+1, k], self.xv[i, j+1, k], 
                                                                                        self.yv[i,j,k], self.yv[i+1, j, k], self.yv[i+1, j+1, k], self.yv[i, j+1, k], 
                                                                                        self.zv[i,j,k], self.zv[i+1, j, k], self.zv[i+1, j+1, k], self.zv[i, j+1, k])

        plt.figure()
        def plot_grid_lines(xgrid, ygrid, color):
            ni, nj = xgrid.shape[0], xgrid.shape[1]
            for i in range(ni):
                plt.plot(xgrid[i,:,0], ygrid[i,:,0], '%s' %(color), lw=0.75, mfc='none')
            for j in range(nj):
                plt.plot(xgrid[:,j,0], ygrid[:,j,0], '%s' %(color), lw=0.75, mfc='none')
            return None
        
        plot_grid_lines(self.X, self.Y, 'ko')
        plot_grid_lines(self.xv, self.yv, '--r^')
        for i in range(self.ni_faces):
            for j in range(self.nj_faces-1):
                plt.quiver(self.CGi[i,j,0,0], self.CGi[i,j,0,1], self.Si[i,j,0,0], self.Si[i,j,0,1])
        for i in range(self.ni_faces-1):
            for j in range(self.nj_faces):
                plt.quiver(self.CGj[i,j,0,0], self.CGj[i,j,0,1], self.Sj[i,j,0,0], self.Sj[i,j,0,1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('k=%i plane' %(0))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()



    def ComputeVolumes(self):
        """
        Build the elements structure, based on an array of CElements objects
        """
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
        
        # if self.verbosity==2:
        #     print('='*20 + ' ELEMENTS INFORMATION ' + '='*20)
        #     for i in range(1, self.ni-1):
        #         for j in range(1, self.nj-1):
        #             for k in range(1, self.nk-1):
        #                 print('For point (%i,%i,%i):' %(i,j,k))
        #                 print('                         Si=[%.2e,%.2e,%.2e]' %(self.Si[i,j,k,0],self.Si[i,j,k,1],self.Si[i,j,k,2]))
        #                 print('                         CGi=[%.2e,%.2e,%.2e]' %(self.CGi[i,j,k,0],self.CGi[i,j,k,1],self.CGi[i,j,k,2]))
        #                 print('                         Sj=[%.2e,%.2e,%.2e]' %(self.Sj[i,j,k,0],self.Sj[i,j,k,1],self.Sj[i,j,k,2]))
        #                 print('                         CGj=[%.2e,%.2e,%.2e]' %(self.CGj[i,j,k,0],self.CGj[i,j,k,1],self.CGj[i,j,k,2]))
        #                 print('                         Sk=[%.2e,%.2e,%.2e]' %(self.Sk[i,j,k,0],self.Sk[i,j,k,1],self.Sk[i,j,k,2]))
        #                 print('                         CGk=[%.2e,%.2e,%.2e]' %(self.CGk[i,j,k,0],self.CGk[i,j,k,1],self.CGk[i,j,k,2]))
        #                 print('                         Vol=%.4e' %(self.V[i,j,k]))
        #                 print()
        #     print('='*20 + ' END ELEMENTS INFORMATION ' + '='*20)
        

        # print('='*25 + ' MESH INFORMATION ' + '='*25)
        # print('Number of elements:                  %i' %(self.n_elements))
        # print('Number of internal faces:            %i' %(self.n_faces))
        # print('Type of elements:                    %s' %('Hexagonal'))

        

        # self.ComputeMeshQuality()
        # print('Max aspect ratio:                    %.2f' %(np.max(self.aspect_ratio)))
        # print('Max skewness:                        %.2f' %(np.max(self.skewness)))
        # print('Max orthogonality:                   %.2f' %(np.max(self.orthogonality)))
        # print('='*25 + ' END MESH INFORMATION ' + '='*25)



    def ComputeDualGrid(self):
        """
        Compute the dual grid points
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
        self.ComputeAspectRatio()
        self.ComputeSkewnessOrthogonality()


    def ComputeAspectRatio(self):
        """
        Compute aspect ratio, and store it in a 1D numpy array
        """
        ni, nj, nk = self.xv.shape
        self.aspect_ratio = []

        # the loop extremes avoid to include data of the ghost cells, which should not be relevant for the CFD simulation
        for i in range(ni-1):
            for j in range(nj-1):
                for k in range(nk-1):
                    pt_0 = np.array([self.xv[i,j,k], self.yv[i,j,k], self.zv[i,j,k]])
                    pt_i = np.array([self.xv[i+1,j,k], self.yv[i+1,j,k], self.zv[i+1,j,k]])
                    pt_j = np.array([self.xv[i,j+1,k], self.yv[i,j+1,k], self.zv[i,j+1,k]])
                    pt_k = np.array([self.xv[i,j,k+1], self.yv[i,j,k+1], self.zv[i,j,k+1]])
                    i_edge = np.linalg.norm(pt_i-pt_0)
                    j_edge = np.linalg.norm(pt_j-pt_0)
                    k_edge = np.linalg.norm(pt_k-pt_0)
                    ar = max(i_edge, j_edge, k_edge)/min(i_edge, j_edge, k_edge)
                    self.aspect_ratio.append(ar)
        self.aspect_ratio = np.array(self.aspect_ratio)
    
    def ComputeSkewnessOrthogonality(self):
        """
        Compute the mesh skewness, defined as the distance between the real center face and the midpoint connecting the two cells, normalized
        by the distance between the two elements center. And the orthogonality, defined as the angle between the connecting line and the face normal.
        """
        ni, nj, nk = self.Si[:,:,:,0].shape
        self.skewness = []
        self.orthogonality = []
        for i in range(0, ni):
            for j in range(0, nj):
                for k in range(0, nk):
                    pt_0 = np.array([self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k]])
                    pt_i = np.array([self.X[i+1,j,k], self.Y[i+1,j,k], self.Z[i+1,j,k]])
                    pt_j = np.array([self.X[i,j+1,k], self.Y[i,j+1,k], self.Z[i,j+1,k]])
                    pt_k = np.array([self.X[i,j,k+1], self.Y[i,j,k+1], self.Z[i,j,k+1]])

                    #face_i
                    midpoint = pt_0 + 0.5*(pt_i-pt_0)   # point in the middle between the cell centers
                    CG = self.CGi[i,j,k,:]              # face center
                    l1 = np.linalg.norm(midpoint-CG)
                    l2 = np.linalg.norm(pt_i-pt_0)
                    if (l1/l2)!=0:
                        pass
                    self.skewness.append(l1/l2)
                    S = self.Si[i,j,k,:]
                    l2 = pt_i-pt_0 
                    angle = np.arccos(np.dot(l2, S)/np.linalg.norm(l2)/np.linalg.norm(S))
                    self.orthogonality.append(angle)

                    #face_j
                    midpoint = pt_0 + 0.5*(pt_j-pt_0)   # point in the middle between the cell centers
                    CG = self.CGj[i,j,k,:]              # face center
                    l1 = np.linalg.norm(midpoint-CG)
                    l2 = np.linalg.norm(pt_j-pt_0)
                    if (l1/l2)!=0:
                        pass
                    self.skewness.append(l1/l2)
                    S = self.Sj[i,j,k,:]
                    l2 = pt_j-pt_0
                    angle = np.arccos(np.dot(l2, S)/np.linalg.norm(l2)/np.linalg.norm(S))
                    self.orthogonality.append(angle)

                    #face_k
                    midpoint = pt_0 + 0.5*(pt_k-pt_0)   # point in the middle between the cell centers
                    CG = self.CGk[i,j,k,:]              # face center
                    l1 = np.linalg.norm(midpoint-CG)
                    l2 = np.linalg.norm(pt_k-pt_0)
                    if (l1/l2)!=0:
                        pass
                    self.skewness.append(l1/l2)
                    S = self.Sk[i,j,k,:]
                    l2 = pt_k-pt_0
                    angle = np.arccos(np.dot(l2, S)/np.linalg.norm(l2)/np.linalg.norm(S))
                    self.orthogonality.append(angle)
        
        self.skewness = np.array(self.skewness)
        self.orthogonality = np.array(self.orthogonality)


    def PlotMeshQuality(self):
        """
        Plot the histograms of aspect ratio, skewness and orthogonality
        """
        fig, ax = plt.subplots(1, 3, figsize=(16,9))

        ax[0].hist(self.aspect_ratio, edgecolor='black', color='C0', align='left')
        ax[0].set_title('AR of %i Elements' %(self.n_elems))
        ax[0].set_xlabel('Aspect Ratio')
        ax[0].set_ylabel('N')        

        ax[1].hist(self.skewness, edgecolor='black', color='C1', align='left')
        ax[1].set_title('Skewness of %i Faces' %(self.n_faces))
        ax[1].set_xlabel('Skewness')
        ax[1].set_ylabel('N')

        ax[2].hist(self.orthogonality, edgecolor='black', color='C2', align='left')
        ax[2].set_title('Orthogonality of %i Faces' %(self.n_faces))
        ax[2].set_xlabel('Orthogonality')
        ax[2].set_ylabel('N')
    
    def VisualizeMesh(self):
        # Create a 3D scatter plot
        mesh = pv.StructuredGrid(self.X, self.Y, self.Z)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, cmap='viridis', show_edges=True)
        plotter.show_axes()
        plotter.show()

                    





        

                    

