import numpy as np

def generate_SU2mesh(*coords, kind_elem, kind_bound, full_annulus=False, filename='mesh.su2') -> None:
    """
    Functions to generate a su2 mesh file starting from coordinates arrays.
    
    Parameters
    -------------------------------

    `coords`: coordinate arrays (x,y,z)
    
    `kind_elem`: element kind of the mesh (9 for 2D Quadrilateral, 12 for 3D Hexahedra)
    
    `kind_bound`: kind of the boundary elements (3 for 2D lines, 9 for 3D quadrilaterals)
    
    `full_annulus`: if True select a full-annulus mesh generation (no periodic boundaries)
    
    `filename`: file name of the mesh file to save (without extension)
    """
    if len(coords) == 2:
        X = coords[0]
        Y = coords[1]
        Z = X * 0
        ndim = 2
    elif len(coords) == 3:
        X = coords[0]
        Y = coords[1]
        Z = coords[2]
        ndim = 3
    else:
        raise ValueError('Too many coordinate values given')

    if ndim == 2 and kind_elem == 9 and kind_bound == 3:
        generate_2Dmesh_quads(X, Y, filename)
    elif ndim == 3 and kind_elem == 12 and kind_bound == 9:
        if full_annulus:
            generate_3Dmesh_full(X, Y, Z, filename)
        else:
            generate_3Dmesh_sector(X, Y, Z, filename)
    else:
        raise ValueError('Mesh type not implemented yet.')


def generate_2Dmesh_quads(X: np.ndarray, Y: np.ndarray, filename: str) -> None:
    """
    Generate 2d su2 mesh file with quads elements.

    Parameters
    ----------------------------
    `X`: 2D array of x coordinates
    
    `Y`: 2D array of y coordinates

    `filename`: name of the mesh to save (without extensions)
    """
    nNode = X.shape[0]
    mNode = X.shape[1]
    Mesh_File = open(filename+'.su2', "w")

    KindElem = 9  # Quad
    KindBound = 3  # Line

    # Write the dimension of the problem and the number of interior elements
    Mesh_File.write("%\n")
    Mesh_File.write("% Problem dimension\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NDIME= 2\n")
    Mesh_File.write("%\n")
    Mesh_File.write("% Inner element connectivity\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NELEM= %s\n" % ((nNode - 1) * (mNode - 1)))

    # Write the element connectivity
    iElem = 0
    for iNode in range(nNode - 1):
        for jNode in range(mNode - 1):
            zero = iNode * mNode + jNode
            one = (iNode + 1) * mNode + jNode
            two = (iNode + 1) * mNode + jNode + 1
            three = iNode * mNode + jNode + 1
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindElem, zero, one, two, three))
            iElem = iElem + 1

    # Compute the number of nodes and write the node coordinates
    nPoint = nNode * mNode
    Mesh_File.write("%\n")
    Mesh_File.write("% Node coordinates\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NPOIN= %s\n" % (nNode * mNode))
    iPoint = 0
    for iNode in range(nNode):
        for jNode in range(mNode):
            Mesh_File.write("%15.14f \t %15.14f \t %s\n" % (X[iNode, jNode], Y[iNode, jNode], iPoint))
            iPoint = iPoint + 1

    # Write the header information for the boundary markers
    Mesh_File.write("%\n")
    Mesh_File.write("% Boundary elements\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NMARK= 4\n")

    # Write the boundary information for each marker
    Mesh_File.write("MARKER_TAG= HUB\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % (nNode - 1))
    for iNode in range(nNode - 1):
        Mesh_File.write("%s \t %s \t %s\n" % (KindBound, iNode * mNode, (iNode + 1) * mNode))

    Mesh_File.write("MARKER_TAG= OUTLET\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % (mNode - 1))
    for jNode in range(mNode - 1):
        Mesh_File.write("%s \t %s \t %s\n" % (KindBound, jNode + (nNode - 1) * mNode, jNode + 1 + (nNode - 1) * mNode))

    Mesh_File.write("MARKER_TAG= SHROUD\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % (nNode - 1))
    for iNode in range(nNode - 1):
        Mesh_File.write("%s \t %s \t %s\n" % (KindBound, iNode * mNode + (mNode - 1), (iNode + 1) * mNode + (mNode - 1)))

    Mesh_File.write("MARKER_TAG= INLET\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % (mNode - 1))
    for jNode in range(mNode - 1):
        Mesh_File.write("%s \t %s \t %s\n" % (KindBound, jNode, jNode + 1))

    # Close the mesh file and exit
    Mesh_File.close()


def generate_3Dmesh_full(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, filename: str) -> None:
    """
    Compute full annulus mesh, starting from 3D arrays. In theory the past k-plane of the arrays should be equal to k-0
    plane if 360deg are covered from the point coordinates.

    Parameters
    ----------------------------
    `X`: 3D array of x coordinates
    
    `Y`: 3D array of y coordinates

    `Z`: 3D array of z coordinates

    `filename`: name of the mesh to save (without extensions)
    """
    X = X[:, :, 0:-1]
    Y = Y[:, :, 0:-1]
    Z = Z[:, :, 0:-1]

    # Set the VTK type for the interior elements and the boundary elements
    KindElem = 12  # Quad
    KindBound = 9  # Line

    # Store the number of nodes and open the output mesh file
    nNode = X.shape[0]
    mNode = X.shape[1]
    lNode = X.shape[2]
    Mesh_File = open(filename, "w")

    # Write the dimension of the problem and the number of interior elements
    Mesh_File.write("%\n")
    Mesh_File.write("% Problem dimension\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NDIME= 3\n")
    Mesh_File.write("%\n")
    Mesh_File.write("% Inner element connectivity\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NELEM= %s\n" % ((nNode - 1) * (mNode - 1) * lNode))

    # Write the element connectivity
    iElem = 0
    for iNode in range(nNode - 1):
        for jNode in range(mNode - 1):
            for kNode in range(lNode):
                zero = kNode + jNode * lNode + iNode * lNode * mNode
                one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
                two = kNode + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
                three = kNode + jNode * lNode + (iNode + 1) * lNode * mNode
                if kNode < lNode - 1:
                    four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
                    five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
                    six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
                    seven = kNode + 1 + jNode * lNode + (iNode + 1) * lNode * mNode
                elif kNode == lNode - 1:
                    four = 0 + jNode * lNode + iNode * lNode * mNode
                    five = 0 + (jNode + 1) * lNode + iNode * lNode * mNode
                    six = 0 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
                    seven = 0 + jNode * lNode + (iNode + 1) * lNode * mNode
                else:
                    raise ValueError('Error in the loop indices')

                Mesh_File.write("%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s\n" % (
                    KindElem, zero, one, two, three, four, five, six, seven))
                iElem = iElem + 1

    # Compute the number of nodes and write the node coordinates
    Mesh_File.write("%\n")
    Mesh_File.write("% Node coordinates\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NPOIN= %s\n" % (nNode * mNode * lNode))
    iPoint = 0
    for iNode in range(nNode):
        for jNode in range(mNode):
            for kNode in range(lNode):
                Mesh_File.write("%15.14f \t %15.14f \t %15.14f \t %s\n" % (
                    X[iNode, jNode, kNode], Y[iNode, jNode, kNode], Z[iNode, jNode, kNode], iPoint))
                iPoint = iPoint + 1

    # Write the header information for the boundary markers
    Mesh_File.write("%\n")
    Mesh_File.write("% Boundary elements\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NMARK= 4\n")

    # Write the boundary information for each marker
    Mesh_File.write("MARKER_TAG= HUB\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((nNode - 1) * lNode))
    for iNode in range(nNode - 1):
        for kNode in range(lNode):
            jNode = 0
            zero = kNode + jNode * lNode + iNode * lNode * mNode
            three = kNode + jNode * lNode + (iNode + 1) * lNode * mNode
            if kNode < lNode - 1:
                four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
                seven = kNode + 1 + jNode * lNode + (iNode + 1) * lNode * mNode
            elif kNode == lNode - 1:
                four = 0 + jNode * lNode + iNode * lNode * mNode
                seven = 0 + jNode * lNode + (iNode + 1) * lNode * mNode
            else:
                raise ValueError('Error in the hub loop')
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, zero, four, seven, three))

    Mesh_File.write("MARKER_TAG= OUTLET\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1) * lNode))
    for jNode in range(mNode - 1):
        for kNode in range(lNode):
            iNode = nNode - 2
            two = kNode + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            three = kNode + jNode * lNode + (iNode + 1) * lNode * mNode
            if kNode < lNode - 1:
                six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
                seven = kNode + 1 + jNode * lNode + (iNode + 1) * lNode * mNode
            elif kNode == lNode - 1:
                six = 0 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
                seven = 0 + jNode * lNode + (iNode + 1) * lNode * mNode
            else:
                raise ValueError('Error in the outlet loop')
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, two, six, seven, three))

    Mesh_File.write("MARKER_TAG= SHROUD\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((nNode - 1) * lNode))
    for iNode in range(nNode - 1):
        for kNode in range(lNode):
            jNode = mNode - 2
            one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
            two = kNode + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            if kNode < lNode - 1:
                five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
                six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            elif kNode == lNode - 1:
                five = 0 + (jNode + 1) * lNode + iNode * lNode * mNode
                six = 0 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            else:
                raise ValueError('Error in the shroud loop')
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, one, two, six, five))

    Mesh_File.write("MARKER_TAG= INLET\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1) * lNode))
    for jNode in range(mNode - 1):
        for kNode in range(lNode):
            iNode = 0
            zero = kNode + jNode * lNode + iNode * lNode * mNode
            one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
            if kNode < lNode - 1:
                four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
                five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
            elif kNode == lNode - 1:
                four = 0 + jNode * lNode + iNode * lNode * mNode
                five = 0 + (jNode + 1) * lNode + iNode * lNode * mNode
            else:
                raise ValueError('Error in the inlet loop')
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, zero, one, five, four))

    Mesh_File.close()


def generate_3Dmesh_sector(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, filename: str) -> None:
    """
    Compute the sector of the annulus mesh, starting from 3D arrays.
    The first and last k planes are related through periodic boundary conditions.
    
    Parameters
    ----------------------------
    `X`: 3D array of x coordinates
    
    `Y`: 3D array of y coordinates

    `Z`: 3D array of z coordinates

    `filename`: name of the mesh to save (without extensions)
    """
    KindElem = 12  # Quad
    KindBound = 9  # Line

    # Store the number of nodes and open the output mesh file
    nNode = X.shape[0]
    mNode = X.shape[1]
    lNode = X.shape[2]
    Mesh_File = open(filename, "w")

    # Write the dimension of the problem and the number of interior elements
    Mesh_File.write("%\n")
    Mesh_File.write("% Problem dimension\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NDIME= 3\n")
    Mesh_File.write("%\n")
    Mesh_File.write("% Inner element connectivity\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NELEM= %s\n" % ((nNode - 1) * (mNode - 1) * (lNode - 1)))

    # Write the element connectivity
    iElem = 0
    for iNode in range(nNode - 1):
        for jNode in range(mNode - 1):
            for kNode in range(lNode - 1):
                zero = kNode + jNode * lNode + iNode * lNode * mNode
                one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
                two = kNode + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
                three = kNode + jNode * lNode + (iNode + 1) * lNode * mNode
                four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
                five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
                six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
                seven = kNode + 1 + jNode * lNode + (iNode + 1) * lNode * mNode
                Mesh_File.write("%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s\n" % (
                    KindElem, zero, one, two, three, four, five, six, seven))
                iElem = iElem + 1

    # Compute the number of nodes and write the node coordinates
    nPoint = nNode * mNode * lNode
    Mesh_File.write("%\n")
    Mesh_File.write("% Node coordinates\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NPOIN= %s\n" % (nNode * mNode * lNode))
    iPoint = 0
    for iNode in range(nNode):
        for jNode in range(mNode):
            for kNode in range(lNode):
                Mesh_File.write("%15.14f \t %15.14f \t %15.14f \t %s\n" % (
                    X[iNode, jNode, kNode], Y[iNode, jNode, kNode], Z[iNode, jNode, kNode], iPoint))
                iPoint = iPoint + 1

    # Write the header information for the boundary markers
    Mesh_File.write("%\n")
    Mesh_File.write("% Boundary elements\n")
    Mesh_File.write("%\n")
    Mesh_File.write("NMARK= 6\n")

    # Write the boundary information for each marker
    Mesh_File.write("MARKER_TAG= HUB\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((nNode - 1) * (lNode - 1)))
    for iNode in range(nNode - 1):
        for kNode in range(lNode - 1):
            jNode = 0
            zero = kNode + jNode * lNode + iNode * lNode * mNode
            three = kNode + jNode * lNode + (iNode + 1) * lNode * mNode
            four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
            seven = kNode + 1 + jNode * lNode + (iNode + 1) * lNode * mNode
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, zero, four, seven, three))

    Mesh_File.write("MARKER_TAG= OUTLET\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1) * (lNode - 1)))
    for jNode in range(mNode - 1):
        for kNode in range(lNode - 1):
            iNode = nNode - 2
            two = kNode + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            three = kNode + jNode * lNode + (iNode + 1) * lNode * mNode
            six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            seven = kNode + 1 + jNode * lNode + (iNode + 1) * lNode * mNode
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, two, six, seven, three))

    Mesh_File.write("MARKER_TAG= SHROUD\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((nNode - 1) * (lNode - 1)))
    for iNode in range(nNode - 1):
        for kNode in range(lNode - 1):
            jNode = mNode - 2
            one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
            two = kNode + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
            six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, one, two, six, five))

    Mesh_File.write("MARKER_TAG= INLET\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1) * (lNode - 1)))
    for jNode in range(mNode - 1):
        for kNode in range(lNode - 1):
            iNode = 0
            zero = kNode + jNode * lNode + iNode * lNode * mNode
            one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
            four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
            five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, zero, one, five, four))

    Mesh_File.write("MARKER_TAG= PER0\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1) * (nNode - 1)))
    for iNode in range(nNode - 1):
        for jNode in range(mNode - 1):
            kNode = 0
            zero = kNode + jNode * lNode + iNode * lNode * mNode
            one = kNode + (jNode + 1) * lNode + iNode * lNode * mNode
            two = kNode + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            three = kNode + jNode * lNode + (iNode + 1) * lNode * mNode
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, zero, three, two, one))

    Mesh_File.write("MARKER_TAG= PER1\n")
    Mesh_File.write("MARKER_ELEMS= %s\n" % ((mNode - 1) * (nNode - 1)))
    for iNode in range(nNode - 1):
        for jNode in range(mNode - 1):
            kNode = lNode - 2
            four = kNode + 1 + jNode * lNode + iNode * lNode * mNode
            five = kNode + 1 + (jNode + 1) * lNode + iNode * lNode * mNode
            six = kNode + 1 + (jNode + 1) * lNode + (iNode + 1) * lNode * mNode
            seven = kNode + 1 + jNode * lNode + (iNode + 1) * lNode * mNode
            Mesh_File.write("%s \t %s \t %s \t %s \t %s\n" % (KindBound, four, seven, six, five))

    # Close the mesh file and exit
    Mesh_File.close()
