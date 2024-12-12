import pickle
from TurboBFM.Preprocess.su2_mesh_generator import generate_2Dmesh_quads

name = 'grid_28_16'
with open("../Grids/"+name+".pik", "rb") as file:
    coords = pickle.load(file)

X = coords['X']
Y = coords['Y']

generate_2Dmesh_quads(X, Y, filename=name)



