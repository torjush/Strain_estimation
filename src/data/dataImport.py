import numpy as np


def read_vtk(filename, verbose=False):
    f = open(filename, 'r')
    line = f.readline()
    if line[2:5] != 'vtk':
        raise RuntimeError('File is not a valid vtk file')
    # Throw header
    for _ in range(3):
        f.readline()
    # Read number of vertices
    n_vertices = int(f.readline().split()[1])
    vertices = np.empty(shape=(n_vertices, 3))

    for i in range(n_vertices):
        vals = np.array([float(x) for x in f.readline().split()])
        vertices[i, :] = vals

    f.close()
    vertices = vertices[:, :-1]
    return vertices
