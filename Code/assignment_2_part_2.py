# Test reconstruction of a set of samples with the RBF method

# Import geometry processing library
import geomproc

# Import numpy for data arrays
import numpy as np

# Import math functions
import math

# Measure execution time
import time


# Define vectorized kernel for reconstruction
def euclidean(x, y):
    dp = -2.0*np.dot(x, y.T) + np.sum(np.square(x), axis=1, keepdims=True) + np.sum(np.square(y), axis=1, keepdims=True).T
    dp = np.where(dp < 0, 0, dp)
    return np.sqrt(dp)

def wendland(x,y,h):
    #np.linalg.norm(x - y)
    first_term = 1 - euclidean(x, y)/h
    np.where(first_term < 0, 0, first_term)
    return (np.power(first_term, 4))*(4.0*euclidean(x, y)/h + 1)
    #return (math.pow(1 - np.linalg.norm(x - y)/h, 4))*(4.0*np.linalg.norm(x - y)/h + 1)

def rbf_reconstruction(mesh, pc, kernel_index = 0, wendland_h = 0):
    if kernel_index == 0: 
        print('using triharmonic kernel')
        kernel = lambda x, y: np.power(euclidean(x, y), 3.0)
    elif kernel_index == 1:
        print('using biharmonic kernel')
        kernel = lambda x, y: euclidean(x, y)
    else:
        print('using Wendland kernel, with h = ' + str(wendland_h))
        #wendland = lambda x, y, h: (math.pow(1 - euclidean(x, y)/h, 4))*(4.0*euclidean(x, y)/h + 1)
        kernel = lambda x, y: wendland(x, y, wendland_h)

    # Define epsilon for displacing samples
    epsilon = 0.01
    
    # Run RBF reconstruction
    print('Reconstructing implicit function')
    start_time = time.time()
    surf = geomproc.impsurf()
    surf.setup_rbf(pc, epsilon, kernel, True)

    # Run marching cubes
    print('Running marching cubes')
    rec = geomproc.marching_cubes(np.array([-1.5, -1.5, -1.5]), np.array([1.5, 1.5, 1.5]), 64, surf.evaluate)

    # Report time
    end_time = time.time()
    print('Execution time = ' + str(end_time - start_time) +'s')

    # Save output mesh
    if kernel_index == 0: 
        rec.save(mesh + '_rec_triharmonic.obj')
    elif kernel_index == 1:
        rec.save(mesh + '_rec_biharmonic.obj')
    else:
        rec.save(mesh + '_rec_Wendland_h_' + str(wendland_h) + '.obj')
                               
def load_and_sample_mesh(mesh):
    print('loading the ' + mesh + ' mesh')
    # Load and normalize the mesh       
    tm = geomproc.load('meshes/' + mesh + '.obj')
    tm.normalize()

    # Save normalized mesh
    tm.save(mesh + '_normalized.obj')

    # Compute normal vectors
    tm.compute_vertex_and_face_normals()

    # Sample a point cloud from the mesh
    n = round(tm.vertex.shape[0]/4)
    pc = tm.sample(n)

    # Save samples
    pnt = geomproc.create_points(pc.point, radius=0.01, color=[1, 0, 0])
    pnt.save(mesh + '_samples.obj')

    return pc
                               
                               
mesh = 'sphere'
pc = load_and_sample_mesh(mesh)
rbf_reconstruction(mesh, pc, 0)
rbf_reconstruction(mesh, pc, 1)
rbf_reconstruction(mesh, pc, 2, 0.001)
rbf_reconstruction(mesh, pc, 2, 0.01)
rbf_reconstruction(mesh, pc, 2, 0.1)
rbf_reconstruction(mesh, pc, 2, 1.0)

mesh = 'bunny'
pc = load_and_sample_mesh(mesh)
rbf_reconstruction(mesh, pc, 0)
rbf_reconstruction(mesh, pc, 1)
rbf_reconstruction(mesh, pc, 2, 0.001)
rbf_reconstruction(mesh, pc, 2, 0.01)
rbf_reconstruction(mesh, pc, 2, 0.1)
rbf_reconstruction(mesh, pc, 2, 1.0)