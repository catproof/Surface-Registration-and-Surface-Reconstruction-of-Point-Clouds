# Align two point clouds with descriptor matching

# Import the geometry processing library
import geomproc

import time

# Import numpy for data arrays
import numpy as np

# Math functions
import math

import random

#changes to geomproc:
#mesh.py: 
#added get_point_cloud function for getting the point cloud of a mesh

#alignment.py: 
#added function best_matches for getting the top n matches of each point
#added functions binary_search_and_add, binary_search_and_add_helper and insert_after_any_duplicates 
#as sub-functions for best_matches

#pcloud.py:
#added function add_noise for adding noise to a point cloud, translating points using gaussian noise

#### Create point samples for the test

# Load and normalize the mesh
tm1 = geomproc.load('meshes/bunny.obj')
tm1.normalize()
tm1.compute_vertex_and_face_normals()

print(tm1.vertex.shape[0])
print(tm1.face.shape[0])

# Apply a transformation to the mesh to create a misaligned version
# Define the transformation
if False: # Choose the transformation by switching True/False
    # Choose a completely random transformation
    rnd = np.random.random((3, 3))
    [q, r] = np.linalg.qr(rnd)
    orig_rot = q
    orig_trans = np.random.random((3, 1))
else:
    # Choose a specific rotation around the x axis, then y axis, then z axis
    x_angle = math.pi/3
    y_angle = math.pi/2
    z_angle = math.pi/4
    orig_rot_x = np.array([[1, 0, 0],
                         [0, math.cos(x_angle), -math.sin(x_angle)],
                         [0,math.sin(x_angle), math.cos(x_angle)]])
    orig_rot_y = np.array([[math.cos(y_angle), 0, math.sin(y_angle)],
                     [0, 1, 0],
                     [-math.sin(y_angle), 0, math.cos(y_angle)]])
    orig_rot_z = np.array([[math.cos(z_angle), -math.sin(z_angle), 0],
                 [math.sin(z_angle), math.cos(z_angle), 0],
                 [0, 0, 1]])
    orig_rot = np.matmul(orig_rot_z, np.matmul(orig_rot_y, orig_rot_x))
    orig_trans = np.array([[0.5],[0.2],[0.3]])

# Copy the mesh and apply the transformation to the mesh
tm2 = tm1.copy()
tm2.vertex = geomproc.apply_transformation(tm1.vertex, orig_rot, orig_trans)
# Update normals: important! As the spin images descriptor depends on them
tm2.compute_vertex_and_face_normals() 

# Sample two sets of points from the surfaces of the meshes
n1 = 1000 # We are going to compute a descriptor for each of these points
n2 = 10000 # We are going to use these points only for reference in the descriptor computation
pc1 = tm1.sample(n1)
pc1full = tm1.sample(n2)
pc2 = tm2.sample(n1)
pc2full = tm2.sample(n2)
# pc1 = tm1.get_point_cloud()
# pc2 = tm2.get_point_cloud()
# pc1full = tm1.get_point_cloud()
# pc2full = tm2.get_point_cloud()

#add noise
pc1.add_noise()
pc2.add_noise()
pc1full.add_noise()
pc2full.add_noise()

# #estimate normals
pc1.estimate_normals(4)
pc2.estimate_normals(4)
pc1full.estimate_normals(4)
pc2full.estimate_normals(4)

# Save input data
tm1.save('bunny_normalized.obj')
pc1.save('bunny_sample1.obj')
pc2.save('bunny_sample2.obj')


#### Align the point clouds

# Compute the spin images descriptor for the two point clouds
opt = geomproc.spin_image_options()
start = time.time()
desc1 = geomproc.spin_images(pc1, pc1full, opt)
desc2 = geomproc.spin_images(pc2, pc2full, opt)
end = time.time()
print("The time of execution of spin_images is :", end-start)

num_matches = 3
# Match the descriptors
start = time.time()
corr_unsampled = geomproc.alignment.best_matches(desc1, desc2,num_matches)
end = time.time()
print("The time of execution of best_matches is :", end-start)

corr_unsampled_best_match = geomproc.alignment.best_match(desc1, desc2)

def sample_correspondences(corr, keep):
    #return corr[np.random.randint(corr.shape[0], size=round(keep * corr.shape[0])), :]
    sample = random.sample(corr, round(keep * len(corr)))
    sample_top_matches = []
    for i in sample:
        random_match = random.sample(i[1], 1)
        sample_top_matches.append([i[0],random_match[0][0],random_match[0][1]])
    return sample_top_matches

#corr = geomproc.filter_correspondences(corr, 0.3)
corr = sample_correspondences(corr_unsampled, 0.1)

def count_inliers(pc1tr, pc2, corr, threshold, full_corr):
    num_inliers = 0
    average_error = 0
    inlier_mask = []
    
    for i in range(corr.shape[0]):
        if np.linalg.norm(pc1tr.point[corr[i,0], :] - pc2.point[corr[i,1], :]) < threshold:
            num_inliers = num_inliers + 1
            inlier_mask.append(True)
        else:
            inlier_mask.append(False)

    #print('num_inliers:' + str(num_inliers))
    return corr[inlier_mask,:]
 
start = time.time()


min_error = math.inf
min_error_transformation = 0
num_trials = 50
threshold = 0.1
min_error_transformation = 0
max_number_of_inliers = 0
best_corr = 0
for i in range(num_trials):
    #print("trial number: " + str(i))
    corr = sample_correspondences(corr_unsampled, 0.1)
    corr = np.array(corr).astype(int)
    full_corr = corr
    previous_num_inliers = corr.shape[0]
    rot = 0
    trans = 0
    while True:
        # Derive a transformation from the point match
        [rot, trans] = geomproc.transformation_from_correspondences(pc1, pc2, corr)

        # Apply the transformation to align the meshes
        pc1tr = pc1.copy()
        pc1tr.point = geomproc.apply_transformation(pc1.point, rot, trans)

        corr = count_inliers(pc1tr, pc2, corr, threshold, full_corr)
        
        if corr.shape[0] == 0 or corr.shape[0] < max_number_of_inliers:
            #print("no inliers!")
            break
    
        if previous_num_inliers == corr.shape[0]:
            break
        else:
            previous_num_inliers = corr.shape[0]
    
    if corr.shape[0] > max_number_of_inliers:
        #print("found a new best transformation! (based on amount of inliers)")
        best_corr = corr
        max_number_of_inliers = corr.shape[0]
        min_error_transformation = [rot, trans]

print("max_number_of_inliers: " + str(max_number_of_inliers))

end = time.time()

print("The time of execution of RANSAC is :", end-start)
        
# Save registration
pc1tr.save('bunny_sample1aligned.obj')

rot = min_error_transformation[0]
trans = min_error_transformation[1]
pc1tr = pc1.copy()
pc1tr.point = geomproc.apply_transformation(pc1.point, rot, trans)

# Save final correspondence so that we can see it
if True: # Turn on/off with True/False
    # Create lines to indicate the correspondences
    line = np.zeros((best_corr.shape[0], 6))
    for i in range(best_corr.shape[0]):
        line[i, :] = np.concatenate((pc1tr.point[best_corr[i][0], :], pc2.point[best_corr[i][1], :]))
    cl = geomproc.create_lines(line, color=[0.7, 0.7, 0.7])
    # Create points for the point sets
    pt1 = geomproc.create_points(pc1tr.point, color=[1, 0, 0])
    pt2 = geomproc.create_points(pc1.point, color=[1, 0, 0])
    pt3 = geomproc.create_points(pc2.point, color=[0, 0, 1])
    
    wo = geomproc.write_options()
    wo.write_vertex_colors = True
    
    # meshes aligned using ransac
    result = geomproc.mesh()
    result.append(cl)
    result.append(pt1)
    result.append(pt3)
    # Save the mesh
    result.save('bunny_corr.obj', wo)
    
    # meshes before being aligned
    result = geomproc.mesh()
    result.append(pt2)
    result.append(pt3)
    # Save the mesh
    result.save('bunny_uncorr.obj', wo)

# Print some information
print('Original transformation = ')
print(orig_rot)
print(orig_trans)
print('Alignment result = ')
print(rot)
print(trans)