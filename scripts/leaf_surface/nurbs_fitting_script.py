# attemted to convert the jupyter notebook to this script with chatgpt, but it does not work. 
# if needed this should be done manually or this script thoroughly debugged. 

# !!!!!!!!!!!!!!!!!!!!!!
# This is NOT working as expected. !!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

# Define parameters
grid_size = 20  # Default grid size
visualize = True  # Flag for visualizations
offset_value = 10  # Default offset value along third axis
surf_delta = 0.1  # Surface delta for fitting
alpha = 0.1  # Alpha for hull fitting
data_file_name = 'data/leaf_area/white_leaf/pc_whiteLeaf_03_s.ply'  # Path to the point cloud data file

# Set up file paths dynamically based on the root of the repository
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '../..'))  # Assuming the script is in /scripts/leaf_surface
output_dir = script_dir

# Adjust the path to the data file relative to the repository root
data_file = os.path.join(repo_root, data_file_name)


# Load point cloud
pcd = o3d.io.read_point_cloud(os.path.join(script_dir, data_file))
points = np.asarray(pcd.points)

# Step 1: Visualize original point cloud if required
if visualize:
    o3d.visualization.draw_geometries([pcd], window_name='Original Point Cloud')

# Step 2: Principal Component Analysis (PCA) for projection
mean_center = points - np.mean(points, axis=0)
cov_matrix = np.cov(mean_center, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
projected_points = mean_center @ eigenvectors[:, :2]  # Project onto first two principal components

# Step 3: Generate a grid over projected points
u_min, u_max = np.min(projected_points[:, 0]), np.max(projected_points[:, 0])
v_min, v_max = np.min(projected_points[:, 1]), np.max(projected_points[:, 1])
u_lin = np.linspace(u_min, u_max, grid_size)
v_lin = np.linspace(v_min, v_max, grid_size)
u_grid, v_grid = np.meshgrid(u_lin, v_lin)
grid_points_2d = np.column_stack((u_grid.flatten(), v_grid.flatten()))

# Step 4: Visualize the grid if required
if visualize:
    plt.figure(figsize=(8, 6))
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=1, label='Projected Points')
    plt.scatter(grid_points_2d[:, 0], grid_points_2d[:, 1], c='red', s=5, label='Grid Points')
    plt.legend()
    plt.title('2D Grid Over Projected Points')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(os.path.join(output_dir, 'grid_visualization.png'))
    plt.show()

# Step 5: Use KDTree for neighbor search
tree = cKDTree(projected_points)
num_neighbors = 100  # Number of neighbors to average
search_radius = max(u_max - u_min, v_max - v_min) / (grid_size * 2)

# Step 6: Initialize control points and perform the search
control_points = []
offset_value = offset_value * np.std(points[:, 2])  # Apply offset scaling

for u_coord, v_coord in grid_points_2d:
    distances, indices = tree.query([u_coord, v_coord], k=num_neighbors)
    within_radius = distances <= search_radius
    valid_indices = indices[within_radius]

    if len(valid_indices) > 0:
        nearby_points = points[valid_indices]
        avg_point = np.mean(nearby_points, axis=0)
    else:
        avg_point = np.array([u_coord, v_coord, offset_value])

    control_points.append(avg_point)

control_points = np.array(control_points)

# Step 7: Save control points as point cloud
control_pcd = o3d.geometry.PointCloud()
control_pcd.points = o3d.utility.Vector3dVector(control_points)
o3d.io.write_point_cloud(os.path.join(output_dir, 'control_points.ply'), control_pcd)

# Step 8: Final visualization of control points if required
if visualize:
    o3d.visualization.draw_geometries([control_pcd], window_name='Control Points Cloud')
