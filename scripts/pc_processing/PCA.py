import open3d as o3d
import numpy as np
from pathlib import Path

# Get the current script's directory
script_dir = Path(__file__).parent.resolve()

# Construct the path to the root of the GitHub project
project_root = script_dir.parent.parent

# Construct the path to the /data folder
data_folder = project_root / Path('data/melonCycle/2024-08-04/A-4_2024-08-04')

# Paths to the files
input_point_cloud_file = data_folder / 'm_pc_A-4_2024-08-04_dense_02.ply'
output_point_cloud_file = data_folder / 'm_ground.ply'

# Load the point cloud
pcd = o3d.io.read_point_cloud(str(input_point_cloud_file))

# Convert point cloud to numpy array
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
normals = np.asarray(pcd.normals)

# Perform PCA using numpy to find the dominant axis
center = points.mean(axis=0)
centered_points = points - center
covariance_matrix = np.cov(centered_points.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvectors by eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Make the most dominant axis (first eigenvector) align with the z-axis
rotation_matrix = np.eye(3)
rotation_matrix[:, 2] = sorted_eigenvectors[:, 0]  # Dominant axis becomes the z-axis
rotation_matrix[:, 0] = sorted_eigenvectors[:, 1]  # Second axis
rotation_matrix[:, 1] = sorted_eigenvectors[:, 2]  # Third axis

# Transform the point cloud to align the dominant axis with the z-axis
transformed_points = np.dot(centered_points, rotation_matrix)

# Update point cloud with transformed points, original colors, and normals
transformed_pcd = o3d.geometry.PointCloud()
transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
transformed_pcd.normals = o3d.utility.Vector3dVector(normals)

# Save the transformed point cloud including colors and normals
o3d.io.write_point_cloud(str(output_point_cloud_file), transformed_pcd)

# Visualize the transformed point cloud
#o3d.visualization.draw_geometries([transformed_pcd])
