############################################################################################################
# Description: This script demonstrates how to use DBSCAN clustering to detect the region around the camera poses and extracts it from the point cloud.
# The script loads the camera poses from a JSON file, rotates them, and combines them with the point cloud data.
# It then applies DBSCAN clustering to identify the cluster containing the majority of camera poses.
# Finally, it filters out the points in that cluster and saves the filtered point cloud.
# Use with conda environment: open3d
################################################################################################################


import open3d as o3d
import json
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt

# Paths to the files
transforms_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/B-4/transforms.json'
point_cloud_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/B-4/point_cloud.ply'
output_filtered_point_cloud_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/B-4/filtered_point_cloud2.ply'

# Load the camera poses from the JSON file
with open(transforms_file, 'r') as f:
    data = json.load(f)

camera_poses = []
for frame in data['frames']:
    transform_matrix = np.array(frame['transform_matrix'])
    camera_poses.append(transform_matrix[:3, 3])  # Extract the translation component

camera_poses = np.array(camera_poses)

# Define the rotation matrix
rotation_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0]
])

# Apply the rotation to the camera poses
rotated_camera_poses = np.dot(camera_poses, rotation_matrix.T)

# Load the point cloud
pcd = o3d.io.read_point_cloud(point_cloud_file)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Combine points with camera poses
combined_points = np.vstack((points, rotated_camera_poses))

# Apply DBSCAN clustering
eps = 0.05  # Adjust epsilon based on your data
min_samples = 10  # Minimum number of points to form a cluster
db = DBSCAN(eps=eps, min_samples=min_samples).fit(combined_points)

# Get cluster labels
labels = db.labels_
unique_labels = set(labels)

# Identify the cluster label that contains the majority of camera poses
camera_pose_labels = labels[-len(rotated_camera_poses):]
most_common_label = Counter(camera_pose_labels).most_common(1)[0][0]

# Filter points that are in the same cluster as the majority of camera poses
filtered_indices = np.where(labels[:len(points)] != most_common_label)[0]  # not too sure, but I think the most common label is the one with camera poses, however it it not the one containing the plant
filtered_points = points[filtered_indices]
filtered_colors = colors[filtered_indices]

# Create a new point cloud with filtered points and colors
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Save and visualize the filtered point cloud
o3d.io.write_point_cloud(output_filtered_point_cloud_file, filtered_pcd)
#o3d.visualization.draw_geometries([filtered_pcd])

print(f"Filtered point cloud saved to {output_filtered_point_cloud_file}")

# # Optional: Visualize all clusters
# colors_map = plt.get_cmap("tab20")(np.linspace(0, 1, len(unique_labels)))
# cluster_colors = np.zeros((len(points), 3))
# for k in unique_labels:
#     if k == -1:
#         color = [0, 0, 0]  # Black for noise
#     else:
#         color = colors_map[k % len(colors_map)][:3]
#     cluster_colors[labels[:len(points)] == k] = color

# Create a new point cloud with cluster colors
# clustered_pcd = o3d.geometry.PointCloud()
# clustered_pcd.points = o3d.utility.Vector3dVector(points)
# clustered_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

# # Visualize the clustered point cloud
# o3d.visualization.draw_geometries([clustered_pcd])
