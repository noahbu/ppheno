import open3d as o3d
import json
import numpy as np

# Path to the transforms.json file and point cloud file
transforms_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/transforms.json'
point_cloud_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/point_cloud.ply'
output_filtered_point_cloud_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/filtered_point_cloud.ply'

# Load the transforms.json file
with open(transforms_file, 'r') as f:
    data = json.load(f)

# Extract camera poses
camera_poses = []
for frame in data['frames']:
    transform_matrix = np.array(frame['transform_matrix'])
    camera_poses.append(transform_matrix[:3, 3])  # Extract the translation component

# Convert to numpy arrays for easier handling
camera_poses = np.array(camera_poses)

# Load the point cloud
pcd = o3d.io.read_point_cloud(point_cloud_file)
points = np.asarray(pcd.points)

# Define the maximum allowable distance from the camera pose
max_distance = 3.0  # Adjust this threshold as needed

# Compute distances from each point to the nearest camera pose
distances = np.min(np.linalg.norm(points[:, np.newaxis] - camera_poses, axis=2), axis=1)

# Filter points based on the distance threshold
filtered_points = points[distances <= max_distance]
filtered_colors = np.asarray(pcd.colors)[distances <= max_distance]

# Create a new point cloud with filtered points and colors
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Save or visualize the new filtered point cloud
o3d.io.write_point_cloud(output_filtered_point_cloud_file, filtered_pcd)
o3d.visualization.draw_geometries([filtered_pcd])

print(f"Filtered point cloud saved to {output_filtered_point_cloud_file}")
