import json
import numpy as np
import open3d as o3d

# Path to the transforms.json file and output point cloud file
transforms_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/transforms.json'
aabb_point_cloud_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/aabb_point_cloud.ply'  # Adjust the path to save the AABB point cloud

# Load the transforms.json file
with open(transforms_file, 'r') as f:
    data = json.load(f)

# Extract every 15th camera pose
camera_poses = []
for i, frame in enumerate(data['frames']):
    if i % 15 == 0:
        transform_matrix = np.array(frame['transform_matrix'])
        camera_poses.append(transform_matrix[:3, 3])  # Extract the translation component

# Convert to numpy array for easier handling
camera_poses = np.array(camera_poses)

# Calculate the axis-aligned bounding box (AABB)
min_corner = np.min(camera_poses, axis=0)
max_corner = np.max(camera_poses, axis=0)

# Define the corners of the AABB
corners = np.array([[min_corner[0], min_corner[1], min_corner[2]],
                    [min_corner[0], min_corner[1], max_corner[2]],
                    [min_corner[0], max_corner[1], min_corner[2]],
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [max_corner[0], min_corner[1], min_corner[2]],
                    [max_corner[0], min_corner[1], max_corner[2]],
                    [max_corner[0], max_corner[1], min_corner[2]],
                    [max_corner[0], max_corner[1], max_corner[2]]])

# Create an Open3D point cloud object for the AABB
aabb_pcd = o3d.geometry.PointCloud()
aabb_pcd.points = o3d.utility.Vector3dVector(corners)

# Save the AABB point cloud to a file
o3d.io.write_point_cloud(aabb_point_cloud_file, aabb_pcd)

print(f"AABB point cloud saved to {aabb_point_cloud_file}")
