import json
import numpy as np

# Path to the transforms.json file and output point cloud file
transforms_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/transforms.json'
output_point_cloud_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/camera_positions_and_normals.ply'

# Load the transforms.json file
with open(transforms_file, 'r') as f:
    data = json.load(f)

# Extract camera poses and normals
camera_poses = []
camera_normals = []
for frame in data['frames']:
    transform_matrix = np.array(frame['transform_matrix'])
    camera_poses.append(transform_matrix[:3, 3])  # Extract the translation component
    # Extract the normal vector (assuming the third column of the rotation matrix is the forward direction)
    normal = transform_matrix[:3, 2]
    camera_normals.append(normal)

# Convert to numpy arrays for easier handling
camera_poses = np.array(camera_poses)
camera_normals = np.array(camera_normals)

# Create PLY file header
ply_header = '''ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property float nx
property float ny
property float nz
end_header
'''.format(vertex_count=len(camera_poses))

# Combine positions and normals
points_and_normals = np.hstack((camera_poses, camera_normals))

# Write camera positions and normals to PLY file
with open(output_point_cloud_file, 'w') as f:
    f.write(ply_header)
    np.savetxt(f, points_and_normals, fmt='%f %f %f %f %f %f')

print(f"Camera positions and normals point cloud saved to {output_point_cloud_file}")
