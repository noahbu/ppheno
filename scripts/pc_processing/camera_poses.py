import json
import numpy as np

# Path to the transforms.json file and output point cloud file
transforms_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-06/A-4_2024-08-06/edits/transforms.json'
output_point_cloud_file = '/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-06/A-4_2024-08-06/edits/camera_positions_and_normals_with_lines.ply'

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

# Define the rotation matrix
rotation_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0]
])

# Apply the rotation to the camera poses and normals
rotated_camera_poses = np.dot(camera_poses, rotation_matrix.T)
rotated_normals = np.dot(camera_normals, rotation_matrix.T)

# Set the length of the normal vector line
normal_length = 1  # You can adjust this value to change the length of the normals
num_points_per_normal = 800  # Number of points along each normal line

# Prepare points for PLY (including the normal lines with evenly spaced points)
points = []
normals = []

for pose, normal in zip(rotated_camera_poses, rotated_normals):
    # Start point is the camera position
    for i in range(num_points_per_normal):
        # Calculate the interpolation factor (0 to 1) for evenly spaced points along the normal
        t = i / (num_points_per_normal - 1)
        point_on_normal = pose + t * normal_length * normal
        points.append(point_on_normal)
        # The normal is the same for all points along this line
        normals.append(normal)

# Convert lists to numpy arrays
points = np.array(points)
normals = np.array(normals)

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
'''.format(vertex_count=len(points))

# Combine positions and normals
points_and_normals = np.hstack((points, normals))

# Write camera positions and normals (with lines for normals) to PLY file
with open(output_point_cloud_file, 'w') as f:
    f.write(ply_header)
    np.savetxt(f, points_and_normals, fmt='%f %f %f %f %f %f')

print(f"Camera positions and normals (with evenly spaced points) saved to {output_point_cloud_file}")
