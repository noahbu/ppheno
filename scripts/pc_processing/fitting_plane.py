import open3d as o3d
import numpy as np
from pathlib import Path

# Load the point clouds for the ground and plant clusters
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4')
pcd_g = data_folder / 'ground_cluster_gmm_blue.ply'
pcd_p = data_folder / 'plant_cluster_gmm_original.ply'

ground_pcd = o3d.io.read_point_cloud(str(pcd_g))
plant_pcd = o3d.io.read_point_cloud(str(pcd_p))

# Fit a plane to the ground cluster
plane_model, inliers = ground_pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
[a, b, c, d] = plane_model

# Calculate the distance of each point in the plant cluster to the plane
points = np.asarray(plant_pcd.points)
distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

# Remove points below the plane (i.e., where distance is negative)
above_plane_indices = np.where(distances < 0)[0]
cleaned_plant_pcd = plant_pcd.select_by_index(above_plane_indices)

# Visualize the normal vector as a line with ~1000 points
plane_normal = np.array([a, b, c])
plane_center = -d * plane_normal / np.linalg.norm(plane_normal)**2

# Define the start and end points of the normal vector line
line_start = plane_center - plane_normal * 2.0  # Adjust length for visibility
line_end = plane_center + plane_normal * 1.0  # Adjust length for visibility

# Create a dense line by interpolating between the start and end points
num_points_on_line = 1000
line_points = np.linspace(line_start, line_end, num_points_on_line)
line_colors = np.tile([1.0, 0.0, 0.0], (num_points_on_line, 1))  # Red color for the line

# Create a point cloud for the normal vector line
line_pcd = o3d.geometry.PointCloud()
line_pcd.points = o3d.utility.Vector3dVector(line_points)
line_pcd.colors = o3d.utility.Vector3dVector(line_colors)

# Combine the ground, plant, and normal vector line into a single point cloud
combined_pcd = ground_pcd + cleaned_plant_pcd + line_pcd

# Save the combined point cloud
output_path = data_folder / 'combined_ground_plant_with_normal_vector.ply'
o3d.io.write_point_cloud(str(output_path), combined_pcd)

# Optionally, visualize the result (if needed)
# o3d.visualization.draw_geometries([combined_pcd])
