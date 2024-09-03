import open3d as o3d
import numpy as np
from pathlib import Path

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4')
point_cloud_file = data_folder / 'cleaned_plant_above_ground_manual.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Step 1: Initial Downsampling to Reduce Point Cloud Size
initial_voxel_size = 0.02  # Increased voxel size for more aggressive downsampling
pcd = pcd.voxel_down_sample(voxel_size=initial_voxel_size)

# Step 2: Estimate Normals with a Smaller Search Radius
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Step 3: Approximate Curvature with Simplified Calculation
kdtree = o3d.geometry.KDTreeFlann(pcd)
normals = np.asarray(pcd.normals)
curvature = np.zeros(len(normals))

# Calculate curvature by comparing each normal to the average normal of its neighbors
for i in range(len(normals)):
    [_, idx, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius=0.05)
    if len(idx) > 1:
        neighbor_normals = normals[idx[1:]]  # Exclude the point itself
        mean_normal = np.mean(neighbor_normals, axis=0)
        curvature[i] = np.linalg.norm(normals[i] - mean_normal)

# Step 4: Identify Edge Points
threshold = np.percentile(curvature, 95)  # Keep top 5% points with highest curvature
edge_indices = np.where(curvature > threshold)[0]
edge_pcd = pcd.select_by_index(edge_indices)

# Step 5: Downsample the Non-Edge Points Aggressively
non_edge_pcd = pcd.select_by_index(edge_indices, invert=True)
target_non_edge_points = 4500
voxel_size = (np.max(non_edge_pcd.get_max_bound() - non_edge_pcd.get_min_bound()) /
              np.cbrt(len(non_edge_pcd.points) / target_non_edge_points))
downsampled_non_edge_pcd = non_edge_pcd.voxel_down_sample(voxel_size=voxel_size)

# Step 6: Combine Edge and Downsampled Non-Edge Points
combined_pcd = edge_pcd + downsampled_non_edge_pcd

# Save the downsampled point cloud
output_path = data_folder / 'edge_preserving_downsampled_point_cloud.ply'
o3d.io.write_point_cloud(str(output_path), combined_pcd)

print(f"Original point cloud size: {len(pcd.points)}")
print(f"Edge points kept: {len(edge_pcd.points)}")
print(f"Downsampled non-edge points: {len(downsampled_non_edge_pcd.points)}")
print(f"Final downsampled point cloud size: {len(combined_pcd.points)}")
print(f"Downsampled point cloud saved to {output_path}")
