import open3d as o3d

# Load your point cloud
pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-03/A-3_2024-08-03/pc_A-3_2024-08-03_dense_02.ply")


# Apply uniform downsampling
downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)  # Adjust voxel size as needed

# Apply RANSAC to detect a plane in the downsampled point cloud
plane_model, inliers = downsampled_pcd.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)

# Extract the plane model parameters
[a, b, c, d] = plane_model
print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

# Visualize the inliers (points fitting the plane)
inlier_cloud = downsampled_pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Color the plane red for visualization

# Visualize the outliers (points not fitting the plane)
outlier_cloud = downsampled_pcd.select_by_index(inliers, invert=True)

# Save the inliers and outliers to PLY files for visualization in CloudCompare
inlier_output_path = "inliers_plane.ply"
outlier_output_path = "outliers_plane.ply"

# Save the point clouds
o3d.io.write_point_cloud(inlier_output_path, inlier_cloud)
o3d.io.write_point_cloud(outlier_output_path, outlier_cloud)

print(f"Inliers saved to {inlier_output_path}")
print(f"Outliers saved to {outlier_output_path}")

