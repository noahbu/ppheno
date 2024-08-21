import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# Set this variable to True if the point cloud contains color information
has_color = False  # Set to True if the point cloud does contain color information

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4/Downsampling')
point_cloud_file = data_folder / '1024_manual_cleaned.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

points = np.asarray(pcd.points)
scaler = StandardScaler()
points_normalized = scaler.fit_transform(points)

# Preprocessing: Downsample and remove outliers (Optional)
# pcd = pcd.voxel_down_sample(voxel_size=0.005)
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# pcd = pcd.select_by_index(ind)

# Extract and normalize features
points = np.asarray(pcd.points)
scaler = StandardScaler()
normalized_points = scaler.fit_transform(points)

if has_color:
    colors = np.asarray(pcd.colors)
    normalized_colors = scaler.fit_transform(colors)
    
    # Apply weights to the spatial and color features
    weight_spatial = 1.0  # Adjust based on your needs
    weight_color = 1.0    # Adjust based on your needs
    
    weighted_points = normalized_points * weight_spatial
    weighted_colors = normalized_colors * weight_color
    
    # Combine weighted features
    weighted_features = np.hstack((weighted_points, weighted_colors))
else:
    # If no color information, only use spatial features
    weighted_features = normalized_points

# k is similar to min_samples, the number of points to form a cluster
# the visualization can be used to choose the best value of eps. It should be at the sharp increase of the graph
# plt.figure(figsize=(10, 6))
# for k in range(20, 51, 2):
#     neighbors = NearestNeighbors(n_neighbors=k)
#     neighbors_fit = neighbors.fit(points_normalized)
#     distances, indices = neighbors_fit.kneighbors(points_normalized)
#     distances = np.sort(distances[:, -1], axis=0)
#     plt.plot(distances, label=f'k={k}')

# plt.xlabel('Points sorted by distance')
# plt.ylabel('Distance to kth nearest neighbor')
# plt.title('K-Distance Graph for Different k Values')
# plt.legend()
# plt.show()


# Apply DBSCAN clustering
eps = 0.5  # Adjust epsilon based on your data
min_samples = 40  # Minimum number of points to form a cluster
dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(weighted_features)
labels = dbscan.labels_

# Get the number of clusters (excluding noise, which is labeled as -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Define a color map for the clusters
# We generate random colors for each cluster
color_map = np.random.rand(n_clusters, 3)

# Apply colors based on the cluster labels
new_colors = np.zeros((points.shape[0], 3))  # Initialize with black

for i in range(n_clusters):
    new_colors[labels == i] = color_map[i % len(color_map)]

# Handle noise (label -1) by coloring it black
new_colors[labels == -1] = [0, 0, 0]

pcd.colors = o3d.utility.Vector3dVector(new_colors)

#o3d.visualization.draw_geometries([pcd])


# Save the colored point cloud
output_path = data_folder / f'dbscan_colored_clusters_{n_clusters}_clusters.ply'
o3d.io.write_point_cloud(str(output_path), pcd)

# Optionally, save each cluster separately if needed
# for i in range(n_clusters):
#     cluster_pcd = pcd.select_by_index(np.where(labels == i)[0])
#     o3d.io.write_point_cloud(str(data_folder / f'cluster_{i+1}_dbscan.ply'), cluster_pcd)

print(f"Clustered point cloud saved to {output_path}")
