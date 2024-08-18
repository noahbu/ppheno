import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load the point cloud
# Get the current script's directory
script_dir = Path(__file__).parent.resolve()

# Construct the path to the root of the GitHub project
project_root = script_dir.parent.parent

# Construct the path to the /data folder
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4')

# Paths to the files
point_cloud_file = data_folder / 'plant_cluster_gmm_original.ply'

# Load your cleaned point cloud
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Extract the colors (Nx3 array of RGB values)
colors = np.asarray(pcd.colors)

# Extract the spatial coordinates (Nx3 array of x, y, z values)
points = np.asarray(pcd.points)

# Normalize the features
scaler = StandardScaler()
normalized_points = scaler.fit_transform(points)
normalized_colors = scaler.fit_transform(colors)

# Apply weights to the spatial and color features
weight_spatial = 1.0  # Adjust this weight as needed
weight_color = 0.0    # Adjust this weight as needed

weighted_points = normalized_points * weight_spatial
weighted_colors = normalized_colors * weight_color

# Combine weighted features
weighted_features = np.hstack((weighted_points, weighted_colors))

# Define the number of clusters
n_clusters = 2  # Adjust the number of clusters as needed

# Apply KMeans clustering on the weighted features
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(weighted_features)
labels = kmeans.labels_

# Define custom colors for the clusters (expand if more clusters are needed)
cluster_colors = np.array([[1.0, 0.5, 0.0],  # Orange
                           [0.0, 0.5, 1.0],  # Blue
                           [0.0, 1.0, 0.0],  # Green
                           [1.0, 0.0, 0.0],  # Red
                           [1.0, 1.0, 0.0]])  # Yellow

# Assign custom colors based on the cluster labels (handle more clusters if needed)
new_colors = cluster_colors[labels % len(cluster_colors)]
pcd.colors = o3d.utility.Vector3dVector(new_colors)

# Save the clustered point cloud for visualization
output_path = data_folder / f'clustered_point_cloud_weighted_{n_clusters}_clusters.ply'
o3d.io.write_point_cloud(str(output_path), pcd)

# Optionally, save each cluster separately
for i in range(n_clusters):
    cluster = pcd.select_by_index(np.where(labels == i)[0])
    o3d.io.write_point_cloud(str(data_folder / f'cluster_{i+1}_weighted.ply'), cluster)
