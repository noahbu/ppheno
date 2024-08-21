import open3d as o3d
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Set this variable to True if the point cloud contains color information
has_color = False  # Set to True if the point cloud contains color information

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4/Downsampling')
point_cloud_file = data_folder / '1024_manual_cleaned.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

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
    weight_spatial = 5.0  # Adjust based on your needs
    weight_color = 0.0    # Adjust based on your needs
    
    weighted_points = normalized_points * weight_spatial
    weighted_colors = normalized_colors * weight_color
    
    # Combine weighted features
    weighted_features = np.hstack((weighted_points, weighted_colors))
else:
    # If no color information, only use spatial features
    weighted_features = normalized_points

# Define the number of clusters (Adjust this value as needed)
n_clusters = 4  # Change this to the desired number of clusters

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(weighted_features)
labels = gmm.predict(weighted_features)

# Define a color map for the clusters (You can expand this for more clusters)
color_map = np.array([
    [1.0, 0.5, 0.0],  # Orange
    [0.0, 0.5, 1.0],  # Blue
    [0.0, 1.0, 0.0],  # Green
    [1.0, 0.0, 1.0],  # Magenta
    [1.0, 1.0, 0.0],  # Yellow
])

# Ensure the color map has enough colors for the number of clusters
if n_clusters > len(color_map):
    color_map = np.vstack([color_map, np.random.rand(n_clusters - len(color_map), 3)])

# Apply colors based on the cluster labels
new_colors = np.zeros((points.shape[0], 3))  # Initialize with black or zero color

for i in range(n_clusters):
    new_colors[labels == i] = color_map[i % len(color_map)]

# If there is no initial color, we still color the clusters differently
pcd.colors = o3d.utility.Vector3dVector(new_colors)

# Save the colored point cloud
output_path = data_folder / f'gmm_colored_clusters_{n_clusters}_clusters.ply'
o3d.io.write_point_cloud(str(output_path), pcd)

# Optionally, save each cluster separately if needed
for i in range(n_clusters):
    cluster_pcd = pcd.select_by_index(np.where(labels == i)[0])
    o3d.io.write_point_cloud(str(data_folder / f'cluster_{i+1}_gmm.ply'), cluster_pcd)

print(f"Clustered point cloud saved to {output_path}")
