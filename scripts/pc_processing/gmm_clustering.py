# Description: This script demonstrates how to use Gaussian Mixture Model (GMM) clustering
# to segment a point cloud into two clusters based on spatial and color features.
# The script loads a point cloud, preprocesses it by downsampling and removing outliers,
# extracts spatial and color features, and applies GMM clustering to segment the point cloud.

# Use with conda environment: open3d

# colours the ground blue and leaves the plant colour as it is. 
# ################################################################################################################


import open3d as o3d
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from colorsys import rgb_to_hsv

def rgb_to_hue(r, g, b):
    h, s, v = rgb_to_hsv(r, g, b)
    return h * 360  # Convert to degrees

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-08-04/C-2_2024-08-04')
point_cloud_file = data_folder / 'pc_C-2_2024-08-04_dense_02.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Preprocessing: Downsample and remove outliers
pcd = pcd.voxel_down_sample(voxel_size=0.005)
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.select_by_index(ind)

# Extract and normalize spatial features
points = np.asarray(pcd.points)
scaler_points = StandardScaler()
normalized_points = scaler_points.fit_transform(points)

# Convert RGB to Hue
colors = np.asarray(pcd.colors)
hue_values = np.array([rgb_to_hue(r, g, b) for r, g, b in colors]).reshape(-1, 1)

# Normalize hue values
scaler_hue = StandardScaler()
normalized_hue = scaler_hue.fit_transform(hue_values)

# Apply weights to the spatial and hue features
weight_spatial = 0.3  # Adjust based on your needs
weight_hue = 1.0      # Adjust based on your needs

weighted_points = normalized_points * weight_spatial
weighted_hue = normalized_hue * weight_hue

# Combine weighted features
weighted_features = np.hstack((weighted_points, weighted_hue))

# Number of clusters
n_clusters = 5  # Adjust the number of clusters as needed

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(weighted_features)
labels = gmm.predict(weighted_features)

# Determine which cluster has the most green hue points
green_hue_min, green_hue_max = 80, 100  # Define the range for green hue
green_counts = []
for i in range(n_clusters):
    green_count = np.sum((labels == i) & (green_hue_min <= hue_values.flatten()) & (hue_values.flatten() <= green_hue_max))
    green_counts.append(green_count)

# Identify the plant cluster (the one with the most green points)
plant_cluster = np.argmax(green_counts)
# For simplicity, consider the rest as ground clusters
ground_clusters = [i for i in range(n_clusters) if i != plant_cluster]

# Preserve original colors for the plant cluster, recolor the ground clusters
new_colors = np.copy(colors)
for ground_cluster in ground_clusters:
    new_colors[labels == ground_cluster] = [0.0, 0.5, 1.0]  # Apply blue to ground clusters

pcd.colors = o3d.utility.Vector3dVector(new_colors)

# Save the colored point cloud
output_path = data_folder / 'gmm_colored_clusters_preserved_hue_improved.ply'
o3d.io.write_point_cloud(str(output_path), pcd)

# Optionally, save each cluster separately if needed
plant_pcd = pcd.select_by_index(np.where(labels == plant_cluster)[0])
o3d.io.write_point_cloud(str(data_folder / 'plant_cluster_gmm_original_hue_improved.ply'), plant_pcd)

for ground_cluster in ground_clusters:
    ground_pcd = pcd.select_by_index(np.where(labels == ground_cluster)[0])
    o3d.io.write_point_cloud(str(data_folder / f'ground_cluster_{ground_cluster}_gmm_blue_hue_improved.ply'), ground_pcd)
