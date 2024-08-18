# ################################################################################################################
# Can be used to remove some remaining ground from the point cloud. 
# removes almost all styrofoam from the ground, small middle strip stays
# adding spatial infromation to the clustering did not improve the results
# ################################################################################################################


import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

# Load the point cloud
# Get the current script's directory
script_dir = Path(__file__).parent.resolve()

# Construct the path to the root of the GitHub project
project_root = script_dir.parent.parent

# Construct the path to the /data folder
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4')

# Paths to the files
point_cloud_file = data_folder / 'filtered_point_cloud2.ply'

# Load your cleaned point cloud
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Extract the colors (Nx3 array of RGB values)
colors = np.asarray(pcd.colors)

# Assuming `colors` is a (N, 3) array of RGB colors
kmeans = KMeans(n_clusters=2, random_state=42).fit(colors)
labels = kmeans.labels_

# Define custom colors for the clusters (Orange and Blue)
cluster_colors = np.array([[1.0, 0.5, 0.0],  # Orange
                           [0.0, 0.5, 1.0]])  # Blue

# Assign custom colors based on the cluster labels
new_colors = cluster_colors[labels]
pcd.colors = o3d.utility.Vector3dVector(new_colors)


# Save the clustered point cloud for visualization
output_path = data_folder / 'clustered_point_cloud.ply'
o3d.io.write_point_cloud(str(output_path), pcd)

# You can also separate the clusters
cluster_1 = pcd.select_by_index(np.where(labels == 0)[0])
cluster_2 = pcd.select_by_index(np.where(labels == 1)[0])

# Save each cluster separately if needed
o3d.io.write_point_cloud(str(data_folder / 'cluster_1.ply'), cluster_1)
o3d.io.write_point_cloud(str(data_folder / 'cluster_2.ply'), cluster_2)
