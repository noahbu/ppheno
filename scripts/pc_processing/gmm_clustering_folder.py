import open3d as o3d
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-07-30_2')




# Define the base directory containing your folders
base_dir = data_folder  # Replace with the correct path

# Loop through each subdirectory and process the point cloud
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith("_02.ply"):  # Adjust to match the file names you want to process
            # Construct paths
            point_cloud_file = os.path.join(root, file)
            #output_path = os.path.join(root, file.replace("_02.ply", "_03.ply"))
            
            output_dir = project_root / Path('data/melonCycle/2024-07-30_2/gmm_clustered')
            output_path = output_dir / file.replace("_02.ply", "_03.ply")

            # Load the point cloud
            pcd = o3d.io.read_point_cloud(str(point_cloud_file))

            # Preprocessing: Downsample and remove outliers
            pcd = pcd.voxel_down_sample(voxel_size=0.005)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)

            # Extract and normalize features
            colors = np.asarray(pcd.colors)
            points = np.asarray(pcd.points)
            scaler = StandardScaler()
            normalized_points = scaler.fit_transform(points)
            normalized_colors = scaler.fit_transform(colors)

            # Apply weights to the spatial and color features
            weight_spatial = 1.0  # Adjust based on your needs
            weight_color = 0.0    # Adjust based on your needs

            weighted_points = normalized_points * weight_spatial
            weighted_colors = normalized_colors * weight_color

            # Combine weighted features
            weighted_features = np.hstack((weighted_points, weighted_colors))

            # Apply Gaussian Mixture Model clustering
            gmm = GaussianMixture(n_components=2, random_state=42).fit(weighted_features)
            labels = gmm.predict(weighted_features)

            # Define colors for the clusters (Orange and Blue)
            cluster_colors = np.array([[1.0, 0.5, 0.0],  # Orange
                                       [0.0, 0.5, 1.0]])  # Blue

            # Preserve original colors for one cluster, and recolor the other
            new_colors = np.copy(colors)  # Start with the original colors
            new_colors[labels == 1] = cluster_colors[1]  # Apply blue to cluster 1

            pcd.colors = o3d.utility.Vector3dVector(new_colors)

            # Save the colored point cloud
            o3d.io.write_point_cloud(str(output_path), pcd)

            # Optionally, save each cluster separately if needed
            # plant_pcd = pcd.select_by_index(np.where(labels == 0)[0])
            # ground_pcd = pcd.select_by_index(np.where(labels == 1)[0])

            # o3d.io.write_point_cloud(str(root) + '/plant_cluster_gmm_original.ply', plant_pcd)
            # o3d.io.write_point_cloud(str(root) + '/ground_cluster_gmm_blue.ply', ground_pcd)

            # print(f"Processed {file} and saved the results.")

print("Processing complete.")
