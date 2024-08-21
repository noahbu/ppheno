# get a segmentation fault. No idea how to solve it. 

import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4')
point_cloud_file = data_folder / 'cleaned_plant_above_ground_manual.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Compute normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Simplified Region Growing Implementation
def simple_region_growing(pcd, search_radius=0.05, smoothness=30.0):
    labels = np.full(len(pcd.points), -1, dtype=int)  # Initialize labels to -1 (unlabeled)
    current_label = 0
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for i in range(len(pcd.points)):
        if labels[i] != -1:
            continue  # Skip already labeled points

        # Start a new region
        labels[i] = current_label
        queue = [i]

        while queue:
            point_idx = queue.pop(0)
            [_, idx, _] = kdtree.search_radius_vector_3d(pcd.points[point_idx], search_radius)

            for j in idx:
                if labels[j] == -1:
                    angle = np.arccos(np.clip(np.dot(pcd.normals[point_idx], pcd.normals[j]), -1.0, 1.0))
                    if np.degrees(angle) < smoothness:
                        labels[j] = current_label
                        queue.append(j)

        current_label += 1  # Move to the next label for a new region

    return labels

# Apply the simplified region growing
labels = simple_region_growing(pcd, search_radius=0.05, smoothness=30.0)

# Colorize the point cloud by region
max_label = labels.max() + 1  # Number of clusters
colors = plt.get_cmap("tab20")(labels / max_label)
colors[labels == -1] = 0  # Unlabeled points are colored black

pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Save the segmented point cloud
output_path = data_folder / f'simple_region_growing_segmentation.ply'
o3d.io.write_point_cloud(str(output_path), pcd)

# Visualize the segmented point cloud
#o3d.visualization.draw_geometries([pcd])

print(f"Segmented point cloud saved to {output_path}")
