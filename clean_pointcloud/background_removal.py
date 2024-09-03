import open3d as o3d
import json
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from pathlib import Path
import os

# Load the point cloud directory
script_dir = Path(__file__).parent.resolve()
project_root = Path('/home/ubuntu/ppheno')


# Define the directories containing the point clouds and transforms
point_cloud_dir = project_root / Path('data/MuskMelon_C/2024-08-01')
transforms_dir = Path('/home/ubuntu/data/custom/MuskMelon_C/2024-08-01')
print(f"Point Cloud Directory: {point_cloud_dir}")
print(f"Transforms Directory: {transforms_dir}")


# Loop through each subdirectory and process the point cloud
for root, dirs, files in os.walk(point_cloud_dir):
    for file in files:
        if file.endswith("_dense_01.ply"):
            # Extract the folder name and construct paths
            folder_name = os.path.basename(root)
            point_cloud_file = os.path.join(root, file)
            
            # Find the corresponding transforms.json in the other directory structure
            transforms_file = os.path.join(transforms_dir, folder_name, 'transforms.json')
            
            if not os.path.exists(transforms_file):
                print(f"transforms.json not found for {folder_name}. Skipping...")
                continue
            
            output_filtered_point_cloud_file = os.path.join(root, file.replace("_01.ply", "_02.ply"))

            print(f"Processing: {point_cloud_file}")
            print(f"Using transforms: {transforms_file}")
            print(f"Output will be saved to: {output_filtered_point_cloud_file}")

            # Load the camera poses from the JSON file
            with open(transforms_file, 'r') as f:
                data = json.load(f)

            camera_poses = []
            for frame in data['frames']:
                transform_matrix = np.array(frame['transform_matrix'])
                camera_poses.append(transform_matrix[:3, 3])  # Extract the translation component

            camera_poses = np.array(camera_poses)

            # Define the rotation matrix
            rotation_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0]
            ])

            # Apply the rotation to the camera poses
            rotated_camera_poses = np.dot(camera_poses, rotation_matrix.T)

            # Load the point cloud
            pcd = o3d.io.read_point_cloud(point_cloud_file)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            # Combine points with camera poses
            combined_points = np.vstack((points, rotated_camera_poses))

            # Apply DBSCAN clustering
            eps = 0.05  # Adjust epsilon based on your data
            min_samples = 10  # Minimum number of points to form a cluster
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(combined_points)

            # Get cluster labels
            labels = db.labels_
            unique_labels = set(labels)

            # Identify the cluster label that contains the majority of camera poses
            camera_pose_labels = labels[-len(rotated_camera_poses):]
            most_common_label = Counter(camera_pose_labels).most_common(1)[0][0]

            # Filter points that are in the same cluster as the majority of camera poses
            filtered_indices = np.where(labels[:len(points)] != most_common_label)[0]
            filtered_points = points[filtered_indices]
            filtered_colors = colors[filtered_indices]

            # Create a new point cloud with filtered points and colors
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

            # Save the filtered point cloud
            o3d.io.write_point_cloud(output_filtered_point_cloud_file, filtered_pcd)

            print(f"Filtered point cloud saved to {output_filtered_point_cloud_file}")
