import open3d as o3d
import numpy as np
import os
from segmentation.segmentation_p import segment_plant_and_ground
from graphs.plane_fitting import align_normal_with_z_axis

def load_pointcloud_relative(relative_path):
    """
    Load a point cloud from a relative file path, going up two directories to the root directory.
    
    Args:
        relative_path (str): The relative path to the point cloud file from the root directory.
    
    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    # Get the current file directory (the script's location)
    current_dir = os.getcwd()    
    # Navigate up one directories to reach the root directory of the repository
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Join the root directory with the provided relative path
    point_cloud_path = os.path.join(root_dir, relative_path)
    
    print(f"Loading point cloud from: {point_cloud_path}")


    # Load the point cloud using Open3D
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    # points = np.asarray(pcd.points)
    
    return pcd

def process_point_clouds_in_folder(folder_path):
    """
    Process all point clouds in a given folder.
    
    Args:
        folder_path (str): The path to the folder containing the point clouds.
    """
    # Get all point cloud files in the folder
    point_cloud_files = [f for f in os.listdir(folder_path) if f.endswith(".ply")]
    
    for file_name in point_cloud_files:
        print(f"Processing file: {file_name}")
        
        # Load the point cloud
        relative_path = os.path.join(folder_path, file_name)
        pcd = load_pointcloud_relative(relative_path)
        
        # Segment the plant and ground
        plant_pcd, ground_pcd = segment_plant_and_ground(pcd)
        
        # Align the normal vector of the fitted plane with the z-axis
        rotation_matrix, centroid = align_normal_with_z_axis(ground_pcd)

        # apply the transformation to the plant_pcd and ground_pcd to check the direction of the normal vector
        plant_pcd = plant_pcd.translate(-centroid)
        plant_pcd = plant_pcd.rotate(rotation_matrix, center=[0, 0, 0])
        ground_pcd = ground_pcd.translate(-centroid)
        ground_pcd = ground_pcd.rotate(rotation_matrix, center=[0, 0, 0])

        plant_centroid = np.mean(np.asarray(plant_pcd.points), axis=0)
        ground_centroid = np.mean(np.asarray(ground_pcd.points), axis=0)

        if plant_centroid[2] > ground_centroid[2]:
            print("The normal vector is pointing upwards")
        else:
            print("The normal vector is pointing downwards")
            rotation_matrix[2, :] = -rotation_matrix[2, :]  # Invert the z-axis row


        # Apply the transformation to the point cloud
        pcd = pcd.translate(-centroid)
        pcd = pcd.rotate(rotation_matrix, center=[0, 0, 0])

        # Visualize the result (optional)
        # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, axes], mesh_show_back_face=True)
        
        # Optionally, save the transformed point cloud
        output_path = os.path.join(folder_path, f"r_{file_name}")
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved processed point cloud to: {output_path}")



if __name__ == "__main__":
    # file_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melon_dataset/scaled/manual_cleaned/r_s_pc_A-2_2024-08-05_dense_02 - Cloud.ply"
    # pcd = load_pointcloud_relative(file_path)
    # # o3d.visualization.draw_geometries([pcd])

    # # Segment the plant and ground
    # plant_pcd, ground_pcd = segment_plant_and_ground(pcd)

    # # Visualize the ground point clouds
    # o3d.visualization.draw_geometries([plant_pcd])

    # # Align the normal vector of the fitted plane with the z-axis
    # rotation_matrix, centroid = align_normal_with_z_axis(plant_pcd)

    # # Apply the transformation to the point cloud
    # pcd = pcd.translate(-centroid)

    # pcd = pcd.rotate(rotation_matrix, center=[0, 0, 0])

    # # Visualize the result with axis visible
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axes], mesh_show_back_face=True)

    # # Optionally, save the transformed point cloud
    # o3d.io.write_point_cloud(file_path, pcd)

    # Process all point clouds in a folder
    folder_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melon_dataset/scaled/scaled"
    process_point_clouds_in_folder(folder_path)

