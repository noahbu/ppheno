from segmentation.segmentation_p import estimate_voxel_size_sklearn
from segmentation.segmentation_p import segment_plant_and_ground
from scipy.spatial import ConvexHull, distance_matrix
import open3d as o3d
import numpy as np
import os
import csv
import pandas as pd

def estimate_volume(pcd):
    """
    Estimate the volume of a point cloud by voxelizing it and counting the number of voxels.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to estimate the volume of.
    
    Returns:
        float: The estimated volume of the point cloud.
    """
    # Estimate the voxel size
    voxel_size = estimate_voxel_size_sklearn(pcd, sample_ratio=1, num_neighbors=10)
    
    if not isinstance(voxel_size, float):
        raise ValueError("Voxel size must be a float.")
    
    # Voxelization
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    # o3d.visualization.draw_geometries([voxel_grid])
    
    # Count the number of voxels
    num_voxels = len(voxel_grid.get_voxels())
    
    # Calculate the volume
    volume = num_voxels * voxel_size**3
    
    return volume

def filter_z_axis(point_cloud, z_threshold=0.03):
    # Convert the point cloud to a numpy array
    points = np.asarray(point_cloud.points)

    # Filter points where the z-value is greater than or equal to the threshold
    filtered_points = points[points[:, 2] >= z_threshold]

    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Copy over any additional data, such as colors or normals if they exist
    if point_cloud.has_colors():
        filtered_colors = np.asarray(point_cloud.colors)[points[:, 2] >= z_threshold]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    if point_cloud.has_normals():
        filtered_normals = np.asarray(point_cloud.normals)[points[:, 2] >= z_threshold]
        filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    
    return filtered_pcd

def process_pointclouds_in_folder(folder_path):
    # List all .ply files in the folder
    pointcloud_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]

    # Check if there are any point clouds to process
    if not pointcloud_files:
        print("No point cloud files found in the folder.")
        return
    
    # Alphabetically sort the files
    pointcloud_files.sort()

    # Initialize lists to store the data
    codes = []
    dates = []
    volumes = []
    heights = []
    max_widths = []


    # Process each point cloud file
    for file_name in pointcloud_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            # Load the point cloud
            pcd = o3d.io.read_point_cloud(file_path)

            # Filter the point cloud
            pcd = filter_z_axis(pcd, z_threshold=0.03)

            # Downsample the point cloud with voxel downsampling
            voxel_size = 0.005  # Adjust as needed
            pcd = pcd.voxel_down_sample(voxel_size)

            # Estimate the volume
            volume = estimate_volume(pcd)

            # Calculate max Z-dimension (height)
            points = np.asarray(pcd.points)  # Convert to NumPy array
            z_values = points[:, 2]  # Extract Z values
            height = np.max(z_values)

            # Calculate max width in x-y plane
            # Project points onto x-y plane
            points_xy = points[:, :2]  # Take x and y columns only (ignore z)

            # Compute convex hull
            hull = ConvexHull(points_xy)

            # Get the hull vertices
            hull_points = points_xy[hull.vertices]

            # Compute pairwise distances between hull vertices
            pairwise_distances = distance_matrix(hull_points, hull_points)

            # Find the maximum distance (max width)
            max_width = np.max(pairwise_distances)


            # Extract code and date from the filename
            parts = file_name.split("_")
            code = parts[3]  # "C-5"
            date = parts[4]  # "2024-08-04"

            # Append the data to the lists
            codes.append(code)
            dates.append(date)
            volumes.append(volume)
            heights.append(height)
            max_widths.append(max_width)


            print(f"{file_name}: {volume:.7f} m^3")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # Create a DataFrame from the collected data
    df = pd.DataFrame({
        'code': codes,
        'date': dates,
        'volume': volumes,
        'height': heights, 
        'max_width': max_widths
    })

    return df


#######################################
def process_pointclouds_no_rotation(folder_path):
    # List all .ply files in the folder
    pointcloud_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]

    # Check if there are any point clouds to process
    if not pointcloud_files:
        print("No point cloud files found in the folder.")
        return
    
    # alphabetically sort the files
    pointcloud_files.sort()

     # Initialize a dictionary to store the volumes
    volumes = {}

    # Process each point cloud file
    for file_name in pointcloud_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            # Load the point cloud
            pcd = o3d.io.read_point_cloud(file_path)

            # Filter the point cloud
            plant_pcd, ground_pcd = segment_plant_and_ground(pcd, 5)

            # Downsample the point cloud with voxel downsampling
            voxel_size = 0.01  # Adjust as needed
            plant_pcd = plant_pcd.voxel_down_sample(voxel_size)

            # o3d.visualization.draw_geometries([plant_pcd])

            # Estimate the volume
            volume = estimate_volume(plant_pcd)

            # Store the volume in the dictionary (use the date part of the filename as the key)
            date = file_name.split("_")[4]  # Assuming the date is the third part in the filename
            volumes[date] = volume

            print(f"{file_name}: {volume:.7f} m^3")
        
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return volumes

def write_volumes_to_csv(volumes, file_path):
    """
    Write the volumes to a CSV file.

    Args:
        volumes (dict): Dictionary where keys are dates (or identifiers) and values are volume estimates.
        file_path (str): The full path where the CSV file should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the data to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Date", "Volume (m^3)"])
        # Write the volume data
        for date, volume in volumes.items():
            writer.writerow([date, f"{volume:.6f}"])

    print(f"Volume estimates saved to {file_path}")

def main():
    # Folder containing the point clouds
    folder_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melon_dataset/scaled/rotated"
    csv_file_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/"

    # file_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/pointcloud_time_series/C-3/rotated/processed_s_pc_C-3_2024-08-07_dense_02.ply"

    # pcd = o3d.io.read_point_cloud(file_path)
    # pcd = filter_z_axis(pcd, z_threshold=0.03)

    # voxel_size = 0.001  # Example value, adjust as needed
    # downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    # # o3d.visualization.draw_geometries([downsampled_pcd])

    # # Estimate the volume
    # volume = estimate_volume(pcd)
    # print(f"Volume: {volume:.6f} m^3")


     # Process all point clouds in the folder and collect the volume estimates
    df_volumes = process_pointclouds_in_folder(folder_path)


        # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(csv_file_path, 'volumes.csv')
    try:
        df_volumes.to_csv(csv_file_path, index=False, float_format='%.7f')
        print(f"\nResults saved to {csv_file_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    # Path to save the CSV file

    # Write the volumes to the CSV file


if __name__ == "__main__":
    main()
