############################################################################################################
# This script removes points from a downsampled point cloud that are too far from the original point cloud.
# It also adds random points from the original point cloud to the downsampled point cloud to keep its original size.
# It saves the filtered point cloud to a new .pcd file.
# It also centers and normalizes the point cloud and saves it to a new .pcd file.
# 
# Usage: after SOM_Downsampling, some random points occur far from the original point cloud. This script removes them.
# alternatively adjustt the threshold to remove more or less points. 
#
# Example usage: adapt the file paths at the bottom, activate open3d conda environment and run the script.
############################################################################################################


import numpy as np
import open3d as o3d
import os

def read_point_cloud_pcd(file_path):
    """
    Reads the point cloud from a .pcd file.
    """
    print(f"Reading point cloud from: {file_path}")
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"Loaded point cloud from {file_path} with {len(pcd.points)} points")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise
    return pcd

def write_point_cloud_pcd(file_path, pcd):
    """
    Saves the point cloud to a .pcd file.

    Args:
        file_path (str): Path to save the point cloud.
        pcd (o3d.geometry.PointCloud): The point cloud object to save.
    """
    print(f"Saving {len(pcd.points)} points to {file_path}")
    o3d.io.write_point_cloud(file_path, pcd)

def filter_downsampled_points(original_pcd, downsampled_pcd, threshold):
    """
    Remove points from the downsampled point cloud that are further away from any point in the original point cloud
    than the given threshold.

    Args:
        original_pcd (o3d.geometry.PointCloud): Original point cloud.
        downsampled_pcd (o3d.geometry.PointCloud): Downsampled point cloud.
        threshold (float): Distance threshold for filtering points.

    Returns:
        o3d.geometry.PointCloud: The filtered downsampled point cloud.
    """
    # print(f"Computing distances between {len(downsampled_pcd.points)} downsampled points and {len(original_pcd.points)} original points")
    
    try:
        distances = downsampled_pcd.compute_point_cloud_distance(original_pcd)
        print(f"Distance computation completed")
    except Exception as e:
        print(f"Error during distance computation: {e}")
        raise

    # Keep points that are within the threshold
    mask = np.array(distances) < threshold
    print(f"Keeping {np.sum(mask)} points out of {len(mask)}")

    removed_points = np.sum(~mask)
    
    filtered_points = np.asarray(downsampled_pcd.points)[mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd, removed_points

def add_random_points(original_pcd, downsampled_pcd, num_points_to_add):
    """
    Randomly adds points from the original point cloud to the downsampled point cloud.

    Args:
        original_pcd (o3d.geometry.PointCloud): The original point cloud.
        downsampled_pcd (o3d.geometry.PointCloud): The downsampled point cloud.
        num_points_to_add (int): The number of points to add to the downsampled point cloud.

    Returns:
        o3d.geometry.PointCloud: The downsampled point cloud with added points from the original point cloud.
    """
    original_points = np.asarray(original_pcd.points)
    downsampled_points = np.asarray(downsampled_pcd.points)

    # Randomly sample points from the original point cloud
    original_point_count = len(original_points)
    indices = np.random.choice(original_point_count, num_points_to_add, replace=False)
    points_to_add = original_points[indices]

    # Combine the downsampled points with the randomly sampled points from the original point cloud
    combined_points = np.vstack((downsampled_points, points_to_add))

    # Create a new point cloud with the combined points
    augmented_pcd = o3d.geometry.PointCloud()
    augmented_pcd.points = o3d.utility.Vector3dVector(combined_points)

    return augmented_pcd


def center_and_normalize_point_cloud(pcd):
    """
    Centers and normalizes the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to be centered and normalized.

    Returns:
        o3d.geometry.PointCloud: The centered and normalized point cloud.
    """
    # Convert points to numpy array for processing
    points = np.asarray(pcd.points)

    # Centering: subtract the centroid (mean of x, y, z) from all points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    print(f"Centroid of point cloud: {centroid}")

    # Normalization: scale so that the maximum distance from the origin is 1
    max_distance = np.linalg.norm(centered_points, axis=1).max()
    normalized_points = centered_points / max_distance
    print(f"Max distance from origin before normalization: {max_distance}")

    # Create a new point cloud with the centered and normalized points
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)

    return normalized_pcd

def transfer_normals_based_on_distance(original_pcd, downsampled_pcd):
    """
    Transfer normals from the original point cloud to the downsampled point cloud 
    by finding the closest point based on Euclidean distance and assigning normals from the original point cloud.

    Parameters:
    original_pcd (open3d.geometry.PointCloud): The original point cloud with accurate normals.
    downsampled_pcd (open3d.geometry.PointCloud): The downsampled point cloud without normals.

    Returns:
    downsampled_pcd (open3d.geometry.PointCloud): The downsampled point cloud with transferred normals.
    """
    # Ensure the original point cloud has normals
    if not original_pcd.has_normals():
        raise ValueError("Original point cloud does not contain normals.")
    
    # Compute distances from each downsampled point to the original point cloud
    distances = downsampled_pcd.compute_point_cloud_distance(original_pcd)
    
    # Prepare array to store the transferred normals
    downsampled_normals = np.zeros((len(downsampled_pcd.points), 3))
    
    # For each downsampled point, find the closest point in the original point cloud
    for i, distance in enumerate(distances):
        # Find the index of the closest point in the original point cloud
        min_distance_index = np.argmin(distance)
        
        # Assign the normal from the closest point in the original point cloud
        downsampled_normals[i] = original_pcd.normals[min_distance_index]
    
    # Set the normals for the downsampled point cloud
    downsampled_pcd.normals = o3d.utility.Vector3dVector(downsampled_normals)
    
    return downsampled_pcd



def main(original_file, downsampled_file, output_file, threshold):
    """
    Main function to process the point clouds.

    Args:
        original_file (str): Path to the original point cloud .pcd file.
        downsampled_file (str): Path to the downsampled point cloud .pcd file.
        output_file (str): Path to save the filtered point cloud .pcd file.
        threshold (float): Distance threshold for filtering points.
    """
    # Read the point clouds from .pcd files
    original_pcd = read_point_cloud_pcd(original_file)
    downsampled_pcd = read_point_cloud_pcd(downsampled_file)

    if not downsampled_pcd.has_normals():
        downsampled_pcd.normals = o3d.utility.Vector3dVector(np.zeros((len(downsampled_pcd.points), 3)))

    if not downsampled_pcd.has_colors():
        downsampled_pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(downsampled_pcd.points), 3)))

    print(type(original_pcd.points[0]), type(downsampled_pcd.points[0]))


    # Filter downsampled points that are too far from the original
    #filtered_pcd = filter_downsampled_points(original_pcd, downsampled_pcd, threshold)
    filtered_pcd, removed_points = filter_downsampled_points(original_pcd, downsampled_pcd, threshold)

    if len(filtered_pcd.points) == 0:
        print("Filtered point cloud is empty.")
        return

    # Transfer normals from original to downsampled point cloud
    downsampled_pcd_with_normals = transfer_normals_based_on_distance(original_pcd, filtered_pcd)

    o3d.io.write_point_cloud(output_file_normals, downsampled_pcd_with_normals)


    # # Save the filtered point cloud to .pcd
    # write_point_cloud_pcd(output_file, filtered_pcd)
    # print(f"Filtered point cloud saved to {output_file} with {len(filtered_pcd.points)} points.")

    # augmented_pcd = add_random_points(original_pcd, filtered_pcd, removed_points)

    # # Center and normalize the point cloud
    # normalized_pcd = center_and_normalize_point_cloud(augmented_pcd)

    # print(f"Centered, normalized and augmented point cloud saved to {output_file_normalized}, with {len(normalized_pcd.points)} points.")
    # # Save the normalized point cloud
    # o3d.io.write_point_cloud(output_file_normalized, normalized_pcd)


if __name__ == '__main__':
    # Provide the original and downsampled file paths
    original_file = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/semantic_segemntation/Downsampling/s_pc_C-3_2024-08-07_dense_02.ply"
    downsampled_file = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/semantic_segemntation/Downsampling/downsampled/512_processed_s_pc_C-3_2024-08-07_dense_02.ply"

    # Extract the directory and base name (without extension) from the downsampled file
    base_dir = os.path.dirname(downsampled_file)
    base_filename = os.path.basename(downsampled_file).replace(".ply", "")  # Remove ".ply"

    # Define output file names based on the downsampled file
    output_file_cleaned = os.path.join(base_dir, f"{base_filename}_cleaned.ply")
    output_file_normals = os.path.join(base_dir, f"{base_filename}_cleaned_normals.ply")
    output_file_normalized = os.path.join(base_dir, f"{base_filename}_cleaned_normalized.ply")

    # Set distance threshold
    threshold = 0.008

    print(f"Original file: {original_file}")
    print(f"Downsampled file: {downsampled_file}")
    print(f"Output (cleaned): {output_file_cleaned}")
    print(f"Output (normals): {output_file_normals}")
    print(f"Output (normalized): {output_file_normalized}")

    # Call the main function with the provided paths
    main(original_file, downsampled_file, output_file_cleaned, threshold)

