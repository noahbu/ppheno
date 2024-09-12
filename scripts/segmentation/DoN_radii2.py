############################################################################################################
# Title: Difference of Normals (DoN) for Point Clouds
# Works so far, however the visualization in open3d leads to a crash.
# So one has to save the point cloud with the DoN values as a scalar value and visualize it in CloudCompare.
############################################################################################################

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def compute_normals_and_stats(pcd, radius):
    """
    Estimate normals for the point cloud using radius-based normal estimation and compute statistics.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        radius (float): The radius for normal estimation.
    
    Returns:
        np.ndarray: The normals as a NumPy array.
        dict: Statistics (mean, min, max, std) of the normals.
    """
    # Estimate normals using radius-based search
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    
    # Retrieve normals as a NumPy array immediately
    normals = np.asarray(pcd.normals).copy()  # Copy to prevent overwriting later
    
    # Compute statistics
    stats = {
        'mean': np.mean(normals, axis=0),
        'min': np.min(normals, axis=0),
        'max': np.max(normals, axis=0),
        'std': np.std(normals, axis=0)
    }
    
    return normals, stats

def compute_DoN(normals_small, normals_large):
    """
    Compute the Difference of Normals (DoN) between two sets of normals.
    
    Args:
        normals_small (np.ndarray): Normals computed at small scale.
        normals_large (np.ndarray): Normals computed at large scale.
    
    Returns:
        np.ndarray: The DoN values (difference between small and large-scale normals).
    """
    DoN = normals_small - normals_large
    return DoN

def inspect_DoN_statistics(DoN):
    """
    Inspect the statistics of the magnitudes of the Difference of Normals (DoN).
    
    Args:
        DoN (np.ndarray): The DoN values for each point in the point cloud.
    
    Returns:
        dict: A dictionary containing min, max, mean, median, and std of DoN magnitudes.
    """
    # Compute the magnitude of the DoN vectors
    DoN_magnitude = np.linalg.norm(DoN, axis=1)

    # Compute statistics
    min_DoN = np.min(DoN_magnitude)
    max_DoN = np.max(DoN_magnitude)
    mean_DoN = np.mean(DoN_magnitude)
    median_DoN = np.median(DoN_magnitude)
    std_DoN = np.std(DoN_magnitude)

    # Print the statistics
    print(f"Minimum DoN magnitude: {min_DoN}")
    print(f"Maximum DoN magnitude: {max_DoN}")
    print(f"Mean DoN magnitude: {mean_DoN}")
    print(f"Median DoN magnitude: {median_DoN}")
    print(f"Standard deviation of DoN magnitude: {std_DoN}")

    # Return statistics as a dictionary
    return {
        'min': min_DoN,
        'max': max_DoN,
        'mean': mean_DoN,
        'median': median_DoN,
        'std': std_DoN
    }

def plot_DoN_histogram(DoN, num_bins=50):
    """
    Compute and plot a histogram of the magnitudes of the Difference of Normals (DoN).
    
    Args:
        DoN (np.ndarray): The DoN values for each point in the point cloud.
        num_bins (int): The number of bins for the histogram.
    """
    # Compute the magnitude of the DoN vectors
    DoN_magnitude = np.linalg.norm(DoN, axis=1)

    # Plot the histogram of DoN magnitudes
    plt.hist(DoN_magnitude, bins=num_bins, color='blue', alpha=0.7)
    plt.title('Histogram of DoN Magnitudes')
    plt.xlabel('Magnitude of DoN')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def print_normals_for_points(normals_small, normals_large, point_indices):
    """
    Print the small and large scale normals for specific points.
    
    Args:
        normals_small (np.ndarray): Small-scale normals.
        normals_large (np.ndarray): Large-scale normals.
        point_indices (list): List of point indices to inspect.
    """
    print("\nNormals for selected points:")
    for idx in point_indices:
        print(f"\nPoint {idx}:")
        print(f"Small-scale normal: {normals_small[idx]}")
        print(f"Large-scale normal: {normals_large[idx]}")

import numpy as np

def compute_DoN_with_orientation_check(normal_small, normal_large):
    """
    Compute the Difference of Normals (DoN) between two normal vectors, checking if they point
    in the same direction. If they point in opposite directions, reorient one normal.
    
    Args:
        normal_small (np.ndarray): The small-scale normal (1x3 vector).
        normal_large (np.ndarray): The large-scale normal (1x3 vector).
    
    Returns:
        float: The magnitude of the Difference of Normals (DoN).
    """
    # Compute the dot product between the two normals
    dot_product = np.dot(normal_small, normal_large)

    # Check if normals are pointing in the opposite direction (dot product < 0)
    if dot_product < 0:
        # Reorient the large normal by flipping its direction
        normal_large = -normal_large

    # Compute the Difference of Normals (DoN) as the difference between the two normals
    DoN = normal_small - normal_large

    # Compute the magnitude of the DoN
    DoN_magnitude = np.linalg.norm(DoN)

    return DoN_magnitude

def compute_DoN_magnitudes_for_all_points(normals_small, normals_large):
    """
    Compute the Difference of Normals (DoN) magnitudes for all points in the point cloud,
    ensuring normal orientation consistency.
    
    Args:
        normals_small (np.ndarray): The small-scale normals for all points (n x 3 array).
        normals_large (np.ndarray): The large-scale normals for all points (n x 3 array).
    
    Returns:
        np.ndarray: An array of DoN magnitudes for all points.
    """
    # Initialize an array to store the DoN magnitudes
    DoN_magnitudes = np.zeros(normals_small.shape[0])

    # Loop through each point and compute the DoN magnitude with orientation check
    for i in range(normals_small.shape[0]):
        DoN_magnitudes[i] = compute_DoN_with_orientation_check(normals_small[i], normals_large[i])

    return DoN_magnitudes


def attach_DoN_as_scalar_to_point_cloud(pcd, DoN_magnitudes):
    """
    Attach the Difference of Normals (DoN) magnitudes as a scalar field to the point cloud,
    normalizing the values to the range [0, 1] to avoid clamping issues.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        DoN_magnitudes (np.ndarray): The DoN magnitudes for each point.
    
    Returns:
        o3d.geometry.PointCloud: The point cloud with DoN values added as a custom scalar field.
    """
    # Normalize the DoN magnitudes to [0, 1] for storage
    DoN_magnitude_normalized = (DoN_magnitudes - DoN_magnitudes.min()) / (DoN_magnitudes.max() - DoN_magnitudes.min())

    # Attach the normalized DoN magnitudes as a color field (3 channels, but all values are identical)
    pcd.colors = o3d.utility.Vector3dVector(DoN_magnitude_normalized[:, np.newaxis].repeat(3, axis=1))
    
    return pcd

def save_point_cloud_with_DoN(pcd, filename):
    """
    Save the point cloud with attached DoN scalar values to a PLY file.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud with DoN values attached.
        filename (str): The output filename (.ply).
    """
    # Save the point cloud to the specified PLY file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud with DoN magnitudes saved to {filename}")

def write_custom_ply_with_DoN(filename, pcd, DoN_magnitudes):
    """
    Write a custom PLY file with an additional scalar field (DoN magnitudes) attached.
    
    Args:
        filename (str): The output filename (.ply).
        pcd (o3d.geometry.PointCloud): The point cloud.
        DoN_magnitudes (np.ndarray): The DoN magnitudes to attach as a scalar field.
    """
    # Get the points and convert them to a NumPy array
    points = np.asarray(pcd.points)

    # Ensure DoN_magnitudes is 1D and has the same number of entries as points
    assert DoN_magnitudes.shape[0] == points.shape[0], "Number of DoN magnitudes must match number of points."

    # Write to a custom PLY file
    with open(filename, 'w') as file:
        # Write the PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float DoN_magnitude\n")  # Add custom scalar field in the PLY header
        file.write("end_header\n")
        
        # Write the point cloud data with DoN magnitudes
        for point, don_magnitude in zip(points, DoN_magnitudes):
            file.write(f"{point[0]} {point[1]} {point[2]} {don_magnitude}\n")
    
    print(f"Point cloud with DoN magnitudes saved to {filename}")

def visualize_DoN_as_color(pcd, DoN_magnitudes):
    """
    Visualize the Difference of Normals (DoN) magnitudes as colors in the point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        DoN_magnitudes (np.ndarray): The DoN magnitudes for each point.
    
    Returns:
        o3d.geometry.PointCloud: The point cloud with DoN values visualized as colors.
    """
    # Normalize the DoN magnitudes to [0, 1] for color mapping
    DoN_magnitude_normalized = (DoN_magnitudes - DoN_magnitudes.min()) / (DoN_magnitudes.max() - DoN_magnitudes.min())

    # Map the normalized DoN magnitudes to colors using a colormap (e.g., 'jet')
    colors = plt.cm.jet(DoN_magnitude_normalized)[:, :3]  # Take only RGB (ignore alpha channel)

    # Ensure that colors are in the format expected by Open3D (float32 in the range [0, 1])
    colors = colors.astype(np.float32)

    # Attach the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud with DoN-based colors
    o3d.visualization.draw_geometries([pcd])

    return pcd


if __name__ == '__main__':
    # Load point cloud
    pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-06/A-4_2024-08-06/edits/m_pc_A-4_2024-08-06_dense_03.pcd")
    output_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-06/A-4_2024-08-06/edits/DoN_normals.ply"

    # Print number of points in the point cloud
    print(f"Number of points in the point cloud: {len(pcd.points)}")
    
    # Define radii for small and large scale normals
    radius_small = 0.05  # Adjust based on point cloud scale
    radius_large = 0.25   # Adjust based on point cloud scale

    # Compute normals at small scale and get statistics
    normals_small, stats_small = compute_normals_and_stats(pcd, radius_small)
    print("\nSmall-scale normals statistics:")
    print(f"Mean: {stats_small['mean']}")
    print(f"Min: {stats_small['min']}")
    print(f"Max: {stats_small['max']}")
    print(f"Standard deviation: {stats_small['std']}")

    # Compute normals at large scale and get statistics
    normals_large, stats_large = compute_normals_and_stats(pcd, radius_large)
    print("\nLarge-scale normals statistics:")
    print(f"Mean: {stats_large['mean']}")
    print(f"Min: {stats_large['min']}")
    print(f"Max: {stats_large['max']}")
    print(f"Standard deviation: {stats_large['std']}")

    # Compute the Difference of Normals (DoN)
    DoN = compute_DoN(normals_small, normals_large)

    # Inspect the statistics of the DoN values
    DoN_statistics = inspect_DoN_statistics(DoN)

    # Plot the histogram of DoN values
    #plot_DoN_histogram(DoN)

    # Print normals for a few points (adjust indices based on point cloud size)
    point_indices = [0, 100, 500, 1000, 5000]  # Change these indices as needed
    print_normals_for_points(normals_small, normals_large, point_indices)

    DoN_magnitudes = compute_DoN_magnitudes_for_all_points(normals_small, normals_large)
    print(f"Computed DoN magnitudes for all points: {DoN_magnitudes}")

        # Attach DoN magnitudes as scalar values (greyscale colors) to the point cloud
    #pcd_with_DoN = attach_DoN_as_scalar_to_point_cloud(pcd, DoN_magnitudes)

    # Save the point cloud with DoN magnitudes to a new location
    #save_point_cloud_with_DoN(pcd_with_DoN, output_path)

    #pcd_with_DoN_colors = visualize_DoN_as_color(pcd, DoN_magnitudes)

    write_custom_ply_with_DoN(output_path, pcd, DoN_magnitudes)


