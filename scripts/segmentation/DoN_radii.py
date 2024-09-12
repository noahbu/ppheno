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

if __name__ == '__main__':
    # Load point cloud
    pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-06/A-4_2024-08-06/edits/m_pc_A-4_2024-08-06_dense_03.pcd")
    
    # Print number of points in the point cloud
    print(f"Number of points in the point cloud: {len(pcd.points)}")
    
    # Define radii for small and large scale normals
    radius_small = 0.01  # Adjust based on point cloud scale
    radius_large = 0.1   # Adjust based on point cloud scale

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
    plot_DoN_histogram(DoN)

    # Print normals for a few points (adjust indices based on point cloud size)
    point_indices = [0, 100, 500, 1000, 5000]  # Change these indices as needed
    print_normals_for_points(normals_small, normals_large, point_indices)
