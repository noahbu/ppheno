##################
# No real difference between small and large scale normals based on k nearest neighbors
##################

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def compute_normals(pcd, radius):
    """
    Compute the normals for a point cloud with a specified radius for the nearest neighbor search.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        radius (float): The search radius for normal estimation.
    
    Returns:
        np.ndarray: The normals computed for each point in the point cloud.
    """
    # Clone the point cloud so we don't modify the original
    
    # Estimate normals for the point cloud with a given radius
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    
    # Return the normals as a numpy array
    return np.asarray(pcd.normals)

# def compute_DoN(pcd, small_knn, large_knn):
#     """
#     Compute the Difference of Normals (DoN) for a point cloud.
    
#     Args:
#         original_pcd (o3d.geometry.PointCloud): The input point cloud.
#         small_radius (float): The radius for estimating small-scale normals.
#         large_radius (float): The radius for estimating large-scale normals.
    
#     Returns:
#         np.ndarray: The DoN values for each point in the point cloud.
#     """
#     # Compute normals at both scales (same point cloud, different radii)
#     small_scale_normals = pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=small_knn))
#     large_scale_normals = pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=large_knn))

#     # Ensure the shapes of the normals are the same
#     assert small_scale_normals.shape == large_scale_normals.shape, \
#         f"Normal shapes do not match: {small_scale_normals.shape} vs {large_scale_normals.shape}"

#     # Compute the Difference of Normals (DoN)
#     DoN = small_scale_normals - large_scale_normals

#     return DoN

def visualize_DoN(original_pcd, DoN):
    """
    Visualize the DoN by mapping the magnitude of the DoN vectors to colors.
    
    Args:
        original_pcd (o3d.geometry.PointCloud): The input point cloud.
        DoN (np.ndarray): The Difference of Normals values for each point.
    """
    # Compute the magnitude of DoN vectors
    DoN_magnitude = np.linalg.norm(DoN, axis=1)

    # Normalize the DoN magnitudes to the range [0, 1]
    DoN_magnitude_normalized = (DoN_magnitude - DoN_magnitude.min()) / (DoN_magnitude.max() - DoN_magnitude.min())

    # Map the normalized DoN magnitude to color
    colors = np.zeros((DoN_magnitude_normalized.shape[0], 3))
    colors[:, 0] = DoN_magnitude_normalized  # Red channel represents the DoN magnitude

    # Set the colors of the point cloud
    original_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud with DoN-based colors
    o3d.visualization.draw_geometries([original_pcd])

def plot_DoN_histogram(DoN, num_bins=100):
    """
    Compute and plot a histogram of the magnitudes of the Difference of Normals (DoN).
    
    Args:
        DoN (np.ndarray): The DoN values for each point in the point cloud.
        num_bins (int): The number of bins for the histogram.
    """
    # Compute the magnitude of the DoN vectors
    DoN_magnitude = np.linalg.norm(DoN, axis=1)

    # Plot the histogram of DoN magnitudes
    plt.hist(DoN_magnitude, bins=num_bins, color='blue', alpha=0.7, range=(-0.1, 0.1))
    plt.title('Histogram of DoN Magnitudes')
    plt.xlabel('Magnitude of DoN')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

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

def compute_average_distance_to_k_neighbors_core(pcd, k):
    """
    Compute the average distance to the nearest k neighbors for a point cloud using `NearestNeighborSearch`.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        k (int): The number of neighbors to consider.
    
    Returns:
        float: The average distance to the k nearest neighbors.
    """
    # Convert the point cloud to an Open3D Tensor
    points = o3d.core.Tensor(np.asarray(pcd.points), o3d.core.Dtype.Float32)
    
    # Create the NearestNeighborSearch object
    nns = o3d.core.nns.NearestNeighborSearch(points)
    
    # Build the index for efficient searching
    nns.knn_index()
    
    distances = []
    
    # Perform k-NN search for all points in the point cloud
    for i in range(len(points)):
        # Reshape the point to (1, 3) before querying
        query_point = points[i:i+1]  # This makes it a 2D tensor with shape (1, 3)
        
        # Perform the k-NN search
        result = nns.knn_search(query_point, k + 1)  # +1 to include the point itself
        
        # Extract distances (ensure there are valid neighbors)
        if result[1].shape[0] > 1:  # At least 1 valid neighbor (excluding the point itself)
            dist = result[1][1:].numpy()  # Exclude the distance to the point itself
            distances.append(np.mean(dist))
        else:
            print(f"Warning: No valid neighbors found for point {i}")
    
    # Check if we found any valid distances
    if len(distances) == 0:
        print("No valid neighbors were found for any points.")
        return float('nan')

    # Calculate the average distance across all points
    avg_distance = np.mean(distances)
    
    print(f"Average distance to the nearest {k} neighbors: {avg_distance}")
    return avg_distance


def compute_DoN(pcd, knn_small=10, knn_large=50):
    """
    Compute the Difference of Normals (DoN) for a point cloud using different k-NN values.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        knn_small (int): The k-NN value for small-scale normal estimation.
        knn_large (int): The k-NN value for large-scale normal estimation.
    
    Returns:
        np.ndarray: The DoN values (difference between small and large-scale normals).
    """
    # Estimate small-scale normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_small))
    small_scale_normals = np.asarray(pcd.normals)

    # Estimate large-scale normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_large))
    large_scale_normals = np.asarray(pcd.normals)

    # Compute the Difference of Normals (DoN)
    DoN = small_scale_normals - large_scale_normals

    return DoN

def compute_normals_and_return_stats(pcd, knn):
    """
    Estimate normals for the point cloud and return statistical information.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        knn (int): The k-NN value for normal estimation.
    
    Returns:
        np.ndarray: The normals.
        dict: Statistics (mean, min, max, std) of the normals.
    """
    # Estimate normals using k-NN
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals = np.asarray(pcd.normals)
    
    # Calculate statistics
    stats = {
        'mean': np.mean(normals, axis=0),
        'min': np.min(normals, axis=0),
        'max': np.max(normals, axis=0),
        'std': np.std(normals, axis=0)
    }
    
    return normals, stats


if __name__ == '__main__':
    # Load point cloud
    pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-06/A-4_2024-08-06/edits/m_pc_A-4_2024-08-06_dense_03.pcd")
    
    # Define radii for small and large scale normals
    # small_radius = 0.0005
    # large_radius = 0.1
    print(f"Number of points in the point cloud: {len(pcd.points)}")
    # o3d.visualization.draw_geometries([pcd])

    # Compute small-scale normals (k-NN = 10)
    small_scale_normals, small_stats = compute_normals_and_return_stats(pcd, knn=5)
    print("Small-scale normals statistics:")
    print(f"Mean: {small_stats['mean']}")
    print(f"Min: {small_stats['min']}")
    print(f"Max: {small_stats['max']}")
    print(f"Standard deviation: {small_stats['std']}")
    
    # Compute large-scale normals (k-NN = 50)
    large_scale_normals, large_stats = compute_normals_and_return_stats(pcd, knn=500)
    print("\nLarge-scale normals statistics:")
    print(f"Mean: {large_stats['mean']}")
    print(f"Min: {large_stats['min']}")
    print(f"Max: {large_stats['max']}")
    print(f"Standard deviation: {large_stats['std']}")

    # Compute the Difference of Normals (DoN)
    DoN = compute_DoN(pcd, knn_small=5, knn_large=500)

    # Inspect the statistics of the DoN values
    DoN_statistics = inspect_DoN_statistics(DoN)

    # Plot the histogram of DoN values
    #plot_DoN_histogram(DoN)
