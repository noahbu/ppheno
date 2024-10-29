import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import open3d as o3d


def estimate_eps(data, k=4):
    """
    Estimate the eps parameter for DBSCAN by identifying the knee in the k-distance graph.
    
    Args:
    - data: Point cloud data (numpy array of shape [n_samples, n_features]).
    - k: The number of neighbors to use (default: 4 for DBSCAN).
    
    Returns:
    - k-distances plot to help identify the knee.
    """
    # Fit nearest neighbors model
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    
    # Distances and indices of the k nearest neighbors
    distances, indices = neighbors_fit.kneighbors(data)
    
    # Sort the distances (k-th nearest distances)
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Plot the sorted k-distance graph
    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.xlabel("Points sorted by distance to {}th nearest neighbor".format(k))
    plt.ylabel("{}th nearest neighbor distance".format(k))
    plt.title("K-distance Graph to Estimate eps for DBSCAN")
    plt.grid(True)
    plt.show()

# Example usage with random 2D points
if __name__ == "__main__":
    # Generate some random data (replace this with your point cloud)
    file_path = '/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/semantic_segemntation/Downsampling/PCA_soft_send/processed_s_pc_C-3_2024-08-07_dense_02 - Cloud.txt'
    # pcd = o3d.io.read_point_cloud(file_path, format='txt')
    points = np.loadtxt(file_path, delimiter=' ')

    
    # Estimate eps
    estimate_eps(points, k=5)  # k=4 is common for DBSCAN
