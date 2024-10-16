import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture




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


def compute_DoN_feature_vector(pcd, radius_small, radius_large):
    """
    Compute the Difference of Normals (DoN) for a point cloud and return the DoN magnitudes.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        radius_small (float): The radius for small-scale normal estimation.
        radius_large (float): The radius for large-scale normal estimation.
    
    Returns:
        np.ndarray: The DoN magnitudes, which can be used as a feature vector for clustering or other analysis.
    """
    
    # Function to compute normals
    def compute_normals(pcd, radius):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
        normals = np.asarray(pcd.normals).copy()
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / norm
    

    # Compute normals at small and large scales
    normals_small = compute_normals(pcd, radius_small)
    normals_large = compute_normals(pcd, radius_large)



    # Ensure normals point in the same direction
    dot_products = np.sum(normals_small * normals_large, axis=1)
    reverse_indices = dot_products < 0
    normals_large[reverse_indices] *= -1  # Reverse normals where dot product is negative

    # Compute DoN (Difference of Normals)
    DoN = (normals_small - normals_large) / 2
    
    # Compute DoN magnitudes
    DoN_magnitudes = np.linalg.norm(DoN, axis=1)

       # Normalize DoN magnitudes to [0, 1]
    max_DoN_value = np.sqrt(2) / 2  # Maximum possible DoN magnitude
    DoN_magnitudes_normalized = DoN_magnitudes / max_DoN_value
    
    return DoN_magnitudes_normalized

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

def filter_based_on_DoN(pcd, DoN_magnitudes, threshold=0.5):
    """
    Filter points from the point cloud based on DoN magnitudes.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        DoN_magnitudes (np.ndarray): The DoN magnitudes for each point in the point cloud.
        threshold (float): The threshold value for filtering. Points with DoN values greater
                           than this threshold will be retained.
    
    Returns:
        o3d.geometry.PointCloud: A new point cloud containing only the filtered points.
    """
    
    # Get the points of the point cloud
    points = np.asarray(pcd.points)
    
    # Filter points where DoN_magnitude exceeds the threshold
    filter_mask = DoN_magnitudes < threshold
    filtered_points = points[filter_mask]
    
    # Create a new point cloud from the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # If the point cloud has colors, also filter the colors
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_colors = colors[filter_mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # If the point cloud has normals, also filter the normals
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_normals = normals[filter_mask]
        filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    
    return filtered_pcd

def visualize_clusters(pcd, labels):
    """
    Visualize the point cloud clusters by coloring each cluster with a different high-contrast color.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        labels (np.ndarray): Cluster labels for each point in the point cloud.
    """
    max_label = labels.max()  # The number of clusters (excluding noise)
    num_clusters = max_label + 1  # Since labels are 0-indexed
    print(f"Point cloud has {num_clusters} clusters (excluding noise)")
    
    # Use a high-contrast colormap (like Set1 or hsv)
    colormap = plt.get_cmap("Set1", num_clusters)  # "Set1" provides high-contrast colors
    
    # Initialize colors array
    colors = np.zeros((labels.shape[0], 3))  # Default color is black (for noise or unclustered points)
    
    # Assign colors based on cluster labels
    for label in range(num_clusters):
        colors[labels == label] = colormap(label)[:3]  # Assign colormap color to each cluster
    
    # Handle noise (label == -1), assigning grey color
    colors[labels == -1] = [0.5, 0.5, 0.5]  # Grey for noise points
    
    # Assign the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize the clustered point cloud
    o3d.visualization.draw_geometries([pcd])

def plot_knn_distance(pcd, min_points):
    points = np.asarray(pcd.points)
    neigh = NearestNeighbors(n_neighbors=min_points)
    nbrs = neigh.fit(points)
    distances, indices = nbrs.kneighbors(points)
    distances = np.sort(distances[:, -1])  # Get the distances to the k-th nearest neighbor
    
    # Plot the distances
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{min_points}-th Nearest Neighbor Distance')
    plt.title(f'k-NN Distance Plot (k = {min_points})')
    plt.show()


if __name__ == '__main__':
    # Load point cloud
    pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/pointcloud_time_series/C-3/rotated/processed_s_pc_C-3_2024-08-08_dense_02.ply")
    output_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/pointcloud_time_series/C-3/DoN/processed_s_pc_C-3_2024-08-08_dense_02_DoN.ply"

    # Downsample the point cloud for greatly improved performance
    # pcd = pcd.voxel_down_sample(voxel_size=0.0001)

    # Print number of points in the point cloud
    print(f"Number of points in the point cloud: {len(pcd.points)}")

    # filter the point cloud
    pcd = filter_z_axis(pcd, z_threshold=0.05)
    # o3d.visualization.draw_geometries([pcd])

    # calculate the average point distance
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"Average point distance: {avg_dist}")

    radius_small = 4 * avg_dist

    radius_large = 8 * radius_small
    # radius_large = 0.005

    print(f"Small-scale radius: {radius_small}")
    print(f"Large-scale radius: {radius_large}")

    DoN_magnitudes = compute_DoN_feature_vector(pcd, radius_small, radius_large)
    
    print("Computed DoN magnitudes:", DoN_magnitudes)
    print(f"DoN magnitudes range: min={DoN_magnitudes.min()}, max={DoN_magnitudes.max()}")

    print(f"Number of points before filtering: {len(pcd.points)}")
    # Filter the point cloud based on DoN magnitudes
    threshold = 0.3
    pcd_filtered = filter_based_on_DoN(pcd, DoN_magnitudes, threshold=threshold)

    # downsample the point cloud if it has more thän 10000 points
    # if len(pcd_filtered.points) > 10000:
    #     pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=0.005)

    
    # Visualize the filtered point cloud
    # o3d.visualization.draw_geometries([pcd_filtered])


    # pcd = pcd.voxel_down_sample(voxel_size=0.002)
    print(f"Number of points after filtering: {len(pcd_filtered.points)}")


    # points = np.asarray(pcd_filtered.points)

############################################
    # # Performs DBSCAN clustering on the plant point cloud
    # labels = np.array(pcd_filtered.cluster_dbscan(eps=0.0025, min_points=10, print_progress=True))
    # max_label = labels.max()
    # print(f"Point cloud has {max_label + 1} clusters")

    # # tuning eps: by finding the knee point in the k-NN distance plot
    # plot_knn_distance(pcd_filtered, min_points=10)

############################################
    # Perform OPTICS clustering
    # clustering = OPTICS(min_samples=10)
    # labels = clustering.fit_predict(points)
############################################
    # Perform Gaussian Mixture Model clustering

    # Try different numbers of clusters and use BIC to select the best model
    # lowest_bic = np.infty
    # best_gmm = None
    # n_components_range = range(1, 15)
    # for n_components in n_components_range:
    #     gmm = GaussianMixture(n_components=n_components)
    #     gmm.fit(points)
    #     bic = gmm.bic(points)
    #     if bic < lowest_bic:
    #         lowest_bic = bic
    #         best_gmm = gmm

    # # Get the best labels from the best model
    # labels = best_gmm.predict(points)



    # visualize_clusters(pcd_filtered, labels)


    # Save the filtered point cloud with DoN magnitudes to a new location
    # o3d.io.write_point_cloud(output_path, pcd_filtered)

    # print(f"Computed DoN magnitudes for all points: {DoN_magnitudes}")

        # Attach DoN magnitudes as scalar values (greyscale colors) to the point cloud
    #pcd_with_DoN = attach_DoN_as_scalar_to_point_cloud(pcd, DoN_magnitudes)

    # Save the point cloud with DoN magnitudes to a new location
    #save_point_cloud_with_DoN(pcd_with_DoN, output_path)

    #pcd_with_DoN_colors = visualize_DoN_as_color(pcd, DoN_magnitudes)



    write_custom_ply_with_DoN(output_path, pcd, DoN_magnitudes)
