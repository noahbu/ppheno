import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from DoN_radii2 import compute_DoN_feature_vector
import sys
import os
import colorsys
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


import numpy as np
import colorsys

def extract_hsi_features(pcd):
    """
    Extract spatial information (x, y, z), normals, and HSI (Hue, Saturation, Intensity) from a point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
    
    Returns:
        np.ndarray: A feature vector for each point, combining spatial info, normals, and HSI (Hue, Saturation, Intensity).
    """
    # Step 1: Extract spatial information (x, y, z)
    points = np.asarray(pcd.points)  # (n_points, 3)
    
    # Step 2: Estimate normals (if not already computed)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)  # (n_points, 3)

    # Step 3: Extract colors and convert RGB to HSV to get Hue, Saturation, and Intensity
    if pcd.has_colors():
        colors_rgb = np.asarray(pcd.colors)  # (n_points, 3) in RGB format (range [0, 1])
        hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in colors_rgb])
        hues = hsv_colors[:, 0]  # Hue
        saturations = hsv_colors[:, 1]  # Saturation
        intensities = hsv_colors[:, 2]  # Intensity (Value)
    else:
        hues = np.zeros(len(points))  # If no colors are available, set all hues to 0
        saturations = np.zeros(len(points))  # Set saturation to 0 if no color
        intensities = np.zeros(len(points))  # Set intensity to 0 if no color

    # Combine spatial, normals, and HSI into a single feature matrix
    # Combine: [x, y, z, normal_x, normal_y, normal_z, Hue, Saturation, Intensity]
    hsi_feature_vector = np.hstack((
        points,
        normals,
        hues[:, np.newaxis], 
        saturations[:, np.newaxis], 
        intensities[:, np.newaxis]
    ))  # (n_points, 9)

    return hsi_feature_vector



# load pointcloud directly, not in main. Use a relative path from the root directory: 
def load_pointcloud_relative(relative_path):
    """
    Load a point cloud from a relative file path, going up two directories to the root directory.
    
    Args:
        relative_path (str): The relative path to the point cloud file from the root directory.
    
    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    # Get the current file directory (the script's location)
    current_dir = os.path.dirname(__file__)
    
    # Navigate up two directories to reach the root directory of the repository
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Join the root directory with the provided relative path
    point_cloud_path = os.path.join(root_dir, relative_path)
    
    # Load the point cloud using Open3D
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    
    return pcd

def save_pointcloud_relative(pcd, relative_dir, filename):
    """
    Save a point cloud to a relative directory path and filename, navigating up two directories to the root directory.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to save.
        relative_dir (str): The relative directory where the point cloud should be saved from the root directory.
        filename (str): The filename to save the point cloud as (e.g., 'output.ply').
    """
    # Get the current file directory (the script's location)
    current_dir = os.path.dirname(__file__)
    
    # Navigate up two directories to reach the root directory of the repository
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Join the root directory with the provided relative directory to create the full directory path
    output_dir = os.path.join(root_dir, relative_dir)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Join the directory path and filename to get the full output path
    point_cloud_path = os.path.join(output_dir, filename)
    
    # Save the point cloud using Open3D
    o3d.io.write_point_cloud(point_cloud_path, pcd)
    
    print(f"Saved point cloud to {point_cloud_path}")



def combine_hsi_and_don_features(pcd, radius_small, radius_large):
    """
    Combine the HSI (Hue, Saturation, Intensity) features and DoN (Difference of Normals) features.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        radius_small (float): Radius for small-scale normal estimation (for DoN).
        radius_large (float): Radius for large-scale normal estimation (for DoN).
    
    Returns:
        np.ndarray: Combined feature vector with HSI and DoN features.
    """
    # Extract HSI features
    hsi_features = extract_hsi_features(pcd)  # (n_points, 9)

    # Compute DoN features
    don_features = compute_DoN_feature_vector(pcd, radius_small, radius_large)  # (n_points, 1)

    # Combine HSI and DoN into a single feature matrix
    combined_features = np.hstack((hsi_features, don_features[:, np.newaxis]))  # (n_points, 10)

    return combined_features



def extract_and_weight_features(pcd, radius_small, radius_large, weights):
    """
    Extract features, normalize them, and apply weights, including spatial, normals, Hue, Saturation, Intensity, and DoN.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        radius_small (float): Radius for small-scale normal estimation (for DoN).
        radius_large (float): Radius for large-scale normal estimation (for DoN).
        weights (dict): Weights for each feature. Keys: 'spatial', 'normals', 'hue', 'saturation', 'intensity', 'don'.

    Returns:
        np.ndarray: Weighted feature matrix.
    """
    # Extract features
    features = combine_hsi_and_don_features(pcd, radius_small, radius_large)
    # features shape: (n_points, 8)

    # Initialize normalized features
    n_points = features.shape[0]
    normalized_features = np.zeros((n_points, 10))  # Added Saturation and Intensity to the features

    # Indices for features
    idx = {
        'spatial': slice(0, 3),
        'normals': slice(3, 6),
        'hue': 6,
        'saturation': 7,   # New index for saturation
        'intensity': 8,    # New index for intensity
        'don': 9           # Shifted DoN index to 9
    }

    # Convert RGB to HSV to extract Saturation and Intensity (Value)
    colors_rgb = np.asarray(pcd.colors)
    hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in colors_rgb])

    hues = hsv_colors[:, 0]  # Hue is already in features[:, 6]
    saturations = hsv_colors[:, 1]  # Saturation
    intensities = hsv_colors[:, 2]  # Value (Intensity)

    # Normalize spatial coordinates
    spatial_features = features[:, idx['spatial']]
    spatial_mean = np.mean(spatial_features, axis=0)
    spatial_std = np.std(spatial_features, axis=0)
    spatial_std[spatial_std == 0] = 1  # Avoid division by zero
    normalized_features[:, idx['spatial']] = (spatial_features - spatial_mean) / spatial_std

    # Normalize normals (ensure they are unit vectors)
    normals = features[:, idx['normals']]
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_norm[normals_norm == 0] = 1
    normalized_features[:, idx['normals']] = normals / normals_norm

    # Normalize hue
    hue_mean = np.mean(hues)
    hue_std = np.std(hues)
    hue_std = hue_std if hue_std != 0 else 1
    normalized_features[:, idx['hue']] = (hues - hue_mean) / hue_std

    # Normalize saturation
    saturation_mean = np.mean(saturations)
    saturation_std = np.std(saturations)
    saturation_std = saturation_std if saturation_std != 0 else 1
    normalized_features[:, idx['saturation']] = (saturations - saturation_mean) / saturation_std

    # Normalize intensity (Value)
    intensity_mean = np.mean(intensities)
    intensity_std = np.std(intensities)
    intensity_std = intensity_std if intensity_std != 0 else 1
    normalized_features[:, idx['intensity']] = (intensities - intensity_mean) / intensity_std

    # Normalize DoN
    don = features[:, idx['don']]
    don_mean = np.mean(don)
    don_std = np.std(don)
    don_std = don_std if don_std != 0 else 1
    normalized_features[:, idx['don']] = (don - don_mean) / don_std

    # Apply weights
    weighted_features = np.zeros_like(normalized_features)
    for key, weight in weights.items():
        indices = idx[key]
        if isinstance(indices, slice):
            weighted_features[:, indices] = normalized_features[:, indices] * weight
        else:
            weighted_features[:, indices] = normalized_features[:, indices] * weight

    return weighted_features


def assign_colors_to_clusters(pcd, labels, cluster_colors):
    """
    Assign colors to clusters in the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        labels (np.ndarray): Cluster labels for each point.
        cluster_colors (dict): Mapping from cluster index to color [R, G, B].

    Returns:
        o3d.geometry.PointCloud: Point cloud with updated colors.
    """
    colors = np.asarray(pcd.colors)
    new_colors = np.zeros_like(colors)

    for cluster_idx, color in cluster_colors.items():
        new_colors[labels == cluster_idx] = color

    pcd.colors = o3d.utility.Vector3dVector(new_colors)
    return pcd

############################################################################################################
# the following functions are used to determine the volumetric IoU score of the segmentation

def estimate_voxel_size_sklearn(pcd, sample_ratio=0.01, num_neighbors=20):
    """
    Estimate voxel size based on nearest neighbor distances using scikit-learn's NearestNeighbors.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        sample_ratio (float): Fraction of points to sample from the point cloud.
        num_neighbors (int): Number of neighbors for distance estimation.
    
    Returns:
        float: Estimated voxel size.
    """
    points = np.asarray(pcd.points)
    n_points = len(points)

    # Sample points if the point cloud is too large
    sample_size = max(1, int(n_points * sample_ratio))
    sample_points = points[np.random.choice(n_points, sample_size, replace=False)]

    # Use scikit-learn's NearestNeighbors for efficient KDTree-like search
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto').fit(sample_points)
    distances, _ = nbrs.kneighbors(sample_points)

    # Compute average distance to neighbors (ignore distance to self)
    avg_distances = np.mean(distances[:, 1:], axis=1)

    return np.mean(avg_distances)

def compute_voxel_iou(pcd_1, pcd_2):
    preprocess_point_cloud(pcd_1)
    voxel_size = estimate_voxel_size_sklearn(pcd_2)


    voxel_grid_1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_1, voxel_size=voxel_size)
    voxel_grid_2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_2, voxel_size=voxel_size)
    
    # Get the voxel coordinates for both grids
    voxels1 = set([tuple(voxel.grid_index) for voxel in voxel_grid_1.get_voxels()])
    voxels2 = set([tuple(voxel.grid_index) for voxel in voxel_grid_2.get_voxels()])

        # Compute intersection and union
    intersection = len(voxels1.intersection(voxels2))
    union = len(voxels1.union(voxels2))

    if union == 0:
        return 0.0

    return intersection / union




def objective_function(pcd, ground_truth_pcd, radius_small, radius_large, weights, n_clusters=5):
    """
    Objective function that evaluates segmentation quality based on given weights.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        ground_truth_pcd (o3d.geometry.PointCloud): The manually segmented plant point cloud (ground truth).
        radius_small (float): Small-scale radius for DoN.
        radius_large (float): Large-scale radius for DoN.
        weights (dict): Weights for the features (spatial, normals, hue, don).
        n_clusters (int): Number of clusters for GMM.
        
    Returns:
        float: The IoU score of the segmentation.
    """
    # Extract and weight features based on current weights
    weighted_features = extract_and_weight_features(pcd, radius_small, radius_large, weights)

    # Perform GMM clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(weighted_features)
    labels = gmm.predict(weighted_features)

    # Assuming we want to find the plant cluster (based on hue, normals, etc.)
    plant_cluster = identify_plant_cluster(pcd, labels)

    # Create a binary mask for the predicted plant points
    predicted_plant_mask = labels == plant_cluster

    # Create a binary mask for the ground truth plant points
    ground_truth_points = np.asarray(ground_truth_pcd.points)
    full_pcd_points = np.asarray(pcd.points)
    
    ground_truth_mask = np.isin(full_pcd_points, ground_truth_points)

    # Compute the IoU between the predicted and ground truth plant points
    iou_score = compute_voxel_iou(predicted_plant_mask, ground_truth_mask)

    return iou_score


def identify_plant_cluster(pcd, labels, n_clusters):
    """
    Identify the plant cluster based on the hue values of the point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        labels (np.ndarray): Cluster labels for each point.
        
    Returns:
        int: The cluster index that corresponds to the plant cluster.
    """
    # Extract hue values from the point cloud colors
    colors = np.asarray(pcd.colors)
    hue_values = np.array([colorsys.rgb_to_hsv(r, g, b)[0] * 360 for r, g, b in colors])

    # Define the range for green hue
    green_hue_min, green_hue_max = 80, 100

    # Count the number of green hue points in each cluster
    green_counts = []
    for i in range(n_clusters):
        cluster_indices = (labels == i)
        green_count = np.sum((cluster_indices) & (green_hue_min <= hue_values) & (hue_values <= green_hue_max))
        green_counts.append(green_count)

    # Identify the plant cluster (the one with the most green points)
    plant_cluster = np.argmax(green_counts)

    return plant_cluster

def preprocess_point_cloud(pcd, voxel_size=0.03, nb_neighbors=20, std_ratio=2.0):
    """Preprocess point cloud by downsampling and removing statistical outliers."""
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def perform_gmm_clustering(pcd, radius_small, radius_large, weights, n_clusters):
    """Perform GMM clustering on the point cloud."""
    weighted_features = extract_and_weight_features(pcd, radius_small, radius_large, weights)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(weighted_features)
    labels = gmm.predict(weighted_features)
    return labels

def filter_cluster_points(pcd, labels, target_cluster):
    """Filter points and normals that belong to a specific cluster."""
    cluster_indices = np.where(labels == target_cluster)[0]
    cluster_points = np.asarray(pcd.points)[cluster_indices]
    cluster_colors = np.asarray(pcd.colors)[cluster_indices]
    
    # Also filter normals
    if pcd.has_normals():
        cluster_normals = np.asarray(pcd.normals)[cluster_indices]
    else:
        cluster_normals = None

    # Create a new point cloud for the cluster with normals
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)
    
    if cluster_normals is not None:
        cluster_pcd.normals = o3d.utility.Vector3dVector(cluster_normals)

    return cluster_pcd

def save_plant_and_clusters(pcd, labels, plant_cluster, pcd_dir, pcd_filename):
    """Assign colors to clusters, save the final clustered and plant-only point clouds."""
    ground_clusters = [i for i in range(np.max(labels) + 1) if i != plant_cluster]
    
    # Define colors for clusters
    cluster_colors = {}
    cluster_colors[plant_cluster] = [pcd.colors[i] for i in np.where(labels == plant_cluster)[0]]
    for ground_cluster in ground_clusters:
        cluster_colors[ground_cluster] = [0.0, 0.5, 1.0]  # Blue color for ground clusters

    # Assign colors to plant cluster after second clustering
    pcd_colored = assign_colors_to_clusters(pcd, labels, cluster_colors)

    # Save the clustered point cloud
    save_pointcloud_relative(pcd_colored, pcd_dir, f"{pcd_filename[:-4]}_clustered.ply")

    # Filter and save plant-only points
    plant_pcd = filter_cluster_points(pcd, labels, plant_cluster)
    save_pointcloud_relative(plant_pcd, pcd_dir, f"{pcd_filename[:-4]}_plant_cluster.ply")

############################################################################################################
# processing a whole directory of point clouds
def load_point_clouds(directory):
    """Load the pc_***_dense_02.ply and m_pc_***_dense_03.ply files in a directory."""
    pc_dense_02 = None
    m_pc_dense_03 = None
    for filename in os.listdir(directory):
        if "pc_" in filename and "_dense_02.ply" in filename:
            pc_dense_02 = o3d.io.read_point_cloud(os.path.join(directory, filename))
        elif "m_pc_" in filename and "_dense_03.ply" in filename:
            m_pc_dense_03 = o3d.io.read_point_cloud(os.path.join(directory, filename))
    
    if pc_dense_02 is None or m_pc_dense_03 is None:
        return None, None
    
    return pc_dense_02, m_pc_dense_03

def process_directory(directory):
    """Process all point clouds in a directory."""
    # Load the point clouds
    pc_dense_02, m_pc_dense_03 = load_point_clouds(directory)
    
    if pc_dense_02 is None or m_pc_dense_03 is None:
        print(f"Skipping directory {directory}: Required files not found.")
        return
    
    print(f"Processing {directory}...")

    # Preprocessing (you can add any specific preprocessing steps)
    pc_dense_02 = preprocess_point_cloud(pc_dense_02)

    # Perform clustering
    radius_small = 0.4
    radius_large = 0.5
    n_clusters = 5
    weights = {
        'spatial': 0.3,
        'normals': 0.0,
        'hue': 1.0,
        'saturation': 0.0,
        'intensity': 0.0,
        'don': 0.0
    }
    
    labels = perform_gmm_clustering(pc_dense_02, radius_small, radius_large, weights, n_clusters)
    
    # Filter plant points (or define any other condition for plant cluster)
    plant_cluster = identify_plant_cluster(pc_dense_02, labels)
    plant_pcd = filter_cluster_points(pc_dense_02, labels, plant_cluster)

    # Save clustered point cloud
    clustered_filename = os.path.join(directory, "clustered.ply")
    o3d.io.write_point_cloud(clustered_filename, plant_pcd)
    print(f"Saved clustered point cloud: {clustered_filename}")

    # Compute voxelized IoU
    iou = voxelized_iou(plant_pcd, m_pc_dense_03)
    print(f"Voxelized IoU for {directory}: {iou}")
    

def main():
    # The relative path to the point cloud file from the root of your repository
    relative_pcd_path = "data/melonCycle/2024-08-01/A-5_2024-08-01/pc_A-5_2024-08-01_dense_02.ply"

    # Split the path into directory and file name
    pcd_dir = os.path.dirname(relative_pcd_path)  # Directory path
    pcd_filename = os.path.basename(relative_pcd_path)  # File name
    
    # Load the point cloud
    pcd = load_pointcloud_relative(relative_pcd_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    # Preprocess the point cloud
    pcd = preprocess_point_cloud(pcd)
    print(f"Downsampled pointcloud to {len(pcd.points)} points.")

    # Define parameters
    radius_small = 0.4
    radius_large = 0.5

    # First GMM clustering
    weights_first = {
        'spatial': 0.3,
        'normals': 0.0,
        'hue': 1.0,
        'saturation': 0.0,
        'intensity': 0.0,
        'don': 0.0
    }
    labels_first = perform_gmm_clustering(pcd, radius_small, radius_large, weights_first, n_clusters = 5)

    # Identify plant cluster based on first clustering
    plant_cluster = identify_plant_cluster(pcd, labels_first, n_clusters = 5)

    # Filter plant points for second clustering
    plant_pcd = filter_cluster_points(pcd, labels_first, plant_cluster)

    # Second GMM clustering (on plant cluster)
    weights_second = {
        'spatial': 0.0,
        'normals': 0.0,
        'hue': 1.0,
        'saturation': 1.0,
        'intensity': 1.0,
        'don': 0.0
    }
    labels_second = perform_gmm_clustering(plant_pcd, radius_small, radius_large, weights_second, n_clusters=2)

    # Identify plant cluster from the second clustering
    plant_cluster_second = identify_plant_cluster(plant_pcd, labels_second, n_clusters=2)

    # Save the clustered and plant-only point clouds
    save_plant_and_clusters(plant_pcd, labels_second, plant_cluster_second, pcd_dir, pcd_filename)

    print(f"Final clustering and plant cloud saved for {pcd_filename}")

if __name__ == "__main__":
    main()




