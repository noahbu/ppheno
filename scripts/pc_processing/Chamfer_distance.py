# Description: This script demonstrates how to align two point clouds using RANSAC-based global 
# registration followed by ICP fine alignment. The script also computes the Chamfer Distance between the two aligned point clouds. 
# The point clouds are downsampled to reduce memory usage during processing. The script uses Open3D for point cloud processing and visualization.

import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

def align_point_clouds(pcd1, pcd2, threshold=0.02):
    """
    Align two point clouds using ICP (Iterative Closest Point).
    """
    # Initial alignment (identity transformation)
    trans_init = np.eye(4)

    # Apply ICP to align pcd2 to pcd1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    # Apply the resulting transformation to align pcd2
    pcd2_aligned = pcd2.transform(reg_p2p.transformation)
    
    return pcd2_aligned, reg_p2p.transformation

def global_registration_ransac(source, target, voxel_size):
    """
    Perform RANSAC-based global registration between two point clouds.
    """
    # Extract FPFH (Fast Point Feature Histograms) features for both clouds
    def preprocess_point_cloud(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    # Preprocess both source and target
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Set RANSAC parameters
    distance_threshold = voxel_size * 1.5
    mutual_filter = False  # Set mutual_filter to False or True as required

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # Number of RANSAC iterations
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result_ransac.transformation

def chamfer_distance(pcd1, pcd2, chunk_size=10000):
    """
    Compute Chamfer Distance between two aligned point clouds using KDTree.
    """
    # Convert point clouds to numpy arrays
    pcd1_points = np.asarray(pcd1.points)
    pcd2_points = np.asarray(pcd2.points)
    
    # Build KD trees for nearest neighbor search
    kdtree_pcd1 = KDTree(pcd1_points)
    kdtree_pcd2 = KDTree(pcd2_points)
    
    # Process points in chunks to reduce memory usage
    def compute_chunked_min_distances(kdtree, points, chunk_size):
        min_distances = []
        for i in range(0, len(points), chunk_size):
            chunk = points[i:i + chunk_size]
            distances, _ = kdtree.query(chunk)
            min_distances.append(distances)
        return np.hstack(min_distances)
    
    # Compute the distances in both directions
    dists_pcd1_to_pcd2 = compute_chunked_min_distances(kdtree_pcd2, pcd1_points, chunk_size)
    dists_pcd2_to_pcd1 = compute_chunked_min_distances(kdtree_pcd1, pcd2_points, chunk_size)
    
    # Chamfer distance is the sum of both directions
    chamfer_dist = np.mean(dists_pcd1_to_pcd2) + np.mean(dists_pcd2_to_pcd1)
    
    return chamfer_dist

def visualize_point_clouds(pcd1, pcd2_aligned):
    """
    Visualize two point clouds: the original and the aligned.
    """
    # Color the point clouds for easier visualization
    pcd1.paint_uniform_color([1, 0.706, 0])  # Yellow for original
    pcd2_aligned.paint_uniform_color([0, 0.651, 0.929])  # Blue for aligned

    # Visualize
    o3d.visualization.draw_geometries([pcd1, pcd2_aligned], 
                                      zoom=0.7, 
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])
    

# Load and downsample two point clouds
pcd1 = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-17/A-1/m_point_cloud_02.ply")
# pcd2 = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-17/A-1_HD/m_point_cloud_02.ply")
pcd2 = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-17/A-1_2024-08-17_iphone/m_point_cloud_02.ply")
# Optionally downsample point clouds to reduce memory usage (adjust voxel size as needed)
pcd1 = pcd1.voxel_down_sample(voxel_size=0.005)  # Adjust voxel size as per requirement
pcd2 = pcd2.voxel_down_sample(voxel_size=0.005)

# Perform global registration using RANSAC
voxel_size = 0.05  # Adjust based on your point cloud scale
global_transformation = global_registration_ransac(pcd1, pcd2, voxel_size)

# Transform the second point cloud using the global transformation
pcd2_aligned_global = pcd2.transform(global_transformation.copy())

# Refine alignment using ICP after global registration
threshold = 0.05  # Set the ICP threshold for fine alignment
pcd2_aligned, transformation = align_point_clouds(pcd1, pcd2_aligned_global, threshold)

# Visualize the alignment result
visualize_point_clouds(pcd1, pcd2_aligned)

# Compute Chamfer Distance with reduced memory usage
cd = chamfer_distance(pcd1, pcd2_aligned, chunk_size=10000)  # Adjust chunk size based on available memory
print(f"Chamfer Distance: {cd}")
