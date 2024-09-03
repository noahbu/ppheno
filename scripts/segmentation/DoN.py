import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys

def compute_normals(pcd, radius):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    return np.asarray(pcd.normals)

def difference_of_normals(pcd, small_radius, large_radius):
    normals_small = compute_normals(pcd, small_radius)
    normals_large = compute_normals(pcd, large_radius)
    
    # Check for differences in normals
    print(f"Sample normals (small scale): {normals_small[:5]}")
    print(f"Sample normals (large scale): {normals_large[:5]}")
    
    don_values = np.linalg.norm(normals_small - normals_large, axis=1)

    # Handle potential NaN or infinite values
    don_values = np.nan_to_num(don_values, nan=0.0, posinf=0.0, neginf=0.0)
    return don_values

def plot_don_histogram(don_values):
    plt.figure(figsize=(10, 6))
    plt.hist(don_values, bins=50, color='blue', alpha=0.7)
    plt.title("Histogram of DoN Values")
    plt.xlabel("DoN Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def visualize_normals(pcd, radius):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    o3d.visualization.draw_geometries([pcd], window_name=f'Normals with radius {radius}')

def compute_avg_distance(pcd, num_neighbors):
    distances = []
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    for point in pcd.points:
        [_, idx, _] = kdtree.search_knn_vector_3d(point, num_neighbors)
        if len(idx) > 1:
            dists = [np.linalg.norm(np.asarray(pcd.points)[i] - point) for i in idx[1:]]
            avg_dist = np.mean(dists)
            distances.append(avg_dist)
    
    return np.mean(distances)

def main():
    try:
        # Load the point cloud
        pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-01/A-1_2024-08-01/m_pc_A-1_2024-08-01_dense_03.ply")
        if pcd.is_empty():
            print("The point cloud is empty!")
            return
        
        # Downsample the point cloud
        voxel_size = 0.02  # Adjust the voxel size according to your needs
        pcd = pcd.voxel_down_sample(voxel_size)

        print(f"Number of points: {len(pcd.points)}")
        print(f"Bounding box: {pcd.get_axis_aligned_bounding_box()}")
        
        # Compute average distances for radii selection
        avg_dist_30 = compute_avg_distance(pcd, 30)
        avg_dist_100 = compute_avg_distance(pcd, 100)
        
        small_radius = 2 * avg_dist_30
        large_radius = 2 * avg_dist_100
        
        print(f"Computed small_radius: {small_radius}")
        print(f"Computed large_radius: {large_radius}")
        
        # Compute DoN
        don_values = difference_of_normals(pcd, small_radius, large_radius)
        
        # Plot histogram of DoN values
        plot_don_histogram(don_values)
        
    except KeyboardInterrupt:
        print("\nScript aborted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
