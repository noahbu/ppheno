import open3d as o3d
import numpy as np

def visualize_voxel_grid(pcd, voxel_size):
    """
    Visualize the voxelized version of a point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The size of each voxel.
    """
    # Create voxel grid from point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    # Visualize the voxel grid
    o3d.visualization.draw_geometries([voxel_grid])

# Load your point cloud
# pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-01/A-4_2024-08-01/m_pc_A-4_2024-08-01_dense_03.ply")
# pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-01/A-4_2024-08-01/pc_A-4_2024-08-01_dense_02_plant_cluster.ply")
pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-01/A-4_2024-08-01/pc_A-4_2024-08-01_dense_02_plant_only_cluster.ply")

# Estimate voxel size or set manually
voxel_size = 0.18  # Replace with your estimated voxel size

# Visualize voxelized point cloud
visualize_voxel_grid(pcd, voxel_size)
