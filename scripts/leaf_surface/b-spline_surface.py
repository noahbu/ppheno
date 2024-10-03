# not working, got some errors
import open3d as o3d
import numpy as np
from scipy.interpolate import griddata
from geomdl import BSpline
from geomdl.visualization import VisMPL as vis
from geomdl.fitting import approximate_surface

# Step 1: Load Point Cloud
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    return pcd

# Step 2: Interpolate Points into a Grid
def create_grid(points, grid_size=50):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create grid points for interpolation
    grid_x, grid_y = np.mgrid[min(x):max(x):complex(grid_size), min(y):max(y):complex(grid_size)]
    
    # Interpolate z values on the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    
    # Ensure no NaN values are present in grid_z
    grid_z = np.nan_to_num(grid_z, nan=0.0)

    return grid_x, grid_y, grid_z

# Step 3: Fit B-Spline Surface to the Grid Points
def fit_b_spline_surface(grid_x, grid_y, grid_z, degree_u=3, degree_v=3, size_u=10, size_v=10):
    # Convert the grid_x, grid_y, and grid_z into a list of tuples (x, y, z)
    grid_points_list = [[(float(grid_x[i, j]), float(grid_y[i, j]), float(grid_z[i, j])) for j in range(grid_x.shape[1])] for i in range(grid_x.shape[0])]

    # Fit B-Spline surface
    surf = approximate_surface(grid_points_list, degree_u, degree_v, size_u, size_v)
    return surf

# Step 4: Visualize B-Spline Surface
def visualize_surface(surf):
    surf.vis = vis.VisSurface()
    surf.render()

# Step 5: Visualize Point Cloud
def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # File path to point cloud
    file_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/leaf_area/blue_leaf/som_output/pc_blueLeaf_03_s_8192_cleaned.ply"

    # Load point cloud
    pcd = load_point_cloud(file_path)

    # Visualize original point cloud
    visualize_point_cloud(pcd)

    # Extract points
    points = np.asarray(pcd.points)

    # Create grid from points
    grid_x, grid_y, grid_z = create_grid(points)

    # Fit B-Spline surface
    b_spline_surface = fit_b_spline_surface(grid_x, grid_y, grid_z, degree_u=3, degree_v=3, size_u=10, size_v=10)

    # Visualize B-Spline surface
    visualize_surface(b_spline_surface)
