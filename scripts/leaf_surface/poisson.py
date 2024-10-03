import open3d as o3d
from pathlib import Path
import numpy as np

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/leaf_area')
point_cloud_file = data_folder / 'blue_leaf/pc_blueLeaf_03_s.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Set a flag to determine if normals should be calculated
calc_normals = False  # Set to False if normals are already present

if calc_normals:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Perform Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Convert densities to a numpy array for processing
densities = np.asarray(densities)

# Remove vertices with low density (e.g., below the 1% quantile)
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# Visualize the original point cloud and reconstructed mesh
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
o3d.visualization.draw_geometries([mesh], window_name="Poisson Surface Reconstruction")

# Calculate the surface area in square meters
surface_area_m2 = mesh.get_surface_area()

# Convert to square centimeters (1 m^2 = 10,000 cm^2)
surface_area_cm2 = surface_area_m2 * 10000

print(f"Surface Area of the Leaf: {surface_area_cm2:.2f} square centimeters")
