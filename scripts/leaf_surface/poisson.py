#works, but need way to verify this is correct

import open3d as o3d
from pathlib import Path
import numpy as np

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4/Downsampling')
point_cloud_file = data_folder / 'single_leaf_4096.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))


# Estimate normals (required for Poisson reconstruction)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Perform Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Optionally, remove low-density vertices (noise)
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# Calculate the surface area
surface_area = mesh.get_surface_area()
print(f"Surface Area of the Leaf: {surface_area} square units")