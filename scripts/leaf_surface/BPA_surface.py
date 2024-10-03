import open3d as o3d
from pathlib import Path

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/leaf_area')
point_cloud_file = data_folder / 'musk_leaf/pc_muskLeaf_03_s.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Estimate normals if the point cloud doesn't have them already
calc_normals = False  # Set to False if normals already exist
if calc_normals:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Perform Ball Pivoting Algorithm
# Specify a list of radii for the ball pivoting process
radii = [0.005, 0.01, 0.02]  # Adjust based on your point cloud size and scale
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

# Visualize the resulting BPA mesh
o3d.visualization.draw_geometries([mesh], window_name="Ball Pivoting Surface Reconstruction")

# Calculate the surface area in square meters
surface_area_m2 = mesh.get_surface_area()

# Convert to square centimeters (1 m^2 = 10,000 cm^2)
surface_area_cm2 = surface_area_m2 * 10000
print(f"Surface Area of the Leaf: {surface_area_cm2:.2f} square centimeters")
