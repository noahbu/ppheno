import open3d as o3d
import numpy as np
from geomdl import BSpline
from geomdl.visualization import VisMPL
from geomdl import utilities
from scipy.spatial import Delaunay
import matplotlib.pyplot

# Path to the point cloud file
pointcloud_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/leaf_area/blue_leaf/pc_blueLeaf_03_s.ply"

# Step 1: Load point cloud using Open3D
pcd = o3d.io.read_point_cloud(pointcloud_path)

# Step 2: Uniformly downsample the point cloud to get evenly spaced points
downsampled_pcd = pcd.uniform_down_sample(every_k_points=10)  # Adjust this value for denser/sparser sampling

o3d.visualization.draw_geometries([downsampled_pcd])

points = np.asarray(downsampled_pcd.points)

# visualize points in 3D
matplotlib.pyplot.scatter(points[:, 0], points[:, 1], points[:, 2])
matplotlib.pyplot.show()

# Step 3: Select the number of control points in U and V directions (grid size)
num_u, num_v = 10, 10

# Step 4: Ensure enough points are available after downsampling
if points.shape[0] >= num_u * num_v:
    # Randomly sample evenly distributed points from the downsampled point cloud
    sampled_indices = np.random.choice(points.shape[0], num_u * num_v, replace=False)
    control_points = points[sampled_indices].tolist()

    # Step 5: Create a NURBS surface
    surf = BSpline.Surface()

    # Set the degrees of the surface (adjust the degree based on the desired smoothness)
    surf.degree_u = 3
    surf.degree_v = 3

    # Set control points for the NURBS surface
    surf.set_ctrlpts(control_points, num_u, num_v)

    # Step 6: Generate knot vectors
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, num_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, num_v)

    # Step 7: Evaluate the NURBS surface
    surf.evaluate()

    # Step 8: Visualize the NURBS surface in Matplotlib using VisMPL
    vis_comp = VisMPL.VisSurface()
    surf.vis = vis_comp

    # Render the surface
    surf.render(fig_size=[10, 10])

    # Step 9: Manually calculate the surface area by sampling the surface
    sample_points = []
    for u in np.linspace(0, 1, num_u):
        for v in np.linspace(0, 1, num_v):
            point = surf.evaluate_single((u, v))
            sample_points.append(point)

    sample_points = np.array(sample_points)

    # Use Delaunay triangulation to calculate areas of the triangles formed by the sample points
    tri = Delaunay(sample_points[:, :2])  # Triangulate based on the X and Y coordinates
    area = 0
    for simplex in tri.simplices:
        pts = sample_points[simplex]
        # Calculate area of each triangle using the cross product method
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        cross_prod = np.cross(v1, v2)
        triangle_area = 0.5 * np.linalg.norm(cross_prod)
        area += triangle_area

    print(f"Surface area of the NURBS surface (approx.): {area} square cm")
else:
    print("Not enough points to create a structured grid.")
