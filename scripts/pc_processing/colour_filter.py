import open3d as o3d
import numpy as np

# Load your point cloud
pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/point_cloud.ply")

# Convert to NumPy array
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Define color thresholds based on the histogram analysis
red_min, red_max = 0.08, 0.5
green_min, green_max = 0.2, 0.6
blue_min, blue_max = 0.1, 0.5

# Filter points based on color ranges
color_filter = (
    (colors[:, 0] >= red_min) & (colors[:, 0] <= red_max) &
    (colors[:, 1] >= green_min) & (colors[:, 1] <= green_max) &
    (colors[:, 2] >= blue_min) & (colors[:, 2] <= blue_max)
)

filtered_points = points[color_filter]
filtered_colors = colors[color_filter]

# Check if the filtering resulted in any points
if filtered_points.size == 0:
    raise ValueError("No points passed the color threshold. Please adjust the thresholds.")

# Create a new point cloud with filtered points and colors
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Save or visualize the new point cloud
o3d.visualization.draw_geometries([filtered_pcd])
