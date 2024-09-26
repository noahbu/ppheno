import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from colorsys import rgb_to_hsv, hsv_to_rgb
from pathlib import Path

# Function to filter point cloud based on hue range
def filter_pointcloud_by_hue(pcd, min_hue, max_hue):
    # Extract RGB values (assuming the point cloud has colors)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)

    # Convert RGB to HSV and extract the Hue component
    hue_values = []
    for color in colors:
        r, g, b = color
        h, s, v = rgb_to_hsv(r, g, b)
        hue_values.append(h * 360)  # Convert to degrees [0, 360]

    hue_values = np.array(hue_values)

    # Filter points based on the hue range
    mask = (hue_values >= min_hue) & (hue_values <= max_hue)
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Create a new point cloud with the filtered points and colors
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd

# Function to filter point cloud based on hue, saturation, and value ranges
def filter_pointcloud_by_hsv(pcd, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value):
    # Extract RGB values (assuming the point cloud has colors)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)

    # Convert RGB to HSV and extract the components
    filtered_points = []
    filtered_colors = []
    
    for point, color in zip(points, colors):
        r, g, b = color
        h, s, v = rgb_to_hsv(r, g, b)  # Convert RGB to HSV
        h *= 360  # Convert hue to degrees [0, 360]

        # Filter by hue, saturation, and value ranges
        if min_hue <= h <= max_hue and min_saturation <= s <= max_saturation and min_value <= v <= max_value:
            filtered_points.append(point)
            filtered_colors.append(color)

    # Create a new point cloud with the filtered points and colors
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
    filtered_pcd.colors = o3d.utility.Vector3dVector(np.array(filtered_colors))

    return filtered_pcd

def density_filter(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Apply a statistical outlier removal filter to remove sparse points.

    Parameters:
    - pcd: The input point cloud (after color filtering).
    - nb_neighbors: The number of neighbors to analyze for each point.
    - std_ratio: The standard deviation multiplier to determine the threshold for filtering.

    Returns:
    - filtered_pcd: The filtered point cloud with outliers removed.
    """
    # Perform statistical outlier removal
    filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Optionally, visualize inliers and outliers
    print(f"Removed {len(pcd.points) - len(filtered_pcd.points)} outliers")

    # Return the filtered point cloud
    return filtered_pcd



# Load your point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-08-01/A-1_2024-08-01')
point_cloud_file = data_folder / 'pc_A-1_2024-08-01_dense_02.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Set hue, saturation, and value thresholds (tune these to remove white points)
min_hue = 70  # Minimum hue for green
max_hue = 100  # Maximum hue for green
min_saturation = 0.2  # Avoid points with very low saturation (near white)
max_saturation = 1.0  # Full saturation
min_value = 0.2  # Avoid very bright points (near white)
max_value = 0.8  # Allow up to medium brightness

o3d.visualization.draw_geometries([pcd])

# Filter the point cloud based on the specified hue range
filtered_pcd = filter_pointcloud_by_hsv(pcd, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value)

o3d.visualization.draw_geometries([filtered_pcd])

filtered_pcd = density_filter(filtered_pcd)

o3d.visualization.draw_geometries([filtered_pcd])

filtered_pcd = density_filter(filtered_pcd)


# # Save the filtered point cloud
# filtered_output_file = data_folder / 'pc_A-4_2024-08-01_dense_02_filtered.ply'
# o3d.io.write_point_cloud(str(filtered_output_file), filtered_pcd)

# print(f"Filtered point cloud saved to {filtered_output_file}")

# Optionally, create the histogram for hue visualization
# colors = np.asarray(pcd.colors)
# hue_values = []
# for color in colors:
#     r, g, b = color
#     h, s, v = rgb_to_hsv(r, g, b)
#     hue_values.append(h * 360)  # Convert to degrees [0, 360]

# hue_values = np.array(hue_values)

# # Create the histogram
# num_bins = 360
# hist, bins = np.histogram(hue_values, bins=num_bins, range=(0, 360))

# # Plot the histogram
# plt.figure(figsize=(12, 6))

# # Color each bin according to its hue
# for i in range(num_bins):
#     color = hsv_to_rgb(bins[i] / 360, 1, 1)  # Convert hue to RGB
#     plt.bar(bins[i], hist[i], width=360/num_bins, color=color, edgecolor='black')

# plt.title('Hue Histogram Colored by Hue Degree')
# plt.xlabel('Hue (degrees)')
# plt.ylabel('Frequency')
# plt.xlim(0, 360)
# plt.grid(True)
# plt.show()
