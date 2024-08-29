import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from colorsys import rgb_to_hsv, hsv_to_rgb
from pathlib import Path


# Load your point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-08-01/gmm_clustered_hue')
point_cloud_file = data_folder / 'pc_A-4_2024-08-01_dense_plant_cluster.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Extract RGB values (assuming the point cloud has colors)
colors = np.asarray(pcd.colors)

# Convert RGB to HSV and extract the Hue component
hue_values = []
for color in colors:
    r, g, b = color
    h, s, v = rgb_to_hsv(r, g, b)
    hue_values.append(h * 360)  # Convert to degrees [0, 360]

hue_values = np.array(hue_values)

# Create the histogram
num_bins = 360
hist, bins = np.histogram(hue_values, bins=num_bins, range=(0, 360))

# Plot the histogram
plt.figure(figsize=(12, 6))

# Color each bin according to its hue
for i in range(num_bins):
    color = hsv_to_rgb(bins[i] / 360, 1, 1)  # Convert hue to RGB
    plt.bar(bins[i], hist[i], width=360/num_bins, color=color, edgecolor='black')

plt.title('Hue Histogram Colored by Hue Degree')
plt.xlabel('Hue (degrees)')
plt.ylabel('Frequency')
plt.xlim(0, 360)
plt.grid(True)
plt.show()
