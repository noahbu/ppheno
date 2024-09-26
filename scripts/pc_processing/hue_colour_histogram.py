import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from colorsys import rgb_to_hsv, hsv_to_rgb
from pathlib import Path

# Load your point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-08-01/A-1_2024-08-01')
point_cloud_file = data_folder / 'pc_A-1_2024-08-01_dense_02.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Extract RGB values (assuming the point cloud has colors)
colors = np.asarray(pcd.colors)

# Convert RGB to HSV and extract Hue, Saturation, and Value components
hue_values = []
saturation_values = []
value_values = []

for color in colors:
    r, g, b = color
    h, s, v = rgb_to_hsv(r, g, b)
    hue_values.append(h * 360)  # Convert hue to degrees [0, 360]
    saturation_values.append(s)
    value_values.append(v)

# Convert lists to numpy arrays
hue_values = np.array(hue_values)
saturation_values = np.array(saturation_values)
value_values = np.array(value_values)

# Create histograms for Hue, Saturation, and Value
num_bins = 360

# 1. Hue Histogram
plt.figure(figsize=(12, 6))
hist, bins = np.histogram(hue_values, bins=num_bins, range=(0, 360))

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

# 2. Saturation Histogram
plt.figure(figsize=(12, 6))
hist, bins = np.histogram(saturation_values, bins=100, range=(0, 1))  # 100 bins for saturation

# Plot the saturation histogram
plt.bar(bins[:-1], hist, width=1/100, color='blue', edgecolor='black')
plt.title('Saturation Histogram')
plt.xlabel('Saturation (0 to 1)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 3. Value (Brightness) Histogram
plt.figure(figsize=(12, 6))
hist, bins = np.histogram(value_values, bins=100, range=(0, 1))  # 100 bins for value

# Plot the value histogram
plt.bar(bins[:-1], hist, width=1/100, color='gray', edgecolor='black')
plt.title('Value (Brightness) Histogram')
plt.xlabel('Value (0 to 1)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
