import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Get the current script's directory
script_dir = Path(__file__).parent.resolve()

# Construct the path to the root of the GitHub project
project_root = script_dir.parent.parent

# Construct the path to the /data folder
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4')

# Paths to the files
point_cloud_file = data_folder / 'filtered_point_cloud2.ply'

# Load your cleaned point cloud
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Convert to NumPy array
colors = np.asarray(pcd.colors)

# Plot histograms for each color channel
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.hist(colors[:, 0], bins=50, color='red', alpha=0.7)
plt.title('Red Channel')

plt.subplot(132)
plt.hist(colors[:, 1], bins=50, color='green', alpha=0.7)
plt.title('Green Channel')

plt.subplot(133)
plt.hist(colors[:, 2], bins=50, color='blue', alpha=0.7)
plt.title('Blue Channel')

plt.show()
