import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load your cleaned point cloud
pcd = o3d.io.read_point_cloud("/Users/noahbucher/Documents_local/Plant_reconstruction/melonCycle/2024-07-30/A-1/point_cloud_edits/cleaned_A-1.ply")

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
