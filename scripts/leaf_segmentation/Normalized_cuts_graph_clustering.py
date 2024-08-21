import open3d as o3d
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from pathlib import Path

# Load the point cloud
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
data_folder = project_root / Path('data/melonCycle/2024-07-30/B-4/Downsampling')
point_cloud_file = data_folder / '1024_manual_cleaned.ply'
pcd = o3d.io.read_point_cloud(str(point_cloud_file))

# Extract points and normalize them
points = np.asarray(pcd.points)
scaler = StandardScaler()
points_normalized = scaler.fit_transform(points)

# Compute the affinity matrix using a Gaussian (RBF) kernel
sigma = 0.1  # Adjust the sigma value based on your data distribution
affinity_matrix = pairwise_kernels(points_normalized, metric='rbf', gamma=1/(2*sigma**2))

# Construct the graph Laplacian
laplacian_matrix = csgraph.laplacian(affinity_matrix, normed=True)

# Compute the first few eigenvectors (this step performs the normalized cut)
n_clusters = 5  # Adjust based on your needs
eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=n_clusters+1, which='SM')

# Discard the first eigenvector (corresponding to the smallest eigenvalue, which is close to 0)
embedding = eigenvectors[:, 1:n_clusters+1]

# Apply k-means clustering on the eigenvectors (embedding)
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
labels = kmeans.labels_

# Set up colors for different clusters
color_map = np.array([
    [1.0, 0.5, 0.0],  # Orange
    [0.0, 0.5, 1.0],  # Blue
    [0.0, 1.0, 0.0],  # Green
    [1.0, 0.0, 1.0],  # Magenta
    [1.0, 1.0, 0.0],  # Yellow
])

# Ensure the color map has enough colors for the number of clusters
if n_clusters > len(color_map):
    color_map = np.vstack([color_map, np.random.rand(n_clusters - len(color_map), 3)])

new_colors = np.zeros((points.shape[0], 3))
for i in range(n_clusters):
    new_colors[labels == i] = color_map[i % len(color_map)]

# Apply the colors to the point cloud
pcd.colors = o3d.utility.Vector3dVector(new_colors)

# Save the segmented point cloud
output_path = data_folder / f'normalized_cuts_segmentation_{n_clusters}_clusters.ply'
o3d.io.write_point_cloud(str(output_path), pcd)

# Visualize the segmented point cloud
o3d.visualization.draw_geometries([pcd])

print(f"Segmented point cloud saved to {output_path}")
