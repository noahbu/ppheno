import open3d as o3d
import numpy as np

# Enable WebRTC server
o3d.visualization.webrtc_server.enable_webrtc()

# Create a sample point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
pcd.colors = o3d.utility.Vector3dVector(np.random.rand(1000, 3))

# Visualize the point cloud
o3d.visualization.draw([pcd])
