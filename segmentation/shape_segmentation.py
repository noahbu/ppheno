import open3d as o3d

# Enable WebRTC
o3d.visualization.webrtc_server.enable_webrtc()

# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/ubuntu/ppheno/data/MuskMelon_00/0121/pc_noiseremoved_downsampled.pcd")

# Compute normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Optionally orient normals
pcd.orient_normals_consistent_tangent_plane(k=30)

# Visualize the point cloud with normals
# Use draw_geometries with the proper parameters
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
