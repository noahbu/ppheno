{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import open3d as o3d\n",
    "\n",
    "# Load your point cloud\n",
    "pcd = o3d.io.read_point_cloud(\"path/to/your/pointcloud.ply\")\n",
    "\n",
    "# Apply RANSAC to detect a plane in the point cloud\n",
    "plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,\n",
    "                                         ransac_n=3,\n",
    "                                         num_iterations=1000)\n",
    "\n",
    "# Extract the plane model parameters\n",
    "[a, b, c, d] = plane_model\n",
    "print(f\"Plane equation: {a}x + {b}y + {c}z + {d} = 0\")\n",
    "\n",
    "# Visualize the inliers (points fitting the plane)\n",
    "inlier_cloud = pcd.select_by_index(inliers)\n",
    "inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Color the plane red\n",
    "\n",
    "# Visualize the outliers (points not fitting the plane)\n",
    "outlier_cloud = pcd.select_by_index(inliers, invert=True)\n",
    "\n",
    "# Show the point cloud with the plane highlighted\n",
    "o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
