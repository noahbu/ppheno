import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
import os

def project_to_xz_plane(pcd):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    mask = points[:, 2] >= 0
    projected_points = points[mask][:, [0, 2]]  # Take only X and Z coordinates
    projected_colors = colors[mask]
    
    return projected_points, projected_colors

def format_z_ticks(value, tick_number):
    return f'{value:.2f}'

def visualize_xz_projections_grid(folder_path, output_file=None):
    point_cloud_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".ply")])
    
    # Use GridSpec for better control over subplot layout
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.2, 0.4, 0.6])
    

    
    all_x_vals = []

    for idx, file_name in enumerate(point_cloud_files[:9]):
        file_path = os.path.join(folder_path, file_name)
        pcd = o3d.io.read_point_cloud(file_path)

        # Remove statistical outliers
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        # pcd = pcd.select_by_index(ind)

        # # Downsample the point cloud
        pcd = pcd.voxel_down_sample(voxel_size=0.003)

        # Project to XZ plane
        projected_points, projected_colors = project_to_xz_plane(pcd)

        # Collect all X values for consistent scaling
        all_x_vals.extend(projected_points[:, 0])

        # Set up subplot in a 3x3 grid
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.scatter(projected_points[:, 0], projected_points[:, 1], s=0.3, c=projected_colors, alpha=0.5)
        ax.set_title(f"Day {idx + 1}")
        ax.set_xlabel('X (meter)')
        ax.set_ylabel('Z (meter)')

        # Format the Z-axis ticks to one decimal place
        ax.yaxis.set_major_formatter(FuncFormatter(format_z_ticks))

    # Set consistent X limits for all subplots
    x_lim = (min(all_x_vals), max(all_x_vals))

    # Apply consistent Z-axis limits based on row
    z_limits = [0.2, 0.4, 0.6]
    for row in range(3):
        for col in range(3):
            ax = fig.axes[row * 3 + col]
            ax.set_xlim(x_lim)  # Set consistent X-axis limits
            ax.set_ylim(0, z_limits[row])  # Apply Z limit based on row
            # Set aspect ratio to be equal
            ax.set_aspect('equal')

            z_lim = z_limits[row]
            ticks = np.arange(0, z_lim + 0.001, 0.05)
            ax.set_yticks(ticks)

            if col != 0:
                ax.set_ylabel('')  # Remove Y-axis label for columns 2 and 3
        
            if row != 2:
                ax.set_xlabel('')  # Remove X-axis label for rows 1 and 2

    # Optionally save the figure
    if output_file:
        plt.savefig(output_file, dpi=300)

    plt.show()

if __name__ == "__main__":
    folder_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/pointcloud_time_series/C-3/rotated"
    visualize_xz_projections_grid(folder_path, output_file="3x3_pointcloud_projections_day_labels.png")
