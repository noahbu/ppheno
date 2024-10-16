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
    
    # Exclude the Day 4 file
    exclude_file = "processed_s_pc_C-3_2024-08-03_dense_02.ply"
    point_cloud_files = [f for f in point_cloud_files if f != exclude_file]

    # Use GridSpec for a 2x4 grid with specified height_ratios
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[0.3, 0.6], wspace=0.3)  # Added wspace=0.3 for column spacing
    
    all_x_vals = []
    
    # Keep track of the day number (start at 0)
    day_count = 0

    for idx, file_name in enumerate(point_cloud_files[:8]):  # Loop through 8 point clouds
        file_path = os.path.join(folder_path, file_name)
        pcd = o3d.io.read_point_cloud(file_path)

        # Downsample the point cloud
        pcd = pcd.voxel_down_sample(voxel_size=0.002)

        # Project to XZ plane
        projected_points, projected_colors = project_to_xz_plane(pcd)

        # Collect all X values for consistent scaling
        all_x_vals.extend(projected_points[:, 0])

        # Set up subplot in a 2x4 grid (idx // 4 for row, idx % 4 for column)
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        ax.scatter(projected_points[:, 0], projected_points[:, 1], s=0.3, c=projected_colors, alpha=0.5)

        # Set the title to reflect the day, skipping the excluded file's day
        if day_count == 4 or day_count ==1:
            day_count += 1  # Skip the missing day (Day 4)
        ax.set_title(f"Day {day_count}")  # Adjusted to start from Day 1
        day_count += 1  # Increment day count

        ax.set_xlabel('X (meter)')
        ax.set_ylabel('Z (meter)')

        # Format the Z-axis ticks to one decimal place
        ax.yaxis.set_major_formatter(FuncFormatter(format_z_ticks))

    # Set consistent X limits for all subplots
    x_lim = (min(all_x_vals), max(all_x_vals))

    # Apply different Z-axis limits based on row
    z_limits = [0.25, 0.6]  # 0.3 for the first row, 0.6 for the second row
    for row in range(2):  # Iterate over 2 rows
        for col in range(4):  # Iterate over 4 columns
            ax = fig.axes[row * 4 + col]
            ax.set_xlim(x_lim)  # Set consistent X-axis limits
            ax.set_ylim(0, z_limits[row])  # Apply Z limit based on the row

            # Enforce equal aspect ratio
            ax.set_aspect('equal')

            z_lim = z_limits[row]
            ticks = np.arange(0, z_lim + 0.001, 0.05)
            ax.set_yticks(ticks)

            if col != 0:
                ax.set_ylabel('')  # Remove Y-axis label for columns 2 to 4

            if row != 1:
                ax.set_xlabel('')  # Remove X-axis label for the first row

    # Optionally save the figure
    if output_file:
        plt.savefig(output_file, dpi=300)

    plt.show()

if __name__ == "__main__":
    folder_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/figures/pointcloud_time_series/C-3/rotated"
    visualize_xz_projections_grid(folder_path, output_file="4x2_pointcloud_projections_day_labels.png")
