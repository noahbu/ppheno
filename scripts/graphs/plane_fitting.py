import open3d as o3d
import numpy as np

# Function to load and convert point cloud to NumPy array
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return pcd, points

# Function to compute the centroid of the point cloud
def compute_centroid(points):
    return np.mean(points, axis=0)

# Function to perform PCA and compute the normal of the best-fitting plane
def compute_normal_vector(points):
    cov_matrix = np.cov(points.T)
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    normal_vector = eigenvectors[:, 0]
    return -normal_vector  # Invert to point upwards

# Function to visualize the normal vector as a line
def create_normal_line(centroid, normal_vector):
    line_start = centroid
    line_end = centroid + normal_vector * 1.0  # Scale for visualization
    line_points = np.linspace(line_start, line_end, 100)
    line_colors = np.tile([1.0, 0.0, 0.0], (100, 1))  # Red color

    normal_line_pcd = o3d.geometry.PointCloud()
    normal_line_pcd.points = o3d.utility.Vector3dVector(line_points)
    normal_line_pcd.colors = o3d.utility.Vector3dVector(line_colors)
    return normal_line_pcd

# Function to compute the rotation matrix to align the normal with the z-axis
def compute_rotation_matrix(normal_vector, centroid):
    target_axis = np.array([0.0, 0.0, 1.0])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    rotation_axis = np.cross(normal_vector, target_axis)

    if np.linalg.norm(rotation_axis) >= 1e-6:  # Rotation needed
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        theta = np.arccos(np.clip(np.dot(normal_vector, target_axis), -1.0, 1.0))
        axis_angle = rotation_axis * theta
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
        return rotation_matrix
    else:
        return np.eye(3)  # Identity matrix, no rotation needed

# Function to construct and print the 4x4 transformation matrix
def print_transformation_matrix(rotation_matrix, centroid):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = -centroid

    # Flatten the matrix and print the values in space-separated format
    for row in transformation_matrix:
        print(' '.join(map(str, row)))
    
    return transformation_matrix

# Function to apply translation to the origin and visualize the result
def transform_pcd(pcd, centroid, rotation_matrix):
    pcd.rotate(rotation_matrix, center=centroid)  # Apply rotation
    pcd.translate(-centroid)  # Translate to the origin

    return pcd

# function that takes a pointcloud as input, 
# computes the normal vector of the fitted plane and 
# returns the pointcloud with the normal vector aligned with z axis
def align_normal_with_z_axis(pcd):
    # Compute centroid and normal vector
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    normal_vector = compute_normal_vector(points)

    # Compute rotation matrix to align normal with the z-axis
    rotation_matrix = compute_rotation_matrix(normal_vector, centroid)

    # Translate the point cloud and visualize the result
    # pcd = transform_pcd(pcd, centroid, rotation_matrix)
    return rotation_matrix, centroid

# Main function
def main():
    # Load point cloud
    file_path = "/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/melonCycle/2024-08-02/B-1_2024-08-02/manual_ground_pc_B-1_2024-08-02_dense_02 - Cloud.ply"
    pcd, points = load_point_cloud(file_path)

    # # Compute centroid and normal vector
    # centroid = compute_centroid(points)
    # normal_vector = compute_normal_vector(points)

    # # Visualize the normal line (optional)
    # normal_line_pcd = create_normal_line(centroid, normal_vector)

    # # Compute rotation matrix to align normal with the z-axis
    # rotation_matrix = compute_rotation_matrix(normal_vector, centroid)

    rotation_matrix, centroid = align_normal_with_z_axis(pcd)

    # Print 4x4 transformation matrix
    print_transformation_matrix(rotation_matrix, centroid)

    # Translate the point cloud and visualize the result
    pcd = transform_pcd(pcd, centroid, rotation_matrix)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axes], mesh_show_back_face=True)

# Run the main function
if __name__ == "__main__":
    main()
