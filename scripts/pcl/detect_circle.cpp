#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <chrono>


int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_point_cloud.ply>" << std::endl;
        return -1;
    }

    // Load the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile(argv[1], *cloud) == -1) {
        PCL_ERROR("Couldn't read the PLY file \n");
        return -1;
    }

    std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << argv[1] << std::endl;

    // Downsample the point cloud if necessary
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*cloud_filtered);

    std::cout << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

    // Set up the circle segmentation using RANSAC
    pcl::ModelCoefficients::Ptr coefficients_circle(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_circle(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CIRCLE2D);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    // Estimate the circle in the point cloud
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers_circle, *coefficients_circle);

    if (inliers_circle->indices.size() == 0) {
        PCL_ERROR("Could not estimate a circle model for the given dataset.");
        return -1;
    }

    std::cout << "Circle coefficients: " << coefficients_circle->values[0] << " "
              << coefficients_circle->values[1] << " "
              << coefficients_circle->values[2] << " "
              << coefficients_circle->values[3] << std::endl;

    // Visualize the point cloud with the detected circle
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud_filtered, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, cloud_color_handler, "cloud");

    // Draw the circle using the coefficients
    viewer->addCircle(*coefficients_circle, "circle");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
