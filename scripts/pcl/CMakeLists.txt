cmake_minimum_required(VERSION 3.10)
project(DetectCircle)

# Find PCL package
find_package(PCL 1.14 REQUIRED)

# Find VTK package
find_package(VTK REQUIRED)

# Include PCL directories
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

# Include VTK directories
include(${VTK_USE_FILE})

# Add the executable
add_executable(detect_circle detect_circle.cpp)

# Link PCL and VTK libraries
target_link_libraries(detect_circle ${PCL_LIBRARIES} ${VTK_LIBRARIES})
