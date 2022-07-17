cmake_minimum_required(VERSION 3.12.0)
project(optimization_ex)

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")

# find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(Python COMPONENTS Development)
find_package(pybind11 REQUIRED)

include_directories(
    include
)

# add_executable(ceres_hello_world src/ceres_hello_world.cpp)
# target_link_libraries(ceres_hello_world Ceres::ceres)

# add_executable(ceres_rosenbrock src/ceres_rosenbrock_num_diff.cpp)
# target_link_libraries(ceres_rosenbrock Ceres::ceres)

add_executable(ex1 src/ex1.cpp)
target_link_libraries(ex1 Eigen3::Eigen pybind11::embed)

install(TARGETS ex1
RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)