cmake_minimum_required(VERSION 3.10)
project(tensorrt_ros)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA 11.8 REQUIRED)


include_directories(
  include
  /usr/include/eigen3
  ${CUDA_INCLUDE_DIRS}
  /home/server/xec/tensorrt-8.6.1/TensorRT-8.6.1.6/include    # tensorRT
)

link_directories(
    /home/server/xec/tensorrt-8.6.1/TensorRT-8.6.1.6/lib
    ${CUDA_LIBRARY_DIRS}
)

add_executable(inference ${CMAKE_CURRENT_SOURCE_DIR}/inference_whole.cpp)
target_link_libraries(inference ${catkin_LIBRARIES} nvinfer ${CUDA_LIBRARIES})