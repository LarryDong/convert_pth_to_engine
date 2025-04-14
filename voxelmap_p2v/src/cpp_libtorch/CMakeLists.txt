cmake_minimum_required(VERSION 3.0)
project(libtorch_test)

include_directories(/home/larry/codeGit/libtorch/include)


# Set LibTorch path (adjust to your installation)
set(Torch_DIR "/home/larry/codeGit/libtorch/share/cmake/Torch")  # e.g., /home/user/libtorch
find_package(Torch REQUIRED)

add_executable(libtorch_test main.cpp)
target_link_libraries(libtorch_test "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_test PROPERTY CXX_STANDARD 17)
