cmake_minimum_required(VERSION 3.18)
project(numerical_pdes_csrc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(Torch) looks for TorchConfig.cmake under CMAKE_PREFIX_PATH
# (and default system prefixes). For a pip/uv-installed torch, configure with:
#   cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
find_package(Torch REQUIRED)

set(CSRC_SOURCES
  mesh.cpp
  boundary_conditions.cpp
  reconstruction.cpp
  equations.cpp
  # burgers_equation.cpp
  # ...
)

add_executable(demo demo.cpp ${CSRC_SOURCES})
target_link_libraries(demo PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(demo PRIVATE "${TORCH_CXX_FLAGS}")