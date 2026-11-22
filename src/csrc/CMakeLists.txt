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
  time_integrators.cpp
  # burgers_equation.cpp
  # ...
)

add_library(numerical_pdes ${CSRC_SOURCES})

target_include_directories(numerical_pdes
  PUBLIC
  $(CMAKE_CURRENT_SOURCE_DIR)
)

target_link_libraries(numerical_pdes PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(numerical_pdes PRIVATE "${TORCH_CXX_FLAGS}")

add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE numerical_pdes)

# tests
enable_testing()
add_subdirectory(tests)
