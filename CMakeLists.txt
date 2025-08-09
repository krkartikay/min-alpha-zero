cmake_minimum_required(VERSION 3.10)
project(min_alpha_zero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Required Libraries
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS system fiber context)
find_package(absl REQUIRED)
find_package(GTest REQUIRED)

# Include dirs
include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${absl_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
)

# Main source files (excluding test)
set(SOURCES
  alpha_zero.h
  evaluator.cpp
  worker.cpp
  logging.cpp
)

# Main binary
add_executable(alpha_zero main.cpp ${SOURCES})

# absl libraries variable
set(ABSL_LIBS
  absl::time
  absl::strings
  absl::str_format
  absl::flags
  absl::flags_parse
)

target_link_libraries(alpha_zero
  ${TORCH_LIBRARIES}
  ${ABSL_LIBS}
  ${Boost_LIBRARIES}
)

# Copy model.pt to build directory
configure_file(${CMAKE_SOURCE_DIR}/model.pt
    ${CMAKE_BINARY_DIR}/model.pt
    COPYONLY)
