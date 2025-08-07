cmake_minimum_required(VERSION 3.10)
project(min_alpha_zero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Required Libraries
find_package(Torch REQUIRED)
find_package(absl REQUIRED)
find_package(GTest REQUIRED)

# Include dirs
include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${absl_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
)

# Main source files (excluding test)
set(SOURCES
  evaluator.cpp
  selfplay.cpp
  worker.cpp
)

# Main binary
add_executable(min_alpha_zero main.cpp ${SOURCES})
target_link_libraries(min_alpha_zero
  ${TORCH_LIBRARIES}
  absl::time
  absl::strings
  absl::synchronization
)

# Test binary
add_executable(test_all test_all.cpp ${SOURCES})
target_link_libraries(test_all
  ${TORCH_LIBRARIES}
  GTest::gtest_main
  absl::time
  absl::strings
  absl::synchronization
)
