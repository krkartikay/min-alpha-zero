cmake_minimum_required(VERSION 3.20)
project(min_alpha_zero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Set debug flags separately
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Required Libraries
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS system fiber context)
find_package(absl CONFIG REQUIRED)
find_package(GTest REQUIRED)

# Include dirs
include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${absl_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
)

# Main source files (excluding test)
set(SOURCES
  include/alpha_zero.h
  src/agents.cpp
  src/evaluator.cpp
  src/worker.cpp
  src/logging.cpp
)

# absl libraries variable
set(ABSL_LIBS
  absl::time
  absl::strings
  absl::str_format
  absl::flags
  absl::flags_parse
  absl::log
  absl::log_initialize
  absl::log_globals
  absl::log_flags
)

# Main binary
add_executable(alpha_zero src/main.cpp ${SOURCES})

target_link_libraries(alpha_zero
  ${ABSL_LIBS}
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(test_board src/test_board.cpp ${SOURCES})
target_link_libraries(test_board
  ${ABSL_LIBS}
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(model_eval src/model_eval.cpp ${SOURCES})

target_link_libraries(model_eval
  ${ABSL_LIBS}
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Test executable
add_executable(test_all test/test_all.cpp ${SOURCES})
target_link_libraries(test_all
  ${ABSL_LIBS}
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
  GTest::gtest
  GTest::gtest_main
)

# Copy model.pt to build directory
configure_file(${CMAKE_SOURCE_DIR}/model.pt
  ${CMAKE_BINARY_DIR}/model.pt
  COPYONLY)
