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
find_package(GTest REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(external/abseil-cpp)
add_subdirectory(external/pybind11)

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

# Removed C++ executables - now using Python scripts in scripts/

# Test executable
add_executable(test_all test/test_all.cpp ${SOURCES})
target_link_libraries(test_all
  ${ABSL_LIBS}
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
  GTest::gtest
  GTest::gtest_main
)

# Python bindings
pybind11_add_module(min_alpha_zero src/python_bindings.cpp ${SOURCES})
target_link_libraries(min_alpha_zero PRIVATE
  ${ABSL_LIBS}
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Install the module to site-packages
execute_process(COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
install(TARGETS min_alpha_zero DESTINATION ${PYTHON_SITE_PACKAGES})

# Copy model.pt to build directory
configure_file(${CMAKE_SOURCE_DIR}/model.pt
  ${CMAKE_BINARY_DIR}/model.pt
  COPYONLY)
