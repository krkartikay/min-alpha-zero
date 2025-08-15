# syntax=docker/dockerfile:1

ARG BASE_IMAGE=pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

ARG TORCH_DEVICE=cu118
ARG TORCH_VERSION=2.2.1

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip \
    libabsl-dev libboost-all-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cmake -S /usr/src/gtest -B /tmp/googletest-build \
    && cmake --build /tmp/googletest-build \
    && cmake --install /tmp/googletest-build

# Build Abseil from source to obtain logging libraries
RUN git clone --depth=1 https://github.com/abseil/abseil-cpp.git /tmp/abseil && \
    cmake -S /tmp/abseil -B /tmp/abseil/build \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/abseil/build --target install

# Download and install libtorch
RUN wget --progress=dot:giga https://download.pytorch.org/libtorch/${TORCH_DEVICE}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_DEVICE}.zip -O /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip

ENV TORCH_HOME=/opt/libtorch
ENV LD_LIBRARY_PATH=${TORCH_HOME}/lib:$LD_LIBRARY_PATH

WORKDIR /app
COPY . /app

# Placeholder model file required by CMake
RUN touch model.pt && \
    cmake --preset release && \
    cmake --build --preset release

# Usage:
#   docker build -t min-alpha-zero .
# For CPU-only build, use:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.2.1-cpu --build-arg TORCH_DEVICE=cpu -t min-alpha-zero .
