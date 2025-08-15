# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG TORCH_VERSION=2.2.1
ARG TORCH_DEVICE=cu118

# Get dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip \
    libabsl-dev libboost-all-dev libgtest-dev

# Download libtorch zip and install it
RUN wget --progress=dot:giga https://download.pytorch.org/libtorch/${TORCH_DEVICE}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_DEVICE}.zip -O /tmp/libtorch.zip \
    && unzip /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip

# Build Abseil from source to obtain logging libraries
RUN git clone --depth=1 https://github.com/abseil/abseil-cpp.git /tmp/abseil && \
    cmake -S /tmp/abseil -B /tmp/abseil/build \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/abseil/build --target install

ENV TORCH_HOME=/opt/libtorch
ENV LD_LIBRARY_PATH=${TORCH_HOME}/lib:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:$PATH

WORKDIR /app
COPY . /app

RUN touch model.pt && \
    cmake --preset build && \
    cmake --build --preset build -j

CMD ["/bin/bash"]
