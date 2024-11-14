# Use the BASE_TAG argument with a default value
ARG CUDA_VERSION=12.4.1
ARG OS_VERSION=ubuntu22.04
ARG BASE_TAG=${CUDA_VERSION}-cudnn-devel-${OS_VERSION}
ARG BASE_STAGE=cuda-only

FROM nvidia/cuda:${BASE_TAG} AS base

# Updating keys and installing dependencies
RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys "$(echo https://developer.download.nvidia.com/compute/cuda/repos/${OS_VERSION}/x86_64/3bf863cc.pub | sed 's/\\./_/g')" && \
    DEBIAN_FRONTEND=noninteractive apt update && apt install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    checkinstall \
    cmake \
    curl \
    doxygen \
    ffmpeg \
    g++ \
    gcc \
    gdb \
    gfortran \
    git \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libeigen3-dev \
    libflann-dev \
    libgdal-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libgtk-3-dev \
    libgtk2.0-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    liblapack-dev \
    libleptonica-dev \
    libpng-dev \
    libprotobuf-dev \
    libswscale-dev \
    libtbb-dev \
    libtbb2 \
    libtesseract-dev \
    libtiff-dev \
    libv4l-dev \
    nano \
    ninja-build \
    pkg-config \
    protobuf-compiler \
    python3-dev \
    python3-pip \
    python3-venv \
    unzip \
    vim \
    wget \
    yasm \
    zlib1g-dev && \
    pip3 install -U numpy==2.0.0 && \
    rm -rf /var/lib/apt/lists/*

# Allowing GUI stuff through X
RUN mkdir /tmp/runtime-root
ENV XDG_RUNTIME_DIR "/tmp/runtime-root"
RUN chmod -R 0700 /tmp/runtime-root
ENV NO_AT_BRIDGE 1

FROM base AS cuda-only
# OpenCV version argument
ARG OPENCV_VERSION=4.10.0
# Set up OpenCV build
WORKDIR /tmp
RUN wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip && unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip

RUN mkdir opencv-${OPENCV_VERSION}/build && \
    cd opencv-${OPENCV_VERSION}/build && \
    cmake -GNinja \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_CUDNN=OFF \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_NVCUVID=OFF \
      -D WITH_NVCUVENC=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_DOCS=OFF \
      -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules .. && \
    ninja -j1 && \
    ninja install && \
    cd /tmp && \
    rm -rf /tmp/* && \
    ldconfig

FROM base AS cuda-cudacodec

ARG VIDEO_CODEC_SDK=12.2.72
WORKDIR /tmp

# Copy SDK from build context
ADD ./Video_Codec_SDK_${VIDEO_CODEC_SDK}.zip /tmp/.
# Unzip and copy headers
RUN unzip Video_Codec_SDK_${VIDEO_CODEC_SDK}.zip && rm Video_Codec_SDK_${VIDEO_CODEC_SDK}.zip && \
    cp /tmp/Video_Codec_SDK_${VIDEO_CODEC_SDK}/Interface/* /usr/local/cuda/include/ && \
    cp /tmp/Video_Codec_SDK_${VIDEO_CODEC_SDK}/Lib/linux/stubs/x86_64/* /usr/lib/x86_64-linux-gnu/ && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/local/cuda/lib64/libnvcuvid.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/local/cuda/lib64/libnvidia-encode.so

# OpenCV version argument
ARG OPENCV_VERSION=4.10.0
# Set up OpenCV build
RUN wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip && unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip

RUN mkdir opencv-${OPENCV_VERSION}/build && \
    cd opencv-${OPENCV_VERSION}/build && \
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -DWITH_CUDA=ON \
        -DWITH_NVCUVID=ON \
        -DWITH_CUDNN=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_TBB=ON \
        -DENABLE_FAST_MATH=ON \
        -DCUDA_FAST_MATH=ON \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -DOPENCV_DNN_CUDA=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_apps=ON \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=ON \
        -DBUILD_DOCS=OFF .. && \
    ninja -j1 && \
    ninja install && \
    rm -rf /tmp/* && \
    ldconfig

FROM base AS cuda-cudacodec-nonfree
ARG VIDEO_CODEC_SDK=12.2.72
WORKDIR /tmp
# Copy SDK from build context
ADD ./Video_Codec_SDK_${VIDEO_CODEC_SDK}.zip /tmp/.
# Unzip and copy headers
RUN unzip Video_Codec_SDK_${VIDEO_CODEC_SDK}.zip && rm Video_Codec_SDK_${VIDEO_CODEC_SDK}.zip && \
    cp /tmp/Video_Codec_SDK_${VIDEO_CODEC_SDK}/Interface/* /usr/local/cuda/include/ && \
    cp /tmp/Video_Codec_SDK_${VIDEO_CODEC_SDK}/Lib/linux/stubs/x86_64/* /usr/lib/x86_64-linux-gnu/ && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/local/cuda/lib64/libnvcuvid.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/local/cuda/lib64/libnvidia-encode.so

# OpenCV version argument
ARG OPENCV_VERSION=4.10.0
# Set up OpenCV build
RUN wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip && unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
RUN mkdir opencv-${OPENCV_VERSION}/build && \
    cd opencv-${OPENCV_VERSION}/build && \
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -DWITH_CUDA=ON \
        -DWITH_NVCUVID=ON \
        -DWITH_CUDNN=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_TBB=ON \
        -DENABLE_FAST_MATH=ON \
        -DCUDA_FAST_MATH=ON \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -DOPENCV_DNN_CUDA=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_apps=ON \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=ON \
        -DBUILD_DOCS=OFF .. && \
    ninja -j1 && \
    ninja install && \
    rm -rf /tmp/* && \
    ldconfig

# Use ARG to set the base stage dynamically
FROM ${BASE_STAGE} AS underlay

# Shared dev stage based on the dynamic base stage
FROM underlay AS dev

# Allowing GUI stuff through X
RUN mkdir /tmp/runtime-root
ENV XDG_RUNTIME_DIR "/tmp/runtime-root"
RUN chmod -R 0700 /tmp/runtime-root
ENV NO_AT_BRIDGE 1

# Create a non-root user and switch to that user
# Add ARG for UID and GID
ARG USERNAME=ubuntu
ARG UID=1000
ARG GID=${UID}

# Create new user and home directory, with the same UID and GID as on the host
RUN groupadd --gid $GID $USERNAME \
 && mkdir -p /etc/sudoers.d \
 && mkdir -p /tmp/runtime-root \
 && useradd --uid ${GID} --gid ${UID} --create-home ${USERNAME} \
 && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME} \
 && mkdir -p /home/${USERNAME} \
 && chown -R ${UID}:${GID} /home/${USERNAME} \
 && chown -R ${UID}:${GID} /tmp/runtime-root

# Set the user and source entrypoint in the user's .bashrc file
USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN python3 -m venv --system-site-packages venv && \
 /bin/bash -c "source /home/${USERNAME}/venv/bin/activate && \
 python -m pip install -U \
 networkx \
 pandas \
 pyyaml \
 scikit-learn \
 scipy \
 shapely"
