#!/bin/bash
##############################################################################
# Example command to build the android target.
##############################################################################
#
# This script shows how one can build a Caffe2 binary for the Android platform
# using android-cmake. A few notes:
#
# (1) This build also does a host build for protobuf. You will need autoconf
#     to carry out this. If autoconf is not possible, you will need to provide
#     a pre-built protoc binary that is the same version as the protobuf
#     version under third_party.
#     If you are building on Mac, you might need to install autotool and
#     libtool. The easiest way is via homebrew:
#         brew install automake
#         brew install libtool
# (2) You will need to have android ndk installed. The current script assumes
#     that you set ANDROID_NDK to the location of ndk.
# (3) The toolchain and the build target platform can be specified with the
#     cmake arguments below. For more details, check out android-cmake's doc.

set -e

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"

if [ -z "$PYTHON" ]; then
  PYTHON=python
  PYTHON_VERSION_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info[0])')
  if [ "${PYTHON_VERSION_MAJOR}" -le 2 ]; then
    echo "Default python executable is Python-2, trying to use python3 alias"
    PYTHON=python3
  fi
fi


echo "Bash: $(/bin/bash --version | head -1)"
echo "Python: $($PYTHON -c 'import sys; print(sys.version)')"
echo "Caffe2 path: $CAFFE2_ROOT"

CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$CAFFE2_ROOT/cmake/cmake_aarch64.toolchain")

#CMAKE_ARGS+=("-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2")
#CMAKE_ARGS+=("-DCUDA_TOOLKIT_TARGET_DIR=/usr/local/cuda-10.2/targets/aarch64-linux/include")
CMAKE_ARGS+=("-DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=/usr/local/bin/protoc")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$($PYTHON -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')")
CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=$($PYTHON -c 'import sys; print(sys.executable)')")
CMAKE_ARGS+=("-DBUILD_CUSTOM_PROTOBUF=ON")
CMAKE_ARGS+=("-DCOMPILER_WORKS_EXITCODE=0")
CMAKE_ARGS+=("-DCOMPILER_WORKS_EXITCODE__TRYRUN_OUTPUT=''")
# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]; then
  CMAKE_ARGS+=("-GNinja")
fi

export TORCH_CUDA_ARCH_LIST='5.3;6.2;7.2'
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda-10.2/targets/aarch64-linux
export CUDA_PATH=/usr/local/cuda-10.2/targets/aarch64-linux
export CUDA_LIB_PATH=/usr/local/cuda-10.2/targets/aarch64-linux/lib/stubs
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-10.2/bin/nvcc 

ARM_CUDA_STUBS=/usr/local/cuda-10.2/targets/aarch64-linux/lib/stubs
export LD_LIBRARY_PATH=$ARM_CUDA_STUBS

#Manually add these lib dependencies for now

CMAKE_ARGS+=("-DUSE_CUDA=ON")
CMAKE_ARGS+=("-DCUDA_64_BIT_DEVICE_CODE=ON")
CMAKE_ARGS+=("-DCUDA_VERSION=10.2")
CMAKE_ARGS+=("-DUSE_NCCL=OFF")
CMAKE_ARGS+=("-DUSE_DISTRIBUTED=OFF")
CMAKE_ARGS+=("-DUSE_QNNPACK=OFF")
CMAKE_ARGS+=("-DUSE_PYTORCH_QNNPACK=OFF")


#CMAKE_ARGS+=("-DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so")
#CMAKE_ARGS+=("-DCUDA_INCLUDE_DIRS=$CUDA_HOME/include")
CMAKE_ARGS+=("-Wno-dev")
CMAKE_ARGS+=($@)

BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build_arm"}
INSTALL_PREFIX=${BUILD_ROOT}/install
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT
echo "${CMAKE_ARGS[@]}"


cmake "$CAFFE2_ROOT" \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_ARGS[@]}"

# # Cross-platform parallel build
# if [ -z "$MAX_JOBS" ]; then
#   if [ "$(uname)" == 'Darwin' ]; then
#     MAX_JOBS=$(sysctl -n hw.ncpu)
#   else
#     MAX_JOBS=$(nproc)
#   fi
# fi

# cmake --build . --target install -- "-j${MAX_JOBS}"
# echo "Installation completed, now you can copy the headers/libs from $INSTALL_PREFIX to your Android project directory."
