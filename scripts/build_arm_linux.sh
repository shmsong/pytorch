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

# Build native sleef
SLEEF_NATIVE_BINARY_DIR=build/sleef/native
cmake -GNinja --DBUILD_QUAD=TRUE -S third_party/sleef -B ${SLEEF_NATIVE_BINARY_DIR}
cmake --build ${SLEEF_NATIVE_BINARY_DIR}
# Sleef cross compile assumes a Makefile in the native build but we want ninja, for sure
printf "all:\n\tninja\n">>${SLEEF_NATIVE_BINARY_DIR}/Makefile
echo "SLEEF DONE"
pwd
# Build native protoc
scripts/build_host_protoc.sh

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
CMAKE_ARGS+=("-DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=$CAFFE2_ROOT/build_host_protoc/bin/protoc")
CMAKE_ARGS+=("-DPROTOBUF_PROTOC_EXECUTABLE=$CAFFE2_ROOT/build_host_protoc/bin/protoc")

ARM_SYS_LIB=/usr/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu
PYTHON_LIB=$($PYTHON -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$ARM_SYS_LIB;$PACKAGE_INSTALL_DIR")

CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=$($PYTHON -c 'import sys; print(sys.executable)')")
CMAKE_ARGS+=("-DCOMPILER_WORKS_EXITCODE=0")
CMAKE_ARGS+=("-DCOMPILER_WORKS_EXITCODE__TRYRUN_OUTPUT=''")
CMAKE_ARGS+=("-DPYTHON_INCLUDE_DIR=$PACKAGE_INSTALL_DIR/python/include/python3.7m")
CMAKE_ARGS+=("-DPYTHON_LIBRARY=$PACKAGE_INSTALL_DIR/python/lib/python3.7")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]; then
  CMAKE_ARGS+=("-GNinja")
fi

export TORCH_CUDA_ARCH_LIST='7.2'
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda-10.2/targets/aarch64-linux
export CUDA_PATH=/usr/local/cuda-10.2/targets/aarch64-linux
export CUDA_LIB_PATH=/usr/local/cuda-10.2/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-10.2/bin/nvcc 
export MAGMA_HOME="$PACKAGE_INSTALL_DIR/magma"

CMAKE_ARGS+=("-DUSE_CUDA=ON")
CMAKE_ARGS+=("-DCUDA_64_BIT_DEVICE_CODE=ON")
CMAKE_ARGS+=("-DCUDA_VERSION=10.2")
CMAKE_ARGS+=("-DUSE_NCCL=OFF")
CMAKE_ARGS+=("-DUSE_DISTRIBUTED=OFF")
CMAKE_ARGS+=("-DUSE_QNNPACK=OFF")
CMAKE_ARGS+=("-DUSE_XNNPACK=OFF")
CMAKE_ARGS+=("-DUSE_PYTORCH_QNNPACK=OFF")
CMAKE_ARGS+=("-Wno-dev")
CMAKE_ARGS+=($@)

BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build"}

INSTALL_PREFIX=$CAFFE2_ROOT/torch
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT
echo "${CMAKE_ARGS[@]}"

cmake "$CAFFE2_ROOT" \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_ARGS[@]}"

echo "config complete"

if [[ "$BUILD_OPTION" != "config-only" ]]
then
  # build libs
  cmake --build . --target install --

  # build wheel
  cd ..
  export _PYTHON_HOST_PLATFORM=linux_aarch64
  export CROSS_ARCH=aarch64
  python setup.py bdist_wheel -p linux_aarch64
fi