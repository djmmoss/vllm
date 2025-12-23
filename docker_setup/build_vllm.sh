#!/bin/bash
# Build vLLM from source and nsys-enabled images, then push to registry
#
# This script:
# 1. Builds vLLM from source using the Dockerfile in the vllm directory
# 2. Builds Docker images based on vllm/vllm-openai:nightly and the custom image
#    with nsys CLI installed
#
# Example of building vLLM from source on Nvidia GH200 server:
#   Memory usage: ~15GB
#   Build time: ~1475s / ~25 min
#   Image size: 6.93GB

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_DIR="$PROJECT_ROOT/vllm"

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker not found. Please install Docker first."
    exit 1
fi

# Step 1: Build vLLM from source
# Change to vLLM directory
pushd "$VLLM_DIR"

# Run use_existing_torch.py
python3 use_existing_torch.py

# Build the image
# Set VLLM_VERSION_OVERRIDE to avoid git version detection issues
# Set VLLM_USE_PRECOMPILED=1 to use precompiled binaries
DOCKER_BUILDKIT=1 docker build . \
    --file docker/Dockerfile \
    --target vllm-openai \
    --platform "linux/arm64" \
    -t "vllm-base" \
    --build-arg max_jobs=66 \
    --build-arg nvcc_threads=2 \
    --build-arg torch_cuda_arch_list="9.0 10.0+PTX" \
    --build-arg VLLM_USE_PRECOMPILED=0 \
    --build-arg VLLM_VERSION_OVERRIDE="0.11.0+custom" \
    --build-arg RUN_WHEEL_CHECK="false"

popd

# Step 2: Build nsys-enabled images

docker build -f Dockerfile.nsys \
    --build-arg BASE_IMAGE="vllm-base" \
    -t "vllm-gb200:custom" \
    "$SCRIPT_DIR"



