#!/bin/bash
# User-local CUDA toolchain (no sudo, post-SSD rebuild).
# Source before any CUDA work:  source scripts/cuda_env.sh
# Provides: nvcc 13.4 + CCCL headers + cublas stubs + Python dev headers
# for triton's JIT shim + the user-local Python stack.
SP="$HOME/.local/lib/python3.12/site-packages"
export PATH="$SP/nvidia/cu13/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages"
export CPATH="$SP/nvidia/cu13/include:$SP/cuda/cccl/headers/include:$HOME/.local/pydev/include/python3.12"
export LIBRARY_PATH="$HOME/.local/cuda-stubs/lib:$SP/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$HOME/.local/cuda-stubs/lib:$SP/nvidia/cu13/lib"
