#!/bin/bash

GPU_NAME=$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | head -n 1)

if [ "$GPU_NAME" = "Tesla K40m" ]; then
    ARCH="-arch=sm_35"
elif [ "$GPU_NAME" = "Tesla K80" ]; then
    ARCH="-arch=sm_37"
elif [ "$GPU_NAME" = "Quadro K620" ]; then
    ARCH="-arch=sm_50"
elif [ "$GPU_NAME" = "NVIDIA GeForce GTX 745" ]; then
    ARCH="-arch=sm_50"
else
    echo "GPU $GPU_NAME is not supported, using default"
    ARCH=""
fi

echo "Compiling for $GPU_NAME with Compute Capability $ARCH"

make -C source/ clean

make ARCH=$ARCH -C source

# /tmp/benchmark_load 1 50,100000,100,100 . 0 0 0