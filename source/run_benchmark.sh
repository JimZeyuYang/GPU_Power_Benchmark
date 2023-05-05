#!/bin/bash

# throw a error if the number of input arguments is not 3
if [ $# -ne 3 ]; then
    echo "Error: invalid number of arguments"
    echo "Usage: $0 <test> <niter> <testLength>"
    echo "  test:       the length of the test in ms"
    echo "  niter:      the number of iterations to run"
    echo "  testLength: the length of the test in ms"
    exit 1
fi

# take 3 input arguments
test=$1
niter=$2
testLength=$3

echo test:       $test ms
echo niter:      $niter
echo testLength: $testLength

# Run the nvidia-smi command in the background
echo "Running nvidia-smi measurement in the background..."
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv,nounits -f results/gpudata.csv -lms 1 &
nvidia_pid=$!

# # Run your code here
echo "Running the GPU benchmark..."
./bin/benchmark_load $test $niter $testLength

# Kill the nvidia-smi process
kill $nvidia_pid