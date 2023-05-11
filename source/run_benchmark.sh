#!/bin/bash

# throw a error if the number of input arguments is not 3
if [ $# -ne 4 ]; then
    echo "Error: invalid number of arguments"
    echo "Usage: $0 <test> <niter> <testLength>"
    echo "  test:       the length of the test in ms"
    echo "  niter:      the number of iterations to run"
    echo "  testLength: the length of the test in ms"
    echo “  result dir: the directory to store the results”
    exit 1
fi

# take 3 input arguments
test=$1
niter=$2
testLength=$3
result_dir=$4

gpudata_dir=$result_dir/gpudata.csv


# echo test:       $test ms
# echo niter:      $niter
# echo testLength: $testLength

# Run the nvidia-smi command in the background
# echo "Running nvidia-smi measurement in the background..."
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv,nounits -f $gpudata_dir -lms 5 &
nvidia_pid=$!

# Run your code here
# echo "Running the GPU benchmark..."
./bin/benchmark_load $test $niter $testLength $result_dir

# Kill the nvidia-smi process
kill $nvidia_pid