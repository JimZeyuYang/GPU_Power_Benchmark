#!/bin/bash

# throw a error if the number of input arguments is not 3
if [ $# -ne 4 ]; then
    echo "Error: invalid number of arguments"
    echo "Usage: $0 <experiment> <config> <result dir> <sampling pd>"
    echo "  experiment:     the experiment to run"
    echo "  config:         the corresponding experiment configuration"
    echo “  result dir:     the directory to store the results”
    echo "  sampling pd:    the nvidia-smi sampling period in ms"

    exit 1
fi

# take 3 input arguments
experiment=$1
config=$2
result_dir=$3
sampling_pd=$4

gpudata_dir=$result_dir/gpudata.csv

# echo Experiment:   $experiment
# echo Config:       $config
# echo Result dir:   $result_dir
# echo Sampling pd:  $sampling_pd


# Run the nvidia-smi command in the background
# echo "Running nvidia-smi measurement in the background..."
nvidia-smi --id=0 --query-gpu=timestamp,power.draw,utilization.gpu,pstate,temperature.gpu,clocks.current.smS --format=csv,nounits -f $gpudata_dir -lms $sampling_pd &
nvidia_pid=$!

# echo "Running the GPU benchmark..."
./bin/benchmark_load $experiment $config $result_dir

# Kill the nvidia-smi process
kill $nvidia_pid