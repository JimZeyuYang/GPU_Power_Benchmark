#!/bin/bash

# throw a error if the number of input arguments is not 7
if [ $# -ne 7 ]; then
    echo "Error: invalid number of arguments"
    echo "Usage: $0 <experiment> <config> <result dir> <sampling pd>"
    echo "  experiment:              the experiment to run"
    echo "  config:                  the corresponding experiment configuration"
    echo “  result dir:              the directory to store the results”
    echo "  sampling pd:             the nvidia-smi/NVML sampling period in ms"
    echo "  nv-smi query options:    the nvidia-smi query options"
    echo "  Software measurement:    nvidia-smi, NVML, or None"
    echo "  PMD:                     if measuring power using PMD"

    exit 1
fi

# take 6 input arguments
experiment=$1
config=$2
result_dir=$3
sampling_pd=$4
nvsmi_query_options=$5
software_measurement=$6
pmd=$7

gpudata_dir=$result_dir/gpudata.csv

# echo Experiment:   $experiment
# echo Config:       $config
# echo Result dir:   $result_dir
# echo Sampling pd:  $sampling_pd

# check if software measurement ($5) equals to nvidia-smi
if [ $software_measurement == "nvidia-smi" ]; then
    # Run the nvidia-smi command in the background
    # echo "Running nvidia-smi measurement in the background..."
    nvidia-smi --id=0 $nvsmi_query_options --format=csv,nounits -f $gpudata_dir -lms $sampling_pd &
    nvidia_pid=$!

    # check if nvdia-smi started successfully
    if [ $? -ne 0 ]; then
        echo "Error: nvidia-smi failed to start"
        exit 1
    fi
fi

if [ $software_measurement == "NVML" ]; then
    NVML=1
else 
    NVML=0
fi

# echo "Running the GPU benchmark..."
/tmp/benchmark_load $experiment $config $result_dir $NVML $pmd

if [ $software_measurement == "nvidia-smi" ]; then
    # Kill the nvidia-smi process
    kill $nvidia_pid
fi