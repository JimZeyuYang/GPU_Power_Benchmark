#!/bin/bash

clear
make

# create a variable called test
# possible test durations:
# 5, 10, 20, 40, 50, 100, 200, 400
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <arg>"
    exit 1
fi

test=$1

# query the GPU name
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo $gpu_name

# if GPU name euqals to "NVIDIA A100-PCIE-40GB"
if [ "$gpu_name" = "NVIDIA A100-PCIE-40GB" ]; then
    if   [ "$test" = "10" ];  then matrix=4500
    elif [ "$test" = "20" ];  then matrix=5725
    elif [ "$test" = "50" ];  then matrix=7700
    elif [ "$test" = "100" ]; then matrix=9555
    elif [ "$test" = "200" ]; then matrix=12000
    elif [ "$test" = "400" ]; then matrix=15000
    else echo "No experiment Data stored for this GPU"
    fi

    let reps=4000/$test

elif [ "$gpu_name" = "NVIDIA V100-PCIE-16GB" ]; then
    if   [ "$test" = "10" ];  then matrix=3000
    elif [ "$test" = "20" ];  then matrix=5725
    elif [ "$test" = "50" ];  then matrix=7500
    elif [ "$test" = "100" ]; then matrix=9555
    elif [ "$test" = "200" ]; then matrix=12500
    elif [ "$test" = "400" ]; then matrix=15000
    else echo "No experiment Data stored for this GPU"
    fi
    let reps=800/$test
else
    echo "No experiment Data stored for this GPU"
fi

echo test:   $test ms
echo matrix: $matrix x $matrix
echo reps:   $reps


# Run the nvidia-smi command in the background
echo "Running nvidia-smi measurement in the background..."
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv,nounits -f gpudata.csv -lms 5 &
nvidia_pid=$!

# # Run your code here
echo "Running the GPU benchmark..."
./benchmark $test $matrix $reps

# Kill the nvidia-smi process
kill $nvidia_pid

# Plot the results
echo "Plotting the results..."
gpu_model=$(echo $gpu_name | cut -d ' ' -f 2)
python result_plotting.py --gpu $gpu_model --tl $test