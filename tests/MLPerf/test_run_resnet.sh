#!/bin/bash

nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm --format=csv -lms 50 -f data.csv &
nvidia_pid=$!


start_time=$(date +%s%3N)

./resnet50.py

end_time=$(date +%s%3N)

kill $nvidia_pid


duration=$((end_time - start_time))

echo "Execution time: $duration ms"
