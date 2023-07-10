#! /bin/bash
#SBATCH --job-name=GPB
#SBACHT --nodes=1
#SBATCH --time=3:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:1

#SBATCH --partition=defq

source bin/env_setup.sh
module load GCC/11.3.0

./source/benchmark.py -b