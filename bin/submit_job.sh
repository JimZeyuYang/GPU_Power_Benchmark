#! /bin/bash
#SBATCH --job-name=GPB_A100
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:1 --constraint='gpu_sku:A100'

#SBATCH --partition=test

source bin/env_setup.sh

./source/benchmark.py