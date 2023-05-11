#! /bin/bash
#SBATCH --job-name=DLEI
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:1

#SBATCH --partition=test
#SBATCH --nodelist=htc-g019

source bin/env_setup.sh

./source/benchmark.py