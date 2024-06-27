# GPU Power Microbenchmark

Microbenchmark that unveals the mechanisms behind power readings reported by nvidia-smi/MVML on your NVIDIA GPU.

These includes power update frequency, transient response, and boxcar averaging window. Details can be found in the paper: "Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU's Built-in Power Sensor"

## Requirements
Make sure the system has NVIDIA Drivers installed for the NVIDIA GPUs. You can check via:
```bash
nvidia-smi
```
If not installed please visit https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html

Also make sure you have conda installed. All other packages and dependencies should be handled by conda using the provided yml file.

The benchmark should work on most Linux distributions. 

## Setup

Clone the repository, and build conda environment from the provided yml file.
```bash
git clone https://github.com/JimZeyuYang/GPU_Power_Benchmark.git
cd GPU_Power_Benchmark
conda env create -p ./env -f bin/project_env.yml
source activate ./env
```

## Usage

```bash
usage: benchmark.py [-h] [-b] [-e EXPERIMENTS] [-p] [-g GPU] [-v] [-i GPU_ID]

GPU Power Benchmark

options:
  -h, --help            Show this help message and exit
  -b, --benchmark       Runs the microbenchmarks
  -e EXPERIMENTS, --experiments EXPERIMENTS
                        Specify the expriments to run, by default runs all the microbenchmarks
  -p, --plot            Plots the results of the benchmark
  -g GPU, --gpu GPU     Select the name of the data to process, For plotting only
  -v, --verbose         Prints out verbose results of the benchmark
  -i GPU_ID, --gpu_id GPU_ID
                        Select which GPU to benchmark. Default is 0
  
```

## Example
To run the benchmark and process the results:
```bash
./source/benchmark.py -b -p
```
Results will be printed in command line, whereas plots could be found in the /results directory.

## Notes
Ensure no other processes are running on the system and uses the GPU for accurate benchmarking results.

Result processing is parallelised. Multi core CPUs will greatly speed up result processing.

On Ubuntu 22.04, to grant access to serial port (if PMD is present), run
```bash
  sudo apt remove brltty
  sudo usermod -aG dialout <username>
```


## Support
For questions and support please email zeyu.yang@eng.ox.ac.uk.

## License

[MIT](https://choosealicense.com/licenses/mit/)
