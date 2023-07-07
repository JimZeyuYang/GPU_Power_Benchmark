#!/usr/bin/env python3

from GPU_pwr_benchmark import GPU_pwr_benchmark

import argparse
import time
import os

def main():
    parser = argparse.ArgumentParser(description='GPU Power Benchmark')

    parser.add_argument('-b', '--benchmark', action='store_true', help='Runs the benchmark')
    parser.add_argument('-e', '--experiments', help='Specify the expriments to run')
    parser.add_argument('-p', '--plot', action='store_true', help='Plots the results of the benchmark')
    parser.add_argument('-g', '--gpu', help='GPU model data to process')
    parser.add_argument('-m', '--sw_meas', choices=['nvidia-smi', 'NVML', 'none'], default='nvidia-smi', help='Software to use for GPU power measurement')
    parser.add_argument('-v', '--verbose', action='store_true', help='Prints out the results of the benchmark')

    args = parser.parse_args()
    experiment(args)


def experiment(args):
    start = time.time()

    # check if the linux server hostname equals 'jim-linux', if yes, then use the PMD
    if os.uname()[1] == 'jim-linux': PMD = 1
    else:                            PMD = 0

    benchmark = GPU_pwr_benchmark(args.sw_meas, PMD, verbose=args.verbose)

    if args.benchmark:
        benchmark.prepare_experiment()

        if args.experiments is not None:
            experiments = [int(x) for x in args.experiments.split(',')]
        else:
            experiments = [1, 2]

        for experiment in experiments:  benchmark.run_experiment(experiment)
        if args.plot:  benchmark.process_results()

    elif args.plot:
        if args.gpu is None:
            raise Exception('GPU model not specified')
        else:
            gpu, run = args.gpu.split(',')
            benchmark.process_results(gpu, run)

    end = time.time()
    print(f'Time taken: {(end - start) // 60} minutes {round((end - start) % 60, 2)} seconds')



if __name__ == "__main__":
    main()