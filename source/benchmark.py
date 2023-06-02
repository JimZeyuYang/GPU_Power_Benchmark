#!/usr/bin/env python3

from GPU_pwr_benchmark import GPU_pwr_benchmark

import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='GPU Power Benchmark')

    parser.add_argument('-b', '--benchmark', action='store_true', help='Runs the benchmark')
    parser.add_argument('-p', '--plot', action='store_true', help='Plots the results of the benchmark')
    parser.add_argument('-g', '--gpu', help='GPU model data to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Prints out the results of the benchmark')

    args = parser.parse_args()

    experiment(args)


def experiment(args):
    start = time.time()

    benchmark = GPU_pwr_benchmark(verbose=args.verbose)
    if args.benchmark:
        benchmark.prepare_experiment()
        benchmark.run_experiment()
        if args.plot:
            benchmark.plot_results()
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