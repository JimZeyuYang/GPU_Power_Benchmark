import subprocess
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import statistics
from functools import partial
import time
from datetime import datetime, timezone
import random
import os
from multiprocessing import Pool
import struct
from scipy.optimize import minimize
import csv
import json
import math

class GPU_pwr_benchmark:
    def __init__(self, sw_meas, gpu_id=0, PMD=0, verbose=False):
        print('_________________________')
        print('Initializing benchmark...')
        self.verbose = verbose
        self.repetitions = 32
        self.nvsmi_smp_pd = 5
        self.sw_meas = sw_meas
        self.gpu_id = gpu_id
        self.PMD = PMD
        self.aliasing_ratios = [1/2, 2/3, 4/5, 6/5, 4/3]
        self.pwr_draw_options = {
            'power.draw': False,
            'power.draw.average': False,
            'power.draw.instant': False
        }
        
    def prepare_experiment(self):
        self._get_machine_info()
        
        run = 0
        while os.path.exists(os.path.join('results', f'{self.gpu_name}_run_#{run}')): run += 1
        self.result_dir = os.path.join('results', f'{self.gpu_name}_run_#{run}')
        os.makedirs(self.result_dir)
        os.makedirs(os.path.join(self.result_dir, 'Preparation'))
        self.log_file = os.path.join(self.result_dir, 'log.txt')

        self._print_general_info()

        self.nvsmi_query_options = '--query-gpu=timestamp,utilization.gpu,pstate,temperature.gpu,clocks.current.sm,'
        for key, value in self.pwr_draw_options.items():
            if value: self.nvsmi_query_options += key + ','

        print('___________________________')
        print('Preparing for experiment...')
        self._log('___________________________')
        self._log('Preparing for experiment...')


        self.nvsmi_time = datetime.strptime(self.nvsmi_time, '%Y/%m/%d %H:%M:%S.%f')
        self.nvsmi_time = self.nvsmi_time.replace(tzinfo=timezone.utc).timestamp()
        diff_hr = round((self.epoch_time - self.nvsmi_time)/3600)
        with open(os.path.join(self.result_dir, 'Preparation', 'jet_lag.txt'), 'w') as f:
            f.write(str(diff_hr))

        if not os.path.exists('/tmp'): os.makedirs('/tmp')
        self._recompile_load()
        self._warm_up()

        self.scale_gradient, self.scale_intercept = self._find_scale_parameter()
        self.pwr_update_freq = self._find_pwr_update_freq()
        # self.scale_gradient, self.scale_intercept = 43877.8, -2322
        # self.pwr_update_freq = 100
        # self.jet_lag = -1
    
    def _log(self, message, end='\n'):
        with open(self.log_file, 'a') as f:
            f.write(message + end)

    def _print_general_info(self):
        print()
        time_ =            'Date and time:        ' + time.strftime('%d/%m/%Y %H:%M:%S')
        gpu =              'Benchmarking on GPU:  ' + self.gpu_name
        serial =           'GPU serial number:    ' + self.gpu_serial
        uuid =             'GPU UUID:             ' + self.gpu_uuid
        host =             'Host machine:         ' + os.uname()[1]
        cuda =             'CUDA version:         ' + self.nvcc_version
        driver =           'Driver version:       ' + self.driver_version
        pwr_draw_options = 'Power draw query options: ' 
        pwr_draw =         '  power.draw:         ' + str(self.pwr_draw_options['power.draw'])
        pwr_draw_avg =     '  power.draw.average: ' + str(self.pwr_draw_options['power.draw.average'])
        pwr_draw_inst =    '  power.draw.instant: ' + str(self.pwr_draw_options['power.draw.instant'])

        max_len = max(len(time_), len(gpu), len(host), len(serial), len(uuid), len(driver), len(cuda), len(pwr_draw_options), len(pwr_draw), len(pwr_draw_avg), len(pwr_draw_inst))
        output = ''
        output += '+ ' + '-'*(max_len) + ' +\n'
        output += '| ' + time_ + ' '*(max_len - len(time_)) + ' |\n'
        output += '| ' + gpu + ' '*(max_len - len(gpu)) + ' |\n'
        output += '| ' + serial + ' '*(max_len - len(serial)) + ' |\n'
        output += '| ' + uuid + ' '*(max_len - len(uuid)) + ' |\n'
        output += '| ' + host + ' '*(max_len - len(host)) + ' |\n'
        output += '| ' + cuda + ' '*(max_len - len(cuda)) + ' |\n'
        output += '| ' + driver + ' '*(max_len - len(driver)) + ' |\n'
        output += '| ' + pwr_draw_options + ' '*(max_len - len(pwr_draw_options)) + ' |\n'
        output += '| ' + pwr_draw + ' '*(max_len - len(pwr_draw)) + ' |\n'
        output += '| ' + pwr_draw_avg + ' '*(max_len - len(pwr_draw_avg)) + ' |\n'
        output += '| ' + pwr_draw_inst + ' '*(max_len - len(pwr_draw_inst)) + ' |\n'
        output += '+ ' + '-'*(max_len) + ' +\n'
        print(output)
        self._log(output)

    def _recompile_load(self):
        print('Recompiling benchmark load...')
        # make clean and make
        # try can catch the error
        return_code = subprocess.call(['./source/compile.sh'])
        if return_code != 0:  raise Exception('Error compiling benchmark')

        print()

    def _get_machine_info(self):
        def is_number(s):
            try:    float(s)
            except ValueError: return False
            return True

        self.epoch_time = time.time()
        result = subprocess.run(['nvidia-smi', f'--id={self.gpu_id}', '--query-gpu=timestamp,name,serial,uuid,driver_version', '--format=csv,noheader'], stdout=subprocess.PIPE)
        output = result.stdout.decode().split('\n')[0].split(', ')
        self.nvsmi_time, self.gpu_name, self.gpu_serial, self.gpu_uuid, self.driver_version = output
        self.gpu_name = self.gpu_name.replace(' ', '_')

        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            output = result.stdout
            nvcc_version = output.split('\n')[3].split(',')[1].strip()
            self.nvcc_version = nvcc_version
        except FileNotFoundError:
            print("CUDA is not installed or 'nvcc' command not found.")
            self.nvcc_version = 'N/A'

        # Check supported power draw query options
        output = subprocess.run(['nvidia-smi', '--help-query-gpu'], stdout=subprocess.PIPE)
        output = output.stdout.decode()
        query_options = '--query-gpu='
        for key, value in self.pwr_draw_options.items():
            if output.find(key) != -1:
                query_options += key + ','
                self.pwr_draw_options[key] = True

        output = subprocess.run(['nvidia-smi', f'--id={self.gpu_id}', query_options, '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        output = output.stdout.decode()[:-1].split(', ')

        for i, (key, value) in enumerate(self.pwr_draw_options.items()):
            if value:    self.pwr_draw_options[key] = is_number(output[i])

    def _run_benchmark(self, experiment, config, store_path, delay=True):
        '''
        Config for Experiment 1:
        <delay>,<niter>,<testlength>,<percentage>
            <delay>         : Length of idle time in ms
            <niter>         : Number of iterations to control the load time
            <testlength>    : Number of the square wave periods
            <percentage>    : Percentage of the SMs to be loaded
        '''
        subprocess.call([
                            './source/run_benchmark_load.sh', 
                            str(experiment), 
                            config,
                            store_path, 
                            str(self.nvsmi_smp_pd),
                            self.nvsmi_query_options,
                            self.sw_meas,
                            str(self.PMD), 
                            str(self.gpu_id)
                        ])
        if delay:  time.sleep(1)

    def _warm_up(self):
        print('Warming up GPU...                    ', end='', flush=True)
        self._log('Warming up GPU...                   ', end='')
        store_path = os.path.join(self.result_dir, 'Preparation', 'warm_up')
        os.makedirs(store_path)

        # While loop that run for at least 200 seconds
        start_time = time.time()
        while time.time() - start_time < 200:
            self._run_benchmark(1, '1,1000000,64,100', store_path)
        print('done')
        self._log('done')

    def _find_scale_parameter(self, store_path=None, percentage=100):
        print('Finding scale parameter...           ', end='', flush=True)
        self._log('Finding scale parameter...          ', end='')

        def f_duration(niter, store_path):
            config = f'50,{niter},30,{percentage}'
            self._run_benchmark(1, config, store_path)
            
            df = pd.read_csv(os.path.join(store_path, 'timestamps.csv'))
            df = df.drop_duplicates(subset=['timestamp'])
            df.reset_index(drop=True, inplace=True)
            df['diff'] = df['timestamp'].diff()
            df = df.iloc[1::2]
            df = df.iloc[10:]
            avg = df['diff'].mean()

            return avg / 1000

        def linear_regression(x, y):
            X = np.column_stack((np.ones(len(x)), x))
            coefficents = np.linalg.inv(X.T @ X) @ X.T @ y

            intercept = coefficents[0]
            gradient = coefficents[1]
            return intercept, gradient

        if store_path is None:
            store_path = os.path.join(self.result_dir, 'Preparation', 'find_scale_param')
        else:
            store_path = os.path.join(store_path, 'find_scale_param')
        os.makedirs(store_path)

        niter = 100000
        duration = 0

        duration_list = []
        niter_list = []

        while duration < 1000:
            if self.verbose: print(f'    {duration:.2f} ms')
            duration = f_duration(niter, store_path)
            if duration > 2:
                duration_list.append(duration)
                niter_list.append(niter)
            niter = int(niter * 2)

        # Linear regression
        intercept, gradient = linear_regression(duration_list, niter_list)
        print(f'{gradient:.2f} | {intercept:.2f}')
        self._log(f'{gradient:.2f} | {intercept:.2f}')

        # plot niter vs duration
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(duration_list, niter_list, 'o')
        # plot a line with gradient and intercept
        x = np.linspace(0, duration_list[-1], 100)
        y = gradient * x + intercept
        ax.plot(x, y, '--')
        ax.set_xlabel('Duration (ms)')
        ax.set_ylabel('Number of iterations')
        ax.set_title('Number of benchload iterations vs duration')
        ax.grid(True, linestyle='--', linewidth=0.5)

        plt.savefig(os.path.join(store_path, 'niter_vs_duration.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(store_path, 'niter_vs_duration.svg'), format='svg', bbox_inches='tight')
        plt.close('all')

        return gradient, intercept

    def _find_pwr_update_freq(self):
        print('Finding power update frequency...    ', end='', flush=True)
        self._log('Finding power update frequency...   ', end='')
        
        store_path = os.path.join(self.result_dir, 'Preparation', 'find_pwr_update_freq')
        os.makedirs(store_path)

        niters = int(10 * self.scale_gradient + self.scale_intercept)
        config = f'10,{niters},500,100'
        self._run_benchmark(1, config, store_path)

        df = pd.read_csv(os.path.join(store_path, 'gpudata.csv'))
        df['timestamp'] = (pd.to_datetime(df['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")

        period_list = []
        last_pwr = df.iloc[0]
        pwr_option = ' power.draw [W]'
        if self.pwr_draw_options['power.draw.instant']:    pwr_option = ' power.draw.instant [W]'
        for index, row in df.iterrows():
            if row[pwr_option] != last_pwr[pwr_option]:
                period_list.append(row['timestamp'] - last_pwr['timestamp'])
                last_pwr = row
        
        avg_period = statistics.mean(period_list)
        median_period = statistics.median(period_list)
        std_period = statistics.stdev(period_list)

        if self.verbose:
            print()
            print(f'avg period:    {avg_period:.2f} ms')
            print(f'median period: {median_period:.2f} ms')
            print(f'std period:    {std_period:.2f} ms')

        # plot the period_list as histogram
        fig, axis = plt.subplots(nrows=1, ncols=1)
        axis.hist(period_list, bins=20)
        axis.set_xlabel('Period (ms)')
        axis.set_ylabel('Frequency')
        axis.grid(True, linestyle='--', linewidth=0.5)

        fig.set_size_inches(8, 6)
        plt.savefig(os.path.join(store_path, 'power_update_freq.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(store_path, 'power_update_freq.svg'), format='svg', bbox_inches='tight')

        plt.close('all')

        print(f'{median_period:.2f} ms')
        self._log(f'{median_period:.2f} ms')

        if self.pwr_draw_options['power.draw.average']:
            same_rows_percentage = (df[' power.draw [W]'] == df[' power.draw.average [W]']).mean() * 100
            if same_rows_percentage > 90:
                print('  power.draw and power.draw.average are the same, treating power.draw as power.draw.average')
                self._log('  power.draw and power.draw.average are the same, treating power.draw as power.draw.average')
                self.pwr_draw_options['power.draw'] = False
        
        if self.pwr_draw_options['power.draw.instant']:
            same_rows_percentage = (df[' power.draw [W]'] == df[' power.draw.instant [W]']).mean() * 100
            if same_rows_percentage > 90:
                print('  power.draw and power.draw.instant are the same, treating power.draw as power.draw.instant')
                self._log('  power.draw and power.draw.instant are the same, treating power.draw as power.draw.instant')
                self.pwr_draw_options['power.draw'] = False

        with open(os.path.join(self.result_dir, 'Preparation', 'pwr_draw_options.txt'), 'w') as f:
            for key, value in self.pwr_draw_options.items():
                if value:    f.write(f'{key},')

        return median_period

    def run_experiment(self, experiment):
        if experiment == 1:
            # Experiment 1: Steady state and trasient response analysis
            print('_____________________________________________________________________')
            print('Running experiment 1: Steady state and transient response analysis...')
            self._log('_____________________________________________________________________')
            self._log('Running experiment 1: Steady state and transient response analysis...')

            os.makedirs(os.path.join(self.result_dir, 'Experiment_1'))
            for percentage in range(0, 101, 20):
                print(f'  Running experiment with {percentage}% load...')
                self._log(f'  Running experiment with {percentage}% load...')
                # create the store path
                percentage_store_path = os.path.join(self.result_dir, 'Experiment_1', f'{percentage}%_load')
                os.makedirs(percentage_store_path)
                # scale_gradient, scale_intercept = self._find_scale_parameter(percentage_store_path, percentage)

                for rep in range(int(self.repetitions/4)):
                    print(f'    Repetition {rep+1} of {int(self.repetitions/4)}...')
                    self._log(f'    Repetition {rep+1} of {int(self.repetitions/4)}...')
                    rep_store_path = os.path.join(percentage_store_path, f'rep_#{rep}')
                    os.makedirs(rep_store_path)

                    niters = int(6000 * self.scale_gradient + self.scale_intercept)
                    config = f'6000,{niters},1,{percentage}'
                    self._run_benchmark(1, config, rep_store_path, delay=False)
                    time.sleep(random.random())

        elif experiment == 2:
            # Experiment 2: Load with different aliasing ratios to find averaging window duration
            print('______________________________________________________________________')
            print('Running experiment 2: Finding averaging window duration by aliasing...')
            self._log('______________________________________________________________________')
            self._log('Running experiment 2: Finding averaging window duration by aliasing...')


            os.makedirs(os.path.join(self.result_dir, 'Experiment_2'))
            for ratio in self.aliasing_ratios:
                load_pd = int(ratio * self.pwr_update_freq)
                print(f'  Running experiment with load period of {load_pd} ms...')
                self._log(f'  Running experiment with load period of {load_pd} ms...')

                # create the store path
                ratio_store_path = os.path.join(self.result_dir, 'Experiment_2', f'load_{load_pd}_ms')    
                os.makedirs(ratio_store_path)
                            
                for rep in range(self.repetitions):
                    print(f'    Repetition {rep+1} of {self.repetitions}...')
                    self._log(f'    Repetition {rep+1} of {self.repetitions}...')

                    rep_store_path = os.path.join(ratio_store_path, f'rep_#{rep}')
                    os.makedirs(rep_store_path)
                    
                    niters = int(load_pd * self.scale_gradient + self.scale_intercept)
                    repetitions = int(4000 / load_pd)
                    config = f'{load_pd},{niters},{repetitions},100'
                    self._run_benchmark(1, config, rep_store_path, delay=False)
                    time.sleep(random.random())
        
        elif experiment == 3:
            # test the number of repetitions needed
            print('_________________________________________________________________________________')
            print('Running experiment 3: Finding the number of repetitions needed for convergence...')
            self._log('_________________________________________________________________________________')
            self._log('Running experiment 3: Finding the number of repetitions needed for convergence...')

            os.makedirs(os.path.join(self.result_dir, 'Experiment_3'))

            workload_length = [0.25, 1, 8]
            grains = [32, 8, 8]
            nums = [21, 19, 11]
            shifts = [1, 4, 8]

            for wl, grain, num in zip(workload_length, grains, nums):
                wl_length = int(self.pwr_update_freq * wl)
                print(f'  Running experiment with workload length of {wl_length} ms...')
                self._log(f'  Running experiment with workload length of {wl_length} ms...')
                
                # create the store path
                wl_store_path = os.path.join(self.result_dir, 'Experiment_3', f'workload_{wl}_pd')
                os.makedirs(wl_store_path)

                for shift in shifts:
                    print(f'    With shift of {shift}...')
                    self._log(f'    With shift of {shift}...')

                    sft_store_path = os.path.join(wl_store_path, f'shift_{shift}')
                    os.makedirs(sft_store_path)


                    rep = stride_gen(mode='lin', grain=grain)
                    for i in range(num):
                        next(iter(rep))
                        if shift > int(rep):  continue
                        if int(rep) % shift != 0: continue
                        print(f'      With number of repetitions of {rep} ({i+1} of {num})', end='', flush=True)
                        self._log(f'      With number of repetitions of {rep} ({i+1} of {num})...')

                        rep_store_path = os.path.join(sft_store_path, f'rep_{rep}')
                        os.makedirs(rep_store_path)

                        for iter_ in range(32):
                            print('.', end='', flush=True)
                        
                            iter_store_path = os.path.join(rep_store_path, f'iter_{iter_}')
                            os.makedirs(iter_store_path)

                            niters = int(wl_length/2 * self.scale_gradient + self.scale_intercept)
                            config = f'{wl_length/2},{niters},{rep},{shift}'
                            self._run_benchmark(3, config, iter_store_path, delay=False)

                            time.sleep(random.random())
                        
                        print('')

        elif experiment == 4:
            print('_____________________________________________________')
            print('Running experiment 4: Energy measurement using PMD...')
            self._log('_____________________________________________________')
            self._log('Running experiment 4: Energy measurement using PMD...')

            os.makedirs(os.path.join(self.result_dir, 'Experiment_4'))

            # create a dictionary of the test name and executable command
            tests_dict = {
                '0.25_period'   : {'reps' : 32,   'config' : f'{int(self.pwr_update_freq/8)},{int(self.pwr_update_freq/8 * self.scale_gradient + self.scale_intercept)},32,100'},
                '1.00_period'   : {'reps' : 8,    'config' : f'{int(self.pwr_update_freq/2)},{int(self.pwr_update_freq/2 * self.scale_gradient + self.scale_intercept)},8,100'},
                '8.00_period'   : {'reps' : 4,    'config' : f'{int(self.pwr_update_freq*4)},{int(self.pwr_update_freq*4 * self.scale_gradient + self.scale_intercept)},4,100'},
                'cublas_sgemm'  : {'reps' : 16,   'config' : 'tests/simpleCUBLAS/,./simpleCUBLAS'},
                'cufft'         : {'reps' : 32,   'config' : 'tests/simpleCUFFT/,./simpleCUFFT'},
                'nvJPEG'        : {'reps' : 8,    'config' : 'tests/nvJPEG/,./nvJPEG'},
                'quasi_rnd_gen' : {'reps' : 1000, 'config' : 'tests/quasirandomGenerator/,./quasirandomGenerator'},
                'stereo_disp'   : {'reps' : 64,   'config' : 'tests/stereoDisparity/,./stereoDisparity'},
                'black_scholes' : {'reps' : 512,  'config' : 'tests/BlackScholes/,./BlackScholes'},
                'resnet_50'     : {'reps' : 4,    'config' : 'tests/MLPerf/,./resnet50.py'},
                'retina_net'    : {'reps' : 16,   'config' : 'tests/MLPerf/,./retinanet.py'},
                'bert'          : {'reps' : 8,    'config' : 'tests/MLPerf/,./bert.py'},
            }

            with open(os.path.join(self.result_dir, 'Experiment_4', 'tests_dict.json'), 'w') as f: json.dump(tests_dict, f)

            # enum through the tests with the enumerae function
            for i, (test_name, value) in enumerate(tests_dict.items()):
                print(f'  Measuring energy for test: {test_name} ', end='', flush=True)

                if i < 3: exp = 1  
                else    : exp = 2

                # create a folder for the test
                test_store_path = os.path.join(self.result_dir, 'Experiment_4', test_name)
                os.makedirs(test_store_path)

                num_repetitions = 16
                for rep in range(num_repetitions):
                    print('.', end='', flush=True)
                    # create a folder for the repetition
                    rep_store_path = os.path.join(test_store_path, f'rep_{rep}')
                    os.makedirs(rep_store_path)

                    if rep == 0:  self._run_benchmark(exp, value['config'], rep_store_path, delay=False)
                    self._run_benchmark(exp, value['config'], rep_store_path, delay=False)
                    time.sleep(random.random())

                print(' Done!')

        elif experiment == 5:
            print('_____________________________________________________')
            print('Running experiment 5: Energy measurement using Nvidia-smi...')
            self._log('_____________________________________________________')
            self._log('Running experiment 5: Energy measurement using Nvidia-smi...')

            os.makedirs(os.path.join(self.result_dir, 'Experiment_5'))

            # create a dictionary of the test name and executable command
            tests_dict = {
                '0.25_period'   : {'reps' : 200,   'config' : f'{int(self.pwr_update_freq/8)},{int(self.pwr_update_freq/8 * self.scale_gradient + self.scale_intercept)},200,1'},
                '1.00_period'   : {'reps' : 50,    'config' : f'{int(self.pwr_update_freq/2)},{int(self.pwr_update_freq/2 * self.scale_gradient + self.scale_intercept)},50,1'},
                '8.00_period'   : {'reps' : 20,    'config' : f'{int(self.pwr_update_freq*4)},{int(self.pwr_update_freq*4 * self.scale_gradient + self.scale_intercept)},20,1'},
                'cublas_sgemm'  : {'reps' : 125,   'config' : 'tests/simpleCUBLAS/,./simpleCUBLAS'},
                'cufft'         : {'reps' : 194,   'config' : 'tests/simpleCUFFT/,./simpleCUFFT'},
                'nvJPEG'        : {'reps' : 20,    'config' : 'tests/nvJPEG/,./nvJPEG'},
                'quasi_rnd_gen' : {'reps' : 10184, 'config' : 'tests/quasirandomGenerator/,./quasirandomGenerator'},
                'stereo_disp'   : {'reps' : 426,   'config' : 'tests/stereoDisparity/,./stereoDisparity'},
                'black_scholes' : {'reps' : 2110,  'config' : 'tests/BlackScholes/,./BlackScholes'},
                'resnet_50'     : {'reps' : 20,    'config' : 'tests/MLPerf/,./resnet50.py'},
                'retina_net'    : {'reps' : 20,    'config' : 'tests/MLPerf/,./retinanet.py'},
                'bert'          : {'reps' : 20,    'config' : 'tests/MLPerf/,./bert.py'},
            }

            with open(os.path.join(self.result_dir, 'Experiment_4', 'tests_dict.json'), 'w') as f: json.dump(tests_dict, f)

            # enum through the tests with the enumerae function
            for i, (test_name, value) in enumerate(tests_dict.items()):
                print(f'  Measuring energy for test: {test_name} ', end='', flush=True)

                if i < 3: exp = 3
                else    : exp = 2

                # create a folder for the test
                test_store_path = os.path.join(self.result_dir, 'Experiment_4', test_name)
                os.makedirs(test_store_path)

                num_repetitions = 4
                for rep in range(num_repetitions):
                    print('.', end='', flush=True)
                    # create a folder for the repetition
                    rep_store_path = os.path.join(test_store_path, f'rep_{rep}')
                    os.makedirs(rep_store_path)

                    if rep == 0:  self._run_benchmark(exp, value['config'], rep_store_path, delay=False)
                    self._run_benchmark(exp, value['config'], rep_store_path, delay=False)
                    time.sleep(random.random())

                print(' Done!')

        else:
            raise ValueError(f'Invalid experiment number {experiment}')

    def process_results(self, GPU_name=None, run=0, exp='all'):
        print('_____________________')
        print('Processing results...')

        if GPU_name is not None:
            # retrive all file name from /results
            file_list = os.listdir('results')
            # get the file name that contains the GPU name
            file_list = [file for file in file_list if GPU_name in file]
            # if there is no file that contains the GPU name, return
            if len(file_list) == 0:
                raise FileNotFoundError(f'No file found for GPU {GPU_name}')
            # get the file name that ends with run number
            file_list = [file for file in file_list if file.endswith(f'_#{run}')]
            if len(file_list) == 0:
                raise FileNotFoundError(f'No file found for GPU {GPU_name} run #{run}')
            self.result_dir = os.path.join('results', file_list[0])
            self.gpu_name = file_list[0].split('_run_')[0]

        print(self.result_dir.split("/")[-1])
        with open(os.path.join(self.result_dir, 'Preparation', 'jet_lag.txt'), 'r') as f:  self.jet_lag = int(f.readline())
        print(f'Jet lag: {self.jet_lag} hours')

        try:
            with open(os.path.join(self.result_dir, 'Preparation', 'pwr_draw_options.txt'), 'r') as f:
                options = f.readline().split(',')[:-1]
                for option in options:
                    self.pwr_draw_options[option] = True
        except FileNotFoundError:
            print('No power draw options file found, using default options')
            self.pwr_draw_options['power.draw'] = True


        dir_list = os.listdir(self.result_dir)
        continue_ = True
        if exp == 'all' or exp == 1:    
            if 'Experiment_1' in dir_list:  continue_ = self.process_exp_1(os.path.join(self.result_dir, 'Experiment_1'))
        if exp == 'all' or exp == 2:
            if 'Experiment_2' in dir_list:  self.process_exp_2(os.path.join(self.result_dir, 'Experiment_2'))
        if exp == 'all' or exp == 3:
            if 'Experiment_3' in dir_list:  self.process_exp_3(os.path.join(self.result_dir, 'Experiment_3'))
        if exp == 'all' or exp == 4:
            if 'Experiment_4' in dir_list:  self.process_exp_4(os.path.join(self.result_dir, 'Experiment_4'))
        if exp == 'all' or exp == 5:
            if 'Experiment_5' in dir_list:  self.process_exp_5(os.path.join(self.result_dir, 'Experiment_5'))

        return continue_

    def _convert_pmd_data(self, dir):
        with open(os.path.join(dir, 'PMD_start_ts.txt'), 'r') as f:  pmd_start_ts = int(f.readline()) - 50000
        with open(os.path.join(dir, 'PMD_data.bin'), 'rb') as f:     pmd_data = f.read()

        bytes_per_sample = 18
        num_samples = int(len(pmd_data) / bytes_per_sample)

        # data_dict = {
        #     'timestamp (us)' : [],
        #     'pcie1_v': [], 'pcie1_i': [], 'pcie1_p': [],
        #     'pcie2_v': [], 'pcie2_i': [], 'pcie2_p': [],
        #     'eps1_v' : [], 'eps1_i' : [], 'eps1_p' : [],
        #     'eps2_v' : [], 'eps2_i' : [], 'eps2_p' : [],
        # }

        data_dict = {'timestamp': [], 'pcie_total_p': [], 'eps_total_p': []}

        last_timestamp = 0
        timestamp = 0
        for i in range(num_samples):
            sample = pmd_data[i*bytes_per_sample:(i+1)*bytes_per_sample]

            values = struct.unpack('<HHHHHHHHH', sample)
            values = list(values)

            timestamp += values[0] - last_timestamp
            if values[0] < last_timestamp:  timestamp += 65536
            last_timestamp = values[0]

            # enumerate through the values and convert to signed values, begin at index 1
            for i, value in enumerate(values[1:]):
                signed = struct.unpack('<h', struct.pack('<H', value))[0]
                signed = (signed >> 4) & 0x0FFF
                if signed & (1 << 11):  signed = 0
                values[i+1] = signed

            # populate the dictionary ###########################################
            data_dict['timestamp'].append(timestamp / 2933.5)
            data_dict['pcie_total_p'].append( values[1] * 0.007568 * values[2] * 0.0488
                                        + values[3] * 0.007568 * values[4] * 0.0488
                                        + values[7] * 0.007568 * values[8] * 0.0488)
            data_dict['eps_total_p'].append(  values[5] * 0.0075 * values[6] * 0.0488)

            # data_dict['timestamp (us)'].append(timestamp / 2.938)
            # data_dict['pcie1_v'].append(values[1] * 0.007568)
            # data_dict['pcie1_i'].append(values[2] * 0.0488)
            # data_dict['pcie1_p'].append(data_dict['pcie1_v'][-1] * data_dict['pcie1_i'][-1])
            # data_dict['pcie2_v'].append(values[3] * 0.007568)
            # data_dict['pcie2_i'].append(values[4] * 0.0488)
            # data_dict['pcie2_p'].append(data_dict['pcie2_v'][-1] * data_dict['pcie2_i'][-1])
            # data_dict['eps1_v'].append(values[5] * 0.007568)
            # data_dict['eps1_i'].append(values[6] * 0.0488)
            # data_dict['eps1_p'].append(data_dict['eps1_v'][-1] * data_dict['eps1_i'][-1])
            # data_dict['eps2_v'].append(values[7] * 0.007568)
            # data_dict['eps2_i'].append(values[8] * 0.0488)
            # data_dict['eps2_p'].append(data_dict['eps2_v'][-1] * data_dict['eps2_i'][-1])

        df = pd.DataFrame(data_dict)

        # df['timestamp (us)'] += (pmd_start_ts - df['timestamp (us)'][0])
        # df['timestamp'] = (df['timestamp (us)'] / 1000)
        df['timestamp'] += (pmd_start_ts/1000 - df['timestamp'][0])

        # df['pcie_total_p'] = df['pcie1_p'] + df['pcie2_p'] + df['eps2_p']
        # df['eps_total_p'] = df['eps1_p']
        df['total_p'] = df['pcie_total_p'] + df['eps_total_p']

        # df.drop(columns=['timestamp (us)', 'pcie1_v', 'pcie1_i', 'pcie1_p', 'pcie2_v', 'pcie2_i', 'pcie2_p', 'eps1_v', 'eps1_i', 'eps1_p', 'eps2_v', 'eps2_i', 'eps2_p'], axis=1, inplace=True)
        # df.drop(columns=['pcie_total_p'], axis=1, inplace=True)

        return df

    def process_exp_1(self, result_dir):
        def linear_regression(x, y):
            X = np.column_stack((np.ones(len(x)), x))
            coefficents = np.linalg.inv(X.T @ X) @ X.T @ y

            intercept = coefficents[0]
            gradient = coefficents[1]

            y_pred = [intercept + gradient * x for x in x]
            y_mean = np.mean(y)
            ss_tot = sum([(y_i - y_mean)**2 for y_i in y])
            ss_res = sum([(y_i - y_pred_i)**2 for y_i, y_pred_i in zip(y, y_pred)])
            r_squared = 1 - (ss_res / ss_tot)

            return intercept, gradient, r_squared

        


        print('  Processing experiment 1...')

        dir_list = os.listdir(result_dir)
        dir_list = [dir for dir in dir_list if os.path.isdir(os.path.join(result_dir, dir))]
        dir_list = sorted(dir_list, key=lambda dir: int(dir.split('%')[0]))
        subdir_list = os.listdir(os.path.join(result_dir, dir_list[0]))
        subdir_list = [dir for dir in subdir_list if 'find_scale_param' not in dir]
        subdir_list = sorted(subdir_list, key=lambda dir: int(dir.split('#')[1]))
        dir_list = [os.path.join(result_dir, dir, subdir) 
                        for dir in dir_list 
                            for subdir in subdir_list]

        self.found_PMD = False
        if os.path.exists(os.path.join(dir_list[0], 'PMD_data.bin')) and os.path.exists(os.path.join(dir_list[0], 'PMD_start_ts.txt')):
            self.found_PMD = True
        
        num_processes = min(self.repetitions, os.cpu_count())
        pool = Pool(processes=num_processes)
        results = pool.map(self._exp_1_plot_result, dir_list)
        pool.close()
        pool.join()

        results = list(map(list, zip(*results)))

        rise_times = []
        for key, value in self.pwr_draw_options.items():
            if value:
                rise_time = results.pop(0)
                rise_times.append(np.mean(rise_time))
                print(f'    {key} rise time: {np.mean(rise_time):.2f} ms')

                delay_time = results.pop(0)
                print(f'    {key} delay time: {np.mean(delay_time):.2f} ms')
        
        if self.found_PMD:
            for key, value in self.pwr_draw_options.items():
                if value:
                    pwr_pair = results.pop(0)
                    pwr_pair = list(map(list, zip(*pwr_pair)))

                    intercept, gradient, r_squared = linear_regression(pwr_pair[1], pwr_pair[0])

                    # avg error
                    avg_err = []
                    for nv_power, pmd_power in zip(pwr_pair[0], pwr_pair[1]):
                        avg_err.append((nv_power - pmd_power) / pmd_power * 100)

                    print(f'    {key} error gradient|intercept|R2|raw percentage(%): {gradient:.6f}|{intercept:.4f}|{r_squared:.4f}|{np.mean(avg_err):.4f}') 

                    # plot the points and the linear regression line
                    fig, ax = plt.subplots(nrows=1, ncols=1)

                    ax.plot(pwr_pair[1], pwr_pair[0], '+', markersize=12, label='Steady state power draw')
                    x = np.linspace(0, max(pwr_pair[1]), 100)
                    y = gradient * x + intercept
                    ax.plot(x, y, ':', label=f'Line of best fit ($R^2$ = {r_squared:.4f})')

                    ax.set_xlim(left=0)
                    ax.set_ylim(bottom=0)

                    ax.set_xlabel('Power draw from PMD (W)')
                    ax.set_ylabel('Power draw from nvidia-smi (W)')
                    # ax.set_title(f'{key} power draw comparison')
                    ax.grid(True, linestyle='--', linewidth=0.5)
                    ax.legend(loc='lower right')
                    fig.set_size_inches(7, 5)
                    plt.savefig(os.path.join(result_dir, f'{key}_power_draw_comparison.jpg'), format='jpg', dpi=256, bbox_inches='tight')
                    plt.savefig(os.path.join(result_dir, f'{key}_power_draw_comparison.eps'), format='eps', bbox_inches='tight')
                    plt.close('all')

        print('  Done')

        if len(rise_times) == 1 and rise_times[0] > 700:
            return False
        else:
            return True

    def _exp_1_plot_result(self, dir):
        def plot_PMD_data(dir, t0, t_max, power, axis):
            df = self._convert_pmd_data(dir)

            df['timestamp'] -= t0
            
            df = df[(df['timestamp'] >= 0) & (df['timestamp'] <= t_max)]
            
            axis[1].fill_between(df['timestamp'], df['total_p'], df['eps_total_p'], alpha=0.5, label='Rower draw from PCIE power cables')
            axis[1].fill_between(df['timestamp'], df['eps_total_p'], 0, alpha=0.5, label='Power draw from PCIE slot')
            
            result = []
            for key, value in self.pwr_draw_options.items():
                if value:
                    last_timestamp = 0
                    for index, row in power.iterrows():
                        df.loc[(df['timestamp'] > last_timestamp) & (df['timestamp'] <= row['timestamp']), 'nv_power'] = row[f' {key} [W]']
                        last_timestamp = row['timestamp']

                    # calculate the average of nv_power_error between timestamp 3000 and 6000
                    nv_pwr_mean = np.mean(df[(df['timestamp'] >= 3000) & (df['timestamp'] <= 6000)]['nv_power'])
                    PMD_pwr_mean = np.mean(df[(df['timestamp'] >= 3000) & (df['timestamp'] <= 6000)]['total_p'])
                    results.append([nv_pwr_mean, PMD_pwr_mean])

                    df['nv_power_error'] = df['nv_power'] - df['total_p']
                    mse = np.mean(df['nv_power_error']**2)
                    axis[2].fill_between(df['timestamp'], df['nv_power_error'], 0, alpha=0.5, label=f'{key} error, MSE={mse:.2f}')
            
            axis[2].set_xlabel('Time (ms)')
            axis[2].set_ylabel('Difference [W]')
            axis[2].grid(True, linestyle='--', linewidth=0.5)
            axis[2].set_xlim(axis[0].get_xlim())
            axis[2].legend()

            return result
            # END OF THE FUNCTION
        
        def find_rise_and_delay_time(power, option):
            reduced_power = power[(power['timestamp'] < 3000) & ((power['timestamp'] > 450))].copy()
            reduced_power = reduced_power[reduced_power[option] != reduced_power[option].shift()]
            reduced_power['pwr_diff'] = reduced_power[option].diff()
            # remove the power.draw column from the dataframe
            reduced_power.drop(columns=[' utilization.gpu [%]', ' pstate', ' temperature.gpu', ' clocks.current.sm [MHz]'], axis=1, inplace=True)
            reduced_power.reset_index(drop=True, inplace=True)

            start_ts, end_ts = 0, 0
            power_0, power_100 = 0, 0
            started = False
            for index, row in reduced_power.iterrows():
                if index == len(reduced_power) - 1:    break

                if started and row['pwr_diff'] < 1 and reduced_power['pwr_diff'].iloc[index+1] < 1:
                    power_100 = row[option]
                    break
                
                if not started and row['pwr_diff'] > 1 and reduced_power['pwr_diff'].iloc[index+1] > 1:
                    power_0 = row[option]
                    started = True


            power_10 = power_0 + (power_100 - power_0) * 0.1
            power_90 = power_0 + (power_100 - power_0) * 0.9

            start_ts = reduced_power[reduced_power[option] < power_10]['timestamp']
            if start_ts.empty:    start_ts = 500
            else:                 start_ts = start_ts.iloc[-1]
            end_ts = reduced_power[reduced_power[option] > power_90]['timestamp'].iloc[0]

            rise_time = end_ts - start_ts

            poewr_50 = power_0 + (power_100 - power_0) * 0.5
            # ts_50_greater = reduced_power[reduced_power[option] >= poewr_50]['timestamp'].iloc[0]
            # ts_50_smaller = reduced_power[reduced_power[option] <= poewr_50]['timestamp'].iloc[-1]
            # ts_50 = (ts_50_greater + ts_50_smaller) / 2
            ts_50 = reduced_power[reduced_power[option] <= poewr_50]['timestamp']
            if ts_50.empty:    ts_50 = 500
            else:              ts_50 = ts_50.iloc[-1]
            delay_time = ts_50 - 500

            return rise_time, delay_time

            # END OF THE FUNCTION
        load_percentage = dir.split('/')[-2].split('%')[0]

        load = pd.read_csv(os.path.join(dir, 'timestamps.csv'))
        load.loc[-1] = load.loc[0] - 500000
        load.index = load.index + 1
        load = load.sort_index()
        load['activity'] = (load.index / 2).astype(int) % 2
        load['timestamp'] = (load['timestamp'] / 1000).astype(int) 
        t0 = load['timestamp'][0]
        load['timestamp'] -= t0
        load.loc[load.index[-1], 'timestamp'] += 500
        load.loc[load.index[-1], 'activity'] = 0
        t_max = load['timestamp'].max()

        power = pd.read_csv(os.path.join(dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        power['timestamp'] += 60*60*1000 * self.jet_lag
        power['timestamp'] -= t0
        power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= t_max+10)]

        results = []
        for key, value in self.pwr_draw_options.items():
            if value:    
                rise_time, delay_time = find_rise_and_delay_time(power, f' {key} [W]')
                results.append(rise_time)
                results.append(delay_time)

        # plotting
        n_rows = 2
        if self.found_PMD:    n_rows = 3
        fig, axis = plt.subplots(nrows=n_rows, ncols=1)

        axis[0].plot(load['timestamp'], load['activity']*100, label='load')
        axis[0].plot(power['timestamp'], power[' utilization.gpu [%]'], label='utilization.gpu [%]')
        axis[0].plot(power['timestamp'], power[' temperature.gpu'], label='temperature.gpu')
        # plot this on the second y axis
        axis_0_1 = axis[0].twinx()
        axis_0_1.plot(power['timestamp'], power[' clocks.current.sm [MHz]'], label='clocks.current.sm [MHz]', color='red')
        axis_0_1.set_ylabel('Clock frequency [MHz]')
        axis_0_1.legend(loc='lower right')

        axis[0].set_xlabel('Time (ms)')
        axis[0].set_ylabel('Load [%] / Temperature [C]')
        axis[0].grid(True, linestyle='--', linewidth=0.5)
        axis[0].set_xlim(0, t_max)
        axis[0].legend()
        axis[0].set_title(f'{self.gpu_name} - {load_percentage}% load')

        # check if PMD_data.bin and PMD_start_ts.txt exist
        if self.found_PMD:    
            ss_errs = plot_PMD_data(dir, t0, t_max, power, axis)
            for err in ss_errs:    results.append(err)

        for key, value in self.pwr_draw_options.items():
            if value:    axis[1].plot(power['timestamp'], power[f' {key} [W]'], label=f'{key} [W]', linewidth=2)        
        axis[1].set_xlabel('Time (ms)')
        axis[1].set_ylabel('Power [W]')
        axis[1].grid(True, linestyle='--', linewidth=0.5)
        axis[1].set_xlim(axis[0].get_xlim())
        axis[1].legend()

        fig.set_size_inches(20, 4*n_rows)
        plt.savefig(os.path.join(dir, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(dir, 'result.svg'), format='svg', bbox_inches='tight')
        plt.close('all')

        return results

    def process_exp_2(self, result_dir):
        print('  Processing experiment 2...')

        if self.pwr_draw_options['power.draw']:    self.exp_2_pwr_option = ' power.draw [W]'
        else:                                      self.exp_2_pwr_option = ' power.draw.instant [W]'

        dir_list = os.listdir(result_dir)
        dir_list = [dir for dir in dir_list if dir.endswith('ms')]
        dir_list = sorted(dir_list, key=lambda dir: int(dir.split('_')[1]))
        
        test_pmd_path = os.path.join(result_dir, dir_list[0], 'rep_#0')
        self.found_PMD = False
        if os.path.exists(os.path.join(test_pmd_path, 'PMD_data.bin')) and os.path.exists(os.path.join(test_pmd_path, 'PMD_start_ts.txt')):
            print('    Found PMD data')
            self.found_PMD = True
            

        plotting_only = dir_list.pop(0)
        self._exp_2_plotting_only(os.path.join(result_dir, plotting_only))    
        
        
        print(f'    Found {len(dir_list)} directories to process...')

        self.pwr_update_freq = int(statistics.median([int(dir.split('_')[1]) for dir in dir_list]))
        #'''
        labels = []
        args = []
        for dir in dir_list:
            load_pd = int(dir.split('_')[1])

            ratio_store_path = os.path.join(result_dir, dir)
            labels.append(f'{load_pd}')

            # self.repetitions = 2
            for rep in range(self.repetitions):
                rep_store_path = os.path.join(ratio_store_path, f'rep_#{rep}')
                args.append((rep_store_path, load_pd))
            # break
            

        num_processes = min(len(args), os.cpu_count())
        # num_processes = 1
        print(f'    Running with {num_processes} processes', end='', flush=True)
        
        pool = Pool(processes=num_processes)
        results = pool.starmap(self._exp_2_process_single_run, args)
        pool.close()
        pool.join()
        print('')

        if not self.found_PMD:
            avg_windows, delays = zip(*results)
            avg_windows = [list(avg_windows)[i:i+self.repetitions] for i in range(0, len(list(avg_windows)), self.repetitions)]
            delays = [list(delays)[i:i+self.repetitions] for i in range(0, len(list(delays)), self.repetitions)]
            
            # store avg_window and delay in a csv file
            with open(os.path.join(result_dir, 'results.csv'), 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(labels)
                for window in avg_windows:    csvwriter.writerow(window)
                for delay  in delays:         csvwriter.writerow(delay)
        
        else:
            avg_windows, delays, PMD_avg_windows, PMD_delays, error = zip(*results)
            avg_windows = [list(avg_windows)[i:i+self.repetitions] for i in range(0, len(list(avg_windows)), self.repetitions)]
            delays = [list(delays)[i:i+self.repetitions] for i in range(0, len(list(delays)), self.repetitions)]
            PMD_avg_windows = [list(PMD_avg_windows)[i:i+self.repetitions] for i in range(0, len(list(PMD_avg_windows)), self.repetitions)]
            PMD_delays = [list(PMD_delays)[i:i+self.repetitions] for i in range(0, len(list(PMD_delays)), self.repetitions)]
            error = [list(error)[i:i+self.repetitions] for i in range(0, len(list(error)), self.repetitions)]

            # store avg_window and delay in a csv file
            with open(os.path.join(result_dir, 'results.csv'), 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(labels)
                for window in avg_windows:        csvwriter.writerow(window)
                for delay  in delays:             csvwriter.writerow(delay)
                for window in PMD_avg_windows:    csvwriter.writerow(window)
                for delay  in PMD_delays:         csvwriter.writerow(delay)
                for err    in error:              csvwriter.writerow(err)
        
        # '''
        self._exp_2_plot_results(result_dir)
        print('  Done')

    def _exp_2_plot_results(self, result_dir):
        # read the avg_window_results and delay_results from the csv file
        with open(os.path.join(result_dir, 'results.csv'), 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            rows = list(csv_reader)
            num_rows = len(rows)
            if not self.found_PMD:
                midpoint = num_rows//2 + 1
                labels = rows[0]
                avg_window_results = [[np.float64(value) for value in row] for row in rows[1:midpoint]]
                delay_results = [[np.float64(value) for value in row] for row in rows[midpoint:]]
            else:
                stride = num_rows//5
                labels = rows[0]
                avg_window_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[1:stride+1]]
                delay_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[stride+1:2*stride+1]]
                PMD_avg_window_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[2*stride+1:3*stride+1]]
                PMD_delay_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[3*stride+1:4*stride+1]]
                error_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[4*stride+1:]]
                
        
        avg_window_flat = [item for sublist in avg_window_results for item in sublist]
        avg_window_mean = statistics.mean(avg_window_flat)
        avg_window_median = statistics.median(avg_window_flat)
        avg_window_std = statistics.stdev(avg_window_flat)

        delay_flat = [item for sublist in delay_results for item in sublist]
        delay_mean = statistics.mean(delay_flat)
        delay_median = statistics.median(delay_flat)
        delay_std = statistics.stdev(delay_flat)

        labels.append('All')
        avg_window_results.append(avg_window_flat)
        delay_results.append(delay_flat)        

        print( '  Overall:')
        print( '            Delay           Avg window')
        print(f'    Mean:   {delay_mean:.2f} ms         {avg_window_mean:.2f} ms')
        print(f'    Median: {delay_median:.2f} ms         {avg_window_median:.2f} ms')
        print(f'    Std:    {delay_std:.2f} ms         {avg_window_std:.2f} ms')        
        
        if self.found_PMD:
            PMD_avg_window_flat = [item for sublist in PMD_avg_window_results for item in sublist]
            PMD_avg_window_mean = statistics.mean(PMD_avg_window_flat)
            PMD_avg_window_median = statistics.median(PMD_avg_window_flat)
            PMD_avg_window_std = statistics.stdev(PMD_avg_window_flat)

            PMD_delay_flat = [item for sublist in PMD_delay_results for item in sublist]
            PMD_delay_mean = statistics.mean(PMD_delay_flat)
            PMD_delay_median = statistics.median(PMD_delay_flat)
            PMD_delay_std = statistics.stdev(PMD_delay_flat)

            error_flat = [item for sublist in error_results for item in sublist]
            error_mean = statistics.mean(error_flat)
            error_median = statistics.median(error_flat)
            error_std = statistics.stdev(error_flat)

            PMD_avg_window_results.append(PMD_avg_window_flat)
            PMD_delay_results.append(PMD_delay_flat)
            error_results.append(error_flat)

            print( '            PMD Delay       PMD Avg window  Error')
            print(f'    Mean:   {PMD_delay_mean:.2f} ms         {PMD_avg_window_mean:.2f} ms         {error_mean:.2f}%')
            print(f'    Median: {PMD_delay_median:.2f} ms         {PMD_avg_window_median:.2f} ms         {error_median:.2f}%')
            print(f'    Std:    {PMD_delay_std:.2f} ms         {PMD_avg_window_std:.2f} ms         {error_std:.2f}%')


        # plot the results
        if self.found_PMD:    n_rows = 5
        else:           n_rows = 2

        fig, axis = plt.subplots(nrows=n_rows, ncols=1)
        self._violin_plot(axis[0], avg_window_results, labels)
        axis[0].set_xlabel('Load period (ms)')
        axis[0].set_ylabel('Averaging windod (ms)')
        axis[0].set_title(f'Averaging windod vs Load Period ({self.gpu_name})')

        self._violin_plot(axis[1], delay_results, labels)
        axis[1].set_xlabel('Load period (ms)')
        axis[1].set_ylabel('Delay (ms)')
        axis[1].set_title(f'Delay vs Load Period ({self.gpu_name})')

        if self.found_PMD:
            self._violin_plot(axis[2], PMD_avg_window_results, labels)
            axis[2].set_xlabel('Load period (ms)')
            axis[2].set_ylabel('Averaging windod (ms)')
            axis[2].set_title(f'PMD Averaging windod vs Load Period ({self.gpu_name})')

            self._violin_plot(axis[3], PMD_delay_results, labels)
            axis[3].set_xlabel('Load period (ms)')
            axis[3].set_ylabel('Delay (ms)')
            axis[3].set_title(f'PMD Delay vs Load Period ({self.gpu_name})')

            self._violin_plot(axis[4], error_results, labels)
            axis[4].set_xlabel('Load period (ms)')
            axis[4].set_ylabel('Error (%)')
            axis[4].set_title(f'Error of Energy, nvidia-smi compared to PMD')

            

        # add some texts at the bottom of the plot
        # axis[1].text(0.05, -0.15, f'Mean averaging window:   {avg_window_mean:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        # axis[1].text(0.05, -0.2,  f'Median averaging window: {avg_window_median:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        # axis[1].text(0.05, -0.25, f'Std averaging window:    {avg_window_std:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        # axis[1].text(0.65,  -0.15, f'Mean delay:   {delay_mean:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        # axis[1].text(0.65,  -0.2,  f'Median delay: {delay_median:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        # axis[1].text(0.65,  -0.25, f'Std delay:    {delay_std:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})

        fig.set_size_inches(12, 5*n_rows)
        fname = f'history_length_{self.gpu_name}'
        plt.savefig(os.path.join(result_dir, fname+'.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, fname+'.svg'), format='svg', bbox_inches='tight')
        plt.close('all')

    def _violin_plot(self, ax, data, labels):
        def adjacent_values(vals, q1, q3):
            IQR = q3 - q1
            upper_adjacent_value = q3 + IQR * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - IQR * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value

        def set_axis_style(ax, labels):
            ax.get_xaxis().set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, ha='center')
            ax.set_xlim(0.25, len(labels) + 0.75)

        # sort the data
        data = [sorted(row) for row in data]

        parts = ax.violinplot(
            data, showmeans=False, showmedians=False,
            showextrema=False)

        for pc in parts['bodies']:
            pc.set_facecolor('#76b900')
            pc.set_edgecolor('#76b900')
            pc.set_alpha(1)
        # set the last violin to be a different color
        parts['bodies'][-1].set_facecolor('#ED1C24')
        parts['bodies'][-1].set_edgecolor('#ED1C24')

        percentiles = np.stack([np.percentile(row, [25, 50, 75]) for row in data], axis=0)
        quartile1, medians, quartile3 = percentiles[:, 0], percentiles[:, 1], percentiles[:, 2]

        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=12, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

        set_axis_style(ax, labels)
        ax.grid(True, linestyle='--', linewidth=0.5)

    def _exp_2_process_single_run(self, result_dir, load_period):
        # Load data
        load = pd.read_csv(os.path.join(result_dir, 'timestamps.csv'))
        load.loc[-1] = load.loc[0] - 500000
        load.index = load.index + 1
        load = load.sort_index()
        load['activity'] = (load.index / 2).astype(int) % 2
        load['timestamp'] = (load['timestamp'] / 1000).astype(int) 
        t0 = load['timestamp'][0]
        load['timestamp'] -= t0
        load.loc[load.index[-1], 'timestamp'] += 500
        load.loc[load.index[-1], 'activity'] = 0
        load.set_index('timestamp', inplace=True)
        load.sort_index(inplace=True)

        # nv_smi power data
        power = pd.read_csv(os.path.join(result_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        power['timestamp'] += 60*60*1000 * self.jet_lag
        power['timestamp'] -= t0
        power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= load.index[-1])]
        power.set_index('timestamp', inplace=True)
        power.sort_index(inplace=True)

        # Optimization
        reduced_power = power[power[self.exp_2_pwr_option] != power[self.exp_2_pwr_option].shift()].copy()
        reduced_power.drop(columns=[' utilization.gpu [%]', ' pstate', ' temperature.gpu', ' clocks.current.sm [MHz]'], axis=1, inplace=True)
        loss_func = partial(self._reconstr_loss, load, reduced_power)

        init_vars = [self.pwr_update_freq/2, self.nvsmi_smp_pd/2]
        avg_window, delay = minimize(loss_func, init_vars, method='Nelder-Mead', 
                                    options={'maxiter': 1000, 'xatol': 1e-3, 'disp': self.verbose}).x

        if self.verbose:    print(f'modeled avg_window: {avg_window:.2f}ms, delay: {delay:.2f}ms')
        reconstructed = self._reconstruction(load, reduced_power, avg_window, delay)

        # PMD data
        PMD_data, PMD_avg_window, PMD_delay, PMD_reconstructed, error , error_msg = None, 0, 0, None, 0, None
        if self.found_PMD:
            PMD_data = self._convert_pmd_data(result_dir)
            PMD_data['timestamp'] -= t0
            PMD_data = PMD_data[(PMD_data['timestamp'] >= 0) & (PMD_data['timestamp'] <= load.iloc[-1].name)]
            PMD_data.set_index('timestamp', inplace=True)
            PMD_data.sort_index(inplace=True)

            PMD_cp = PMD_data[['total_p']].copy()
            PMD_cp.rename(columns={'total_p': 'activity'}, inplace=True)
            loss_func = partial(self._reconstr_loss, PMD_cp, reduced_power)

            PMD_avg_window, PMD_delay = minimize(loss_func, init_vars, method='Nelder-Mead',
                                        options={'maxiter': 1000, 'xatol': 1e-3, 'disp': self.verbose}).x

            if self.verbose:    print(f'modeled avg_window: {PMD_avg_window:.2f}ms, delay: {PMD_delay:.2f}ms')
            PMD_reconstructed = self._reconstruction(PMD_cp, reduced_power, PMD_avg_window, PMD_delay)

            # compare what you get in energy when integrating the power signal
            nv_energy = np.trapz(power[self.exp_2_pwr_option], power.index) / 1000
            pmd_energy = np.trapz(PMD_cp['activity'], PMD_cp.index) / 1000
            error = (nv_energy - pmd_energy) / pmd_energy * 100
            error_msg = f'nv_energy: {nv_energy:.2f} J, pmd_energy: {pmd_energy:.2f} J, diff: {nv_energy - pmd_energy:.2f} J, error: {error:.2f}%'
            if self.verbose:    print(error_msg)
            if abs(error) > 15:      print(f'error is too high: {error:.2f}%, check dir {result_dir}')
        
        # Plot resontruction result
        self._plot_reconstr_result(load_period, load, power, avg_window, delay, reconstructed, result_dir,
                                                PMD_data, PMD_avg_window, PMD_delay, PMD_reconstructed, error_msg)

        # Plot the loss function if the first repetition
        if result_dir.split('/')[-1].split('_')[-1] == '#0':
            avg_windows = [i for i in range(1, self.pwr_update_freq+10, 2)]
            losses = []
            loss_func = partial(self._reconstr_loss, load, reduced_power)
            for window in avg_windows:
                loss = loss_func([window, 0])
                losses.append(loss)
            
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(avg_windows, losses, label='Loss from nvidia_smi')

            if self.found_PMD:
                loss_func = partial(self._reconstr_loss, PMD_cp, reduced_power)
                pmd_losses = []
                for window in avg_windows:
                    loss = loss_func([window, 0])
                    pmd_losses.append(loss)

                ax.plot(avg_windows, pmd_losses, label='Loss from PMD')


            ax.set_xlabel('avg_window (ms)')
            ax.set_ylabel('loss')
            ax.set_title('Loss function')
            ax.legend()
            fig.set_size_inches(8, 6)
            plt.savefig(os.path.join(result_dir, 'loss.jpg'), format='jpg', dpi=256, bbox_inches='tight')
            plt.savefig(os.path.join(result_dir, 'loss.svg'), format='svg', bbox_inches='tight')
            plt.close(fig)
                            
        print('.', end='', flush=True)
        if abs(error) > 15:  return avg_window, delay, np.nan, np.nan, np.nan
        if self.found_PMD:    return avg_window, delay, PMD_avg_window, PMD_delay, error
        else:           return avg_window, delay

    def _reconstr_loss(self, load, power, variables, loss_type='MSE'):
        if variables[0] < 0 or variables[1] < 0:    return 100

        reconstructed = self._reconstruction(load, power, variables[0], variables[1])

        # discard the beginning 20% and the end 10% of the data
        power = power.iloc[int(len(power)*0.2):int(len(power)*0.9)]
        reconstructed = reconstructed.iloc[int(len(reconstructed)*0.2):int(len(reconstructed)*0.9)]

        # norm the 2 signals before calculating the loss
        power_norm = (power[self.exp_2_pwr_option] - power[self.exp_2_pwr_option].mean()) / power[self.exp_2_pwr_option].std()
        reconstructed_norm = (reconstructed[self.exp_2_pwr_option] - reconstructed[self.exp_2_pwr_option].mean()) / reconstructed[self.exp_2_pwr_option].std()

        if loss_type == 'MSE':
            return np.mean((power_norm - reconstructed_norm)**2)
        elif loss_type == 'MAE':
            return np.mean(np.abs(power[self.exp_2_pwr_option] - reconstructed[self.exp_2_pwr_option]))
        elif loss_type == 'EMSE':
            return np.sqrt(np.mean((power[self.exp_2_pwr_option] - reconstructed[self.exp_2_pwr_option])**2))
        else:
            raise Exception('Invalid loss type')    

    def _reconstruction(self, load, power, history_length, delay=0):
        reconstructed = power.copy()

        for idx, row in reconstructed.iterrows():
            if idx <= 500:
                # find the first row in load that is after the current timestamp
                after = load[load.index >= idx].iloc[0]
                before = load[load.index <= idx]
                if before.empty:    reconstructed.loc[idx, self.exp_2_pwr_option] = after['activity']
                else:               reconstructed.loc[idx, self.exp_2_pwr_option] = (after['activity'] + before.iloc[-1]['activity']) / 2
                
            else:
                # find rows in load df that are within the past history_length of the current timestamp
                load_window = load[(load.index >= idx - history_length - delay) & (load.index < idx - delay)].copy()

                # interpolate the lower bound of the load window
                lb_t = idx - history_length - delay
                lb_0 = load[load.index < lb_t].iloc[-1]
                lb_1 = load[load.index >= lb_t]
                if lb_1.empty:    lb_p = lb_0['activity']
                else:
                    lb_1 = lb_1.iloc[0]
                    gradient = (lb_1['activity'] - lb_0['activity']) / (lb_1.name - lb_0.name)
                    lb_p = lb_0['activity'] + gradient * (lb_t - lb_0.name)

                # interpolate the upper bound of the load window
                ub_t = idx - delay
                ub_0 = load[load.index < ub_t].iloc[-1]
                ub_1 = load[load.index >= ub_t]
                if ub_1.empty:    ub_p = ub_0['activity']
                else:
                    ub_1 = ub_1.iloc[0]
                    gradient = (ub_1['activity'] - ub_0['activity']) / (ub_1.name - ub_0.name)
                    ub_p = ub_0['activity'] + gradient * (ub_t - ub_0.name)
                
                # take the average of the load window
                t = np.concatenate((np.array([lb_t]), load_window.index.to_numpy(), np.array([ub_t])))
                p = np.concatenate((np.array([lb_p]), load_window['activity'].to_numpy(), np.array([ub_p])))
                reconstr_pwr = np.trapz(p, t) / history_length
                
                reconstructed.loc[idx, self.exp_2_pwr_option] = reconstr_pwr

        return reconstructed

    def _plot_reconstr_result(self, load_period, load, power, avg_window, delay, reconstructed, store_path,
                                PMD_data, PMD_avg_window, PMD_delay, PMD_reconstructed, error_msg):

        
        if self.found_PMD:    n_rows = 4
        else:           n_rows = 3
        # Plot the results
        fig, axis = plt.subplots(nrows=n_rows, ncols=1)

        axis[0].plot(load.index, load['activity']*100, label='load')
        axis[0].plot(power.index, power[' utilization.gpu [%]'], label='utilization.gpu [%]')
        axis[0].set_xlabel('Time (ms)')
        axis[0].set_ylabel('Load')
        axis[0].grid(True, linestyle='--', linewidth=0.5)
        axis[0].set_xlim(0, load.index.max())
        axis[0].legend(loc='upper right')
        axis[0].set_title(f'{self.gpu_name} - {load_period} ms load window')
        
        axis[1].plot(power.index, power[self.exp_2_pwr_option], label='Power draw')
        axis[1].set_xlabel('Time (ms)')
        axis[1].set_ylabel('Power [W]')
        axis[1].grid(True, linestyle='--', linewidth=0.5)
        axis[1].set_xlim(axis[0].get_xlim())
        axis[1].legend(loc='lower center')

        reconstructed = reconstructed.loc[reconstructed.index.repeat(2)]
        reconstructed[self.exp_2_pwr_option] = reconstructed[self.exp_2_pwr_option].shift()
        reconstructed = reconstructed.dropna()

        axis[2].plot(reconstructed.index, reconstructed[self.exp_2_pwr_option], label='Reconstructed power draw')
        axis[2].set_xlabel('Time (ms)')
        axis[2].set_ylabel('Power [W]')
        axis[2].grid(True, linestyle='--', linewidth=0.5)
        axis[2].set_xlim(axis[0].get_xlim())
        axis[2].legend(loc='lower center')

        if self.found_PMD:
            # plot a secondary axis for axies[0]
            ax0_2 = axis[0].twinx()

            ax0_2.fill_between(PMD_data.index, PMD_data['total_p'], PMD_data['eps_total_p'], alpha=0.5, label='Rower draw from PCIE power cables')
            ax0_2.fill_between(PMD_data.index, PMD_data['eps_total_p'], 0, alpha=0.5, label='Power draw from PCIE slot')
            ax0_2.set_xlabel('Time (ms)')
            ax0_2.set_ylabel('Power [W]')
            ax0_2.set_xlim(axis[0].get_xlim())
            ax0_2.legend(loc='lower right')
            

            PMD_reconstructed = PMD_reconstructed.loc[PMD_reconstructed.index.repeat(2)]
            PMD_reconstructed[self.exp_2_pwr_option] = PMD_reconstructed[self.exp_2_pwr_option].shift()
            PMD_reconstructed = PMD_reconstructed.dropna()

            axis[3].plot(PMD_reconstructed.index, PMD_reconstructed[self.exp_2_pwr_option], label='Reconstructed power draw from PMD data')
            axis[3].set_xlabel('Time (ms)')
            axis[3].set_ylabel('Power [W]')
            axis[3].grid(True, linestyle='--', linewidth=0.5)
            axis[3].set_xlim(axis[0].get_xlim())
            axis[3].legend(loc='lower center')

            
            axis[3].text(0.05, -0.15, f'Modeled history length: {avg_window:.2f} ms, delay: {delay:.2f} ms', transform=axis[3].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
            axis[3].text(0.05, -0.2,  f'PMD modeled history length: {PMD_avg_window:.2f} ms, delay: {PMD_delay:.2f} ms', transform=axis[3].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
            axis[3].text(0.05, -0.25, error_msg, transform=axis[3].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        else:
            axis[2].text(0.05, -0.15, f'Modeled history length: {avg_window:.2f} ms, delay: {delay:.2f} ms', transform=axis[2].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})


        fig.set_size_inches(20, 5*n_rows)
        plt.savefig(os.path.join(store_path, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(store_path, 'result.svg'), format='svg', bbox_inches='tight')
        plt.close('all')

    def _exp_2_plotting_only(self, dir):
        dirs = []
        for rep in range(self.repetitions):
            rep_store_path = os.path.join(dir, f'rep_#{rep}')
            dirs.append(rep_store_path)

        num_processes = min(len(dirs), os.cpu_count())
        pool = Pool(processes=num_processes)
        pool.map(self._exp_2_plot_power_data, dirs)
        pool.close()
        pool.join()

    def _exp_2_plot_power_data(self, result_dir):
        # Load data
        load = pd.read_csv(os.path.join(result_dir, 'timestamps.csv'))
        load.loc[-1] = load.loc[0] - 500000
        load.index = load.index + 1
        load = load.sort_index()
        load['activity'] = (load.index / 2).astype(int) % 2
        load['timestamp'] = (load['timestamp'] / 1000).astype(int) 
        t0 = load['timestamp'][0]
        load['timestamp'] -= t0
        load.loc[load.index[-1], 'timestamp'] += 500
        load.loc[load.index[-1], 'activity'] = 0
        load.set_index('timestamp', inplace=True)
        load.sort_index(inplace=True)

        # nv_smi power data
        power = pd.read_csv(os.path.join(result_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        power['timestamp'] += 60*60*1000 * self.jet_lag
        power['timestamp'] -= t0
        power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= load.index[-1])]
        power.set_index('timestamp', inplace=True)
        power.sort_index(inplace=True)

        if self.found_PMD:
            PMD_data = self._convert_pmd_data(result_dir)
            PMD_data['timestamp'] -= t0
            PMD_data = PMD_data[(PMD_data['timestamp'] >= 0) & (PMD_data['timestamp'] <= load.iloc[-1].name)]
            PMD_data.set_index('timestamp', inplace=True)
            PMD_data.sort_index(inplace=True)

        fig, axis = plt.subplots(nrows=2, ncols=1)

        axis[0].plot(load.index, load['activity']*100, label='load')
        axis[0].plot(power.index, power[' utilization.gpu [%]'], label='utilization.gpu [%]')
        axis[0].set_xlabel('Time (ms)')
        axis[0].set_ylabel('Load')
        axis[0].grid(True, linestyle='--', linewidth=0.5)
        axis[0].set_xlim(0, load.index.max())
        axis[0].legend(loc='upper right')
        axis[0].set_title(f'{self.gpu_name} - load window as half of power update period')
        
        axis[1].plot(power.index, power[self.exp_2_pwr_option], label='Power draw')
        axis[1].set_xlabel('Time (ms)')
        axis[1].set_ylabel('Power [W]')
        axis[1].grid(True, linestyle='--', linewidth=0.5)
        axis[1].set_xlim(axis[0].get_xlim())
        axis[1].legend(loc='lower center')

        if self.found_PMD:
            # plot a secondary axis for axies[0]
            ax0_2 = axis[0].twinx()

            ax0_2.fill_between(PMD_data.index, PMD_data['total_p'], PMD_data['eps_total_p'], alpha=0.5, label='Rower draw from PCIE power cables')
            ax0_2.fill_between(PMD_data.index, PMD_data['eps_total_p'], 0, alpha=0.5, label='Power draw from PCIE slot')
            ax0_2.set_xlabel('Time (ms)')
            ax0_2.set_ylabel('Power [W]')
            ax0_2.set_xlim(axis[0].get_xlim())
            ax0_2.legend(loc='lower right')

        fig.set_size_inches(20, 10)
        plt.savefig(os.path.join(result_dir, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'result.svg'), format='svg', bbox_inches='tight')
        plt.close('all')


    def process_exp_3(self, result_dir):
        color_palette =   {1 : '#BDCCFF', 4 : '#8D9DCE', 8 : '#5F709F'}
        correct_palette = {1 : '#91c732', 4 : '#76b900', 8 : '#5e9400'}
        # A100
        # gnd_truth = {'workload_0.25_pd' : 3.153843, 'workload_1_pd' : 12.629575, 'workload_8_pd' : 102.533766}
        # for key, value in gnd_truth.items():
        #     gnd_truth[key] = value * 1.16
        
        # 3090

        if self.gpu_name == 'NVIDIA_GeForce_RTX_3090':
            gnd_truth = {'workload_0.25_pd' : 7.861696, 'workload_1_pd' : 30.446554, 'workload_8_pd' : 238.511096}



        tests_list = os.listdir(result_dir)
        tests_list = [test for test in tests_list if os.path.isdir(os.path.join(result_dir, test))]
        tests_list.sort(key=lambda x: float(x.split('_')[-2]))


        fig, ax = plt.subplots(nrows=2, ncols=3)

        for plot_num, test in enumerate(tests_list):
            print(f'Processing test: {test}')

            ax[0, plot_num].set_title(f'{test} - {self.gpu_name}')
            ax[0, plot_num].set_xlabel('# of repetitions')
            ax[0, plot_num].set_ylabel('Standard Deviation [% of groung truth]')
            # create secondary axis
            ax[1, plot_num].set_xlabel('# of repetitions')
            ax[1, plot_num].set_ylabel('Error [% of groung truth]')

            shifts_list = os.listdir(os.path.join(result_dir, test))
            shifts_list.sort(key=lambda x: int(x.split('_')[-1]))

            for shift in shifts_list:
                print(f'  Processing shift: {shift}')
                num_shift = int(shift.split('_')[-1])

                reps_list = os.listdir(os.path.join(result_dir, test, shift))
                reps_list.sort(key=lambda x: int(x.split('_')[-1]))

                rep_result =  []
                std_result = []
                err_result = []
                correct_rep_result = []
                correct_std_result = []
                correct_err_result = []

                for reps in reps_list:
                    print(f'    Processing reps: {reps}')
                    num_reps = int(reps.split('_')[-1])

                    iters_list = os.listdir(os.path.join(result_dir, test, shift, reps))
                    iters_list.sort(key=lambda x: int(x.split('_')[-1]))

                    iters_dir = [os.path.join(result_dir, test, shift, reps, iters) for iters in iters_list]


                    num_processes = min(len(iters_dir), os.cpu_count())
                    pool = Pool(num_processes)
                    results = pool.map(self._exp_3_calculate_power, iters_dir)
                    results = list(map(list, zip(*results)))
                    energy_list = results[0]
                    correct_list = results[1]

                    rep_result.append(num_reps)
                    std_result.append(np.std(energy_list) / gnd_truth[test] * 100)
                    err_result.append((np.mean(energy_list) - gnd_truth[test]) / gnd_truth[test] * 100)

                    if np.mean(correct_list) > 0:
                        correct_rep_result.append(num_reps)
                        correct_std_result.append(np.std(correct_list) / gnd_truth[test] * 100)
                        correct_err_result.append((np.mean(correct_list) - gnd_truth[test]) / gnd_truth[test] * 100)

                
                ax[0, plot_num].plot(rep_result, std_result, '--o', color=color_palette[num_shift], label=f'{num_shift} shifts', linewidth=2)
                ax[1, plot_num].plot(rep_result, err_result, '--o', color=color_palette[num_shift], label=f'{num_shift} shifts', linewidth=2)

                ax[0, plot_num].plot(correct_rep_result, correct_std_result, '-o', color=correct_palette[num_shift], linewidth=2, label=f'{num_shift} shifts (Corrected)')
                ax[1, plot_num].plot(correct_rep_result, correct_err_result, '-o', color=correct_palette[num_shift], linewidth=2, label=f'{num_shift} shifts (Corrected)')


            ax[0, plot_num].legend(loc='upper right')
            ax[1, plot_num].legend(loc='lower right')

        fig.set_size_inches(20, 10)
        plt.savefig(os.path.join(result_dir, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'result.svg'), format='svg', bbox_inches='tight')
        plt.close('all')
            
            
    def _exp_3_calculate_power(self, result_dir):
        duration = float(result_dir.split('/')[-4].split('_')[-2]) * 100
        num_shifts = int(result_dir.split('/')[-3].split('_')[-1])
        num_reps = int(result_dir.split('/')[-2].split('_')[-1])

        power_option = ' power.draw.instant [W]'
        power_option = ' power.draw [W]'

        load = pd.read_csv(os.path.join(result_dir, 'timestamps.csv'))
        load['timestamp'] = (load['timestamp'] / 1000)
        t0 = load['timestamp'][0]
        load['timestamp'] -= t0
        load.set_index('timestamp', inplace=True)
        load.sort_index(inplace=True)

        # nv_smi power data
        power = pd.read_csv(os.path.join(result_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        power['timestamp'] += 60*60*1000 * self.jet_lag
        power['timestamp'] -= t0
        correct_power = power.copy()
        correct_power['timestamp'] = correct_power['timestamp'] - 100
        power.set_index('timestamp', inplace=True)
        power.sort_index(inplace=True)
        correct_power.set_index('timestamp', inplace=True)
        correct_power.sort_index(inplace=True)

        energy_nvsmi = 0
        correct_energy = 0

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        for i in range(num_shifts):
            start_ts = load.iloc[2*i].name
            end_ts   = load.iloc[2*i+1].name

            power_window = power[(power.index >= start_ts) & (power.index <= end_ts)]
            # interpolate the lowerbound of the power data
            lb_0 = power[power.index <= start_ts].iloc[-1]
            lb_1 = power[power.index > start_ts].iloc[0]
            gradient = (lb_1[power_option] - lb_0[power_option]) / (lb_1.name - lb_0.name)
            lb_p = lb_0[power_option] + gradient * (start_ts - lb_0.name)

            # interpolate the upperbound of the power data
            ub_0 = power[power.index < end_ts].iloc[-1]
            ub_1 = power[power.index >= end_ts].iloc[0]
            gradient = (ub_1[power_option] - ub_0[power_option]) / (ub_1.name - ub_0.name)
            ub_p = ub_0[power_option] + gradient * (end_ts - ub_0.name)

            # create the power data frame
            t = np.concatenate((np.array([start_ts]), power_window.index.to_numpy(), np.array([end_ts])))
            p = np.concatenate((np.array([lb_p]), power_window[power_option].to_numpy(), np.array([ub_p])))
            energy_nvsmi += np.trapz(p, t) / 1000 / (num_reps / num_shifts)

            # plot 2 verical lines at start_ts and end_ts
            ax.axvline(start_ts, color='g', linestyle='--', linewidth=1)
            ax.axvline(end_ts, color='r', linestyle='--', linewidth=1)

            ########################################################################
            reps_to_ignore = math.ceil(1200 / duration)
            if reps_to_ignore >= (num_reps / num_shifts) :
                correct_energy += 0
            else:
                start_ts += reps_to_ignore * (end_ts - start_ts) / (num_reps / num_shifts)

                power_window = correct_power[(correct_power.index >= start_ts) & (correct_power.index <= end_ts)]
                # interpolate the lowerbound of the correct_power data
                lb_0 = correct_power[correct_power.index <= start_ts].iloc[-1]
                lb_1 = correct_power[correct_power.index > start_ts].iloc[0]
                gradient = (lb_1[power_option] - lb_0[power_option]) / (lb_1.name - lb_0.name)
                lb_p = lb_0[power_option] + gradient * (start_ts - lb_0.name)

                # interpolate the upperbound of the correct_power data
                ub_0 = correct_power[correct_power.index < end_ts].iloc[-1]
                ub_1 = correct_power[correct_power.index >= end_ts].iloc[0]
                gradient = (ub_1[power_option] - ub_0[power_option]) / (ub_1.name - ub_0.name)
                ub_p = ub_0[power_option] + gradient * (end_ts - ub_0.name)

                # create the correct_power data frame
                t = np.concatenate((np.array([start_ts]), power_window.index.to_numpy(), np.array([end_ts])))
                p = np.concatenate((np.array([lb_p]), power_window[power_option].to_numpy(), np.array([ub_p])))
                correct_energy += np.trapz(p, t) / 1000 / (num_reps / num_shifts - reps_to_ignore)


        ax.plot(power.index, power[power_option], label='nvidia-smi Power Draw Reading', linewidth=3)

        ax.set_xlim(load.iloc[0].name-100, load.iloc[-1].name+100)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power draw from nv_smi and PMD')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', linewidth=0.5)

        
        fig.set_size_inches(7.5, 5)
        plt.savefig(os.path.join(result_dir, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        # plt.savefig(os.path.join(result_dir, 'result.eps'), format='eps', bbox_inches='tight')
        plt.close('all')

        return [energy_nvsmi / num_shifts, correct_energy / num_shifts]


    def process_exp_4(self, result_dir):
        with open(os.path.join(result_dir, 'tests_dict.json'), 'r') as f:  tests_dict = json.load(f)

        tests_list = os.listdir(result_dir)
        tests_list.remove('tests_dict.json')

        
        for test in tests_list:
            print(f'Processing test: {test}')
            args = []
            reps_list = os.listdir(os.path.join(result_dir, test))
            reps_list.sort(key=lambda x: int(x.split('_')[-1]))
            for reps in reps_list:
                args.append(os.path.join(result_dir, test, reps))

            E_PMD_avg = []
            E_nvsmi_avg = []
            for arg in args:
                reps = int(tests_dict[test]['reps'] * 0.6)
                ratio = reps / tests_dict[test]['reps']
                E_PMD, E_nvsmi = self._exp_4_process_single_run(arg, ratio)
                E_PMD /= reps
                E_nvsmi /= reps
                E_PMD_avg.append(E_PMD)
                E_nvsmi_avg.append(E_nvsmi)
                print(f'  rep: {arg.split("_")[-1]}    Energy PMD: {E_PMD:.6f} J    Energy nv_smi: {E_nvsmi:.6f} J')
            
            E_PMD_avg = np.mean(E_PMD_avg)
            E_nvsmi_avg = np.mean(E_nvsmi_avg)

            print(f'  Average Energy PMD: {E_PMD_avg:.6f} J    Average Energy nv_smi: {E_nvsmi_avg:.6f} J\n')

    def _exp_4_process_single_run(self, result_dir, ratio):
        load = pd.read_csv(os.path.join(result_dir, 'timestamps.csv'))
        load['timestamp'] = (load['timestamp'] / 1000)
        t0 = load['timestamp'][0]
        load['timestamp'] -= t0
        load.set_index('timestamp', inplace=True)
        load.sort_index(inplace=True)

        # nv_smi power data
        power = pd.read_csv(os.path.join(result_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        power['timestamp'] += 60*60*1000 * self.jet_lag
        power['timestamp'] -= t0
        # power['timestamp'] -= 25
        power.set_index('timestamp', inplace=True)
        power.sort_index(inplace=True)

        PMD_data = self._convert_pmd_data(result_dir)
        PMD_data['timestamp'] -= t0
        PMD_data.set_index('timestamp', inplace=True)
        PMD_data.sort_index(inplace=True)
    
        start_ts = load.iloc[0].name
        end_ts = load.iloc[-1].name
        start_ts += (1 - ratio) * (end_ts - start_ts)


        power_option = ' power.draw.instant [W]'

        power_window = power[(power.index >= start_ts) & (power.index <= end_ts)]
        # interpolate the lowerbound of the power data
        lb_0 = power[power.index < start_ts].iloc[-1]
        lb_1 = power[power.index > start_ts].iloc[0]
        gradient = (lb_1[power_option] - lb_0[power_option]) / (lb_1.name - lb_0.name)
        lb_p = lb_0[power_option] + gradient * (start_ts - lb_0.name)

        # interpolate the upperbound of the power data
        ub_0 = power[power.index < end_ts].iloc[-1]
        ub_1 = power[power.index > end_ts].iloc[0]
        gradient = (ub_1[power_option] - ub_0[power_option]) / (ub_1.name - ub_0.name)
        ub_p = ub_0[power_option] + gradient * (end_ts - ub_0.name)

        # create the power data frame
        t = np.concatenate((np.array([start_ts]), power_window.index.to_numpy(), np.array([end_ts])))
        p = np.concatenate((np.array([lb_p]), power_window[power_option].to_numpy(), np.array([ub_p])))
        energy_nvsmi = np.trapz(p, t) / 1000

        # calculate energy from PMD data
        PMD_window = PMD_data[(PMD_data.index >= start_ts) & (PMD_data.index <= end_ts)]
        energy_PMD = np.trapz(PMD_window['total_p'].to_numpy(), PMD_window.index.to_numpy()) / 1000

        # plot the data
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # plot 2 verical lines at start_ts and end_ts
        ax.axvline(start_ts, color='r', linestyle='--')
        ax.axvline(end_ts, color='r', linestyle='--')

        ax.plot(power.index, power[power_option], label='nv_smi')
        ax.fill_between(PMD_data.index, PMD_data['total_p'], PMD_data['eps_total_p'], alpha=0.5, label='Rower draw from PCIE power cables')
        ax.fill_between(PMD_data.index, PMD_data['eps_total_p'], 0, alpha=0.5, label='Power draw from PCIE slot')

        ax.set_xlim(start_ts-100, end_ts+100)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power draw from nv_smi and PMD')
        ax.legend(loc='upper right')

        
        fig.set_size_inches(15, 9)
        plt.savefig(os.path.join(result_dir, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'result.svg'), format='svg', bbox_inches='tight')
        plt.close('all')

        return energy_PMD, energy_nvsmi

    def process_exp_5(self, result_dir):
        with open(os.path.join(result_dir, 'tests_dict.json'), 'r') as f:  tests_dict = json.load(f)

        tests_list = os.listdir(result_dir)
        tests_list.remove('tests_dict.json')

        
        for test in tests_list:
            print(f'Processing test: {test}')
            args = []
            reps_list = os.listdir(os.path.join(result_dir, test))
            reps_list.sort(key=lambda x: int(x.split('_')[-1]))
            for reps in reps_list:
                args.append(os.path.join(result_dir, test, reps))

            naive_energy_list = []
            correct_energy_list = []
            for arg in args:
                reps = int(tests_dict[test]['reps'])
                naive_energy, correct_energy = self._exp_5_process_single_run(arg, reps)

                naive_energy_list.append(naive_energy)
                correct_energy_list.append(correct_energy)
                print(f'  rep: {arg.split("_")[-1]}    naive energy: {naive_energy:.6f} J    correct energy: {correct_energy:.6f} J')
            
            naive_energy_avg = np.mean(naive_energy_list)
            correct_energy_avg = np.mean(correct_energy_list)

            print(f'  Average naive energy: {naive_energy_avg:.6f} J    Average correct energy: {correct_energy_avg:.6f} J')

    def _exp_5_process_single_run(self, result_dir, num_reps):
        load = pd.read_csv(os.path.join(result_dir, 'timestamps.csv'))
        load['timestamp'] = (load['timestamp'] / 1000)
        t0 = load['timestamp'][0]
        load['timestamp'] -= t0
        load.set_index('timestamp', inplace=True)
        load.sort_index(inplace=True)

        num_shifts = int(load.shape[0]/2)

        # nv_smi power data
        power = pd.read_csv(os.path.join(result_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        power['timestamp'] += 60*60*1000 * self.jet_lag
        power['timestamp'] -= t0
        correct_power = power.copy()
        correct_power['timestamp'] = correct_power['timestamp'] - 100
        power.set_index('timestamp', inplace=True)
        power.sort_index(inplace=True)
        correct_power.set_index('timestamp', inplace=True)
        correct_power.sort_index(inplace=True)
        power_option = ' power.draw.instant [W]'
        power_option = ' power.draw [W]'
                
        naive_energy = 0
        correct_energy = 0

        for i in range(num_shifts):
            start_ts = load.iloc[2*i].name
            end_ts   = load.iloc[2*i+1].name

            power_window = power[(power.index >= start_ts) & (power.index <= end_ts)]
            # interpolate the lowerbound of the power data
            lb_0 = power[power.index <= start_ts].iloc[-1]
            lb_1 = power[power.index > start_ts].iloc[0]
            gradient = (lb_1[power_option] - lb_0[power_option]) / (lb_1.name - lb_0.name)
            lb_p = lb_0[power_option] + gradient * (start_ts - lb_0.name)

            # interpolate the upperbound of the power data
            ub_0 = power[power.index < end_ts].iloc[-1]
            ub_1 = power[power.index >= end_ts].iloc[0]
            gradient = (ub_1[power_option] - ub_0[power_option]) / (ub_1.name - ub_0.name)
            ub_p = ub_0[power_option] + gradient * (end_ts - ub_0.name)

            # create the power data frame
            t = np.concatenate((np.array([start_ts]), power_window.index.to_numpy(), np.array([end_ts])))
            p = np.concatenate((np.array([lb_p]), power_window[power_option].to_numpy(), np.array([ub_p])))
            naive_energy += np.trapz(p, t) / 1000 / (num_reps / num_shifts)

            ########################################################################
            duration = (end_ts - start_ts) / (num_reps / num_shifts)
            reps_to_ignore = math.ceil(1200 / duration)
            if reps_to_ignore >= (num_reps / num_shifts) :
                correct_energy += 0
            else:
                start_ts += reps_to_ignore * (end_ts - start_ts) / (num_reps / num_shifts)

                power_window = correct_power[(correct_power.index >= start_ts) & (correct_power.index <= end_ts)]
                # interpolate the lowerbound of the correct_power data
                lb_0 = correct_power[correct_power.index <= start_ts].iloc[-1]
                lb_1 = correct_power[correct_power.index > start_ts].iloc[0]
                gradient = (lb_1[power_option] - lb_0[power_option]) / (lb_1.name - lb_0.name)
                lb_p = lb_0[power_option] + gradient * (start_ts - lb_0.name)

                # interpolate the upperbound of the correct_power data
                ub_0 = correct_power[correct_power.index < end_ts].iloc[-1]
                ub_1 = correct_power[correct_power.index >= end_ts].iloc[0]
                gradient = (ub_1[power_option] - ub_0[power_option]) / (ub_1.name - ub_0.name)
                ub_p = ub_0[power_option] + gradient * (end_ts - ub_0.name)

                # create the correct_power data frame
                t = np.concatenate((np.array([start_ts]), power_window.index.to_numpy(), np.array([end_ts])))
                p = np.concatenate((np.array([lb_p]), power_window[power_option].to_numpy(), np.array([ub_p])))
                correct_energy += np.trapz(p, t) / 1000 / (num_reps / num_shifts - reps_to_ignore)

        


        return naive_energy, correct_energy



class stride_gen:
    def __init__(self, mode, grain):
        self.mode = mode
        self.grain = grain
        self.i = 0
        self.size = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.mode == 'lin':
            if self.size <= self.grain:
                self.size = 2**self.i
            else:
                self.size += self.grain
        elif self.mode == 'exp':
            if self.i < self.grain:
                self.size += 1
            else:
                self.size += 2**(int(self.i/self.grain)-1)
        self.i += 1
    
    def __str__(self):
        return str(self.size)

    def __int__(self):
        return self.size

    def __add__(self, x):
        return self.size + x

    def __rsub__(self, x):
        return x - self.size

    def __mul__(self, x):
        return self.size * x

    def __rtruediv__(self, x):
        return x / self.size

    def __itruediv__(self, x):
        return self.size / x

    def reset(self, grain = None):
        self.i = 0
        self.size = 0
        if grain is not None:
            self.grain = grain




