import subprocess
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics
from functools import partial
import time
import random
import os
from multiprocessing import Pool
import struct
from scipy.optimize import minimize
import csv

class GPU_pwr_benchmark:
    def __init__(self, sw_meas, PMD=0, verbose=False):
        print('_________________________')
        print('Initializing benchmark...')
        self.verbose = verbose
        self.BST = True
        self.repetitions = 64
        self.nvsmi_smp_pd = 5
        self.sw_meas = sw_meas
        self.PMD = PMD
        self.aliasing_ratios = [2/3, 3/4, 4/5, 1, 6/5, 5/4, 4/3]
        
    def prepare_experiment(self):
        self._get_machine_info()
        
        run = 0
        while os.path.exists(os.path.join('results', f'{self.gpu_name}_run_#{run}')): run += 1
        self.result_dir = os.path.join('results', f'{self.gpu_name}_run_#{run}')
        os.makedirs(self.result_dir)
        os.makedirs(os.path.join(self.result_dir, 'Preparation'))
        self.log_file = os.path.join(self.result_dir, 'log.txt')

        self._print_general_info()

        print('___________________________')
        print('Preparing for experiment...')
        self._log('___________________________')
        self._log('Preparing for experiment...')


        if not os.path.exists('/tmp'): os.makedirs('/tmp')
        self._recompile_load()
        self._warm_up()

        self.scale_gradient, self.scale_intercept = self._find_scale_parameter()
        self.pwr_update_freq = self._find_pwr_update_freq()
    
    def _log(self, message, end='\n'):
        with open(self.log_file, 'a') as f:
            f.write(message + end)

    def _print_general_info(self):
        print()
        time_ =  'Date and time:        ' + time.strftime('%d/%m/%Y %H:%M:%S')
        gpu =    'Benchmarking on GPU:  ' + self.gpu_name
        serial = 'GPU serial number:    ' + self.gpu_serial
        uuid =   'GPU UUID:             ' + self.gpu_uuid
        driver = 'Driver version:       ' + self.driver_version
        cuda =   'CUDA version:         ' + self.nvcc_version
        host =   'Host machine:         ' + os.uname()[1]

        max_len = max(len(time_), len(gpu), len(host), len(serial), len(uuid), len(driver))
        output = ''
        output += '+ ' + '-'*(max_len) + ' +\n'
        output += '| ' + time_ + ' '*(max_len - len(time_)) + ' |\n'
        output += '| ' + gpu + ' '*(max_len - len(gpu)) + ' |\n'
        output += '| ' + serial + ' '*(max_len - len(serial)) + ' |\n'
        output += '| ' + uuid + ' '*(max_len - len(uuid)) + ' |\n'
        output += '| ' + driver + ' '*(max_len - len(driver)) + ' |\n'
        output += '| ' + cuda + ' '*(max_len - len(cuda)) + ' |\n'
        output += '| ' + host + ' '*(max_len - len(host)) + ' |\n'
        output += '+ ' + '-'*(max_len) + ' +\n'
        print(output)
        self._log(output)

    def _recompile_load(self):
        print('Recompiling benchmark load...')
        # make clean and make
        # try can catch the error
        subprocess.call(['make', '-C', 'source/', 'clean'])
        return_code = subprocess.call(['make', '-C', 'source/'])
        if return_code != 0:  raise Exception('Error compiling benchmark')

        print()

    def _get_machine_info(self):
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=name,serial,uuid,driver_version', '--format=csv,noheader'], stdout=subprocess.PIPE)
        output = result.stdout.decode().split('\n')[0].split(', ')
        self.gpu_name, self.gpu_serial, self.gpu_uuid, self.driver_version = output
        self.gpu_name = self.gpu_name.replace(' ', '_')

        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            output = result.stdout
            nvcc_version = output.split('\n')[3].split(',')[1].strip()
            self.nvcc_version = nvcc_version
        except FileNotFoundError:
            print("CUDA is not installed or 'nvcc' command not found.")
            self.nvcc_version = 'N/A'

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
                            self.sw_meas, 
                            str(self.PMD)
                        ])
        if delay:  time.sleep(1)

    def _warm_up(self):
        print('Warming up GPU...                    ', end='', flush=True)
        self._log('Warming up GPU...                   ', end='')
        store_path = os.path.join(self.result_dir, 'Preparation', 'warm_up')
        os.makedirs(store_path)

        # While loop that run for at least 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            self._run_benchmark(1, '50,1000000,20,100', store_path)
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
            niter = int(niter * 1.5)


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
        config = f'10,{niters},100,100'
        self._run_benchmark(1, config, store_path)

        df = pd.read_csv(os.path.join(store_path, 'gpudata.csv'))
        df['timestamp'] = (pd.to_datetime(df['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")

        period_list = []
        last_pwr = df.iloc[0]
        for index, row in df.iterrows():
            if row[' power.draw [W]'] != last_pwr[' power.draw [W]']:
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
        return median_period

    def run_experiment(self, experiment):
        if experiment == 1:
            # Experiment 1: Steady state and trasient response analysis
            print('_____________________________________________________________________')
            print('Running experiment 1: Steady state and transient response analysis...')
            self._log('_____________________________________________________________________')
            self._log('Running experiment 1: Steady state and transient response analysis...')

            os.makedirs(os.path.join(self.result_dir, 'Experiment_1'))
            for percentage in range(25, 101, 25):
                print(f'  Running experiment with {percentage}% load...')
                self._log(f'  Running experiment with {percentage}% load...')
                # create the store path
                percentage_store_path = os.path.join(self.result_dir, 'Experiment_1', f'{percentage}%_load')
                os.makedirs(percentage_store_path)
                scale_gradient, scale_intercept = self._find_scale_parameter(percentage_store_path, percentage)

                for rep in range(int(self.repetitions/4)):
                    print(f'    Repetition {rep+1} of {int(self.repetitions/4)}...')
                    self._log(f'    Repetition {rep+1} of {int(self.repetitions/4)}...')
                    rep_store_path = os.path.join(percentage_store_path, f'rep_#{rep}')
                    os.makedirs(rep_store_path)

                    niters = int(6000 * scale_gradient + scale_intercept)
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
                    print(f'  Repetition {rep+1} of {self.repetitions}...')
                    self._log(f'  Repetition {rep+1} of {self.repetitions}...')

                    rep_store_path = os.path.join(ratio_store_path, f'rep_#{rep}')
                    os.makedirs(rep_store_path)
                    
                    niters = int(load_pd * self.scale_gradient + self.scale_intercept)
                    repetitions = int(4000 / load_pd)
                    config = f'{load_pd},{niters},{repetitions},100'
                    self._run_benchmark(1, config, rep_store_path, delay=False)
                    time.sleep(random.random())
        else:
            raise ValueError(f'Invalid experiment number {experiment}')

    def process_results(self, GPU_name=None, run=0, notes=None):
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

        dir_list = os.listdir(self.result_dir)
        # if 'Experiment_1' in dir_list:  self.process_exp_1(os.path.join(self.result_dir, 'Experiment_1'))
        if 'Experiment_2' in dir_list:  self.process_exp_2(os.path.join(self.result_dir, 'Experiment_2'), notes)

    def _convert_pmd_data(self, dir):
        pass

    def process_exp_1(self, result_dir):
        print('  Processing experiment 1...')

        dir_list = os.listdir(result_dir)
        dir_list = sorted(dir_list, key=lambda dir: int(dir.split('%')[0]))
        subdir_list = os.listdir(os.path.join(result_dir, dir_list[0]))
        subdir_list = [dir for dir in subdir_list if 'find_scale_param' not in dir]
        subdir_list = sorted(subdir_list, key=lambda dir: int(dir.split('#')[1]))
        dir_list = [os.path.join(result_dir, dir, subdir) 
                        for dir in dir_list 
                            for subdir in subdir_list]
        
        num_processes = min(self.repetitions, os.cpu_count())
        pool = Pool(processes=num_processes)
        pool.map(self._exp_1_plot_result, dir_list)
        pool.close()
        pool.join()

        # print(dir_list[0])
        # self._exp_1_plot_result(dir_list[0])

    def _exp_1_plot_result(self, dir):
        def plot_PMD_data(dir, t0, t_max, power, axis):
            with open(os.path.join(dir, 'PMD_start_ts.txt'), 'r') as f:  pmd_start_ts = int(f.readline()) - 50000
            with open(os.path.join(dir, 'PMD_data.bin'), 'rb') as f:     pmd_data = f.read()

            bytes_per_sample = 18
            num_samples = int(len(pmd_data) / bytes_per_sample)

            data_dict = {
                'timestamp (us)' : [],
                'pcie1_v': [], 'pcie1_i': [], 'pcie1_p': [],
                'pcie2_v': [], 'pcie2_i': [], 'pcie2_p': [],
                'eps1_v' : [], 'eps1_i' : [], 'eps1_p' : [],
                'eps2_v' : [], 'eps2_i' : [], 'eps2_p' : [],
            }

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

                # populate the dictionary
                data_dict['timestamp (us)'].append(timestamp / 3)
                data_dict['pcie1_v'].append(values[1] * 0.007568)
                data_dict['pcie1_i'].append(values[2] * 0.0488)
                data_dict['pcie1_p'].append(data_dict['pcie1_v'][-1] * data_dict['pcie1_i'][-1])
                data_dict['pcie2_v'].append(values[3] * 0.007568)
                data_dict['pcie2_i'].append(values[4] * 0.0488)
                data_dict['pcie2_p'].append(data_dict['pcie2_v'][-1] * data_dict['pcie2_i'][-1])
                data_dict['eps1_v'].append(values[5] * 0.007568)
                data_dict['eps1_i'].append(values[6] * 0.0488)
                data_dict['eps1_p'].append(data_dict['eps1_v'][-1] * data_dict['eps1_i'][-1])
                data_dict['eps2_v'].append(values[7] * 0.007568)
                data_dict['eps2_i'].append(values[8] * 0.0488)
                data_dict['eps2_p'].append(data_dict['eps2_v'][-1] * data_dict['eps2_i'][-1])

            df = pd.DataFrame(data_dict)

            df['timestamp (us)'] += (pmd_start_ts - df['timestamp (us)'][0])
            df['timestamp (ms)'] = (df['timestamp (us)'] / 1000)

            df['timestamp (ms)'] -= t0
            
            df = df[(df['timestamp (ms)'] >= 0) & (df['timestamp (ms)'] <= t_max)]

            df['pcie_total_p'] = df['pcie1_p'] + df['pcie2_p']
            df['eps_total_p'] = df['eps1_p'] + df['eps2_p']
            df['total_p'] = df['pcie_total_p'] + df['eps_total_p']
            
            axis[1].fill_between(df['timestamp (ms)'], df['total_p'], df['eps_total_p'], alpha=0.5, label='Rower draw from PCIE power cables')
            axis[1].fill_between(df['timestamp (ms)'], df['eps_total_p'], 0, alpha=0.5, label='Power draw from PCIE slot')
            
            
            # iterate through df using iterrows
            last_timestamp = 0
            for index, row in power.iterrows():
                df.loc[(df['timestamp (ms)'] > last_timestamp) & (df['timestamp (ms)'] <= row['timestamp']), 'nv_power'] = row[' power.draw [W]']
                last_timestamp = row['timestamp']

            df['nv_power_error'] = df['nv_power'] - df['total_p']
            mse = np.mean(df['nv_power_error']**2)
            # print('MSE: {}'.format(mse))
            
            axis[2].fill_between(df['timestamp (ms)'], df['nv_power_error'], 0, alpha=1, label=f'Power reading error, MSE={mse:.2f}', color='red')
            axis[2].set_xlabel('Time (ms)')
            axis[2].set_ylabel('Difference [W]')
            axis[2].grid(True, linestyle='--', linewidth=0.5)
            axis[2].set_xlim(axis[0].get_xlim())
            axis[2].legend()
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
        if self.BST:    power['timestamp'] -= 60*60*1000
        power['timestamp'] -= t0
        power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= t_max+10)]

        PMD_exists = False
        if os.path.exists(os.path.join(dir, 'PMD_data.bin')) and os.path.exists(os.path.join(dir, 'PMD_start_ts.txt')):
            PMD_exists = True

        # plotting
        n_rows = 2
        if PMD_exists:    n_rows = 3
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
        if PMD_exists:    plot_PMD_data(dir, t0, t_max, power, axis)

        axis[1].plot(power['timestamp'], power[' power.draw [W]'], label='Power draw reported by nvidia-smi', linewidth=2, color='black')
        axis[1].set_xlabel('Time (ms)')
        axis[1].set_ylabel('Power [W]')
        axis[1].grid(True, linestyle='--', linewidth=0.5)
        axis[1].set_xlim(axis[0].get_xlim())
        axis[1].legend()

        fig.set_size_inches(20, 4*n_rows)
        plt.savefig(os.path.join(dir, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(dir, 'result.svg'), format='svg', bbox_inches='tight')
        plt.close('all')

    def process_exp_2(self, result_dir, notes):
        dir_list = os.listdir(result_dir)
        dir_list = [dir for dir in dir_list if dir.endswith('ms')]
        dir_list = sorted(dir_list, key=lambda dir: int(dir.split('_')[1]))
        print(f'  Found {len(dir_list)-1} directories to process...')

        avg_window_results = []
        delay_results = []
        labels = []

        self.pwr_update_freq = int(statistics.median([int(dir.split('_')[1]) for dir in dir_list]))

        
        for dir in dir_list:
            load_pd = int(dir.split('_')[1])

            if load_pd == self.pwr_update_freq:    continue

            print(f'  Processing results for load period of {load_pd} ms...')
            ratio_store_path = os.path.join(result_dir, dir)
            labels.append(f'{load_pd}')
            
            args = []
            for rep in range(self.repetitions):
                rep_store_path = os.path.join(ratio_store_path, f'rep_#{rep}')
                args.append((rep_store_path, load_pd))

            num_processes = min(self.repetitions, os.cpu_count())
            pool = Pool(processes=num_processes)
            results = pool.starmap(self._process_single_run, args)
            pool.close()
            pool.join()

            avg_windows, delays = zip(*results)
            avg_windows = list(avg_windows)
            delays = list(delays) 

            # find mean median and std of results
            avg_window_mean = statistics.mean(avg_windows)
            avg_window_median = statistics.median(avg_windows)
            avg_window_std = statistics.stdev(avg_windows)

            delay_mean = statistics.mean(delays)
            delay_median = statistics.median(delays)
            delay_std = statistics.stdev(delays)

            print(f'            Delay            Avg window')
            print(f'    Mean:   {delay_mean:.2f} ms          {avg_window_mean:.2f} ms')
            print(f'    Median: {delay_median:.2f} ms          {avg_window_median:.2f} ms')
            print(f'    Std:    {delay_std:.2f} ms          {avg_window_std:.2f} ms')
            
            avg_window_results.append(avg_windows)
            delay_results.append(delays)

        # store avg_window_results and delay_results in a csv file
        with open(os.path.join(result_dir, 'results.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(labels)
            for window in avg_window_results:    csvwriter.writerow(window)
            for delay  in delay_results:         csvwriter.writerow(delay)
        
        self._exp_2_plot_results(result_dir, notes)

    def _exp_2_plot_results(self, result_dir, notes):
        # read the avg_window_results and delay_results from the csv file
        with open(os.path.join(result_dir, 'results.csv'), 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            rows = list(csv_reader)
            num_rows = len(rows)
            midpoint = num_rows//2 + 1
            labels = rows[0]
            avg_window_results = [[np.float64(value) for value in row] for row in rows[1:midpoint]]
            delay_results = [[np.float64(value) for value in row] for row in rows[midpoint:]]
        
        avg_window_flat = [item for sublist in avg_window_results for item in sublist]
        avg_window_mean = statistics.mean(avg_window_flat)
        avg_window_median = statistics.median(avg_window_flat)
        avg_window_std = statistics.stdev(avg_window_flat)

        delay_flat = [item for sublist in delay_results for item in sublist]
        delay_mean = statistics.mean(delay_flat)
        delay_median = statistics.median(delay_flat)
        delay_std = statistics.stdev(delay_flat)

        print( '  Overall:')
        print( '            Delay            Avg window')
        print(f'    Mean:   {delay_mean:.2f} ms          {avg_window_mean:.2f} ms')
        print(f'    Median: {delay_median:.2f} ms          {avg_window_median:.2f} ms')
        print(f'    Std:    {delay_std:.2f} ms          {avg_window_std:.2f} ms')
        
        avg_window_results.append(avg_window_flat)
        delay_results.append(delay_flat)
        labels.append('All')

        # plot the results
        fig, axis = plt.subplots(nrows=2, ncols=1)
        self._violin_plot(axis[0], avg_window_results, labels)
        axis[0].set_xlabel('Load period (ms)')
        axis[0].set_ylabel('Averaging windod (ms)')
        axis[0].set_title(f'Averaging windod vs Load Period ({self.gpu_name})')

        self._violin_plot(axis[1], delay_results, labels)
        axis[1].set_xlabel('Load period (ms)')
        axis[1].set_ylabel('Delay (ms)')
        axis[1].set_title(f'Delay vs Load Period ({self.gpu_name})')

        # add some texts at the bottom of the plot
        axis[1].text(0.05, -0.15, f'Mean averaging window:   {avg_window_mean:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        axis[1].text(0.05, -0.2,  f'Median averaging window: {avg_window_median:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        axis[1].text(0.05, -0.25, f'Std averaging window:    {avg_window_std:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        axis[1].text(0.65,  -0.15, f'Mean delay:   {delay_mean:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        axis[1].text(0.65,  -0.2,  f'Median delay: {delay_median:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        axis[1].text(0.65,  -0.25, f'Std delay:    {delay_std:.2f} ms', transform=axis[1].transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})

        fig.set_size_inches(12, 10)
        fname = f'history_length_{self.gpu_name}{notes}'
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

    def _process_single_run(self, result_dir, load_period):
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

        power = pd.read_csv(os.path.join(result_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        if self.BST:    power['timestamp'] -= 60*60*1000
        power['timestamp'] -= t0
        power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= load['timestamp'].iloc[-1])]

        reduced_power = power[power[' power.draw [W]'] != power[' power.draw [W]'].shift()]

        loss_func = partial(self._reconstr_loss, load, reduced_power)

        init_vars = [self.pwr_update_freq/2, self.nvsmi_smp_pd/200*self.pwr_update_freq]

        avg_window, delay = minimize(loss_func, init_vars, method='Nelder-Mead', 
                                    options={'maxiter': 1000, 'xatol': 1e-3, 'disp': self.verbose}).x

        if self.verbose:    print(f'modeled avg_window: {avg_window}ms, delay: {delay}ms')

        reconstructed = self._reconstruction(load, reduced_power, avg_window, delay)
        self._plot_reconstr_result(load_period, load, power, avg_window, delay, reconstructed, result_dir)

        return avg_window, delay

    def _reconstr_loss(self, load, power, variables, loss_type='MSE'):
        if variables[0] < 0 or variables[1] < 0:    return 100

        reconstructed = self._reconstruction(load, power, variables[0], variables[1])

        # discard the beginning 20% and the end 10% of the data
        power = power.iloc[int(len(power)*0.2):int(len(power)*0.9)]
        reconstructed = reconstructed.iloc[int(len(reconstructed)*0.2):int(len(reconstructed)*0.9)]

        # norm the 2 signals before calculating the loss
        power_norm = (power[' power.draw [W]'] - power[' power.draw [W]'].mean()) / power[' power.draw [W]'].std()
        reconstructed_norm = (reconstructed[' power.draw [W]'] - reconstructed[' power.draw [W]'].mean()) / reconstructed[' power.draw [W]'].std()

        if loss_type == 'MSE':
            return np.mean((power_norm - reconstructed_norm)**2)
        elif loss_type == 'MAE':
            return np.mean(np.abs(power[' power.draw [W]'] - reconstructed[' power.draw [W]']))
        elif loss_type == 'EMSE':
            return np.sqrt(np.mean((power[' power.draw [W]'] - reconstructed[' power.draw [W]'])**2))
        else:
            raise Exception('Invalid loss type')    

    def _reconstruction(self, load, power, history_length, delay=0):
        reconstructed = power.copy()

        for index, row in reconstructed.iterrows():
            if row['timestamp'] <= 500:
                reconstructed.loc[index, ' power.draw [W]'] = 0
            else:
                # find rows in load df that are within the past history_length of the current timestamp
                load_window = load[(load['timestamp'] >= row['timestamp'] - history_length - delay) 
                            & (load['timestamp'] < row['timestamp'] - delay)].copy()

                # interpolate the lower bound of the load window
                lb_t = row['timestamp'] - history_length - delay
                lb_0 = load[load['timestamp'] < lb_t].iloc[-1]
                lb_1 = load[load['timestamp'] >= lb_t].iloc[0]
                gradient = (lb_1['activity'] - lb_0['activity']) / (lb_1['timestamp'] - lb_0['timestamp'])
                lb_p = lb_0['activity'] + gradient * (lb_t - lb_0['timestamp'])

                # interpolate the upper bound of the load window
                ub_t = row['timestamp'] - delay
                ub_0 = load[load['timestamp'] < ub_t].iloc[-1]
                ub_1 = load[load['timestamp'] >= ub_t].iloc[0]
                gradient = (ub_1['activity'] - ub_0['activity']) / (ub_1['timestamp'] - ub_0['timestamp'])
                ub_p = ub_0['activity'] + gradient * (ub_t - ub_0['timestamp'])
                
                # take the average of the load window
                t = np.concatenate((np.array([lb_t]), load_window['timestamp'].to_numpy(), np.array([ub_t])))
                p = np.concatenate((np.array([lb_p]), load_window['activity'].to_numpy(), np.array([ub_p])))
                reconstr_pwr = np.trapz(p, t) / history_length
                
                reconstructed.loc[index, ' power.draw [W]'] = reconstr_pwr

        return reconstructed

    def _plot_reconstr_result(self, load_period, load, power, avg_window, delay, reconstructed, store_path):
        # Plot the results
        fig, axis = plt.subplots(nrows=3, ncols=1)

        axis[0].plot(load['timestamp'], load['activity']*100, label='load')
        axis[0].plot(power['timestamp'], power[' utilization.gpu [%]'], label='utilization.gpu [%]')
        axis[0].set_xlabel('Time (ms)')
        axis[0].set_ylabel('Load')
        axis[0].grid(True, linestyle='--', linewidth=0.5)
        axis[0].set_xlim(0, load['timestamp'].max())
        axis[0].legend()
        axis[0].set_title(f'{self.gpu_name} - {load_period} ms load window - modeled history length: {avg_window:.2f} ms, delay: {delay:.2f} ms')
        
        axis[1].plot(power['timestamp'], power[' power.draw [W]'], label='Power draw')
        axis[1].set_xlabel('Time (ms)')
        axis[1].set_ylabel('Power [W]')
        axis[1].grid(True, linestyle='--', linewidth=0.5)
        axis[1].set_xlim(axis[0].get_xlim())
        axis[1].legend(loc='lower center')

        reconstructed = reconstructed.loc[reconstructed.index.repeat(2)].reset_index(drop=True)
        reconstructed[' power.draw [W]'] = reconstructed[' power.draw [W]'].shift()
        reconstructed = reconstructed.dropna().reset_index(drop=True)

        axis[2].plot(reconstructed['timestamp'], reconstructed[' power.draw [W]'], label='Reconstructed power draw')
        axis[2].set_xlabel('Time (ms)')
        axis[2].set_ylabel('Power [W]')
        axis[2].grid(True, linestyle='--', linewidth=0.5)
        axis[2].set_xlim(axis[0].get_xlim())
        axis[2].legend(loc='lower center')

        fig.set_size_inches(20, 12)
        plt.savefig(os.path.join(store_path, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(store_path, 'result.svg'), format='svg', bbox_inches='tight')
        plt.close('all')