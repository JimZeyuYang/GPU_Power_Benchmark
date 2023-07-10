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

class GPU_pwr_benchmark:
    def __init__(self, sw_meas, PMD=0, verbose=False):
        print('_________________________')
        print('Initializing benchmark...')
        self.verbose = verbose
        self.repetitions = 32
        self.nvsmi_smp_pd = 5
        self.sw_meas = sw_meas
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
        subprocess.call(['make', '-C', 'source/', 'clean'])
        return_code = subprocess.call(['make', '-C', 'source/'])
        if return_code != 0:  raise Exception('Error compiling benchmark')

        print()

    def _get_machine_info(self):
        def is_number(s):
            try:    float(s)
            except ValueError: return False
            return True

        self.epoch_time = time.time()
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=timestamp,name,serial,uuid,driver_version', '--format=csv,noheader'], stdout=subprocess.PIPE)
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
            if bool(output.find(key)):
                query_options += key + ','

        output = subprocess.run(['nvidia-smi', query_options, '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        output = output.stdout.decode()[:-1].split(', ')

        for i, (key, value) in enumerate(self.pwr_draw_options.items()):
            self.pwr_draw_options[key] = is_number(output[i])

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
        config = f'10,{niters},100,100'
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
            for percentage in range(25, 101, 25):
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

        with open(os.path.join(self.result_dir, 'Preparation', 'pwr_draw_options.txt'), 'r') as f:
            options = f.readline().split(',')[:-1]
            for option in options:
                self.pwr_draw_options[option] = True

        dir_list = os.listdir(self.result_dir)
        continue_ = True
        if exp == 'all' or exp == 1:    
            if 'Experiment_1' in dir_list:  continue_ = self.process_exp_1(os.path.join(self.result_dir, 'Experiment_1'))
        if exp == 'all' or exp == 2:
            if 'Experiment_2' in dir_list:  self.process_exp_2(os.path.join(self.result_dir, 'Experiment_2'))

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

            # populate the dictionary
            data_dict['timestamp'].append(timestamp / 2938)
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
        print('  Processing experiment 1...')

        dir_list = os.listdir(result_dir)
        dir_list = sorted(dir_list, key=lambda dir: int(dir.split('%')[0]))
        subdir_list = os.listdir(os.path.join(result_dir, dir_list[0]))
        subdir_list = [dir for dir in subdir_list if 'find_scale_param' not in dir]
        subdir_list = sorted(subdir_list, key=lambda dir: int(dir.split('#')[1]))
        dir_list = [os.path.join(result_dir, dir, subdir) 
                        for dir in dir_list 
                            for subdir in subdir_list]

        self.PMD = False
        if os.path.exists(os.path.join(dir_list[0], 'PMD_data.bin')) and os.path.exists(os.path.join(dir_list[0], 'PMD_start_ts.txt')):
            self.PMD = True
        
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
                rise_times.append(statistics.mean(rise_time))
                print(f'    {key} rise time: {statistics.mean(rise_time):.2f} ms')
        
        if self.PMD:
            for key, value in self.pwr_draw_options.items():
                if value:
                    ss_err = results.pop(0)
                    print(f'    {key} steady state error: {statistics.mean(ss_err):.2f} W')

        print('  Done')

        if len(rise_times) == 1 and rise_times[0] > 900:
            return False

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

                    df['nv_power_error'] = df['nv_power'] - df['total_p']
                    # calculate the average of nv_power_error between timestamp 3000 and 6000
                    result.append(np.mean(df[(df['timestamp'] >= 3000) & (df['timestamp'] <= 6000)]['nv_power_error']))
                    mse = np.mean(df['nv_power_error']**2)
                    axis[2].fill_between(df['timestamp'], df['nv_power_error'], 0, alpha=0.5, label=f'{key} error, MSE={mse:.2f}')
            
            axis[2].set_xlabel('Time (ms)')
            axis[2].set_ylabel('Difference [W]')
            axis[2].grid(True, linestyle='--', linewidth=0.5)
            axis[2].set_xlim(axis[0].get_xlim())
            axis[2].legend()

            return result
            # END OF THE FUNCTION
        
        def find_rise_time(power, option):
            reduced_power = power[power['timestamp'] < 3000].copy()
            reduced_power = reduced_power[reduced_power[option] != reduced_power[option].shift()]
            reduced_power['pwr_diff'] = reduced_power[option].diff()
            # remove the power.draw column from the dataframe
            reduced_power.drop(columns=[option, ' utilization.gpu [%]', ' pstate', ' temperature.gpu', ' clocks.current.sm [MHz]'], axis=1, inplace=True)
            reduced_power.reset_index(drop=True, inplace=True)

            start_ts, end_ts = 0, 0
            started = False
            for index, row in reduced_power.iterrows():
                if index == len(reduced_power) - 1:    break

                if started and row['pwr_diff'] < 1 and reduced_power['pwr_diff'].iloc[index+1] < 1:
                    end_ts = row['timestamp']
                    break
                
                if not started and row['pwr_diff'] > 1 and reduced_power['pwr_diff'].iloc[index+1] > 1:
                    start_ts = row['timestamp']
                    started = True

            return end_ts - start_ts

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
            if value:    results.append(find_rise_time(power, f' {key} [W]'))

        # plotting
        n_rows = 2
        if self.PMD:    n_rows = 3
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
        if self.PMD:    ss_errs = plot_PMD_data(dir, t0, t_max, power, axis)
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
        self.PMD = False
        if os.path.exists(os.path.join(test_pmd_path, 'PMD_data.bin')) and os.path.exists(os.path.join(test_pmd_path, 'PMD_start_ts.txt')):
            print('    Found PMD data')
            self.PMD = True
            

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

        if not self.PMD:
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
            if not self.PMD:
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
        
        if self.PMD:
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
        if self.PMD:    n_rows = 5
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

        if self.PMD:
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
        if self.PMD:
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
            if abs(error) > 10:      print(f'error is too high: {error:.2f}%, check dir {result_dir}')
        
        # Plot resontruction result
        self._plot_reconstr_result(load_period, load, power, avg_window, delay, reconstructed, result_dir,
                                                PMD_data, PMD_avg_window, PMD_delay, PMD_reconstructed, error_msg)
        print('.', end='', flush=True)
        if abs(error) > 10:  return avg_window, delay, np.nan, np.nan, np.nan
        if self.PMD:    return avg_window, delay, PMD_avg_window, PMD_delay, error
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

        
        if self.PMD:    n_rows = 4
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

        if self.PMD:
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

        if self.PMD:
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

        if self.PMD:
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







