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

class GPU_pwr_benchmark:
    def __init__(self, verbose=False):
        print('_________________________')
        print('Initializing benchmark...')
        self.verbose = verbose
        self.BST = True
        self.repetitions = 64
        self.nvsmi_smp_pd = 5
        self.aliasing_ratios = [2/3, 3/4, 4/5, 6/5, 5/4, 4/3]
        
    def prepare_experiment(self):
        print('___________________________')
        print('Preparing for experiment...')

        self.gpu_name = self._get_gpu_name()
        
        run = 0
        while os.path.exists(os.path.join('results', f'{self.gpu_name}_run_#{run}')): run += 1
        self.result_dir = os.path.join('results', f'{self.gpu_name}_run_#{run}')
        os.makedirs(self.result_dir)
        os.makedirs(os.path.join(self.result_dir, 'Preparation'))

        self._recompile_load()
        self._warm_up()

        self.scale_gradient, self.scale_intercept = self._find_scale_parameter()
        # self.scale_gradient, self.scale_intercept = 1386, -1637
        self.pwr_update_freq = self._find_pwr_update_freq()
        
        self._print_general_info()

    def _print_general_info(self):
        print()
        time_ = 'Date and time:        ' + time.strftime('%d/%m/%Y %H:%M:%S')
        gpu =   'Benchmarking on GPU:  ' + self.gpu_name
        host =  'Host machine:         ' + os.uname()[1]

        max_len = max(len(time_), len(gpu), len(host))
        print('+', '-'*(max_len), '+')
        print('|', time_ + ' '*(max_len - len(time_)), '|')
        print('|', gpu + ' '*(max_len - len(gpu)), '|')
        print('|', host + ' '*(max_len - len(host)), '|')
        print('+', '-'*(max_len), '+')

    def _recompile_load(self):
        print('Recompiling benchmark load...')
        # make clean and make
        # try can catch the error
        subprocess.call(['make', '-C', 'source/', 'clean'])
        return_code = subprocess.call(['make', '-C', 'source/'])
        if return_code != 0:  raise Exception('Error compiling benchmark')

        print()

    def _get_gpu_name(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE)
        output = result.stdout.decode().split('\n')
        gpu_name = output[1]
        gpu_name = gpu_name.replace(' ', '_')
        gpu_name = gpu_name[7:]

        return gpu_name
    
    def _benchload(self, load_pd, test_duration, store_path, delay=True):
        repetitions = str(int(test_duration / load_pd))
        niters = str(int(load_pd * self.scale_gradient + self.scale_intercept))
        subprocess.call(['./source/run_benchmark_load.sh', str(load_pd), niters, repetitions, store_path])
        if delay:  time.sleep(1)

    def _run_benchmark(self, experiment, config, store_path, delay=True):
        '''
        Config for Experiment 1:
        <delay>,<niter>,<testlength>,<percentage>
            <delay>         : Length of idle time in ms
            <niter>         : Number of iterations to control the load time
            <testlength>    : Number of the square wave periods
            <percentage>    : Percentage of the SMs to be loaded
        '''
        subprocess.call(['./source/run_benchmark_load.sh', str(experiment), config, store_path, str(self.nvsmi_smp_pd)])
        if delay:  time.sleep(1)

    def _warm_up(self):
        print('Warming up GPU...                    ', end='', flush=True)
        store_path = os.path.join(self.result_dir, 'Preparation', 'warm_up')
        os.makedirs(store_path)

        # While loop that run for at least 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            self._run_benchmark(1, '50,1000000,20,100', store_path)
        print('done')

    def _find_scale_parameter(self, store_path=None, percentage=100):
        print('Finding scale parameter...           ', end='', flush=True)

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

        niter = 10000
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
        return median_period

    def run_experiment(self, experiment):
        if experiment == 1:
            # Experiment 1: Steady state and trasient response analysis
            print('_____________________________________________________________________')
            print('Running experiment 1: Steady state and transient response analysis...')
            os.makedirs(os.path.join(self.result_dir, 'Experiment_1'))
            for percentage in range(25, 101, 25):
                print(f'  Running experiment with {percentage}% load...')
                # create the store path
                percentage_store_path = os.path.join(self.result_dir, 'Experiment_1', f'{percentage}%_load')
                os.makedirs(percentage_store_path)
                scale_gradient, scale_intercept = self._find_scale_parameter(percentage_store_path, percentage)

                for rep in range(int(self.repetitions/4)):
                    print(f'    Repetition {rep+1} of {int(self.repetitions/4)}...')
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
            os.makedirs(os.path.join(self.result_dir, 'Experiment_2'))
            for ratio in self.aliasing_ratios:
                load_pd = int(ratio * self.pwr_update_freq)
                print(f'  Running experiment with load period of {load_pd} ms...\n  ', end='', flush=True)
                # create the store path
                ratio_store_path = os.path.join(self.result_dir, 'Experiment_2', f'load_{load_pd}_ms')    
                os.makedirs(ratio_store_path)
                            
                for rep in range(self.repetitions):
                    print(f'  Repetition {rep+1} of {self.repetitions}...    ')
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
        if 'Experiment_1' in dir_list:
            self.process_exp_1(os.path.join(self.result_dir, 'Experiment_1'))

    def process_exp_1(self, result_dir):
        def plot_result(dir):
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

            power = pd.read_csv(os.path.join(dir, 'gpudata.csv'))
            power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
            if self.BST:    power['timestamp'] -= 60*60*1000
            power['timestamp'] -= t0
            power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= load['timestamp'].iloc[-1])]

            # plotting
            fig, axis = plt.subplots(nrows=3, ncols=1)

            axis[0].plot(load['timestamp'], load['activity']*100, label='load')
            axis[0].plot(power['timestamp'], power[' utilization.gpu [%]'], label='utilization.gpu [%]')
            axis[0].plot(power['timestamp'], power[' temperature.gpu'], label='temperature.gpu')
            axis[0].set_xlabel('Time (ms)')
            axis[0].set_ylabel('Load')
            axis[0].grid(True, linestyle='--', linewidth=0.5)
            axis[0].set_xlim(0, load['timestamp'].max())
            axis[0].legend()
            axis[0].set_title(f'{self.gpu_name} - {load_percentage}% load')
            
            axis[1].plot(power['timestamp'], power[' power.draw [W]'], label='power')
            axis[1].set_xlabel('Time (ms)')
            axis[1].set_ylabel('Power [W]')
            axis[1].grid(True, linestyle='--', linewidth=0.5)
            axis[1].set_xlim(axis[0].get_xlim())
            axis[1].legend()

            axis[2].plot(power['timestamp'], power[' clocks.current.sm [MHz]'], label='clocks.current.sm [MHz]')
            axis[2].set_xlabel('Time (ms)')
            axis[2].set_ylabel('Clock [MHz]')
            axis[2].grid(True, linestyle='--', linewidth=0.5)
            axis[2].set_xlim(axis[0].get_xlim())
            axis[2].legend()

            fig.set_size_inches(20, 12)
            plt.savefig(os.path.join(dir, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
            plt.savefig(os.path.join(dir, 'result.svg'), format='svg', bbox_inches='tight')
            plt.close('all')

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
        pool.map(plot_result, dir_list)
        pool.close()
        pool.join()

        # for dir in dir_list:
        #     print(f'  Processing {dir}...')
        #     plot_result(dir)

    def process_exp_2(self):
        # list the directories in the result directory
        dir_list = os.listdir(self.result_dir)
        # get the directories that ends with 'ms'
        dir_list = [dir for dir in dir_list if dir.endswith('ms')]
        # sort the directories based on the middle split
        dir_list = sorted(dir_list, key=lambda dir: int(dir.split('_')[1]))
        print(f'  Found {len(dir_list)} directories to process...')

        results = []
        labels = []

        self.pwr_update_freq = int(np.mean([int(dir.split('_')[1]) for dir in dir_list]))


        ctr = 0
        for dir in dir_list:
            ctr += 1
            # if ctr < 3:
            #     continue
            
            load_pd = int(dir.split('_')[1])

            print(f'  Processing results for load period of {load_pd} ms...')
            ratio_store_path = os.path.join(self.result_dir, dir)
            labels.append(f'{load_pd}')

            args = []

            self.repetitions = 1
            for rep in range(self.repetitions):
                rep_store_path = os.path.join(ratio_store_path, f'rep_#{rep+15}')
                args.append((rep_store_path, load_pd))

            num_processes = min(self.repetitions, os.cpu_count())
            num_processes = 1
            pool = Pool(processes=num_processes)
            ratio_results = pool.starmap(self._process_single_run, args)
            pool.close()
            pool.join()

            continue
            
            # find mean median and std of results
            mean = statistics.mean(ratio_results)
            median = statistics.median(ratio_results)
            std = statistics.stdev(ratio_results)
            print(f'    Mean:   {mean:.2f} ms')
            print(f'    Median: {median:.2f} ms')
            print(f'    Std:    {std:.2f} ms')
            
            results.append(ratio_results)
            
        
        return 0

        results_flat = [item for sublist in results for item in sublist]
        mean = statistics.mean(results_flat)
        median = statistics.median(results_flat)
        std = statistics.stdev(results_flat)
        print('  Overall:')
        print(f'    Mean:   {mean:.2f} ms')
        print(f'    Median: {median:.2f} ms')
        print(f'    Std:    {std:.2f} ms')

        results.append(results_flat)
        labels.append('all')

        # plot the results
        fig, axis = plt.subplots(nrows=1, ncols=1)
        self._violin_plot(axis, results, labels)
        axis.set_xlabel('Load period (ms)')
        axis.set_ylabel('History Length (ms)')
        axis.set_title(f'History Length vs Load Period ({self.gpu_name})')
        # add some texts at the bottom of the plot
        axis.text(0.05, -0.1,  f'Mean history Length:   {mean:.2f} ms', transform=axis.transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        axis.text(0.05, -0.15, f'Median history Length: {median:.2f} ms', transform=axis.transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        axis.text(0.05, -0.2,  f'Std history Length:    {std:.2f} ms', transform=axis.transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})
        axis.text(0.5, -0.15, f'GPU: {self.gpu_name}', transform=axis.transAxes, ha='left', va='center', fontdict={'fontfamily': 'monospace'})

        fig.set_size_inches(10, 6)
        fname = f'history_length_{self.gpu_name}{notes}'
        plt.savefig(os.path.join(self.result_dir, fname+'.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(self.result_dir, fname+'.svg'), format='svg', bbox_inches='tight')
        plt.close('all')

    def _violin_plot(self, ax, data, labels):
        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value

        def set_axis_style(ax, labels):
            ax.get_xaxis().set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, ha='center')
            ax.set_xlim(0.25, len(labels) + 0.75)

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

        loss_func = partial(self._reconstr_loss, load, power)
        avg_window, delay = self._gradient_descent(
                                loss_func,
                                self.pwr_update_freq/2,
                                self.nvsmi_smp_pd/200*self.pwr_update_freq,
                                lr=load_period/10*self.pwr_update_freq)

        if self.verbose:    print(f'modeled avg_window: {avg_window}ms, delay: {delay}ms')

        self._plot_loss_function(loss_func, avg_window, int(np.ceil(delay)), result_dir)

        # reconstructed = self._reconstruction(load, power, history_length, delay)
        # self._plot_reconstr_result(load_period, load, power, history_length, reconstructed, result_dir)

        return 20

        return history_length

    def _plot_loss_function(self, loss_func, window_range, delay_range, store_path):        
        fig, ax = plt.subplots(1, 1)
        color_codes = ["#000080", "#0B2545", "#154360", "#1F618D", "#2980B9",
                        "#3498DB", "#5DADE2", "#87CEEB", "#ADD8E6", "#D6EAF8", "#F0F8FF"]

        window_range = 100
        window_range = round(window_range / 5) * 5
        if window_range <= 10:
            window_range = (1, 21, 1)
            ax.set_xticks(1, 21, 2)
        else:
            window_range = (window_range-99, window_range+101, 2)
            ax.set_xticks(np.arange(window_range[0], window_range[1], 20))
        windows = list(range(*window_range))

        delay_range += 4
        if delay_range >= 11:    delay_range = 11

        for delay in range(0, delay_range):
            print(f'plotting delay = {delay}ms')
            losses = []
            for window in windows:
                losses.append(loss_func(window, delay))
            ax.plot(windows, losses, label=f'delay = {delay}ms', color=color_codes[delay], zorder=10-delay)

        ax.set_xlim(windows[0], windows[-1])
        # ax.set_ylim(None, loss_func(windows[0], 2))
        ax.set_xlabel('Average window duration (ms)')
        ax.set_ylabel('Reconstruction loss (MSE)')
        ax.set_title('Reconstruction loss vs. window duration and delay')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', linewidth=0.5)

        fig.set_size_inches(8, 6)
        plt.savefig(os.path.join(store_path, 'loss_function.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(store_path, 'loss_function.svg'), format='svg', bbox_inches='tight')
        plt.close('all')

    def _gradient_descent(self, loss_func, x_init, y_init, lr=200, h=0.05, threshold=1e-6, max_iter=200):
        if self.verbose:    print('    Coarse search')
        x = x_init
        y = y_init

        for i in range(10):
            dx = (loss_func(x + h, y_init) - loss_func(x - h, y_init)) / (2 * h)
            x -= lr * dx 
            if x < 2*h:    x = 2*h
            if self.verbose:    print(f'      Iteration {i:3}: x = {x:<.6f}, y = {y:<.6f}, dx = {abs(dx):<.4e}, loss = {loss_func(x, y_init):<.8f}')
            if abs(dx) < threshold:    break

        if self.verbose:    print('    Fine search')
        var_track = [[x, y]]
        loss_track = [loss_func(x, y)]
        osc_ctr = 0

        for i in range(max_iter):
            dx = (loss_func(x + h, y) - loss_func(x - h, y)) / (2 * h)
            dy = (loss_func(x, y + h) - loss_func(x, y - h)) / (2 * h)

            x -= lr / (i*0.001+1.5) * dx
            y -= lr / (i*0.005+1.5) * dy
            if x < h:   x = h
            if y < h:   y = h

            # if the last 8 vartrac contains more than 3 items equal to h, then stop
            osc_ctr += 1
            if osc_ctr > 8 and np.sum(np.array(var_track[-8:]) == h) >= 3:
                lr *= 0.75
                osc_ctr = 0
                print('reducing lr')

            gm_grad = np.sqrt(dx**2 + dy**2)

            loss = loss_func(x, y)
            if self.verbose:    print(f'      Iteration {i:3}: x = {x:<.6f}, y = {y:<.6f}, dxdy = {gm_grad:.4e}, loss = {loss:<.8f}')
            
            var_track.append([x, y])
            loss_track.append(loss)

            if gm_grad < threshold:    break
            if i > 5 and np.var(loss_track[-5:]) < 1e-18:    break

        return var_track[np.argmin(loss_track)]

    def _reconstr_loss(self, load, power, history_length, delay, loss_type='MSE'):
        reconstructed = self._reconstruction(load, power, history_length, delay)

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
        # copy the power df
        reconstructed = power.copy()

        self.idle_pwr = 0
        self.load_pwr = 1

        load['power_draw'] = load['activity'].apply(lambda x: self.idle_pwr if x == 0 else self.load_pwr)
        pwr_track = reconstructed[' power.draw [W]'].iloc[0]
        reconstr_pwr = self.idle_pwr
        for index, row in reconstructed.iterrows():
            if row['timestamp'] <= 500:
                reconstructed.loc[index, ' power.draw [W]'] = self.idle_pwr
            else:
                if row[' power.draw [W]'] != pwr_track:
                    pwr_track = row[' power.draw [W]']
                    
                    # find rows in load df that are within the past history_length of the current timestamp
                    load_window = load[(load['timestamp'] >= row['timestamp'] - history_length - delay) 
                                & (load['timestamp'] < row['timestamp'] - delay)].copy()

                    # interpolate the lower bound of the load window
                    lb_t = row['timestamp'] - history_length - delay
                    lb_0 = load[load['timestamp'] < lb_t].iloc[-1]
                    lb_1 = load[load['timestamp'] >= lb_t].iloc[0]
                    gradient = (lb_1['power_draw'] - lb_0['power_draw']) / (lb_1['timestamp'] - lb_0['timestamp'])
                    lb_p = lb_0['power_draw'] + gradient * (lb_t - lb_0['timestamp'])

                    # interpolate the upper bound of the load window
                    ub_t = row['timestamp'] - delay
                    ub_0 = load[load['timestamp'] < ub_t].iloc[-1]
                    ub_1 = load[load['timestamp'] >= ub_t].iloc[0]

                    gradient = (ub_1['power_draw'] - ub_0['power_draw']) / (ub_1['timestamp'] - ub_0['timestamp'])
                    ub_p = ub_0['power_draw'] + gradient * (ub_t - ub_0['timestamp'])
                    
                    # take the average of the load window
                    t = np.concatenate((np.array([lb_t]), load_window['timestamp'].to_numpy(), np.array([ub_t])))
                    p = np.concatenate((np.array([lb_p]), load_window['power_draw'].to_numpy(), np.array([ub_p])))
                    reconstr_pwr = np.trapz(p, t) / history_length
                
                reconstructed.loc[index, ' power.draw [W]'] = reconstr_pwr

        return reconstructed

    def _plot_reconstr_result(self, load_period, load, power, history_length, reconstructed, store_path):
        # Plot the results
        fig, axis = plt.subplots(nrows=3, ncols=1)

        axis[0].plot(load['timestamp'], load['activity']*100, label='load')
        axis[0].plot(power['timestamp'], power[' utilization.gpu [%]'], label='utilization.gpu [%]')
        axis[0].set_xlabel('Time (ms)')
        axis[0].set_ylabel('Load')
        axis[0].grid(True, linestyle='--', linewidth=0.5)
        axis[0].set_xlim(0, load['timestamp'].max())
        axis[0].legend()
        axis[0].set_title(f'{self.gpu_name} - {load_period} ms load window - modeled history length: {history_length:.2f} ms')
        
        axis[1].plot(power['timestamp'], power[' power.draw [W]'], label='power')
        axis[1].set_xlabel('Time (ms)')
        axis[1].set_ylabel('Power [W]')
        axis[1].grid(True, linestyle='--', linewidth=0.5)
        axis[1].set_xlim(axis[0].get_xlim())

        axis[2].plot(reconstructed['timestamp'], reconstructed[' power.draw [W]'], label='reconstructed')
        axis[2].set_xlabel('Time (ms)')
        axis[2].set_ylabel('Power [W]')
        axis[2].grid(True, linestyle='--', linewidth=0.5)
        axis[2].set_xlim(axis[0].get_xlim())

        fig.set_size_inches(20, 12)
        plt.savefig(os.path.join(store_path, 'result.jpg'), format='jpg', dpi=256, bbox_inches='tight')
        plt.savefig(os.path.join(store_path, 'result.svg'), format='svg', bbox_inches='tight')
        plt.close('all')