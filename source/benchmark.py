#!/usr/bin/env python3

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

# Steps
# -. Fix GPU core and mem clock
# -. Recompile code in case of changes
# -. Warn up GPU
# 0. Find the GPU power update frequency using sinewave load
# 1. Find the scale parameter
# 2. Find the power consumption duing load and idle
# 3. Try loading the GPU at different frequencies of square wave load
# 4. estimate the average window by optimization
# 5. Plot the results and summarize the statistics

class GPU_pwr_benchmark:
    def __init__(self, verbose=False):
        print('Initializing benchmark...')
        self.verbose = verbose
        self.BST = True
        self.repetitions = 64

        self.gpu_name = self._get_gpu_name()

        self.scale_gradient, self.scale_intercept = 1375, -1500
        self.pwr_update_freq = 100
        self.idle_pwr, self.load_pwr = 77, 208

        self.aliasing_ratios = [1/3, 2/3, 3/4, 1, 5/4, 4/3, 3/2]
        # 100ms: 50    67   75   100  125  133  150
        # 20ms:  10    13   15   20   25   27   30

        
    def prepare_experiment(self):
        print('Preparing for experiment...    ')

        run = 0
        while os.path.exists(os.path.join('results', f'{self.gpu_name}_run_#{run}')): run += 1
        self.result_dir = os.path.join('results', f'{self.gpu_name}_run_#{run}')
        os.makedirs(self.result_dir)

        self._recompile_load()
        self._warm_up()

        self.scale_gradient, self.scale_intercept = self._find_scale_parameter()
        self.pwr_update_freq = self._find_pwr_update_freq()
        self.idle_pwr, self.load_pwr = self._find_idle_load_pwr()
        
        self._print_general_info()

    def _print_general_info(self):
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
        subprocess.call(['make', '-C', 'source/'])
        print()

    def _get_gpu_name(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE)
        output = result.stdout.decode().split('\n')
        gpu_name = output[1]
        gpu_name = gpu_name.replace(' ', '_')

        return gpu_name
    
    def _benchload(self, load_pd, test_duration, store_path, delay=True):
        repetitions = str(int(test_duration / load_pd))
        niters = str(int(load_pd * self.scale_gradient + self.scale_intercept))
        subprocess.call(['./source/run_benchmark_load.sh', str(load_pd), niters, repetitions, store_path])
        if delay:  time.sleep(1)
    
    def _random_sleep(self):
        time.sleep(random.random())

    def _warm_up(self):
        print('  Warming up GPU...                    ', end='', flush=True)
        store_path = os.path.join(self.result_dir, 'warm_up')
        os.makedirs(store_path)
        for i in range(5):  self._benchload(100, 5000, store_path)
        print('done')

    def _f_duration(self, niter, store_path):
        subprocess.call(['./source/run_benchmark_load.sh', '50', str(niter), '30', store_path])
        
        df = pd.read_csv(os.path.join(store_path, 'timestamps.csv'))
        df = df.drop_duplicates(subset=['timestamp'])
        df.reset_index(drop=True, inplace=True)
        df['diff'] = df['timestamp'].diff()
        df = df.iloc[1::2]
        df = df.iloc[10:]
        avg = df['diff'].mean()

        return avg / 1000

    def _linear_regression(self, x, y):
        X = np.column_stack((np.ones(len(x)), x))
        coefficents = np.linalg.inv(X.T @ X) @ X.T @ y

        intercept = coefficents[0]
        gradient = coefficents[1]
        return intercept, gradient

    def _find_scale_parameter(self):
        print('  Finding scale parameter...       ', end='', flush=True)

        store_path = os.path.join(self.result_dir, 'find_scale_param')
        os.makedirs(store_path)

        niter = 1000
        duration = 0

        duration_list = []
        niter_list = []

        while duration < 200:
            if self.verbose: print(f'    {duration:.2f} ms')
            duration = self._f_duration(niter, store_path)
            if duration > 2:
                duration_list.append(duration)
                niter_list.append(niter)
            niter = int(niter * 1.5)


        # Linear regression
        intercept, gradient = self._linear_regression(duration_list, niter_list)
        print(f'{gradient:.2f} {intercept:.2f}')


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

    def _find_idle_load_pwr(self):
        print('  Finding idle and load power...       ', end='', flush=True)

        store_path = os.path.join(self.result_dir, 'find_idle_load_pwr')
        os.makedirs(store_path)

        load_pwr_list = []
        idle_pwr_list = []

        # warm up
        for i in range(3): self._benchload(2000, 2000, store_path)

        # measure
        for i in range(5):
            self._benchload(2000, 2000, store_path)

            load = pd.read_csv(os.path.join(store_path, 'timestamps.csv'))
            load['activity'] = (load.index / 2).astype(int) % 2
            load['timestamp'] = (load['timestamp'] / 1000).astype(int) 

            df = pd.read_csv(os.path.join(store_path, 'gpudata.csv'))
            df['timestamp'] = (pd.to_datetime(df['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
            df['timestamp'] -= 60*60*1000

            # find power during load
            load_begin = load['timestamp'][1] + 500
            load_end = load['timestamp'][2] - 500
            load_window = df[(df['timestamp'] >= load_begin) & (df['timestamp'] <= load_end)]
            load_pwr = load_window[' power.draw [W]'].mean()
            load_pwr_list.append(load_pwr)

            # find power during idle
            idle_begin = load['timestamp'][3] + 500
            idle_end = load['timestamp'][4] - 500
            idle_window = df[(df['timestamp'] >= idle_begin) & (df['timestamp'] <= idle_end)]
            idle_pwr = idle_window[' power.draw [W]'].mean()
            idle_pwr_list.append(idle_pwr)

            if self.verbose:
                print()
                print(f'load power: {load_pwr}')
                print(f'idle power: {idle_pwr}')
        
        avg_load_pwr = sum(load_pwr_list) / len(load_pwr_list)
        avg_idle_pwr = sum(idle_pwr_list) / len(idle_pwr_list)
        if self.verbose:
            print(f'avg load power: {avg_load_pwr}')
            print(f'avg idle power: {avg_idle_pwr}')

        print(f'{avg_idle_pwr:.2f} W | {avg_load_pwr:.2f} W')
        return avg_idle_pwr, avg_load_pwr

    def _find_pwr_update_freq(self):
        print('  Finding power update frequency...    ', end='', flush=True)
        
        store_path = os.path.join(self.result_dir, 'find_pwr_update_freq')
        os.makedirs(store_path)

        self._benchload(10, 1000, store_path)

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

    def run_experiment(self):
        for ratio in self.aliasing_ratios:
            load_pd = int(ratio * self.pwr_update_freq)
            print(f'Running experiment with load period of {load_pd} ms...')
            # create the store path
            ratio_store_path = os.path.join(self.result_dir, f'load_{load_pd}_ms')    
            os.makedirs(ratio_store_path)
                        
            for rep in range(self.repetitions):
                print(f'  Repetition {rep+1} of {self.repetitions}...    ')
                rep_store_path = os.path.join(ratio_store_path, f'rep_#{rep}')
                os.makedirs(rep_store_path)
                self._random_sleep()
                self._benchload(load_pd, 4000, rep_store_path, delay=False)
            
    def process_results(self, data_dir=None, notes=None):
        print('Processing results...')

        if data_dir is not None:
            self.result_dir = self.result_dir = os.path.join('results', f'{self.gpu_name}_run_#{data_dir}')

        results = []
        labels = []

        for ratio in self.aliasing_ratios:
            load_pd = int(ratio * self.pwr_update_freq)
            print(f'  Processing results for load period of {load_pd} ms...')
            ratio_store_path = os.path.join(self.result_dir, f'load_{load_pd}_ms')
            labels.append(f'{load_pd}')

            args = []

            for rep in range(self.repetitions):
                rep_store_path = os.path.join(ratio_store_path, f'rep_#{rep}')
                args.append((rep_store_path, load_pd))

            num_processes = min(self.repetitions, os.cpu_count())
            pool = Pool(processes=num_processes)
            ratio_results = pool.starmap(self._process_single_run, args)
            pool.close()
            pool.join()

            # find mean median and std of results
            mean = statistics.mean(ratio_results)
            median = statistics.median(ratio_results)
            std = statistics.stdev(ratio_results)
            print(f'    Mean:   {mean:.2f} ms')
            print(f'    Median: {median:.2f} ms')
            print(f'    Std:    {std:.2f} ms')
            
            results.append(ratio_results)
        
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

    def _adjacent_values(self, vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def _set_axis_style(self, ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, ha='center')
        ax.set_xlim(0.25, len(labels) + 0.75)

    def _violin_plot(self, ax, data, labels):
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
            self._adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=12, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

        self._set_axis_style(ax, labels)
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
        load.loc[load.index[-1], 'timestamp'] += 500 # NEED TO TAKE A LOOK AT THIS
        load.loc[load.index[-1], 'activity'] = 0

        power = pd.read_csv(os.path.join(result_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        if self.BST:    power['timestamp'] -= 60*60*1000
        power['timestamp'] -= t0
        power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= load['timestamp'].iloc[-1])]

        loss_func = partial(self._reconstr_loss, load, power)
        history_length = self._gradient_descent(loss_func, self.pwr_update_freq/2, lr=load_period*5)
        if self.verbose:    print(f'modeled history length: {history_length:.2f} ms')

        reconstructed = self._reconstruction(load, power, history_length)
        self._plot_reconstr_result(load_period, load, power, history_length, reconstructed, result_dir)

        return history_length

    def _gradient_descent(self, loss, x_init, lr=200, dx=0.1, error=1e-6, max_iter=30):
        x = x_init
        x_track = [x]
        loss_track = [loss(x)]
        for i in range(max_iter):
            dy = (loss(x + dx) - loss(x)) / dx
            x -= lr * dy

            if self.verbose:    print(f'Iteration {i}: x = {x}, loss = {loss(x)}')
            x_track.append(x)
            loss_track.append(loss(x))

            if np.abs(loss(x + dx) - loss(x)) < error:
                break
        
        return x_track[np.argmin(loss_track)]

    def _reconstr_loss(self, load, power, history_length, loss_type='MSE'):
        reconstructed = self._reconstruction(load, power, history_length)

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

    def _reconstruction(self, load, power, history_length):
        # copy the power df
        delay = 1
        reconstructed = power.copy()
        load['power_draw'] = load['activity'].apply(lambda x: self.idle_pwr if x == 0 else self.load_pwr)
        pwr_track = reconstructed[' power.draw [W]'].iloc[0]
        reconstr_pwr = self.idle_pwr
        for index, row in reconstructed.iterrows():
            if row['timestamp'] <= 500:
                reconstructed.loc[index, ' power.draw [W]'] = self.idle_pwr
            else:
                if row[' power.draw [W]'] != pwr_track:
                    pwr_track = row[' power.draw [W]']
                                
                    # need to take a look at the greater and greater equals shit !!!!!!!!!!!!!
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


def main():
    start = time.time()
    benchmark = GPU_pwr_benchmark()
    # benchmark.prepare_experiment()
    # benchmark.run_experiment()
    benchmark.process_results(0, 'x_init=50,delay=1')
    end = time.time()
    print(f'Time taken: {(end - start) // 60} minutes {round((end - start) % 60, 2)} seconds')



if __name__ == "__main__":
    main()