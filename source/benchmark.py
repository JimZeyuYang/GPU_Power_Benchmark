#!/usr/bin/env python3

import subprocess
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics
from functools import partial
import time
import random

# Steps
# -. Fix GPU core and mem clock
# -. Recompile code in case of changes
# 0. Find the GPU power update frequency using sinewave load
# 1. Find the scale parameter
# 2. Find the power consumption duing load and idle
# 3. Try loading the GPU at different frequencies of square wave load
# 4. estimate the average window by optimization
# 5. Plot the results and summarize the statistics

# 25 50 75 100 125 150 175 200
# 35 70

def main():
    # recomplie_load()
    # scale_param = find_scale_parameter()
    scale_param = 1375
    random_sleep()
    # find_idle_load_pwr(scale_param)

    '''
    find_pwr_update_freq(scale_param)
    # measure how long this takes
    start = time.time()
    
    window_size = 35
    repetetion = 2000 / window_size
    subprocess.call(['./source/run_benchmark.sh', str(window_size), str(window_size*scale_param), '50'])
    time.sleep(1)
    process_results('A100', window_size)

    end = time.time()
    print(f'Time taken: {end - start}')

    '''

def process_results(gpu_name, window_size):
    load = pd.read_csv('results/timestamps.csv')
    load.loc[-1] = load.loc[0] - 500000
    load.index = load.index + 1
    load = load.sort_index()
    load['activity'] = (load.index / 2).astype(int) % 2
    load['timestamp'] = (load['timestamp'] / 1000).astype(int) 
    t0 = load['timestamp'][0]
    load['timestamp'] -= t0
    load.loc[load.index[-1], 'timestamp'] += 500 # NEED TO TAKE A LOOK AT THIS
    load.loc[load.index[-1], 'activity'] = 0

    power = pd.read_csv('results/gpudata.csv')
    power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
    power['timestamp'] -= 60*60*1000
    power['timestamp'] -= t0
    power = power[power['timestamp'] >= 0]



    loss_func = partial(reconstr_loss, load, power, lowP=75, highP=200)
    history_length = gradient_descent(loss_func, 50)
    print(f'History length: {history_length}')

    reconstructed = reconstruction(load, power, history_length, 75, 200)
    plot_reconstr_result(gpu_name, window_size, load, power, reconstructed)


def gradient_descent(loss, x_init, lr=0.15, dx=0.02, error=1e-3, max_iter=10):
    x = x_init
    x_track = [x]
    loss_track = [loss(x)]
    for i in range(max_iter):
        dy = (loss(x + dx) - loss(x)) / dx
        x -= lr * dy

        print(f'Iteration {i}: x = {x}, loss = {loss(x)}')
        x_track.append(x)
        loss_track.append(loss(x))

        if np.abs(loss(x + dx) - loss(x)) < error:
            break
    
    return x_track[np.argmin(loss_track)]

def reconstr_loss(load, power, history_length, lowP, highP, loss_type='MSE'):
    reconstructed = reconstruction(load, power, history_length, lowP, highP)

    if loss_type == 'MSE':
        return np.mean((power[' power.draw [W]'] - reconstructed[' power.draw [W]'])**2)
    elif loss_type == 'MAE':
        return np.mean(np.abs(power[' power.draw [W]'] - reconstructed[' power.draw [W]']))
    elif loss_type == 'EMSE':
        return np.sqrt(np.mean((power[' power.draw [W]'] - reconstructed[' power.draw [W]'])**2))
    else:
        raise Exception('Invalid loss type')

def reconstruction(load, power, history_length, lowP, highP):
    # copy the power df
    reconstructed = power.copy()
    load['power_draw'] = load['activity'].apply(lambda x: lowP if x == 0 else highP)
    pwr_track = reconstructed[' power.draw [W]'].iloc[0]
    reconstr_pwr = lowP
    for index, row in reconstructed.iterrows():
        if row['timestamp'] <= 500:
            reconstructed.loc[index, ' power.draw [W]'] = lowP
        else:
            if row[' power.draw [W]'] != pwr_track:
                pwr_track = row[' power.draw [W]']

                # find rows in load df that are within the past history_length of the current timestamp
                load_window = load[(load['timestamp'] >= row['timestamp'] - history_length) 
                            & (load['timestamp'] < row['timestamp'])].copy()

                # interpolate the lower bound of the load window
                lb_t = row['timestamp'] - history_length
                lb_0 = load[load['timestamp'] < lb_t].iloc[-1]
                lb_1 = load[load['timestamp'] > lb_t].iloc[0]
                gradient = (lb_1['power_draw'] - lb_0['power_draw']) / (lb_1['timestamp'] - lb_0['timestamp'])
                lb_p = lb_0['power_draw'] + gradient * (lb_t - lb_0['timestamp'])

                # interpolate the upper bound of the load window
                ub_t = row['timestamp']
                ub_0 = load[load['timestamp'] < ub_t].iloc[-1]
                ub_1 = load[load['timestamp'] > ub_t].iloc[0]
                gradient = (ub_1['power_draw'] - ub_0['power_draw']) / (ub_1['timestamp'] - ub_0['timestamp'])
                ub_p = ub_0['power_draw'] + gradient * (ub_t - ub_0['timestamp'])
                
                # take the average of the load window
                t = np.concatenate((np.array([lb_t]), load_window['timestamp'].to_numpy(), np.array([ub_t])))
                p = np.concatenate((np.array([lb_p]), load_window['power_draw'].to_numpy(), np.array([ub_p])))
                reconstr_pwr = np.trapz(p, t) / history_length
            
            reconstructed.loc[index, ' power.draw [W]'] = reconstr_pwr

    return reconstructed

def plot_reconstr_result(gpu_name, window_size, load, power, reconstructed):
    # Plot the results
    fig, axis = plt.subplots(nrows=3, ncols=1)

    axis[0].plot(load['timestamp'], load['activity']*100, label='load')
    axis[0].plot(power['timestamp'], power[' utilization.gpu [%]'], label='utilization.gpu [%]')
    axis[0].set_xlabel('Time (ms)')
    axis[0].set_ylabel('Load')
    axis[0].grid(True, linestyle='--', linewidth=0.5)
    axis[0].set_xlim(0, load['timestamp'].max())
    axis[0].legend()
    axis[0].set_title(f'{gpu_name} - {window_size} ms load window')
    
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
    plt.savefig('results/result.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('results/result.svg', format='svg', bbox_inches='tight')
    plt.close('all')

def get_gpu_name():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE)
    output = result.stdout.decode().split('\n')
    gpu_name = output[1]
    
    return gpu_name

def recomplie_load():
    subprocess.call(['make', '-C', 'source/'])

def find_scale_parameter():
    niter = 1000
    duration = 0

    duration_list = []
    niter_list = []

    while duration < 200:
        duration = f_duration(niter)
        if duration > 1:
            duration_list.append(duration)
            niter_list.append(niter)
        niter *= 2

    print(duration_list)
    print(niter_list)

    gradient_list = []
    for i in range(len(niter_list) - 1):
        gradient_list.append((niter_list[i] - niter_list[i+1])/(duration_list[i] - duration_list[i+1]))

    print(gradient_list)
    # find average gradient
    avg_gradient = sum(gradient_list) / len(gradient_list)
    print(avg_gradient)
    # find the x intercept using the middle date point
    midpoint = int(len(duration_list)/2)
    x_intercept = duration_list[midpoint] - niter_list[midpoint] / avg_gradient
    print(x_intercept)
    return avg_gradient

def f_duration(niter):
    subprocess.call(['./source/run_benchmark.sh', '100', str(niter), '30'])

    df = pd.read_csv('results/timestamps.csv')
    df = df.drop_duplicates(subset=['timestamp'])
    df.reset_index(drop=True, inplace=True)
    df['diff'] = df['timestamp'].diff()
    df = df.iloc[1::2]
    df = df.iloc[10:]
    avg = df['diff'].mean()

    return avg / 1000

def find_idle_load_pwr(scale_param):
    load_pwr_list = []
    idle_pwr_list = []

    # warm up
    for i in range(9): benchload(2000, scale_param, 2000)

    # measure
    for i in range(5):
        benchload(2000, scale_param, 2000)

        load = pd.read_csv('results/timestamps.csv')
        load['activity'] = (load.index / 2).astype(int) % 2
        load['timestamp'] = (load['timestamp'] / 1000).astype(int) 

        df = pd.read_csv('results/gpudata.csv')
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
        print(f'load power: {load_pwr}')
        print(f'idle power: {idle_pwr}')
        print()
    
    avg_load_pwr = sum(load_pwr_list) / len(load_pwr_list)
    avg_idle_pwr = sum(idle_pwr_list) / len(idle_pwr_list)
    print(f'avg load power: {avg_load_pwr}')
    print(f'avg idle power: {avg_idle_pwr}')

    return avg_idle_pwr, avg_load_pwr

def find_pwr_update_freq(scale_param):
    benchload(10, scale_param, 1000)
    df = pd.read_csv('results/gpudata.csv')
    df['timestamp'] = (pd.to_datetime(df['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")

    period_list = []
    last_pwr = df.iloc[0]
    for index, row in df.iterrows():
        if row[' power.draw [W]'] != last_pwr[' power.draw [W]']:
            period_list.append(row['timestamp'] - last_pwr['timestamp'])
            last_pwr = row
            
    avg_period = sum(period_list) / len(period_list)
    median_period = statistics.median(period_list)
    std_period = statistics.stdev(period_list)
    print(avg_period)
    print(median_period)
    print(std_period)

    # plot the period_list as histogram
    fig, axis = plt.subplots(nrows=1, ncols=1)
    axis.hist(period_list, bins=20)
    axis.set_xlabel('Period (ms)')
    axis.set_ylabel('Frequency')
    axis.grid(True, linestyle='--', linewidth=0.5)

    fig.set_size_inches(8, 6)
    plt.savefig('results/power_update_freq.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('results/power_update_freq.svg', format='svg', bbox_inches='tight')

    plt.close('all')

    return avg_period

def benchload(window, scale_param, test_duration):
    repetitions = str(test_duration / window)
    subprocess.call(['./source/run_benchmark.sh', str(window), str(window*scale_param), repetitions])
    time.sleep(1)


def random_sleep():
    time.sleep(random.random())

if __name__ == "__main__":
    main()