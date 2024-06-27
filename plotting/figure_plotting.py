#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
import os
import struct
import csv
import statistics
from matplotlib.ticker import ScalarFormatter



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

def convert_pmd_data(dir):
    with open(os.path.join(dir, 'PMD_start_ts.txt'), 'r') as f:  pmd_start_ts = int(f.readline()) - 50000
    with open(os.path.join(dir, 'PMD_data.bin'), 'rb') as f:     pmd_data = f.read()

    bytes_per_sample = 18
    num_samples = int(len(pmd_data) / bytes_per_sample)

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
        data_dict['eps_total_p'].append(  values[5] * 0.007568 * values[6] * 0.0488)


    df = pd.DataFrame(data_dict)

    df['timestamp'] += (pmd_start_ts/1000 - df['timestamp'][0])
    df['total_p'] = df['pcie_total_p'] + df['eps_total_p']

    return df

def violin_plot(ax, data, labels):
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

def steam_gpu_share():
    labels =  ['Other',   'Intel',   'AMD',     'Older',   'Maxwell', 'Pascal',  'Turing',  'Ampere',  'Ada Lovelace']
    
    sizes   = [ 0.25,      8.66,      15.04,     5.26,      2.37,      14.12,     24.77,     26.62,     2.91         ]
    colors  = ['#f2d600', '#0071c5', '#ed1c24', '#2f4a00', '#466f00', '#5e9400', '#76b900', '#91c732', '#acd566'     ]
    explode = ( 0.17,      0.12,       0.12,       0,         0,         0,         0,         0,         0            )

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, explode=explode, pctdistance=0.8)


    labels_t5 = ['Intel',   'AMD',     'Older', 'Pascal', 'Volta', 'Ampere', 'Hopper']

    sizes_t5 =  [3.28,      6.01,      3.83,    3.28,     35.52,   42.62,     5.46       ]
    colors_ts = ['#0071c5', '#ed1c24', '#2f4a00', '#5e9400', '#76b900', '#91c732', '#acd566']
    explode_t5 = (0.15,       0.15,       0,       0,        0,        0,       0    )

    ax[1].pie(sizes_t5, labels=labels_t5, autopct='%1.1f%%', colors=colors_ts, startangle=90, explode=explode_t5, pctdistance=0.8)

    ax[0].set_xlabel('Steam Hardware Survey', fontweight='bold', labelpad=-15)
    ax[1].set_xlabel('Top500 List', fontweight='bold', labelpad=-15)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.0, wspace=0.1, hspace=0)
    # plt.tight_layout()

    fig.set_size_inches(7, 3)
    plt.savefig('steam_gpu_share.jpg', format='jpg', dpi=512, bbox_inches='tight')
    plt.savefig('steam_gpu_share.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()

def steady_state_accuracy():
    GPUs = [
        "RTX 3090 #1", "RTX 3090 #2", "RTX 3090 #3", "RTX 3090 #4", "RTX 3090 #5",
        "RTX 4090",
        "RTX 3070 Ti",
        "RTX 2080 Ti", "RTX 2060 S",
        "GTX TITAN Xp", "GTX 1080 Ti", "GTX 1080",
        "GTX TITAN X",
        "RTX A5000",
        "A100 40G", "P100 16G", "M40", "K80", "K40m",
    ]

    Data = {
        'Offset': (
                    3.468566667, -1.6615, 3.9206, -5.0714, -8.5276,
                    -12.354,
                    -4.801566667,
                    -5.7953, 11.8485,
                    2.5377, -0.334266667, -5.155333333,
                    1.2307,
                    8.764,
                    -1.210333333, 0.883166667, 1.4497, -6.2932, 0.8057,
                     ),

        'Gradient':  (
                    -5.493, -0.6464, -1.772, 1.4763, 1.7246,
                    4.450333333,
                    0.393966667,
                    -3.3428, -1.303366667,
                    -2.5911, -0.122033333, 6.363033333,
                    1.057666667,
                    -9.4997,
                    -4.582566667, -0.722133333, -2.874166667, -3.6307, 2.4105,
                     )
            }
    colors = {'Offset': '#008FC2', 'Gradient': '#76b900'}

    # reversed order
    GPUs = GPUs[::-1]
    Data['Gradient'] = Data['Gradient'][::-1]
    Data['Offset'] = Data['Offset'][::-1]

    '''
    x = np.arange(len(GPUs))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in Data.items():
        offset = width * multiplier
        rects = ax.barh(x + offset, measurement, width, label=attribute, color=colors[attribute])
        ax.bar_label(rects, padding=3, labels=[f'{i:.2f}' for i in measurement])
        multiplier += 1


    ax.axvline(0, color='grey', linestyle='dotted')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Offset / y-intercept (W)', fontweight='bold')
    ax.set_xlim(-17, 17)
    # ax.set_ylim(-0.55, 11)
    # print ylim
    print(ax.get_ylim())
    ax.set_ylim(-0.44, 18.84)

    ax.set_yticks(x + width/2)
    ax.set_yticklabels(GPUs)
    ax.xaxis.label.set_color(colors['Offset'])
    ax.tick_params(axis='x', colors=colors['Offset'])
    
    # Add the legend and get the handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    plt.legend(handles, labels, loc='lower right')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Gradient / Percentage Error (%)', fontweight='bold')
    ax2.xaxis.label.set_color(colors['Gradient'])
    ax2.tick_params(axis='x', colors=colors['Gradient'])
    ax2.spines['top'].set_color(colors['Gradient'])
    ax2.spines['bottom'].set_color(colors['Offset'])

    fig.set_size_inches(6, 8.5)
    '''
    fig, axis = plt.subplots(nrows=1, ncols=2, sharey=True)
    
    x = np.arange(len(GPUs))  # the label locations
    width = 0.9  # the width of the bars

    gradient_rects = axis[0].barh(x, Data['Gradient'], width, label='Gradient', color=colors['Gradient'])
    offset_rects = axis[1].barh(x, Data['Offset'], width, label='Offset', color=colors['Offset'])

    axis[0].bar_label(gradient_rects, padding=3, labels=[f'{i:.2f}' if abs(i) < 7 else '' for i in Data['Gradient']])
    axis[0].bar_label(gradient_rects, padding=-30, labels=[f'{i:.2f}' if abs(i) >= 7 else '' for i in Data['Gradient']])
    axis[1].bar_label(offset_rects, padding=3, labels=[f'{i:.2f}' if abs(i) < 9 else '' for i in Data['Offset']])
    axis[1].bar_label(offset_rects, padding=-35, labels=[f'{i:.2f}' if abs(i) >= 9 else '' for i in Data['Offset']])

    axis[0].axvline(0, color='grey')
    axis[1].axvline(0, color='grey')

    axis[0].set_xlabel('Gradient / Percentage Error (%)')
    axis[1].set_xlabel('Offset / y-intercept (W)')

    axis[0].set_xlim(-10, 10)
    axis[1].set_xlim(-13, 13)
    axis[0].set_yticks(x)
    axis[0].set_yticklabels(GPUs)

    axis[0].set_ylim(-0.75, 18.75)

    axis[0].xaxis.label.set_color(colors['Gradient'])
    axis[0].tick_params(axis='x', colors=colors['Gradient'])
    axis[1].xaxis.label.set_color(colors['Offset'])
    axis[1].tick_params(axis='x', colors=colors['Offset'])
    axis[0].spines['bottom'].set_color(colors['Gradient'])
    axis[1].spines['bottom'].set_color(colors['Offset'])
    
    axis[0].xaxis.grid(True, linestyle='--', linewidth=0.5)
    axis[1].xaxis.grid(True, linestyle='--', linewidth=0.5)

    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.1, top=0.9, wspace=0.065, hspace=0)

    fig.set_size_inches(6.5, 5)

    plt.savefig('steady_state_accuracy.jpg', format='jpg', dpi=512, bbox_inches='tight')
    plt.savefig('steady_state_accuracy.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()

def pwr_update_freq():
    def process_data(df):
        df['timestamp'] = (pd.to_datetime(df['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")

        period_list = []
        last_pwr = df.iloc[0]
        pwr_option = ' power.draw [W]'
        for index, row in df.iterrows():
            if row[pwr_option] != last_pwr[pwr_option]:
                period_list.append(row['timestamp'] - last_pwr['timestamp'])
                last_pwr = row
        
        avg_period = np.mean(period_list)
        median_period = np.median(period_list)
        std_period = np.std(period_list)

        print(f'avg period:    {avg_period:.2f} ms')
        print(f'median period: {median_period:.2f} ms')
        print(f'std period:    {std_period:.2f} ms')

        # revove outliers
        period_list = [i for i in period_list if i < 200]

        return period_list


    V100 = pd.read_csv('data/pwr_update_freq/V100_gpudata.csv')
    A100 = pd.read_csv('data/pwr_update_freq/A100_gpudata.csv')

    V100_period_list = process_data(V100)
    V100_period_list = V100_period_list[::5]
    A100_period_list = process_data(A100)

    V100_bins = np.arange(min(V100_period_list), max(V100_period_list) + 1, 1)
    A100_bins = np.arange(min(A100_period_list), max(A100_period_list) + 1, 1)

    # plot the period_list as histogram
    fig, axis = plt.subplots(nrows=1, ncols=1)
    axis.hist(V100_period_list, bins=V100_bins, label='Tesla V100-SXM2-16GB', color='#008FC2')
    axis.hist(A100_period_list, bins=A100_bins, label='Tesla A100-PCIe-40GB', color='#76b900')
    axis.set_xlabel('Power Update Period (ms)')
    axis.set_ylabel('Frequency')
    axis.set_xlim(0, 120)
    axis.legend(loc='upper center')
    axis.grid(True, linestyle='--', linewidth=0.5)

    fig.set_size_inches(6.5, 2)
    plt.savefig('power_update_freq.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('power_update_freq.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)

    plt.close('all')

def scaling_params():
    niters = [225000, 337500, 506250, 759375, 1139062, 1708593, 2562889, 3844333, 5766499, 8649748, 12974622,
                19461933, 29192899, 43789348, 65684022, 98526033, 147789049, 221683573, 332525359, 498788038]

    RTX3090 = [2.8207, 4.2046, 6.2817, 8.398, 12.4752, 19.088849999999997, 29.25215, 43.754949999999994, 
                65.92405000000001, 99.268, 149.206, 224.1428, 336.42955, 507.37435, 763.25755, 1148.2991000000002,
                1723.8624499999999, 2589.2348500000003, 3885.3249, 5828.4539]

    A100 = [5.1573, 7.71185, 11.5556, 17.32705, 25.9817, 38.97465, 58.4456, 87.6444, 131.45495, 197.16775, 295.7412,
            443.5997, 665.3719, 998.05955, 1497.0885, 2245.5759, 3368.32845, 5052.5001, 7578.8348, 11367.46275]

    niters = niters[:-3]
    RTX3090 = [i / 1000 for i in RTX3090[:-3]]
    A100 = [i / 1000 for i in A100[:-3]]
    rtx3090_intercept, rtx3090_gradient, rtx3090_r2 = linear_regression(niters, RTX3090)
    a100_intercept, a100_gradient, a100_r2 = linear_regression(niters, A100)

    x = np.linspace(0, max(niters), 100)
    rtx3090_y = rtx3090_gradient * x + rtx3090_intercept
    a100_y = a100_gradient * x + a100_intercept

    fig, axis = plt.subplots(nrows=1, ncols=1)

    axis.plot(niters, RTX3090, 'o', color='#008FC2', label='GeForce RTX 3090')
    axis.plot(x, rtx3090_y, ':', label=f'Line of best fit ($R^2$={rtx3090_r2:.2f})', color='#008FC2')

    axis.plot(niters, A100, 'o', color='#76b900', label='Tesla A100-PCIe-40GB')
    axis.plot(x, a100_y, ':', label=f'Line of best fit ($R^2$={a100_r2:.2f})', color='#76b900')

    axis.set_xlabel('Iterations')
    axis.set_ylabel('Kernel Execution Time (s)')
    axis.set_ylim([-0.15,3.6])
    axis.legend(loc='lower right')

    axis.grid(True, linestyle='--', linewidth=0.5)

    # Here is the part for the inset
    axins = inset_axes(axis, width="35%", height="40%", loc=2)  # loc=2 corresponds to 'upper left'
    axins.plot(niters, RTX3090, 'o', color='#008FC2')
    axins.plot(x, rtx3090_y, ':', color='#008FC2')
    axins.plot(niters, A100, 'o', color='#76b900')
    axins.plot(x, a100_y, ':', color='#76b900')

    # Sub region of the original image
    axins.set_xlim(-100000, 3000000)
    axins.set_ylim(-0.003, 0.050)
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    mark_inset(axis, axins, loc1=3, loc2=4, fc="none", ec="0.25")





    fig.set_size_inches(7.25, 4.25)
    plt.savefig('scaling_param.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('scaling_param.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)

    plt.close('all')

def transient_response():
    GPUs = {
        'Tesla A100' : 'data/Transient_Response/A100_TR',
        'RTX 4090' : 'data/Transient_Response/4090_TR',
        'RTX 3090' : 'data/Transient_Response/3090_TR',
        'Tesla K40m' : 'data/Transient_Response/K40m_TR'
    }

    fig, axis = plt.subplots(nrows=4, ncols=1)

    # iterate over the axis and GPUs
    for i, (gpu, data_dir) in enumerate(GPUs.items()):
        print(f'Processing {gpu}...')
        axis[i].plot(np.nan, np.nan, label=gpu, color='white')
        
        load = pd.read_csv(os.path.join(data_dir, 'timestamps.csv'))
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

        power = pd.read_csv(os.path.join(data_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        power['timestamp'] += 60*60*1000 * -1
        power['timestamp'] -= t0
        power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= t_max+10)]



        PMD = convert_pmd_data(data_dir)
        PMD['timestamp'] -= t0
        PMD = PMD[(PMD['timestamp'] >= 0) & (PMD['timestamp'] <= t_max)]

        print(PMD)
        # delete 9 rows every 10 rows
        PMD = PMD.iloc[::16]

        axis[i].fill_between(PMD['timestamp'], PMD['total_p'], PMD['eps_total_p'], label='Power cables', color='#002147')
        axis[i].fill_between(PMD['timestamp'], PMD['eps_total_p'], 0, label='PCIE x16 slot', color='#008FC2')


        axis[i].plot(power['timestamp'], power[f' power.draw [W]'], label=f'nvidia-smi', linewidth=3, color='#76b900')

        axis[i].set_xlim(0, 3000)
        axis[i].legend(loc='lower right', framealpha=1)
        axis[i].set_ylabel('Power draw (W)')

    axis[0].set_xticklabels([])
    axis[1].set_xticklabels([])
    axis[2].set_xticklabels([])
    axis[3].set_xlabel('Time (ms)')

    axis[0].set_xlim(0, 2999)
    axis[1].set_xlim(0, 2999)
    axis[2].set_xlim(0, 2999)
    axis[3].set_xlim(0, 2999)

    fig.set_size_inches(7, 6.25)
    plt.subplots_adjust(hspace=0.075)
    plt.savefig('transient_response.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('transient_response.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')

def plot_5050():
    GPUs = {
        'RTX 3090' : 'data/5050/3090',
        'Tesla A100' : 'data/5050/A100',
    }

    fig, axis = plt.subplots(nrows=2, ncols=1)

    # iterate over the axis and GPUs
    for i, (gpu, data_dir) in enumerate(GPUs.items()):
        print(f'Processing {gpu}...')
        axis[i].plot(np.nan, np.nan, label=gpu, color='white')
        
        load = pd.read_csv(os.path.join(data_dir, 'timestamps.csv'))
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

        power = pd.read_csv(os.path.join(data_dir, 'gpudata.csv'))
        power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
        power['timestamp'] += 60*60*1000 * -1
        power['timestamp'] -= t0
        power = power[(power['timestamp'] >= 0) & (power['timestamp'] <= t_max+10)]



        PMD = convert_pmd_data(data_dir)
        PMD['timestamp'] -= t0
        PMD = PMD[(PMD['timestamp'] >= 0) & (PMD['timestamp'] <= t_max)]

        # delete 9 rows every 10 rows
        PMD = PMD.iloc[::16]
        axis[i].fill_between(PMD['timestamp'], PMD['total_p'], PMD['eps_total_p'], label='Power cables', color='#002147')
        axis[i].fill_between(PMD['timestamp'], PMD['eps_total_p'], 0, label='PCIE x16 slot', color='#008FC2')


        axis[i].plot(power['timestamp'], power[f' power.draw.instant [W]'], label=f'nvidia-smi', linewidth=3, color='#76b900')

        axis[i].set_xlim(1000, 8000)
        axis[i].legend(loc='lower right', framealpha=1)
        axis[i].set_ylabel('Power draw (W)')

    axis[0].set_xticklabels([])
    axis[1].set_xlabel('Time (ms)')
    fig.set_size_inches(7, 3.25)
    plt.subplots_adjust(hspace=0.075)
    plt.savefig('5050.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('5050.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')

def loss_functions():
    GPUs = {
        'GTX 1080 Ti' : 'data/loss_func/1080ti.csv',
        'Tesla A100'  : 'data/loss_func/A100.csv',
        'RTX 3090'    : 'data/loss_func/3090.csv',
    }

    palette = {
        'GTX 1080 Ti' : '#76B900',
        'Tesla A100'  : '#008FC2',
        'RTX 3090'    : '#002147', 
    }

    fig, axis = plt.subplots(nrows=1, ncols=1)

    for gpu, data_dir in GPUs.items():
        df = pd.read_csv(data_dir)
        axis.plot(df['avg_windows'], df['losses'], label=f'{gpu}, square wave', color=palette[gpu])
        axis.plot(df['avg_windows'], df['pmd_losses'], '--', label=f'{gpu}, PMD data', color=palette[gpu])

    axis.set_xlabel('Average window size')
    axis.set_ylabel('Loss (Mean Squared Error)')
    axis.legend(loc='upper right')
    axis.set_ylim(0, 1.55)
    axis.grid(True, linestyle='--', linewidth=0.5)

    fig.set_size_inches(7.25, 2.5)
    plt.savefig('loss_func.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('loss_func.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')

def violin_plots():
    GPUs = {
        'GTX 1080 Ti' : 'data/violin_plot/1080ti.csv',
        'Tesla A100'  : 'data/violin_plot/A100.csv',
        'RTX 3090'    : 'data/violin_plot/3090.csv',
    }

    fig, axis = plt.subplots(nrows=1, ncols=3)


    for i, (gpu_name, result_dir) in enumerate(GPUs.items()):
        # read the avg_window_results and delay_results from the csv file
        with open(result_dir, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            rows = list(csv_reader)
            num_rows = len(rows)

            stride = num_rows//5
            labels = rows[0]
            avg_window_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[1:stride+1]]
            delay_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[stride+1:2*stride+1]]
            PMD_avg_window_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[2*stride+1:3*stride+1]]
            PMD_delay_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[3*stride+1:4*stride+1]]
            error_results = [[np.float64(value) for value in row if not np.isnan(np.float64(value))] for row in rows[4*stride+1:]]
                
        labels.append('All')    

        avg_window_flat = [item for sublist in avg_window_results for item in sublist]
        avg_window_results.append(avg_window_flat)

        PMD_avg_window_flat = [item for sublist in PMD_avg_window_results for item in sublist]
        PMD_avg_window_results.append(PMD_avg_window_flat)


        violin_plot(axis[i], [PMD_avg_window_flat, avg_window_flat], ['PMD', 'Square wave'])
        axis[i].set_title(gpu_name)

    axis[0].set_ylim(5, 15)
    axis[1].set_ylim(0, 40)
    axis[2].set_ylim(94, 106)
    axis[0].set_ylabel('Averaging window (ms)')
    fig.subplots_adjust(wspace=0.275)
    fig.set_size_inches(7.25, 2.253)
    plt.savefig('violin_plot.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('violin_plot.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')

def temp():
    color_palette =   {1 : '#BDCCFF', 4 : '#8D9DCE', 8 : '#5F709F'}
    correct_palette = {1 : '#acd566', 4 : '#76b900', 8 : '#466f00'}

    fig, axis = plt.subplots(nrows=3, ncols=2)

    # 0.25
    # shift 1
    reps = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    std = [8.748141457686504, 10.981006175325746, 12.82323917107489, 8.665459249935498, 6.592528756232953, 1.8271128946599478, 1.1419074305309984, 0.9450544804604851, 0.7442087556062388, 0.7110457399029881, 0.556063126541129, 0.6414114657116644, 0.5877217325055653]
    err = [-45.39191543114509, -43.02275877831194, -26.098181256724363, -20.760652848571873, -14.339694190314786, -11.390194976132584, -8.95507266823782, -8.162334706463573, -7.692901718274485, -7.512514888277654, -7.281200937609296, -6.9457441027942854, -6.7455848780589465]
    reps_c = [16, 32, 64, 96, 128, 160, 192, 224, 256]
    std_c = [9.802824082155588, 3.6798715527835104, 1.9201090189760148, 1.2204466089620185, 0.8994223641594458, 0.7958364003980004, 0.5896368532552585, 0.605251890099717, 0.5020576713430375]
    err_c = [-8.913920283015184, -8.617772186058586, -7.334760744905052, -7.103825361088506, -6.9254661859798965, -6.899051067150608, -6.837204423869029, -6.4920800009080475, -6.440999575588082]

    axis[0, 0].plot(reps, err, ':x', color=color_palette[1])
    axis[0, 0].plot(reps_c, err_c, '-o', color=correct_palette[1])
    axis[0, 1].plot(reps, std, ':x', color=color_palette[1])
    axis[0, 1].plot(reps_c, std_c, '-o', color=correct_palette[1])

    # shift 4
    reps = [4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    std = [12.91518944253861, 6.81723150001094, 5.973810184271936, 1.3564601392214992, 1.3663213196273214, 1.4519013129154676, 1.5334139243427132, 1.0740890628049056, 0.9014470907983715, 0.7051961219389127, 0.607438963816315]
    err = [-34.91129121385194, -26.031080333289836, -21.24738523861145, -14.232157429691814, -11.002277854407842, -9.312128450460873, -8.45804899689405, -7.966721115974311, -7.655296171185045, -7.11491344117625, -7.161451208928228]
    reps_c = [64, 96, 128, 160, 192, 224, 256]
    std_c = [0.8540840286705572, 1.3262392182673255, 0.5675972380590157, 0.5830978750301808, 0.3618243755192216, 0.5533254553132174, 1.0701715792284161]
    err_c = [-7.851640207657873, -7.228348611108607, -6.905514941129414, -6.631917570581463, -6.457359878949457, -6.327580045419847, -6.523122538512176]

    axis[0, 0].plot(reps, err, ':x', color=color_palette[4])
    axis[0, 0].plot(reps_c, err_c, '-o', color=correct_palette[4])
    axis[0, 1].plot(reps, std, ':x', color=color_palette[4])
    axis[0, 1].plot(reps_c, std_c, '-o', color=correct_palette[4])

    # shift 8
    reps = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    std = [13.013300127836624, 4.919220731575211, 1.0509633183412126, 1.0668821303420457, 0.7583620919692432, 0.689359935052336, 0.9633470344330125, 1.0300073393099187, 1.1496542496622588, 1.2992354839133013]
    err = [-34.5279870578071, -24.184815830686507, -18.78181367491024, -13.114904559351562, -11.154252644826594, -10.070864927609263, -9.155593771553727, -8.535859642778677, -8.329113527455787, -7.653640283587533]
    reps_c = [96, 128, 160, 192, 224, 256]
    std_c = [0.9906529623438924, 0.9330348121386164, 0.9326018034769266, 0.9818893848266157, 1.0986972010077745, 1.4157036892471327]
    err_c = [-7.340749846686538, -7.2537491823313855, -7.078401833812004, -6.708577620504965, -6.795725701969198, -6.69211152799303]

    axis[0, 0].plot(reps, err, ':x', color=color_palette[8])
    axis[0, 0].plot(reps_c, err_c, '-o', color=correct_palette[8])
    axis[0, 1].plot(reps, std, ':x', color=color_palette[8])
    axis[0, 1].plot(reps_c, std_c, '-o', color=correct_palette[8])


    # 1
    # shift 1
    reps = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128]
    err = [-22.015150951646735, -11.584876503553506, -21.56694249597727, -6.797820476758052, -8.5309656923534, -12.975377622598803, -7.692747113625978, -6.584948741802387, -13.902074271573554, -8.1185767292805, -12.7423272333462, -9.96758728757933, -10.178071551593812, -10.324789760086079, -10.621184872336542, -7.020205715275106, -16.506336007155884, -6.57976912319714, -3.459322301827103, -6.141703317989997, -8.58195716952207, -9.796576232449635, -9.127086922804168, -7.2941957075085035, -8.998863130990822, -9.097539699240965, -8.514517166478697, -9.785383598853675, -2.8003944437791146, -11.2782856645586, -8.04603749654388, -7.8564414785249115, -7.399860685679978, -5.611461541016259]
    err_c = [-18.509812184452024, -5.0045495859861235, -7.538211424553857, -11.957243775625958, -6.757418698641218, -5.7967586925827455, -13.51885876063056, -7.573490178637063, -12.416514155157888, -9.701336596655377, -9.616654125156252, -10.027195754224948, -10.397822814783016, -7.039901020727305, -16.45665931133497, -6.282493950292147, -3.2088699268019996, -6.0528841176266255, -8.322369638559097, -9.63160410023248, -9.081524752248814, -6.832417730189561, -8.673880268948377, -9.112275565610888, -8.413637931213906, -9.618535827611577, -2.4833187294549015, -11.112051931828788, -7.976686471336539, -7.815740314097979, -7.306820603582014, -5.460016305606299]
    reps_c = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128]
    std =   [23.070417507900302, 26.3097588263208, 26.869077044484314, 28.327417668701337, 26.62738662923771, 28.297575199315773, 25.679600808079723, 29.110045383528824, 24.785065398400956, 26.044403433212896, 28.40455147721905, 27.968674947192312, 26.189045999409842, 26.730464623175305, 28.495440591915138, 27.074544854849236, 28.77038039375334, 28.6242349620483, 26.393034340328917,    26.483473172349456, 22.876113066311248, 20.459100489907573, 23.060486863642385, 28.035875207110603, 23.07993034163076, 26.166151112842602, 24.675783504666567, 24.702112584391882, 24.385904138880104, 22.026088649898025, 21.33626116002413, 25.649313460308354, 20.944270431652143, 19.158869373179925]
    std_c =                                       [28.349973237357894, 29.27382136231083, 27.344632834422768, 28.866773191696527, 25.92562856153778, 29.450572910688336, 24.040464809673093, 26.310357672131776, 28.82455909795994, 28.24352245657408, 26.52655316036747, 27.0762439608928,      28.804106993662877, 27.378143158062745, 28.982107111396182, 28.925730670769326, 26.625053561813182, 26.681065739634697, 23.06985011626896, 20.79163429943734,   23.346392106036856, 28.40892896796829, 23.642020483330594, 26.531623986315058, 25.047073473615203, 25.148936824144792, 24.743111338264464, 22.398906551039875, 21.698979598403579, 25.83516917196123, 21.387378401579262, 19.541165203549157]

    axis[1, 0].plot(reps, err, ':x', color=color_palette[1])
    axis[1, 0].plot(reps_c, err_c, '-o', color=correct_palette[1])
    axis[1, 1].plot(reps, std, ':x', color=color_palette[1])
    axis[1, 1].plot(reps_c, std_c, '-o', color=correct_palette[1])

    # shift 4
    reps = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128]
    err = [-18.26395439457757, -13.727847107405477, -12.121295466820877, -10.097693585980345, -10.176255873183374, -11.553320442672353, -9.763399639475216, -8.557908579607428, -10.058472286445227, -8.334083722964811, -6.857553567987662, -7.552009445569946, -6.882456480189263, -9.587019087351491, -14.67547702713991, -11.559447187852575, -3.2978789618669997, -6.368269079765461, -9.690047123338484, -6.446927401121119, -5.04152165572679, -8.31069404868086, -9.230289838968412, -5.82051424386187, -6.74788870804301, -14.805817589489074, -3.450446841426945, -12.908692454445948, -12.220856404219468, -1.916376257798348, -9.985709270730416, -4.6217158370092064]
    err_c =                                                             [-8.841791099674188, -7.722164984701278, -8.28638771169666, -10.037488487684957, -8.362100204741484, -7.3459036300373635, -8.931036366877082, -7.3532717644735195, -5.868825830996038, -6.691791852625327, -6.230344821345335, -8.897684687569098, -14.090395885346732, -10.997825353929672, -2.734384164192097, -5.830797367413237, -9.24346935094144, -5.823554715873225, -4.746923530426632, -7.807446606520989, -8.750189133270757, -5.227416078790816, -6.307548777967065, -14.43258799837362, -2.9580825057089113, -12.643331918502987, -12.114574506659583, -1.527415000047471, -9.752650095234019, -3.167191557598988]
    reps_c = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128]
    std = [13.3476730648347024, 11.318443331531118, 12.046782284665448, 10.247409992476381, 9.943604665900287, 10.72145876208153, 11.661308990714796, 12.313473123598381, 11.837604057079043, 8.681799034467982, 10.363298091167668, 9.403179312852401, 9.964290455020656, 13.107527687640241, 11.109137662539766, 13.284466852350377, 12.778985046118438, 9.752464875031173, 8.404328158405242,   12.346365887395784, 10.53040272882784, 11.677131554546076, 14.470759967609045, 14.046481837705485, 10.39540344484524, 9.92468930082086, 12.33037883560525, 10.31258028715234, 11.651643017729636, 8.612899067440875, 7.325990291299206, 8.505174723617817]
    std_c =                                       [12.0108981332132196, 10.237060059200354, 9.964810390886165, 10.810406370213616, 11.665600536301497, 12.307178374936744, 11.902044307681775, 8.695029431810502, 10.455157302964784, 9.474853744302845, 10.185949531993232, 13.142393320469745, 11.170706942264314, 13.351766916575025, 12.857516792412415, 9.834337544099414, 8.467318247764823, 12.540076097201222, 10.621976081125776, 11.71291923317684, 14.56318641790169, 14.209156486310548, 10.610777498479333, 10.156987101945685, 12.63849355070414, 10.459480613975558, 11.872294258701363, 8.771016201994843, 7.598816248948893, 8.578502459623344]

    axis[1, 0].plot(reps, err, ':x', color=color_palette[4])
    axis[1, 0].plot(reps_c, err_c, '-o', color=correct_palette[4])
    axis[1, 1].plot(reps, std, ':x', color=color_palette[4])
    axis[1, 1].plot(reps_c, std_c, '-o', color=correct_palette[4])

    # shift 8
    reps = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
    err = [-16.894972989671114, -12.745857871304004, -11.4088048177483, -9.609667781832961, -7.91143713760421, -8.75815634762723, -8.993251428392963, -7.868750013997801, -8.091185736867084, -7.250431049995274, -8.261936488405441, -7.828667882718521, -7.204066180521757, -6.20919403301849, -7.081524696287643, -6.210479457636335]
    err_c = [-8.477561720974167, -7.359618882149205, -5.99173335191624, -7.21355983379938, -7.754006254489076, -6.7251298012375464, -7.1297690799080184, -6.308607635228701, -7.352159168294454, -7.079251605328439, -6.547301684885079, -5.567753066822501, -6.508946515192407, -5.6469593437842756]
    reps_c = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
    std = [5.7761676602554449, 4.3800013333310193, 3.4690248233984518, 3.9502635282046508, 3.8801933156300947, 4.231080277258048, 4.4065524249070975, 4.637922269927579, 4.143375995909731, 4.451849472787304, 4.226228320655528, 4.8407959163865435, 4.85218261371444, 4.57547450889414, 4.262094346167947, 4.27066096210541]
    std_c =                                       [3.5845027186772422, 3.9780525809650973, 3.8918026313622303, 4.315724202763391, 4.455002993710256, 4.692313753359566, 4.114984374181699, 4.5897230404967075, 4.334323060677953, 4.891102562616596, 4.946475577351895, 4.661423818318308, 4.357754510818831, 4.3375495924979415]
    
    axis[1, 0].plot(reps, err, ':x', color=color_palette[8])
    axis[1, 0].plot(reps_c, err_c, '-o', color=correct_palette[8])
    axis[1, 1].plot(reps, std, ':x', color=color_palette[8])
    axis[1, 1].plot(reps_c, std_c, '-o', color=correct_palette[8])


    # 8
    # shift 1
    reps = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    std = [0.3835313060405669, 0.8873186121043387, 0.8730077835910183, 0.5422750344037134, 0.552091307121159, 0.5783365014544634, 0.4656410207226464, 0.5392118334588654, 0.5564930900894325, 0.5649203185589173, 0.4939592554483708, 0.3738688931767285, 0.418476123607375, 0.5070594731678456, 0.4541456669075412, 0.44536746148937256, 0.4689096918550534, 0.45748408433451393]
    err = [-8.31947552784437, -8.154296662830939, -7.525351206866425, -7.052447972312219, -6.655202392320815, -6.255352173590377, -6.386448329651399, -5.936868190797819, -6.087159586700188, -5.891837344318867, -5.9569360381102525, -6.151227850153272, -6.08518722290193, -5.870083248713419, -6.108143962319446, -5.912688718709525, -5.697765428188979, -6.009317865718067]
    reps_c = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    std_c = [1.132407971158666, 1.0492690072940996, 0.5822324418202831, 0.5648528637261021, 0.5866481387410065, 0.4787874211719714, 0.5503338656244415, 0.5547707782636854, 0.5812507529591722, 0.5020505172243044, 0.37938980365110764, 0.41967609616172485, 0.511273838103262, 0.4562797978715042, 0.44742159249879415, 0.4690559138728233, 0.46277739747085733]
    err_c = [-7.877146966164858, -7.434780721041838, -7.013040825188406, -6.625781217510751, -6.21960129681339, -6.356105021430143, -5.907097169448546, -6.06714651056289, -5.867507873851713, -5.937420478238706, -6.140138558517555, -6.075518677536518, -5.86006875603331, -6.094867654933919, -5.897518463939249, -5.6875522057633905, -5.998975151347014]

    axis[2, 0].plot(reps, err, ':x', color=color_palette[1])
    axis[2, 0].plot(reps_c, err_c, '-o', color=correct_palette[1])
    axis[2, 1].plot(reps, std, ':x', color=color_palette[1])
    axis[2, 1].plot(reps_c, std_c, '-o', color=correct_palette[1])

    # shift 4
    reps =   [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    std =    [0.39548317305839326, 0.5793819303865213, 0.4508630243111845, 0.7390295483679837, 0.5208037860910646, 0.511968735413259, 0.5391981619014544, 0.4168501636102156, 0.5387100965854384, 0.43039520410872817, 0.3609368609655578, 0.39658145543614864, 0.5685886418756247, 0.4602516191563433, 0.4876620972645649, 0.4933462152942768]
    reps_c = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    std_c =  [0.5916058899977595, 0.47807145669450035, 0.69243404554863, 0.5435311340206052, 0.5408471340897122, 0.5255377993299484, 0.42520397093799306, 0.5383873390745242, 0.42342308166419973, 0.367049487282254, 0.3966227634283977, 0.5782362241195385, 0.4749632953180609, 0.4809586932282225, 0.4991243758262365]
    err =    [-6.908461954992144, -6.081797860673741, -6.162122952217219, -6.120619369384416, -6.109331050755001, -6.0012550896143996, -6.048576986832474, -6.24017136031205, -6.131621482711517, -6.029228257051017, -5.939984288373635, -5.95018709046845, -5.949125047554092, -5.859910420828178, -6.042716734808167, -6.11510645669486]
    err_c =                      [-6.055502796677065, -6.114144126948748, -6.073911656670321, -6.099211176573637, -5.97510677626148,  -6.0225798996081785, -6.218757535786614, -6.123817780391316, -6.007230414035062, -5.933623466024123, -5.939007154274909, -5.935807583787275, -5.861738243195674, -6.0241184479602005, -6.108070601891533]

    axis[2, 0].plot(reps, err, ':x', color=color_palette[4])
    axis[2, 0].plot(reps_c, err_c, '-o', color=correct_palette[4])
    axis[2, 1].plot(reps, std, ':x', color=color_palette[4])
    axis[2, 1].plot(reps_c, std_c, '-o', color=correct_palette[4])

    # shift 8
    reps = [8, 16, 24, 32, 40, 48, 56, 64]
    std = [0.6668384545930417, 0.3636104018761191, 0.4604917747080874, 0.40333410323630847, 0.41410320363871184, 0.40440267225483595, 0.3773599126645604, 0.3571345237188322]
    reps_c = [16, 24, 32, 40, 48, 56, 64]
    std_c = [0.3663424353855015, 0.4787743743156263, 0.3948880755617882, 0.40885767621048885, 0.4045168673353055, 0.388368473171724, 0.35641048082344884]
    err = [-6.471533075103815, -6.153755902754613, -6.102161776224639, -5.984363625346028, -6.012814208753487, -5.984616699024825, -5.927790956735522, -5.945765096222749]
    err_c =                   [-6.124206776763604, -6.052601155521944, -5.922951923124435, -6.008176203588936, -5.978629331328609, -5.917472762681982, -5.945182727381886]

    axis[2, 0].plot(reps, err, ':x', color=color_palette[8])
    axis[2, 0].plot(reps_c, err_c, '-o', color=correct_palette[8])
    axis[2, 1].plot(reps, std, ':x', color=color_palette[8])
    axis[2, 1].plot(reps_c, std_c, '-o', color=correct_palette[8])

    

    for i in range(3):
        axis[i, 0].set_ylabel('Error [% of groung truth]')
        axis[i, 1].set_ylabel('Std. [% of groung truth]')
        for j in range (2):
            axis[i, j].set_xlabel('# of Repetitions')


    titles = ['25ms period', '100ms period', '800ms period']
    for i in range(3):
        ax = fig.add_subplot(3, 1, i + 1)  # add subplot in position i+1
        ax.axis('off')  # hide axis
        ax.set_title(titles[i], fontsize=12, y=1)  # add title
    
    
    axis[0, 1].plot(np.nan, np.nan, ':x', color=color_palette[1], label='0 shifts (original)')
    axis[0, 1].plot(np.nan, np.nan, ':x', color=color_palette[4], label='4 shifts (original)')
    axis[0, 1].plot(np.nan, np.nan, ':x', color=color_palette[8], label='8 shifts (original)')
    axis[0, 1].plot(np.nan, np.nan, '-o', color=correct_palette[1], label='0 shifts (corrected)')
    axis[0, 1].plot(np.nan, np.nan, '-o', color=correct_palette[4], label='4 shifts (corrected)')
    axis[0, 1].plot(np.nan, np.nan, '-o', color=correct_palette[8], label='8 shifts (corrected)')

    axis[0, 1].legend(loc='upper right')

    axis[0, 1].set_yticks([0, 2, 4, 6, 8, 10, 12])


    fig.set_size_inches(8.5, 8)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig('A100_25_100.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('A100_25_100.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close('all')

def energy_meas():
    tasks = ['CUBLAS', 'CUFFT', 'nvJPEG', 'ST Disp', 'Black-S', 'RandGen', 'ResNet50', 'RetinaNet', 'Bert', 'Average']
    case_1 = [-21.5411, -52.3756, -16.5776, -43.9343, -58.5065, -65.9410, -6.9455, -15.2858, -7.3298, -32.0486]
    case_2 = [-65.9273, -66.0321, -46.6689, -44.1017, -61.0129, -68.5986, -20.3972, -51.0333, -36.1704, -51.1047]
    case_3 = [-24.0376, -52.3905, 37.0111, -42.9111, -41.5792, -61.2106, -10.1076, -26.8691, -15.7750, -34.6546]
    colors = ['#7fc7e0'] + ['#badc7f'] * (len(tasks)-1)
    # colors = ['#008FC2'] + ['#76b900'] * (len(tasks)-1)

    case_1_correct = [-4.8317, -4.7683, -4.8480, -4.5292, -4.9850, -4.1906, -4.3935, -4.7966, -4.9396, -4.6981]
    case_2_correct = [-4.7424, -4.4905, -4.8454, -4.3106, -4.8157, -4.0126, -4.2830, -4.3987, -4.8263, -4.5250]
    case_3_correct = [-5.4367, -5.0770, -5.3964, -5.6047, -5.4821, -5.7508, -5.6179, -5.4528, -5.0909, -5.4344]
    colors_correct = ['#008FC2'] + ['#76b900'] * (len(tasks)-1)

    # calculate the std of all 3 case_correct
    case_correct_all = case_1_correct[:-1] + case_2_correct[:-1] + case_3_correct[:-1]
    print(case_correct_all)
    print(np.std(case_correct_all))


    tasks = tasks[::-1]
    case_1 = case_1[::-1]
    case_2 = case_2[::-1]
    case_3 = case_3[::-1]

    case_1 = [abs(i) for i in case_1]
    case_2 = [abs(i) for i in case_2]
    case_3 = [abs(i) for i in case_3]

    case_1_correct = case_1_correct[::-1]
    case_2_correct = case_2_correct[::-1]
    case_3_correct = case_3_correct[::-1]

    case_1_correct = [abs(i) for i in case_1_correct]
    case_2_correct = [abs(i) for i in case_2_correct]
    case_3_correct = [abs(i) for i in case_3_correct]

    fig, axis = plt.subplots(nrows=1, ncols=3, sharey=True)

    x = np.arange(len(tasks))  # the label locations
    width = 0.9  # the width of the bars

    rect_1 = axis[0].barh(x, [a-b for a,b in zip(case_1, case_1_correct)], width, color=colors, left=case_1_correct, hatch='///', edgecolor='white')
    rect_1_correct = axis[0].barh(x, case_1_correct, width, color=colors_correct, edgecolor='black')    

    rect_2 = axis[1].barh(x, [a-b for a,b in zip(case_2, case_2_correct)], width, color=colors, left=case_2_correct, hatch='///', edgecolor='white')
    rect_2_correct = axis[1].barh(x, case_2_correct, width, color=colors_correct, edgecolor='black')

    rect_3 = axis[2].barh(x, [a-b for a,b in zip(case_3, case_3_correct)], width, color=colors, left=case_3_correct, hatch='///', edgecolor='white')
    rect_3_correct = axis[2].barh(x, case_3_correct, width, color=colors_correct, edgecolor='black')


    axis[0].bar_label(rect_1_correct, padding=3, labels=[f'{i:.2f}' for i in case_1_correct], fontweight='bold')
    axis[0].bar_label(rect_1, padding=-35, labels=[f'{i:.2f}' if i >55 else '' for i in case_1])
    axis[0].bar_label(rect_1, padding=3, labels=[f'{i:.2f}' if i <=55 and i>20 else '' for i in case_1])
    axis[0].bar_label(rect_1, padding=15, labels=[f'{i:.2f}' if i <=20 and i>10 else '' for i in case_1])
    axis[0].bar_label(rect_1, padding=30, labels=[f'{i:.2f}' if i <=10 else '' for i in case_1])

    axis[1].bar_label(rect_2_correct, padding=3, labels=[f'{i:.2f}' for i in case_2_correct], fontweight='bold')
    axis[1].bar_label(rect_2, padding=-35, labels=[f'{i:.2f}' if i >55 else '' for i in case_2])
    axis[1].bar_label(rect_2, padding=3, labels=[f'{i:.2f}' if i <=55 else '' for i in case_2])

    axis[2].bar_label(rect_3_correct, padding=3, labels=[f'{i:.2f}' for i in case_3_correct], fontweight='bold')
    axis[2].bar_label(rect_3, padding=-35, labels=[f'{i:.2f}' if i >50 else '' for i in case_3])
    axis[2].bar_label(rect_3, padding=3, labels=[f'{i:.2f}' if i <=50 and i>20 else '' for i in case_3])
    axis[2].bar_label(rect_3, padding=10, labels=[f'{i:.2f}' if i <=20 and i>15 else '' for i in case_3])
    axis[2].bar_label(rect_3, padding=20, labels=[f'{i:.2f}' if i <=15 and i>10 else '' for i in case_3])


    axis[0].set_xlabel('Error (%)')
    axis[1].set_xlabel('Error (%)')
    axis[2].set_xlabel('Error (%)')
    axis[0].set_title('Case 1: 100/100')
    axis[1].set_title('Case 2: 1000/100')
    axis[2].set_title('Case 3: 25/100')

    axis[0].set_yticks(x)
    axis[0].set_yticklabels(tasks)


    axis[2].barh(x, -2, width, left=-1, color='grey', hatch='///', edgecolor='white', label='Naive')
    axis[2].barh(x, -3, width, left=-1, color='grey', edgecolor='black', label='Fixed')
    axis[2].set_xlim(0, 65)

    axis[0].set_ylim(-0.6, 9.6)

    axis[2].legend(ncols=1, loc='center right', bbox_to_anchor=(1.03, 0.16), frameon=True, edgecolor='black')

    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.1, top=0.9, wspace=0.065, hspace=0)

    fig.set_size_inches(7.5, 4.25)
    plt.savefig('energy_meas.jpg', format='jpg', dpi=512, bbox_inches='tight')
    plt.savefig('energy_meas.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()


def main():
    # steam_gpu_share()
    # steady_state_accuracy()
    # pwr_update_freq()
    # scaling_params()
    # transient_response()
    # plot_5050()
    # loss_functions()
    # violin_plots()
    # temp()
    energy_meas()

if __name__ == "__main__":
    main()