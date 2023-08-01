#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd

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


def steam_gpu_share():
    labels = ['Other',   'Intel',   'AMD',     'Older',
              'Maxwell\n(GeForce 700/900\nSeries)', 'Pascal\n(GeForce 10 Series)',  'Turing\n(GeForce 16/20 Series)',
              'Ampere\n(GeForce 30 Series)',  'Ada Lovelace\n(GeForce 40 Series)']
    
    
    sizes   = [ 0.25,      8.66,      15.04,     5.26,      2.37,      14.12,     24.77,     26.62,     2.91         ]
    colors  = ['#f2d600', '#0071c5', '#ed1c24', '#2f4a00', '#466f00', '#5e9400', '#76b900', '#91c732', '#acd566'     ]
    explode = ( 0.15,      0.1,       0.1,       0,         0,         0,         0,         0,         0            )

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.2f%%', colors=colors, startangle=90, explode=explode, pctdistance=0.75)

    fig.set_size_inches(5, 5)
    plt.savefig('steam_gpu_share.jpg', format='jpg', dpi=512, bbox_inches='tight')
    plt.savefig('steam_gpu_share.eps', format='eps', bbox_inches='tight')
    plt.close()


def steady_state_accuracy():
    GPUs = [
        "GeForce RTX 3090 #1", "GeForce RTX 3090 #2", "GeForce RTX 3090 #3", "GeForce RTX 3090 #4", "GeForce RTX 3090 #5",
        "GeForce RTX 4090",
        "GeForce RTX 3070 Ti",
        "GeForce RTX 2080 Ti", "GeForce RTX 2060 SUPER",
        "GeForce GTX TITAN X Pascal", "GeForce RTX 1080 Ti", "GeForce RTX 1080",
        "GeForce GTX TITAN X",
        "Quadro RTX A5000",
        "Tesla A100 PCIE 40GB", "Tesla P100 PCIE 16GB", "Tesla M40", "Tesla K80", "Tesla K40m",
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
    ax.set_yticks(x + width/2)
    ax.set_yticklabels(GPUs)
    ax.xaxis.label.set_color(colors['Offset'])
    ax.tick_params(axis='x', colors=colors['Offset'])
    
    # Add the legend and get the handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    plt.legend(handles, labels, loc='lower left')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Gradient / Percentage Error (%)', fontweight='bold')
    ax2.xaxis.label.set_color(colors['Gradient'])
    ax2.tick_params(axis='x', colors=colors['Gradient'])
    ax2.spines['top'].set_color(colors['Gradient'])
    ax2.spines['bottom'].set_color(colors['Offset'])

    fig.set_size_inches(6, 9)
    plt.savefig('steady_state_accuracy.jpg', format='jpg', dpi=512, bbox_inches='tight')
    plt.savefig('steady_state_accuracy.eps', format='eps', bbox_inches='tight')
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
    axis.legend(loc='upper right')
    axis.grid(True, linestyle='--', linewidth=0.5)

    fig.set_size_inches(7.5, 5)
    plt.savefig('power_update_freq.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('power_update_freq.eps', format='eps', bbox_inches='tight')

    plt.close('all')

def scaling_params():
    niters = [225000, 337500, 506250, 759375, 1139062, 1708593, 2562889, 3844333, 5766499, 8649748, 12974622,
                19461933, 29192899, 43789348, 65684022, 98526033, 147789049, 221683573, 332525359, 498788038]

    RTX3090 = [2.8207, 4.2046, 6.2817, 8.398, 12.4752, 19.088849999999997, 29.25215, 43.754949999999994, 
                65.92405000000001, 99.268, 149.206, 224.1428, 336.42955, 507.37435, 763.25755, 1148.2991000000002,
                1723.8624499999999, 2589.2348500000003, 3885.3249, 5828.4539]

    A100 = [5.1573, 7.71185, 11.5556, 17.32705, 25.9817, 38.97465, 58.4456, 87.6444, 131.45495, 197.16775, 295.7412,
            443.5997, 665.3719, 998.05955, 1497.0885, 2245.5759, 3368.32845, 5052.5001, 7578.8348, 11367.46275]

    rtx3090_intercept, rtx3090_gradient, rtx3090_r2 = linear_regression(niters, RTX3090)
    a100_intercept, a100_gradient, a100_r2 = linear_regression(niters, A100)

    x = np.linspace(0, max(niters), 100)
    rtx3090_y = rtx3090_gradient * x + rtx3090_intercept
    a100_y = a100_gradient * x + a100_intercept

    fig, axis = plt.subplots(nrows=1, ncols=1)

    axis.plot(niters, RTX3090, 'o', color='#008FC2', label='GeForce RTX 3090')
    axis.plot(x, rtx3090_y, ':', label=f'Line of best fit ($R^2$ = {rtx3090_r2:.4f})', color='#008FC2')

    axis.plot(niters, A100, 'o', color='#76b900', label='Tesla A100-PCIe-40GB')
    axis.plot(x, a100_y, ':', label=f'Line of best fit ($R^2$ = {a100_r2:.4f})', color='#76b900')

    axis.set_xlabel('Iterations')
    axis.set_ylabel('Kernel Execution Time (ms)')
    axis.legend(loc='lower right')

    axis.grid(True, linestyle='--', linewidth=0.5)



    # Here is the part for the inset
    axins = inset_axes(axis, width="35%", height="35%", loc=2)  # loc=2 corresponds to 'upper left'
    axins.plot(niters, RTX3090, 'o', color='#008FC2')
    axins.plot(x, rtx3090_y, ':', color='#008FC2')
    axins.plot(niters, A100, 'o', color='#76b900')
    axins.plot(x, a100_y, ':', color='#76b900')

    # Sub region of the original image
    axins.set_xlim(-100000, 3000000)
    axins.set_ylim(-3, 50)
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    mark_inset(axis, axins, loc1=3, loc2=4, fc="none", ec="0.25")





    fig.set_size_inches(7.5, 5)
    plt.savefig('scaling_param.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('scaling_param.eps', format='eps', bbox_inches='tight')

    plt.close('all')





def main():
    # steam_gpu_share()
    steady_state_accuracy()
    # pwr_update_freq()
    # scaling_params()

if __name__ == "__main__":
    main()