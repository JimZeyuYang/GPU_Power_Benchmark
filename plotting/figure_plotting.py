#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


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
        "Tesla A100 PCIE 40GB", "Tesla P100 PCIE 16GB", "Tesla M40", "Tesla K80", "Tesla K40m",
        "GeForce RTX 4090",
        "GeForce RTX 3090", "GeForce RTX 3070 Ti",
        "GeForce RTX 2080 Ti", "GeForce RTX 2060 SUPER",
        "GeForce RTX 1080 Ti", "GeForce RTX 1080",
        "GeForce GTX TITAN X",
        "Quadro RTX A5000"
    ]

    Data = {
        'Intercept': (
                    -1.210333333, 0.883166667, 1.4497, -6.2932, 0.8057,
                    -12.354,
                    3.468566667, -4.801566667,
                    -5.7953, 11.8485,
                    -0.334266667, -5.155333333,
                    1.2307,
                    8.764,
                     ),

        'Gradient':  (
                    -4.582566667, -0.722133333, -2.874166667, -3.6307, 2.4105,
                    4.450333333,
                    -5.493, 0.393966667,
                    -3.3428, -1.303366667,
                    -0.122033333, 6.363033333,
                    1.057666667,
                    -9.4997,

                     )
            }
    colors = {'Intercept': '#008FC2', 'Gradient': '#76b900'}

    # reversed order
    GPUs = GPUs[::-1]
    Data['Gradient'] = Data['Gradient'][::-1]
    Data['Intercept'] = Data['Intercept'][::-1]
    
    
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
    ax.xaxis.label.set_color(colors['Intercept'])
    ax.tick_params(axis='x', colors=colors['Intercept'])
    
    # Add the legend and get the handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    plt.legend(handles, labels, loc='upper right')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Gradient / Percentage Error (%)', fontweight='bold')
    ax2.xaxis.label.set_color(colors['Gradient'])
    ax2.tick_params(axis='x', colors=colors['Gradient'])
    ax2.spines['top'].set_color(colors['Gradient'])
    ax2.spines['bottom'].set_color(colors['Intercept'])

    fig.set_size_inches(6, 8)
    plt.savefig('steady_state_accuracy.jpg', format='jpg', dpi=512, bbox_inches='tight')
    plt.savefig('steady_state_accuracy.eps', format='eps', bbox_inches='tight')
    plt.close()

def main():
    steam_gpu_share()
    steady_state_accuracy()

if __name__ == "__main__":
    main()