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
        'Intercept': (1.271166667, -0.89405, -1.4907, 6.5436, -0.7797,
                      11.820375,
                      -2.9074, 4.664825,
                      6.011833333, -11.8559,
                      0.3365, 4.848566667,
                      -1.2105,
                      -7.600125
                     ),
        'Gradient':  (4.800666667, 0.69855, 2.957433333, 3.7553, -2.3619,
                      -4.485375,
                      5.66245, -0.341375,
                      3.450833333, -0.109633333,
                      0.120966667, -5.9841,
                      -1.051566667,
                      9.6619
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
    plt.legend(handles, labels, loc='upper left')

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