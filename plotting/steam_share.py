#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def steam_gpu_share():
    labels = ['Other',   'Intel',   'AMD',     'Older',
              'Maxwell\n(GeForce 700/900 Series)', 'Pascal\n(GeForce 10 Series)',  'Turing\n(GeForce 16/20 Series)',
              'Ampere\n(GeForce 30 Series)',  'Ada Lovelace\n(GeForce 40 Series)']
    
    
    sizes   = [ 0.25,      8.66,      15.04,     5.26,      2.37,      14.12,     24.77,     26.62,     2.91         ]
    colors  = ['#f2d600', '#0071c5', '#ed1c24', '#2f4a00', '#466f00', '#5e9400', '#76b900', '#91c732', '#acd566'     ]
    explode = ( 0.15,      0.1,       0.1,       0,         0,         0,         0,         0,         0          )

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.2f%%', colors=colors, startangle=90, explode=explode, pctdistance=0.75)

    fig.set_size_inches(6, 6)
    plt.savefig('steam_gpu_share.jpg', format='jpg', dpi=512, bbox_inches='tight')
    plt.savefig('steam_gpu_share.svg', format='svg', bbox_inches='tight')
    plt.close()

def steady_state_accuracy():
    species = ("Adelie", "Chinstrap", "Gentoo")
    penguin_means = {
        'Bill Length': (-38.79, 48.83, 47.50),
        'Flipper Length': (189.95, 195.82, 217.19),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    # ax.set_ylim(0, 250)


    fig.set_size_inches(6, 6)
    plt.savefig('steady_state_accuracy.jpg', format='jpg', dpi=512, bbox_inches='tight')
    plt.savefig('steady_state_accuracy.svg', format='svg', bbox_inches='tight')
    plt.close()

def main():
    steam_gpu_share()
    steady_state_accuracy()

if __name__ == "__main__":
    main()