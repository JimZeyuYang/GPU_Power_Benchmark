import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--tl', type=int, required=True)
    args = parser.parse_args()

    load = pd.read_csv('timestamps.csv')
    load['acticity'] = (load.index / 2).astype(int) % 2
    load['timestamp'] = (load['timestamp'] / 1000).astype(int) 
    t0 = load['timestamp'][0]
    load['timestamp'] -= t0

    power = pd.read_csv('gpudata.csv')
    power['timestamp'] = (pd.to_datetime(power['timestamp']) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
    power['timestamp'] -= 60*60*1000
    power['timestamp'] -= t0

    plt.figure()
    fig, axis = plt.subplots(nrows=2, ncols=1)

    axis[0].plot(load['timestamp'], load['acticity']*100, label='load')
    axis[0].plot(power['timestamp'], power[' utilization.gpu [%]'], label='utilization.gpu [%]')
    axis[0].set_xlabel('Time (ms)')
    axis[0].set_ylabel('Load')
    axis[0].grid(True, linestyle='--', linewidth=0.5)
    axis[0].set_xlim(0, load['timestamp'].max())
    axis[0].legend()
    axis[0].set_title(f'{args.gpu} - {args.tl} ms load window')
    
    axis[1].plot(power['timestamp'], power[' power.draw [W]'], label='power')
    axis[1].set_xlabel('Time (ms)')
    axis[1].set_ylabel('Power [W]')
    axis[1].grid(True, linestyle='--', linewidth=0.5)
    axis[1].set_xlim(axis[0].get_xlim())

    fig.set_size_inches(20, 10)
    plt.savefig('result.jpg', format='jpg', dpi=256, bbox_inches='tight')
    plt.savefig('result.svg', format='svg', bbox_inches='tight')


if __name__ == '__main__':
    main()