import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--datapath', type=str)
    args = parser.parse_args()
    return args

def plot():
    args = parse_args()

    data = pd.read_csv(args.datapath + "/log.csv")
    cols = list(data.columns)

    # Get rid of columns we don't plot
    cols.remove('wall_time')
    cols.remove('num_episodes')
    cols.remove('num_episodes_val')
    cols.remove('max_episode_len')
    cols.remove('min_episode_len')
    cols.remove('val_max_episode_len')
    cols.remove('val_min_episode_len')
    cols.remove('max_episode_rewards')
    cols.remove('min_episode_rewards')
    cols.remove('val_max_episode_rewards')
    cols.remove('val_min_episode_rewards')
    cols.remove('timesteps')

    # Plot episode len with some preprocessing, then remove
    plt.subplots(figsize=[10, 7])
    for name in ['mean_episode_len', 'val_mean_episode_len']:
        plt.plot(data['timesteps'], data[name]/100, label=name+"/100",
                 alpha=0.8)
    cols.remove('mean_episode_len')
    cols.remove('val_mean_episode_len')

    # Plot the rest
    for name in cols:
        plt.plot(data['timesteps'], data[name], label=name)
    plt.xlabel('timesteps')
    plt.legend()
    plt.savefig(args.datapath + "/plot.png")

if __name__ == '__main__':
    plot()