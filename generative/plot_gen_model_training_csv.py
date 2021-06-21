"""
When the generative model is training, the data are logged to a CSV file. This
script plots the relevant variables.
"""
# TODO fix these plots
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
    """
    Gets the csv and plots the desired data.

    The result is the generative model training curve.
    """
    # Get CLI args
    args = parse_args()
    data = pd.read_csv(args.datapath + "/progress.csv")

    # Get rid of columns we don't want to plot
    cols = list(data.columns)
    cols.remove('epoch')
    cols.remove('batches')
    cols.remove('loss/total recon w/o KL')

    # cols.remove('loss/act_log_probs')
    # cols.remove('loss/KL')
    # cols.remove('loss/reward')
    # cols.remove('loss/hx')
    # cols.remove('loss/done')
    scales = {'loss/act_log_probs': 5e-3,
              'loss/KL': 1e-2,
              'loss/reward': 1e-2,
              'loss/hx': 1e-1,
              'loss/done': 1e-1,
              'loss/obs': 1.}

    # Plot
    plt.subplots(figsize=[11, 11])
    max_y = data['loss/obs'][100:].max().max()  # largest value in any column
    plt.yticks(np.arange(0, max_y, max_y/50))

    for name in cols:
        plt.plot(data.index[100:]*10, data[name][100:] * scales[name], label=name)
        plt.grid(b=True, which='both')

    plt.xlabel('# batches')
    plt.legend()
    plt.savefig(args.datapath + "/plot_gen_model_training.png")
    data_name = args.datapath[-15:]
    parent_dir = args.datapath[:100]
    plt.savefig(parent_dir + f"plot_gen_model_training_{data_name}.png")

if __name__ == '__main__':
    plot()