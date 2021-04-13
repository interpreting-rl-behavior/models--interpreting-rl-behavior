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
    args = parse_args()

    data = pd.read_csv(args.datapath + "/progress.csv")
    cols = list(data.columns)

    # Get rid of columns we don't plot
    cols.remove('epoch')
    cols.remove('batches')
    cols.remove('loss/total recon w/o KL')

    # Plot episode len with some preprocessing, then remove
    plt.subplots(figsize=[11, 11])
    # plt.xticks(np.arange(0, max(data['batches']*max(data['epoch'] + 1)), 100))
    max_y = data[cols].max().max()
    plt.yticks(np.arange(0, max_y, max_y/71))
    # Plot the rest
    for name in cols:
        plt.plot(data.index*10, data[name], label=name)
        plt.grid(b=True, which='both')
        # plt.hlines(min(data[name]), xmin=0, xmax=max(data.index*10), alpha=0.5, colors='black')
    plt.xlabel('# batches')
    plt.legend()
    plt.savefig(args.datapath + "/plot_gen_model_training.png")

if __name__ == '__main__':
    plot()