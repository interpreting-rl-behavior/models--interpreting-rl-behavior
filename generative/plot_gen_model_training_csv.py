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
    cols.remove('loss/dissim_loss')

    # Plot
    plt.subplots(figsize=[11, 11])
    max_y = data[cols].max().max()  # largest value in any column
    plt.yticks(np.arange(0, max_y, max_y/71))

    for name in cols:
        plt.plot(data.index*10, data[name], label=name)
        plt.grid(b=True, which='both')

    plt.xlabel('# batches')
    plt.legend()
    plt.savefig(args.datapath + "/plot_gen_model_training.png")

if __name__ == '__main__':
    plot()