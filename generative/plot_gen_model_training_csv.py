"""
When the generative model is training, the data are logged to a CSV file. This
script plots the relevant variables.
"""
# TODO fix these plots
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def run():
    # Get CLI args
    args = parse_args()
    if args.datapath == 'all':
        # get all experiment dirs
        dirs = next(os.walk('results'))[1]

        # for loop that plots all exps
        for dir in dirs:
            path = './results/' + dir
            sub_dirs = next(os.walk(path))[1]
            for sub_dir in sub_dirs:
                print(path + '/' + sub_dir)
                plot(os.path.join(path, sub_dir))



def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--datapath', type=str)
    args = parser.parse_args()
    return args

def plot(data_path):
    """
    Gets the csv and plots the desired data.

    The result is the generative model training curve.
    """
    data = pd.read_csv(data_path + "/progress.csv")

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
    max_y = data['loss/hx'][100:].max().max()  # largest value in any column
    plt.yticks(np.arange(0, max_y, max_y/50))

    for name in cols:
        plt.plot(data.index[100:]*10, data[name][100:] * scales[name], label=name)
        plt.grid(b=True, which='both')

    plt.xlabel('# batches')
    plt.legend()
    plt.savefig(data_path + "/plot_gen_model_training.png")
    exp_name, data_date = data_path.split('/')[2:4]
    parent_dir = data_path[:10]
    plt.savefig(parent_dir + f"plot_gen_model_training_{exp_name + data_date}.png")

if __name__ == '__main__':
    run()