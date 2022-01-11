"""
When the generative model is training, the data are logged to a CSV file. This
script plots the relevant variables.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def run():
    # Get CLI args
    args = parse_args()
    if args.experiment_name is not '':
        print("Only plotting %s" % args.experiment_name)
        path = './results/' + args.experiment_name
        plot(path)
    if args.datapath == 'all':
        # get all experiment dirs
        dirs = next(os.walk('results'))[1]

        # for loop that plots all exps
        for dir in dirs:
            path = './results/' + dir
            sub_dirs = next(os.walk(path))[1]
            for sub_dir in sub_dirs:
                print(path + '/' + sub_dir)
                if os.path.exists(path + '/' + sub_dir + "/progress.csv"):
                    plot(os.path.join(path, sub_dir))
                else:
                    print("Skipping")


def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--datapath', type=str)
    parser.add_argument(
        '--experiment_name', type=str, default='')
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

    # Plot
    plt.subplots(figsize=[11, 11])
    # max_y = data['loss/hx'][100:].max().max()  # largest value in any column
    # plt.yticks(np.arange(0, max_y, max_y/50))

    for name in cols: #100 is log interval
        plt.plot(data.index[10:]*100, data[name][10:], label=name)
        plt.grid(b=True, which='both')

    plt.xlabel('# batches')
    plt.legend()
    plt.savefig(data_path + "/plot_gen_model_training.png")
    exp_name, data_date = data_path.split('/')[2:4]
    parent_dir = data_path[:10]
    plt.savefig(parent_dir + f"plot_gen_model_training_{exp_name + data_date}.png")

if __name__ == '__main__':
    run()