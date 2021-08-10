import json
import matplotlib.pyplot as plt
import argparse
import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting error over time')
    parser.add_argument(
        '--presaved_data_path', type=str, default="analysis/loss_over_time/20210730_124423.json")
    args = parser.parse_args()
    return args

def run():
    args = parse_args()
    loss_keys = ['obs', 'hx', 'done', 'reward', 'act_log_probs']#, 'total recon w/o KL']

    with open(args.presaved_data_path, "r") as f:
        data = json.load(f)

    # All keys should have the same length
    x_vals = range(1, len(next(iter(data.values()))) + 1)

    # Create plot and save to directory
    plt.style.use("ggplot")
    for key in loss_keys:
        mean = np.array(data[key+'_mean'])
        higher = np.array(data[key+'_max'])
        lower = np.array(data[key+'_min'])
        plt.plot(x_vals, mean, label=key)
        plt.fill_between(x_vals, mean - lower, mean + higher, alpha=0.3)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Simulation Step")
    plt.ylabel("MSE")

    # Get the stem of the input data filename
    filestem = args.presaved_data_path.split("/")[-1].split(".")[0]
    plt.savefig(f"analysis/loss_over_time/{filestem}.png")
    plt.show()

if __name__ == "__main__":
    run()