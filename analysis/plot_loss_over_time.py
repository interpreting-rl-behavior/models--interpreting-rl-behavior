import json
import matplotlib.pyplot as plt
import argparse
import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting error over time')
    parser.add_argument(
        '--presaved_data_path', type=str, default="generative/analysis/loss_over_time/20210730_124423.json")
    args = parser.parse_args()
    return args

def run():
    args = parse_args()

    with open(args.presaved_data_path, "r") as f:
        data = json.load(f)

    # All keys should have the same length
    x_vals = range(1, len(next(iter(data.values()))) + 1)

    # Create plot and save to directory
    plt.style.use("ggplot")
    for key in data:
        # if key == "total recon w/o KL":
            # continue
        # mean_centered = data[key] - np.mean(data[key])
        # plt.plot(x_vals, mean_centered, label=key)
        plt.plot(x_vals, data[key], label=key)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.xlabel("Simulation Step")
    plt.ylabel("MSE")

    # Get the stem of the input data filename
    filestem = args.presaved_data_path.split("/")[-1].split(".")[0]
    plt.savefig(f"generative/analysis/loss_over_time/{filestem}.png")
    plt.show()

if __name__ == "__main__":
    run()