import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Change this to the folder where the output of get-fig2-data.sh is stored
results_dir = "/home/lauro/projects/aisc2021/train-procgen-pytorch/experiments/hpc-results/"
vanilla_coinrun_resdir = results_dir + "test_rand_percent_0/train_rand_percent_0/"
van_df = pd.read_csv(vanilla_coinrun_resdir + "metrics.csv")
max_collect_freq = 0.1


def listdir(path):
    return [os.path.join(path, d) for d in os.listdir(path)]


def path_to_rand_percent(path):
    """extract integer rand_percent from path info
    path must end with integer rand_percent"""
    out = path[-3:]
    while not out.isdigit():
        out = out[1:]
    return int(out)


def seed_collect_freq(seed):
    """returns frequency at which agent collects the inv coin
    in the vanilla environment, for env seed"""
    idx = van_df['seed'] == seed
    return np.mean(van_df.loc[idx]["inv_coin_collected"])


collect_freqs = list(map(seed_collect_freq, range(np.max(van_df['seed']))))
collect_freqs = np.array(collect_freqs)
(good_seeds,) = np.nonzero(collect_freqs < max_collect_freq)


def get_good_seed_df(df):
    """
    given a dataframe with column 'seed', return new dataframe that is a subset
    of the old one, and includes only rows with good seeds.
    """
    good_seed_idx = [seed in good_seeds for seed in df['seed']]
    return df.loc[good_seed_idx]


test_rp100_resdir = os.path.join(results_dir, "test_rand_percent_100")
rand_percents = [path_to_rand_percent(file) for file in os.listdir(results_dir) if file.startswith("test_rand")]
rand_percents.sort()
joint_rp_paths = [os.path.join(results_dir, f"test_rand_percent_{rp}", f"train_rand_percent_{rp}") for rp in rand_percents[:-1]]


# sweep over training rand_percent
csv_files = {path_to_rand_percent(path): os.path.join(path, "metrics.csv") for path in listdir(test_rp100_resdir)}
dataframes = {k: pd.read_csv(v, on_bad_lines="skip") for k, v in csv_files.items()}
dataframes = {k: get_good_seed_df(df) for k, df in dataframes.items()}
reach_end_freqs = {k: np.mean(df["inv_coin_collected"]) for k, df in dataframes.items()}

data = list(reach_end_freqs.items())
data.sort()
data = np.array(data)


# sweep over training & test rand_percent jointly
# measure how often model dies or times out, ie not gets coin
csv_files = {path_to_rand_percent(path): os.path.join(path, "metrics.csv") for path in joint_rp_paths}
dataframes = {k: pd.read_csv(v) for k, v in csv_files.items()}
dataframes = {k: get_good_seed_df(df) for k, df in dataframes.items()}
fail_to_get_coin_freq = {k: 1 - np.mean(df["coin_collected"]) for k, df in dataframes.items()}

joint_data = list(fail_to_get_coin_freq.items())
joint_data.sort()
joint_data = np.array(joint_data)


baseline_vanilla_df = get_good_seed_df(van_df)
prob_of_reaching_end_without_inv_coin = np.mean(baseline_vanilla_df["coin_collected"] == 1)


figpath = "./"

fig, ax = plt.subplots(figsize=[6, 2.5])
plt.axhline(y=prob_of_reaching_end_without_inv_coin * 100, linestyle="--", color="tab:grey", label="Maximum possible OR frequency")

x, y = joint_data.T
ax.plot(x, y*100, "--o", label="IID Robustness Failure", color="tab:orange")

x, y = data.T
ax.plot(x, y*100, "--o", label="Objective Robustness Failure", color="tab:blue")

# plt.ylim(50, 101)
plt.ylabel("Frequency (%)")
plt.xlabel("Probability (%) of a level with randomized coin.")
plt.legend()

plt.savefig(figpath + "coinrun_freq.pdf")
