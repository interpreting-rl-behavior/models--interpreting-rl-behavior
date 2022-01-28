import os
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

    num_evaluated_ims = 1001

    metadata = pd.read_csv(args.datapath + "/obs_metadata.csv")
    cols = list(metadata.columns)
    metadata = metadata[0:num_evaluated_ims]

    # Add value data
    value_files = os.listdir(args.datapath)
    value_file_cond = lambda file: ('val' in file) and ('.npy' in file)
    value_files = [file for file in value_files if value_file_cond(file)]
    value_files = sorted(value_files)
    value_files = value_files[0:len(metadata)]
    values = [np.load(os.path.join(args.datapath, file)) for file in value_files]
    values = np.array(values)
    metadata['value'] = values.tolist()


    # Make the labels and indices you'll use to identify different categories
    coin_categs_inds = [0, 1] #set(metadata['coin_visible'])
    bme_categs_inds = [0, 1, 2, 3] #set(metadata['begin_middle_end'])
    coin_categs_strs = ['(No coin)', '(with coin)']
    bme_categs_strs = ['Beginning\n', 'Middle\n', 'End\n', 'After End\n']
    full_categs_strs = []
    full_categs_inds = []
    for bme_categ in bme_categs_strs:
        for coin_categ in coin_categs_strs:
            full_categs_strs.append(f"{bme_categ} {coin_categ}")

    for bme_categ in bme_categs_inds:
        for coin_categ in coin_categs_inds:
            full_categs_inds.append( (bme_categ, coin_categ) )

    # Separate the data into their categories
    match_dict = {arr: label for (arr, label) in zip(full_categs_inds, full_categs_strs)}
    categ_metadata = {}
    for coin_categs_ind in coin_categs_inds:
        for bme_categs_ind in bme_categs_inds:
            condition = (metadata['coin_visible'] == coin_categs_ind) & (metadata['begin_middle_end'] == bme_categs_ind)
            label = match_dict[ (bme_categs_ind, coin_categs_ind) ]
            categ_metadata[label] = metadata[condition]


    # Get the mean value for each category and compute confidence intervals
    results_low_mean_high = {}
    for full_categ in full_categs_strs:
        # Draw 15000 bootstrap replicates
        categ_values = categ_metadata[full_categ]['value']
        bs_replicates_values = draw_bs_replicates(categ_values, np.mean, 10000)#15000)

        categ_mean = np.mean(categ_values)
        # Print empirical mean
        print(f"{full_categ} | Empirical mean: " + str(categ_mean))

        # Print the mean of bootstrap replicates
        print(f"{full_categ} | Bootstrap replicates mean: " + str(np.mean(bs_replicates_values)))
        categ_low = np.percentile(bs_replicates_values,[5.])
        categ_high = np.percentile(bs_replicates_values,[95.])
        results_low_mean_high[full_categ] = (categ_low, categ_mean, categ_high)




    # # And we do 'post end-wall' manually because we collected that data later
    # value_files_post = os.listdir(args.datapath + "/for_post_endwall_bars")
    # value_files_post = [file for file in value_files_post if value_file_cond(file)]
    # value_files_post = sorted(value_files_post)
    # values_post = [np.load(os.path.join(args.datapath + "/for_post_endwall_bars", file)) for file in value_files_post]
    # values_post = np.array(values_post)
    # bs_replicates_values_post = draw_bs_replicates(values_post, np.mean,
    #                                           10000)  # 15000)
    # values_post_mean = np.mean(values_post)
    # print(f"Post | Empirical mean: " + str(values_post))
    # # Print the mean of bootstrap replicates
    # print(f"Post | Bootstrap replicates mean: " + str(
    #     np.mean(bs_replicates_values_post)))
    # values_post_low = np.percentile(bs_replicates_values_post, [5.])
    # values_post_high = np.percentile(bs_replicates_values_post, [95.])
    # post_name = 'After End\n(No coin)'
    # results_low_mean_high[post_name] = (values_post_low, values_post_mean, values_post_high)
    #
    # full_categs_strs.append(post_name)

    # Then do a bit of processing
    xticks = np.arange(0,len(full_categs_strs))
    means = [v[1] for v in results_low_mean_high.values()]
    lows = [v[0] for v in results_low_mean_high.values()]
    highs = [v[2] for v in results_low_mean_high.values()]
    errs = [(v[0], v[2]) for v in results_low_mean_high.values()]
    errs = [np.concatenate(e) for e in errs]
    errs = np.stack(errs, axis=0).transpose()
    errs = errs - means
    errs = np.abs(errs)


    fig, ax = plt.subplots(figsize=(7, 4))
    ax.yaxis.grid(True)
    ax.bar(xticks, means, yerr=errs, align='center', alpha=0.95,
           ecolor='black', color=['orangered', 'lightsalmon', 'olivedrab', 'yellowgreen', 'deepskyblue', 'skyblue', 'darkslategray'], capsize=10)
    plt.ylim([5, 10])
    ax.set_ylabel('Value function output')
    plt.box(False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(full_categs_strs)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(args.datapath + '/bar_plot_with_error_bars.png')
    plt.close()



def draw_bs_replicates(data, func, size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)

    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data, size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)

    return bs_replicates




if __name__ == '__main__':
    plot()