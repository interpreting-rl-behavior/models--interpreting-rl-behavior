import pandas as pd
import numpy as np
from precomput_analysis_funcs import scale_then_pca_then_save, plot_variance_expl_plot, clustering_after_pca, tsne_after_pca, nmf_then_save
import argparse
import os
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--agent_env_data_dir', type=str,
        default="data/")
    parser.add_argument(
        '--generated_data_dir', type=str,
        default='generative/recorded_informinit_gen_samples')

    args = parser.parse_args()
    return args


# EPISODE_STRINGS = {v:str(v) for v in range(3431)}
def run():
    args = parse_args()
    num_samples = 4000 # number of generated samples to use
    n_components_hx = 64
    n_components_env_h = 64
    env_components = np.load('analysis/env_analysis_precomp/pcomponents_env_20000.npy')
    hx_components = np.load('analysis/hx_analysis_precomp/pcomponents_4000.npy')
    ts_per_sample = 28


    # Prepare load and save dirs
    generated_data_path = args.generated_data_dir
    save_path = 'xcorr_matrices/'
    save_path = os.path.join(os.getcwd(), "analysis", save_path)
    os.makedirs(save_path, exist_ok=True)

    # Collect env and hxs and put them into a list of arrays, where each
    # array is one sample.

    # Get hidden states the were produced by the generative model
    print("Collecting env data together...")
    samples_data_env = []
    samples_data_hx = []
    hx_mu = np.load(os.path.join('analysis', 'hx_analysis_precomp') + '/hx_mu_4000.npy')
    hx_std = np.load(os.path.join('analysis', 'hx_analysis_precomp') + '/hx_std_4000.npy')
    env_pca_data = np.load(os.path.join('analysis', 'env_analysis_precomp') + '/pca_data_env_4000.npy')

    print("Collecting data together...")
    for ep in range(0, num_samples):
        # Env
        start_ts = (ep * ts_per_sample)
        stop_ts = (ep * ts_per_sample) + ts_per_sample
        env_vecs = env_pca_data[start_ts:stop_ts]
        samples_data_env.append(env_vecs.transpose())

        # Agent hx
        hx = np.load(os.path.join(generated_data_path,
                                  f'sample_{ep:05d}/agent_hxs.npy'))
        hx = (hx - hx_mu) / hx_std
        hx_vecs = hx_components @ hx.transpose()
        samples_data_hx.append(hx_vecs)
    print("Done collecting data together...")

    # For k in {-k, ... , -1, 0, 1, ... , k}
    K = 20
    for k in np.arange(0,K+1,1):
        ## Collect together the pairs of vectors that we'll be using to make the xcorr matrix
        xcorrs_hx = []
        xcorrs_env = []
        xcorrs_both = []
        for hx_sample, env_sample in zip(samples_data_hx, samples_data_env):
            # Clip samples so pathologically large PC loadings don't have as much influence
            hx_sample = np.clip(hx_sample, a_min=-6, a_max=6)
            env_sample = np.clip(env_sample, a_min=-6, a_max=6)
            if k == 0:
                set_a_hx = hx_sample
                set_a_env = env_sample
                set_b_hx = hx_sample
                set_b_env = env_sample
            else:
                set_a_hx = hx_sample[:,:-k]
                set_a_env = env_sample[:,:-k]
                set_b_hx = hx_sample[:,k:]
                set_b_env = env_sample[:,k:]
            set_a_hx_env = np.concatenate([set_a_hx, set_a_env])
            set_b_hx_env = np.concatenate([set_b_hx, set_b_env])

            ## Calculate the xcorr matrices
            n = set_a_hx.shape[0] # num elements in set
            xcorr_hx = (set_a_hx @ set_b_hx.transpose()) / n
            xcorr_env = (set_a_env @ set_b_env.transpose()) / n
            xcorr_both = (set_a_hx_env @ set_b_hx_env.transpose()) / n

            xcorrs_hx.append(xcorr_hx)
            xcorrs_env.append(xcorr_env)
            xcorrs_both.append(xcorr_both)

        xcorrs_hx = np.stack(xcorrs_hx)
        xcorrs_env = np.stack(xcorrs_env)
        xcorrs_both = np.stack(xcorrs_both)

        xcorrs_hx = np.mean(xcorrs_hx, axis=0)
        xcorrs_env = np.mean(xcorrs_env, axis=0)
        xcorrs_both = np.mean(xcorrs_both, axis=0)

        ## Plot and save the xcorr matrices
        print("Plotting for k=%i" % int(k))
        plot_heatmap(xcorrs_hx, save_path, "xcorr_hx_t+%i"%int(k), k)
        plot_heatmap(xcorrs_env, save_path, "xcorr_env_t+%i" % int(k), k)
        plot_heatmap(xcorrs_both, save_path, "xcorr_hx_and_env_t+%i" % int(k), k)

        plot_heatmap(xcorrs_hx[:11,:11], save_path, "first_ten_xcorr_hx_t+%i" % int(k), k)
        plot_heatmap(xcorrs_hx[:21,:21], save_path, "first_twenty_xcorr_hx_t+%i" % int(k), k)



def plot_heatmap(matrix, save_path, name, k):
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.imshow(matrix, cmap='seismic_r',
               interpolation='nearest')
    plt.colorbar()
    plt.clim(-1., 1.)
    plt.xticks(np.arange(0, matrix.shape[0], 2.0))
    plt.yticks(np.arange(0, matrix.shape[0], 2.0))
    plt.tight_layout()
    plt.title("Cross-correlation matrix comparing t (y-axis) with t+%i (x-axis)" % k)
    plt.savefig(os.path.join(save_path, "%s.png" % name))
    plt.clf()
    plt.close()



if __name__ == "__main__":
    run()

