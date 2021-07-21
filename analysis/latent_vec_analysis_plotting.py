"""Make sure you've run latent_vec_analysis_precompute.py before running this
because it generates data that this script uses."""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import imageio

pd.options.mode.chained_assignment = None  # default='warn'


def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--agent_env_data_dir', type=str,
        default="data")
    parser.add_argument(
        '--precomputed_analysis_data_path', type=str, default="latent_vec_analysis_precomp/")
    parser.add_argument(
        '--presaved_data_path', type=str, default="/media/lee/DATA/DDocs/AI_neuro_work/assurance_project_stuff/data/precollected/")
    args = parser.parse_args()
    return args

def run():
    args = parse_args()
    # number of episodes to make plots for. Needs to be
    num_generated_samples = 200
    # the same as the precomputed data you want to use
    plot_pca = True
    plot_3d_pca_all = True
    plot_clusters = True
    plot_tsne = True
    first_PC_ind = 0
    second_PC_ind = 1

    # Set up saving and loading dirs
    precomp_data_path = args.precomputed_analysis_data_path
    save_path = 'latent_vec_plots'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load lv data
    lv_inf_pca = np.load(os.path.join(precomp_data_path,
                         f'pca_data_lv_inf_{num_generated_samples}.npy'))
    lv_rand_pca_proj = np.load(os.path.join(precomp_data_path,
                               f'pca_data_lv_rand_projected_{num_generated_samples}.npy'))
    aux_data = pd.read_csv(os.path.join(precomp_data_path,
                           f'lv_inf_aux_data_{num_generated_samples}.csv'))

    # Make a few extra variables for plotting
    lv_clusters = aux_data['lv_cluster']
    cluster_labels = list(range(max(lv_clusters)))
    cluster_cmap = sns.color_palette("husl", max(lv_clusters), as_cmap=True)

    plotting_variables = ['sample_reward_sum',
                          'sample_has_done', 'sample_avg_value', 'lv_cluster']
    plot_cmaps = {'lv_cluster':               cluster_cmap,
                  'sample_has_done':         'autumn_r',
                  'sample_avg_value':        'cool',
                  'sample_reward_sum':       'cool',}


    # Plotting
    if plot_pca:
        print("Plotting PCAs")

        lv_inf_pca_x = lv_inf_pca[:, first_PC_ind]
        lv_inf_pca_y = lv_inf_pca[:, second_PC_ind]

        # Create grid of plots
        pca_alpha = 0.95
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 18.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            print(col)
            ax = fig.add_subplot(2, 2, plot_idx)
            splot = plt.scatter(lv_inf_pca_x,
                                lv_inf_pca_y,
                                c=aux_data[col],
                                cmap=plot_cmaps[col],
                                s=5., alpha=pca_alpha)
            # if plot_gen_hx_pca:
            #     splot = plt.scatter(
            #         gen_hx_pca[:, first_PC_ind],
            #         gen_hx_pca[:, second_PC_ind],
            #         c='black',
            #         s=0.05, alpha=0.9)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1),borderaxespad=0)
            ax.set_frame_on(False)

        fig.tight_layout()
        fig.savefig(f'{save_path}/lv_pca_epsd{num_generated_samples}_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()

    if plot_tsne:
        lv_inf_tsne = np.load(args.precomputed_analysis_data_path + \
                          'tsne_lv_inf_%i.npy' % num_generated_samples)
        print('Starting tSNE...')

        lv_inf_tsne_x = lv_inf_tsne[:, 0]
        lv_inf_tsne_y = lv_inf_tsne[:, 1]

        # Create grid of plots
        pca_alpha = 0.95
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 18.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            print(col)
            ax = fig.add_subplot(2, 2, plot_idx)
            splot = plt.scatter(lv_inf_tsne_x,
                                lv_inf_tsne_y,
                                c=aux_data[col],
                                cmap=plot_cmaps[col],
                                s=5., alpha=pca_alpha)
            # if plot_gen_hx_pca:
            #     splot = plt.scatter(
            #         gen_hx_pca[:, first_PC_ind],
            #         gen_hx_pca[:, second_PC_ind],
            #         c='black',
            #         s=0.05, alpha=0.9)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1),borderaxespad=0)
            ax.set_frame_on(False)

        fig.tight_layout()
        fig.savefig(f'{save_path}/lv_tsne_epsd{num_generated_samples}_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()


if __name__ == "__main__":
    run()