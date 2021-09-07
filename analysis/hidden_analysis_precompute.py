import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from precomput_analysis_funcs import plot_variance_expl_plot, clustering_after_pca, tsne_after_pca, nmf_then_save
import argparse
import os

pd.options.mode.chained_assignment = None  # default='warn'

COINRUN_ACTIONS = {0: 'downleft', 1: 'left', 2: 'upleft', 3: 'down', 4: None, 5: 'up',
                   6: 'downright', 7: 'right', 8: 'upright', 9: None, 10: None, 11: None,
                   12: None, 13: None, 14: None}
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
    num_episodes = 1000#2000  # number of episodes to make plots for
    num_generated_samples = 200 # number of generated samples to use
    num_epi_paths = 9  # Number of episode to plot paths through time for. Arrow plots.
    n_components_pca = 64
    n_components_tsne = 2
    n_components_nmf = 32
    n_clusters = 70
    path_epis = list(range(num_epi_paths))

    seed = 42  # for the tSNE algo

    # Prepare load and save dirs
    main_data_path = args.agent_env_data_dir
    generated_data_path = args.generated_data_dir
    save_path = 'hx_analysis_precomp/'
    save_path = os.path.join(os.getcwd(), "analysis", save_path)
    plot_save_path = 'hx_plots'
    plot_save_path = os.path.join(os.getcwd(), "analysis", plot_save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)

    # Get (true) hidden states that were generated by the agent interacting
    # with the real environment
    hx = np.load(os.path.join(main_data_path, 'episode0/hx.npy'))
    for ep in range(1, num_episodes):
        hx_to_cat = np.load(os.path.join(main_data_path,
                                         f'episode{ep}/hx.npy'))
        hx = np.concatenate((hx, hx_to_cat))

    # Get hidden states the were produced by the generative model
    gen_hx = np.load(os.path.join(generated_data_path, 'sample_00000/agent_hxs.npy'))
    for ep in range(1, num_generated_samples):
        gen_hx_to_cat = np.load(os.path.join(generated_data_path,
                                         f'sample_{ep:05d}/agent_hxs.npy'))
        gen_hx = np.concatenate((gen_hx, gen_hx_to_cat))

    # PCA
    print('Starting PCA...')
    uncentred_unscaled_hx = hx
    uncentred_unscaled_gen_hx = gen_hx

    mu = np.mean(uncentred_unscaled_hx, axis=0)
    std = np.std(uncentred_unscaled_hx, axis=0)

    centred_scaled_hx = (uncentred_unscaled_hx - mu ) / std
    centred_scaled_gen_hx = (uncentred_unscaled_gen_hx - mu) / std

    pca_obj = PCA(n_components=n_components_pca)
    centred_scaled_hx_pca = pca_obj.fit_transform(centred_scaled_hx)
    centred_scaled_gen_hx_projected = pca_obj.transform(centred_scaled_gen_hx)

    # For future reference the following returns an array of mostly True:
    np.isclose(pca_obj.transform(centred_scaled_hx),
               centred_scaled_hx @ (pca_obj.components_.transpose()),
               atol=1e-3)

    print('PCA finished.')

    # Save PCs and the projections of each of the hx onto those PCs.
    np.save(save_path + 'hx_mu_%i.npy' % num_episodes, mu)
    np.save(save_path + 'hx_std_%i.npy' % num_episodes, std)
    np.save(save_path + 'hx_pca_%i.npy' % num_episodes, centred_scaled_hx_pca)
    np.save(save_path + 'pcomponents_%i.npy' % num_episodes,pca_obj.components_)
    # And save the projections of the generated data onto the PCs of true data
    np.save(save_path + 'gen_hx_projected_real%i_gen%i.npy' % (num_episodes,
                                                         num_generated_samples),
            centred_scaled_gen_hx_projected)

    # Plot variance explained plot
    above95explained = \
        plot_variance_expl_plot(pca_obj, n_components_pca, plot_save_path,
                                "hx", num_episodes)

    # k-means clustering
    print('Starting clustering...')
    clustering_after_pca(hx, above95explained, n_clusters, save_path,
                         "hx", num_episodes)
    print("Clustering finished.")

    # tSNE
    print('Starting tSNE...')
    tsne_after_pca(hx, above95explained, n_components_tsne,
                   save_path, "hx", num_episodes)
    print("tSNE finished.")

    # print('Starting UMAP...')
    # pca_for_umap = pca_for_tsne
    # reducer = umap.UMAP()
    # env_h_umap = reducer.fit_transform(pca_for_umap)
    # np.save(save_path + 'env_h_umap_%i.npy' % num_samples, env_h_umap)
    # print("tSNE finished.")

    print('Starting NMF...')
    nmf_then_save(hx, n_components_nmf, save_path, "hx", num_episodes)
    print("NMF finished.")


if __name__ == "__main__":
    run()