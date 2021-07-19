
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
#import umap
from sklearn.preprocessing import StandardScaler


import argparse

import matplotlib
import matplotlib.pyplot as plt
import os
import time

#TODO: think about t-SNE initialization
# https://www.nature.com/articles/s41587-020-00809-z 
# https://jlmelville.github.io/smallvis/init.html

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
    num_samples = 3999 # number of generated samples to use
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
    save_path = 'analysis/env_analysis_precomp/'
    os.makedirs(save_path, exist_ok=True)
    plot_save_path = 'analysis/env_plots'
    os.makedirs(plot_save_path, exist_ok=True)

    # Get hidden states the were produced by the generative model
    # print("Collecting env data together...")
    # env_hx = np.load(os.path.join(generated_data_path, 'sample_00000/env_hid_states.npy'))
    # env_c = np.load(os.path.join(generated_data_path, 'sample_00000/env_cell_states.npy'))
    # env_h = np.concatenate((env_hx, env_c), axis=1)
    # for ep in range(1, num_samples):
    #     env_hx = np.load(os.path.join(generated_data_path,
    #                                  f'sample_{ep:05d}/env_hid_states.npy'))
    #     env_c = np.load(os.path.join(generated_data_path,
    #                                  f'sample_{ep:05d}/env_cell_states.npy'))
    #     env_h_to_cat = np.concatenate((env_hx, env_c), axis=1)
    #     env_h = np.concatenate((env_h, env_h_to_cat))

    # only hidden states, no cell state:
    print("Collecting env data together...")
    env_h = np.load(os.path.join(generated_data_path, 'sample_00000/env_hid_states.npy'))
    for ep in range(1, num_samples):
        env_h_to_cat = np.load(os.path.join(generated_data_path,
                                     f'sample_{ep:05d}/env_hid_states.npy'))
        env_h = np.concatenate((env_h, env_h_to_cat))

    # PCA
    print('Starting PCA...')
    env_h_prescaling = env_h
    scaler = StandardScaler()
    scaler = scaler.fit(env_h_prescaling)
    env_h = scaler.transform(env_h_prescaling)
    mean_env_h = scaler.mean_
    var_env_h = scaler.var_
    pca = PCA(n_components=n_components_pca)
    env_h_pca = pca.fit_transform(env_h)
    print('PCA finished.')

    # Save PCs and the projections of each of the env_h onto those PCs.
    np.save(save_path + 'env_h_pca_%i.npy' % num_samples, env_h_pca)
    np.save(save_path + 'pcomponents_%i.npy' % num_samples,
            scaler.inverse_transform(pca.components_))

    # Plot variance explained plot
    pca_percent = 100 * pca.explained_variance_/sum(pca.explained_variance_)
    above95explained = np.argmax(pca_percent.cumsum() > 95)
    plt.bar(list(range(n_components_pca)),
            pca_percent,
            color='blue')
    plt.xlabel("Principle Component")
    plt.ylabel("Variance Explained (%)")
    plt.savefig(plot_save_path + "/pca_variance_explained_env_h_epis%i.png" % num_samples)

    # k-means clustering
    print('Starting clustering...')
    pca_for_clust = PCA(n_components=above95explained)
    pca_for_clust = pca_for_clust.fit_transform(env_h)
    knn_graph = kneighbors_graph(pca_for_clust, n_clusters, include_self=False)
    agc_model = AgglomerativeClustering(linkage='ward',
                                    connectivity=knn_graph,
                                    n_clusters=n_clusters)
    agc_model.fit(pca_for_clust)
    clusters = agc_model.labels_
    cluster_means = []
    for cluster_id in list(set(clusters)):
        cluster_mask = clusters == cluster_id
        cluster_eles = env_h[cluster_mask]
        cluster_mean = cluster_eles.mean(axis=0)
        cluster_means.append(cluster_mean)
    cluster_means = np.array(cluster_means)
    np.save(save_path + 'clusters_%i.npy' % num_samples, clusters)
    np.save(save_path + 'cluster_means_%i.npy' % num_samples, cluster_means)
    print("Clustering finished.")

    # tSNE
    print('Starting tSNE...')
    pca_for_tsne = PCA(n_components=above95explained)
    pca_for_tsne = pca_for_tsne.fit_transform(env_h)
    env_h_tsne = TSNE(n_components=n_components_tsne, random_state=seed).fit_transform(pca_for_tsne)
    np.save(save_path + 'env_h_tsne_%i.npy' % num_samples, env_h_tsne)
    print("tSNE finished.")

    # print('Starting UMAP...')
    # pca_for_umap = pca_for_tsne
    # reducer = umap.UMAP()
    # env_h_umap = reducer.fit_transform(pca_for_umap)
    # np.save(save_path + 'env_h_umap_%i.npy' % num_samples, env_h_umap)
    # print("tSNE finished.")

    print('Starting NMF...')
    env_h_nonneg = env_h_prescaling - np.min(env_h_prescaling) # TODO is this the best way to do this? Surely we subtract the min from each dim, and maybe also normalize. OR OR OR we could 'a-score', a term I coined that is like z-scoring but where you subtract the min instead of the mean
    model = NMF(n_components=n_components_nmf,
                init='random', random_state=0, max_iter=3000)
    env_h_nmf = model.fit(env_h_nonneg)
    np.save(save_path + 'env_h_nmf_%i.npy' % num_samples,
            env_h_nmf.transform(env_h_nonneg))
    np.save(save_path + 'nmf_components_%i.npy' % num_samples,
            env_h_nmf.components_ + np.min(env_h_prescaling))
    print("NMF finished.")




if __name__ == "__main__":
    run()