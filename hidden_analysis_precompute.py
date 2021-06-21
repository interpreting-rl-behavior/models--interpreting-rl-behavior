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
    args = parser.parse_args()
    return args


# EPISODE_STRINGS = {v:str(v) for v in range(3431)}
def run():
    args = parse_args()
    num_episodes = 2000  # number of episodes to make plots for
    num_epi_paths = 9  # Number of episode to plot paths through time for. Arrow plots.
    n_components_pca = 64
    n_components_tsne = 2
    n_components_nmf = 32
    n_clusters = 70
    path_epis = list(range(num_epi_paths))

    seed = 42  # for the tSNE algo

    # Prepare load and save dirs
    main_data_path = args.agent_env_data_dir
    save_path = 'analysis/hx_analysis_precomp/'
    os.makedirs(save_path, exist_ok=True)

    # Get hidden states
    hx = np.load(os.path.join(main_data_path, 'episode0/hx.npy'))
    for ep in range(1, num_episodes):
        hx_to_cat = np.load(os.path.join(main_data_path,
                                         f'episode{ep}/hx.npy'))
        hx = np.concatenate((hx, hx_to_cat))

    # PCA
    print('Starting PCA...')
    hx_prescaling = hx
    scaler = StandardScaler()
    scaler = scaler.fit(hx_prescaling)
    hx = scaler.transform(hx_prescaling)
    mean_hx = scaler.mean_
    var_hx = scaler.var_
    pca = PCA(n_components=n_components_pca)
    hx_pca = pca.fit_transform(hx)
    print('PCA finished.')

    # Save PCs and the projections of each of the hx onto those PCs.
    np.save(save_path + 'hx_pca_%i.npy' % num_episodes, hx_pca)
    np.save(save_path + 'pcomponents_%i.npy' % num_episodes,
            scaler.inverse_transform(pca.components_))

    # Plot variance explained plot
    pca_percent = 100 * pca.explained_variance_/sum(pca.explained_variance_)
    above95explained = np.argmax(pca_percent.cumsum() > 95)
    plt.bar(list(range(n_components_pca)),
            pca_percent,
            color='blue')
    plt.xlabel("Principle Component")
    plt.ylabel("Variance Explained (%)")
    plt.savefig("pca_variance_explained_epis%i.png" % num_episodes)

    # k-means clustering
    print('Starting clustering...')
    pca_for_clust = PCA(n_components=above95explained)
    pca_for_clust = pca_for_clust.fit_transform(hx)
    knn_graph = kneighbors_graph(pca_for_clust, n_clusters, include_self=False)
    agc_model = AgglomerativeClustering(linkage='ward',
                                    connectivity=knn_graph,
                                    n_clusters=n_clusters)
    agc_model.fit(pca_for_clust)
    clusters = agc_model.labels_
    cluster_means = []
    for cluster_id in list(set(clusters)):
        cluster_mask = clusters == cluster_id
        cluster_eles = hx[cluster_mask]
        cluster_mean = cluster_eles.mean(axis=0)
        cluster_means.append(cluster_mean)
    cluster_means = np.array(cluster_means)
    np.save(save_path + 'clusters_%i.npy' % num_episodes, clusters)
    np.save(save_path + 'cluster_means_%i.npy' % num_episodes, cluster_means)
    print("Clustering finished.")

    # tSNE
    print('Starting tSNE...')
    pca_for_tsne = PCA(n_components=above95explained)
    pca_for_tsne = pca_for_tsne.fit_transform(hx)
    hx_tsne = TSNE(n_components=n_components_tsne, random_state=seed).fit_transform(pca_for_tsne)
    np.save(save_path + 'hx_tsne_%i.npy' % num_episodes, hx_tsne)
    print("tSNE finished.")

    # print('Starting UMAP...')
    # pca_for_umap = pca_for_tsne
    # reducer = umap.UMAP()
    # hx_umap = reducer.fit_transform(pca_for_umap)
    # np.save(save_path + 'hx_umap_%i.npy' % num_episodes, hx_umap)
    # print("tSNE finished.")

    print('Starting NMF...')
    hx_nonneg = hx_prescaling - np.min(hx_prescaling)
    model = NMF(n_components=n_components_nmf,
                init='random', random_state=0, max_iter=3000)
    hx_nmf = model.fit(hx_nonneg)
    np.save(save_path + 'hx_nmf_%i.npy' % num_episodes,
            hx_nmf.transform(hx_nonneg))
    np.save(save_path + 'nmf_components_%i.npy' % num_episodes,
            hx_nmf.components_ + np.min(hx_prescaling))
    print("NMF finished.")




if __name__ == "__main__":
    run()