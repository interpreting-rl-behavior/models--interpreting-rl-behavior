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


def scale_then_pca_then_save(data, num_pcs, save_path, aux_name1, aux_name2):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)
    mean_sml_vecs = scaler.mean_
    var_sml_vecs = scaler.var_
    pca_object = PCA(n_components=num_pcs)
    data_pcaed = pca_object.fit_transform(data_scaled)

    np.save(save_path + f'pca_data_{aux_name1}_{aux_name2}.npy', data_pcaed)
    np.save(save_path + f'pcomponents_{aux_name1}_{aux_name2}.npy',
            scaler.inverse_transform(pca_object.components_))
    return data_pcaed, data_scaled, pca_object, scaler

def plot_variance_expl_plot(pca_object, num_pcs, plot_save_path, aux_name1,
                            aux_name2):
    pca_percent = 100 * pca_object.explained_variance_/sum(pca_object.explained_variance_)
    above95explained = np.argmax(pca_percent.cumsum() > 95)
    plt.figure(figsize=(15, 5))
    plt.bar(list(range(num_pcs)),
            pca_percent,
            color='blue',
            )
    plt.xlabel("Principle Component")
    plt.ylabel("Variance Explained (%)")
    plt.savefig(plot_save_path + f"/pca_variance_explained_{aux_name1}_epis{aux_name2}.png")
    plt.close()
    return above95explained


def clustering_after_pca(data, num_pcs, num_clusters, save_path, aux_name1,
                     aux_name2):

    # Scale data
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)

    # PCA on scaled data (just to make the job of clustering easier by reducing
    # dimensions)
    pca_for_clust = PCA(n_components=num_pcs)
    pca_for_clust = pca_for_clust.fit_transform(data)

    # Clustering
    knn_graph = kneighbors_graph(pca_for_clust, num_clusters, include_self=False)
    agc_model = AgglomerativeClustering(linkage='ward',
                                        connectivity=knn_graph,
                                        n_clusters=num_clusters)
    agc_model.fit(pca_for_clust)

    # Calculate the means of the clusters
    clusters = agc_model.labels_
    cluster_means = []
    for cluster_id in list(set(clusters)):
        cluster_mask = clusters == cluster_id
        cluster_eles = data[cluster_mask]
        cluster_mean = cluster_eles.mean(axis=0)
        cluster_means.append(cluster_mean)
    cluster_means = np.array(cluster_means)

    # Save
    np.save(save_path + f'clusters_{aux_name1}_{aux_name2}.npy', clusters)
    np.save(save_path + f'cluster_means_{aux_name1}_{aux_name2}.npy', cluster_means)

    return clusters, cluster_means

def tsne_after_pca(data, num_pcs, num_tsne_components, save_path, aux_name1,
                     aux_name2):
    seed = 42

    # Scale data
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)

    # PCA on scaled data (just to make the job of clustering easier by reducing
    # dimensions)
    pca_for_tsne = PCA(n_components=num_pcs)
    pca_for_tsne = pca_for_tsne.fit_transform(data)
    data_tsneed = TSNE(n_components=num_tsne_components,
                      random_state=seed).fit_transform(pca_for_tsne)
    np.save(save_path + f'tsne_{aux_name1}_{aux_name2}.npy', data_tsneed)

def nmf_then_save(data, num_factors, save_path, aux_name1, aux_name2):
    data_nonneg = data - np.min(data, axis=0) # TODO is this the best way to do this? Surely we subtract the min from each dim, and maybe also normalize. OR OR OR we could 'a-score', a term I coined that is like z-scoring but where you subtract the min instead of the mean
    model = NMF(n_components=num_factors,
                init='random', random_state=0, max_iter=3000)
    env_h_nmf = model.fit(data_nonneg)
    np.save(save_path + f'nmf_{aux_name1}_{aux_name2}.npy',
            env_h_nmf.transform(data_nonneg))
    np.save(save_path + f'nmf_components_{aux_name1}_{aux_name2}.npy',
            env_h_nmf.components_ + np.min(data, axis=0))