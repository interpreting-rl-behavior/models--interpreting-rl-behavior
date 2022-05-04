import os

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
#import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import dump, load


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

def plot_cum_variance_expl_plot(pca_object, num_pcs, plot_save_path, aux_name1,
                            aux_name2):
    pca_percent = 100 * pca_object.explained_variance_/sum(pca_object.explained_variance_)
    above95explained = np.argmax(pca_percent.cumsum() > 95)
    above90explained = np.argmax(pca_percent.cumsum() > 90)
    plt.figure(figsize=(15, 5))
    plt.bar(list(range(num_pcs)),
            pca_percent.cumsum(),
            color='blue',
            )

    plt.xlabel("Principle Component")
    plt.ylabel("Variance Explained (%)")
    plt.savefig(plot_save_path + f"/pca_cum_variance_explained_{aux_name1}_epis{aux_name2}.png")
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

def clustering_after_tsne(data, num_pcs, num_tsne_components, num_clusters, save_path, aux_name1,
                     aux_name2):
    seed = 42

    # Scale data
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)

    # PCA on scaled data (just to make the job of clustering easier by reducing
    # dimensions)
    pca_for_clust = PCA(n_components=num_pcs)
    pca_for_clust = pca_for_clust.fit_transform(data)
    tsne_for_clust = TSNE(n_components=num_tsne_components,
                      random_state=seed,
                      init=pca_for_clust[:,0:num_tsne_components]).fit_transform(pca_for_clust)

    # Clustering
    knn_graph = kneighbors_graph(tsne_for_clust, num_clusters, include_self=False)
    agc_model = AgglomerativeClustering(linkage='ward',
                                        connectivity=knn_graph,
                                        n_clusters=num_clusters)
    agc_model.fit(tsne_for_clust)

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
                      random_state=seed,
                      init=pca_for_tsne[:,0:num_tsne_components]).fit_transform(pca_for_tsne)
    np.save(save_path + f'tsne_{aux_name1}_{aux_name2}.npy', data_tsneed)

def nmf_then_save(data, num_factors, save_path, aux_name1, aux_name2, max_iter=5000, tol=1e-5):
    min_per_dim = np.min(data, axis=0)
    data_nonneg = data - min_per_dim
    model = NMF(n_components=num_factors,
                init='random', random_state=0, max_iter=max_iter, verbose=1, tol=tol)
    vecs_nmf = model.fit(data_nonneg)
    np.save(save_path + f'nmf_min_per_dim_{aux_name1}_{aux_name2}.npy',
            min_per_dim)
    np.save(save_path + f'nmf_{aux_name1}_{aux_name2}.npy',
            vecs_nmf.transform(data_nonneg))
    np.save(save_path + f'nmf_components_{aux_name1}_{aux_name2}.npy',
            vecs_nmf.components_)
    model_name = save_path + f'nmf_model_{aux_name1}_{aux_name2}.joblib'
    dump(model, model_name)

def nmf_crossvalidation(data, save_path, aux_name1, aux_name2):
    # X' = WH (X' is NxD; W is NxQ; H is QxD)
    min_per_dim = np.min(data, axis=0)
    data_nonneg = data - min_per_dim

    # Split data into train and validation
    len_data = data_nonneg.shape[0]
    len_test = int(0.15*len_data)
    data_test = data_nonneg[:len_test]
    data_train = data_nonneg[len_test:]

    # Set up configs for cross validation
    num_component_min = 10
    num_component_max = 61
    num_component_step = 10
    num_component_range = np.arange(num_component_min, num_component_max, num_component_step)
    nmf_models = []
    nmf_train_residuals = []
    nmf_test_residuals = []
    num_repeats = 4
    max_iter = 5000

    # Begin cross validation
    for num_comp in num_component_range:
        print("Beginning NMF for %i factors" % num_comp)
        model_repeats = []
        train_residual_repeats = []
        test_residual_repeats = []
        for repeat_id in range(num_repeats):
            model = NMF(n_components=num_comp,
                        init='random', random_state=repeat_id,
                        max_iter=max_iter, verbose=1,
                        tol=1e-4)

            W_train = model.fit_transform(data_train)
            data_train_hat = model.inverse_transform(W_train)

            W_test = model.transform(data_test)
            data_test_hat = model.inverse_transform(W_test)

            train_residual = np.linalg.norm(data_train - data_train_hat, ord='fro')
            test_residual = np.linalg.norm(data_test - data_test_hat, ord='fro')

            model_repeats.append(model)
            train_residual_repeats.append(train_residual)
            test_residual_repeats.append(test_residual)

        nmf_models.append(model_repeats)
        nmf_train_residuals.append(train_residual_repeats)
        nmf_test_residuals.append(test_residual_repeats)

    nmf_train_residuals = [np.array(v).mean() for v in nmf_train_residuals]
    nmf_test_residuals = [np.array(v).mean() for v in nmf_test_residuals]
    print(nmf_test_residuals)

    # get best model and save
    min_value = min(nmf_test_residuals)
    min_index = nmf_test_residuals.index(min_value)
    best_model = nmf_models[min_index][0] # may not be exactly optimal but we at least get a fixed random state==0
    np.save(save_path + f'nmf_xv_min_per_dim_{aux_name1}_{aux_name2}.npy',
            min_per_dim)
    np.save(save_path + f'nmf_xv_{aux_name1}_{aux_name2}.npy',
            best_model.fit_transform(data_nonneg))
    np.save(save_path + f'nmf_xv_components_{aux_name1}_{aux_name2}.npy',
            best_model.components_)
    model_name = save_path + f'nmf_xv_model_{aux_name1}_{aux_name2}.joblib'
    dump(best_model, model_name)
    # load(model_name)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Crossvalidation NMF')

    ax1.plot(num_component_range, nmf_train_residuals, '.-')
    ax1.set_ylabel('Train residual')

    ax2.plot(num_component_range, nmf_test_residuals, '.-')
    ax2.set_xlabel('Number of factors')
    ax2.set_ylabel('Test residual')
    plot_name = os.path.join(os.getcwd(),'analysis','hx_plots', "Crossvalidation NMF.png")
    plt.savefig(plot_name)



def ica_then_save(whitened_data, save_path, aux_name1, aux_name2,
                  max_iter=5000, tol=1e-4):

    model = FastICA(whiten=False, random_state=0, max_iter=max_iter, tol=tol)
    data_ica = model.fit_transform(whitened_data)
    np.save(save_path + f'ica_unmixing_matrix_{aux_name1}_{aux_name2}.npy',
            model.components_)
    np.save(save_path + f'ica_mixing_matrix_{aux_name1}_{aux_name2}.npy',
            model.mixing_)
    np.save(save_path + f'ica_source_signals_{aux_name1}_{aux_name2}.npy',
            data_ica)