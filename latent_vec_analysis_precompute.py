"""To run this script, make sure you've already run record.py and
generative/record_informinit_gen_samples.py and
generative/record_random_gen_samples.py
because they generate data that is used
here. """
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

def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--agent_env_data_dir', type=str,
        default="data/")
    parser.add_argument(
        '--generated_data_dir_inf', type=str,
        default='generative/recorded_informinit_gen_samples')
    parser.add_argument(
        '--generated_data_dir_rand', type=str,
        default='generative/recorded_random_gen_samples')

    args = parser.parse_args()
    return args


# EPISODE_STRINGS = {v:str(v) for v in range(3431)}
def run():
    args = parse_args()
    num_generated_samples = 4000 # number of generated samples to use
    n_components_pca = 128
    n_components_tsne = 2
    n_clusters = 100

    seed = 42  # for the tSNE algo

    # Prepare load and save dirs
    main_data_path = args.agent_env_data_dir
    generated_data_path_inf = args.generated_data_dir_inf
    generated_data_path_rand = args.generated_data_dir_rand

    save_path = 'analysis/latent_vec_analysis_precomp/'
    os.makedirs(save_path, exist_ok=True)

    plot_save_path = 'analysis/latent_vec_plots'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path, exist_ok=True)

    aux_var_names = ['sample_reward_sum',
                     'sample_has_done', 'sample_avg_value', 'lv_cluster']
    aux_data = pd.DataFrame(columns=aux_var_names)

    # Get latent vectors the were produced by the generative model using
    #  informed initialization and also construct dataframe of auxiliary
    #  information to understand the LV space

    lv_inf = np.load(os.path.join(generated_data_path_inf, 'sample_00000/latent_vec.npy'))
    lv_inf = np.expand_dims(lv_inf, axis=0)

    aux_data_rew = np.load(os.path.join(generated_data_path_inf,
                                         'sample_00000/rews.npy')).sum()
    aux_data_done = np.load(os.path.join(generated_data_path_inf,
                                         'sample_00000/dones.npy')).sum()
    aux_data_val = np.load(os.path.join(generated_data_path_inf,
                                        'sample_00000/agent_values.npy')).mean()
    aux_data.at[0, 'sample_reward_sum'] = aux_data_rew
    aux_data.at[0, 'sample_has_done'] = aux_data_done
    aux_data.at[0, 'sample_avg_value'] = aux_data_val

    print("Getting informed init latent vectors and auxiliary data...")
    for ep in range(1, num_generated_samples):
        # Get and concat latent vecs
        lv_to_cat = np.load(os.path.join(generated_data_path_inf,
                                         f'sample_{ep:05d}/latent_vec.npy'))
        lv_to_cat = np.expand_dims(lv_to_cat, axis=0)
        lv_inf = np.concatenate((lv_inf, lv_to_cat))

        # Load then add aux data to df
        aux_data_rew = np.load(os.path.join(generated_data_path_inf,
                                            'sample_00000/rews.npy')).max()
        aux_data_done = np.load(os.path.join(generated_data_path_inf,
                                             'sample_00000/dones.npy')).max()
        aux_data_val = np.load(os.path.join(generated_data_path_inf,
                                            'sample_00000/agent_values.npy'))[-1]
        aux_data.at[ep, 'sample_reward_sum'] = aux_data_rew
        aux_data.at[ep, 'sample_has_done'] = aux_data_done
        aux_data.at[ep, 'sample_avg_value'] = aux_data_val

    # Now same for random latent vecs, but no auxiliary data
    print("Getting random latent vectors...")
    lv_rand = np.load(os.path.join(generated_data_path_rand, 'sample_00000/latent_vec.npy'))
    lv_rand = np.expand_dims(lv_rand, axis=0)
    for ep in range(1, num_generated_samples):
        # Get and concat latent vecs
        lv_to_cat = np.load(os.path.join(generated_data_path_rand,
                                         f'sample_{ep:05d}/latent_vec.npy'))
        lv_to_cat = np.expand_dims(lv_to_cat, axis=0)

        lv_rand = np.concatenate((lv_rand, lv_to_cat))

    # PCA informative init vecs
    print('Starting PCA...')
    lv_inf_prescaling = lv_inf
    scaler_inf = StandardScaler()
    scaler_inf = scaler_inf.fit(lv_inf_prescaling)
    lv_inf = scaler_inf.transform(lv_inf_prescaling)
    mean_lv_inf = scaler_inf.mean_
    var_lv_inf = scaler_inf.var_
    pca_inf = PCA(n_components=n_components_pca)
    lv_inf_pca = pca_inf.fit_transform(lv_inf)
    print('PCA for latent vecs that use informative init finished.')

    # PCA random vecs
    lv_rand_prescaling = lv_rand
    scaler_rand = StandardScaler()
    scaler_rand = scaler_rand.fit(lv_rand_prescaling)
    lv_rand = scaler_rand.transform(lv_rand_prescaling)
    lv_rand_pca_projected = pca_inf.transform(lv_rand)

    # Save PCs and the projections of each of the hx onto those PCs.
    np.save(save_path + 'lv_inf_pca_%i.npy' % num_generated_samples, lv_inf_pca)
    np.save(save_path + 'pcomponents_lv_inf_%i.npy' % num_generated_samples,
            scaler_inf.inverse_transform(pca_inf.components_))
    # And save the projections of the generated data onto the PCs of true data
    np.save(save_path + 'lv_rand_pca_projected_%i.npy' % (num_generated_samples),
            lv_rand_pca_projected)

    # Clustering on informative init vecs
    print("Starting clustering")
    knn_graph = kneighbors_graph(lv_inf, n_clusters, include_self=False)
    agc_model = AgglomerativeClustering(linkage='ward',
                                    connectivity=knn_graph,
                                    n_clusters=n_clusters)
    agc_model.fit(lv_inf)
    clusters = agc_model.labels_
    aux_data['lv_cluster'] = clusters

    cluster_means = []
    for cluster_id in list(set(clusters)):
        cluster_mask = clusters == cluster_id
        cluster_eles = lv_inf[cluster_mask]
        cluster_mean = cluster_eles.mean(axis=0)
        cluster_means.append(cluster_mean)
    cluster_means = np.array(cluster_means)
    np.save(save_path + 'cluster_means_%i.npy' % num_generated_samples, cluster_means)
    print("Clustering finished.")


    # And save the auxiliary dataframe
    aux_data.to_csv(save_path + 'lv_inf_aux_data_%i.csv' % num_generated_samples)

    # Plot variance explained plot
    pca_percent = 100 * pca_inf.explained_variance_/sum(pca_inf.explained_variance_)
    above95explained = np.argmax(pca_percent.cumsum() > 95)
    plt.bar(list(range(n_components_pca)),
            pca_percent,
            color='blue')
    plt.xlabel("Principle Component")
    plt.ylabel("Variance Explained (%)")
    plt.savefig(plot_save_path + "/pca_var_expl_for_infinit_latent_vecs__epis%i.png" % num_generated_samples)

    # tSNE
    print('Starting tSNE...')
    pca_for_tsne = PCA(n_components=above95explained)
    pca_for_tsne = pca_for_tsne.fit_transform(lv_inf)
    lv_inf_tsne = TSNE(n_components=n_components_tsne, random_state=seed).fit_transform(pca_for_tsne)
    np.save(save_path + 'lv_inf_tsne_%i.npy' % num_generated_samples, lv_inf_tsne)
    print("tSNE finished.")


if __name__ == "__main__":
    run()