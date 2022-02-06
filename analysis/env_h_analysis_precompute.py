import pandas as pd
import numpy as np
from precomput_analysis_funcs import scale_then_pca_then_save, plot_variance_expl_plot, clustering_after_pca, tsne_after_pca, nmf_then_save
import argparse
import os
import hyperparam_functions as hpf



pd.options.mode.chained_assignment = None  # default='warn'

COINRUN_ACTIONS = {0: 'downleft', 1: 'left', 2: 'upleft', 3: 'down', 4: None, 5: 'up',
                   6: 'downright', 7: 'right', 8: 'upright', 9: None, 10: None, 11: None,
                   12: None, 13: None, 14: None}
def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--interpreting_params_name', type=str,
        default='defaults')
    args = parser.parse_args()
    return args


def run():
    args = parse_args()
    hp = hpf.load_interp_configs(args.interpreting_params_name)

    num_samples = hp.analysis.env_h.num_samples # number of generated samples to use
    num_epi_paths = hp.analysis.env_h.num_epi_paths  # Number of episode to plot paths through time for. Arrow plots.
    n_components_pca = hp.analysis.env_h.n_components_pca
    n_components_tsne = hp.analysis.env_h.n_components_tsne
    n_components_nmf = hp.analysis.env_h.n_components_nmf
    n_clusters = hp.analysis.env_h.n_clusters

    # Prepare load and save dirs
    generated_data_path = os.path.join(hp.generated_data_dir, 'informed_init')
    save_path = 'env_analysis_precomp/'
    save_path = os.path.join(os.getcwd(), "analysis", save_path)
    plot_save_path = 'env_plots'
    plot_save_path = os.path.join(os.getcwd(), "analysis", plot_save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)

    # Get hidden states the were produced by the generative model
    print("Collecting env data together...")
    env_h = np.load(os.path.join(generated_data_path, 'sample_00000/env_hs.npy'))
    for ep in range(1, num_samples):
        env_h_to_cat = np.load(os.path.join(generated_data_path,
                                     f'sample_{ep:05d}/env_hs.npy'))
        env_h = np.concatenate((env_h, env_h_to_cat))

    # only hidden states, no cell state or global context vec:
    # print("Collecting env data together...")
    # env_vecs = np.load(os.path.join(generated_data_path, 'sample_00000/env_cell_states.npy'))
    # for ep in range(1, num_samples):
    #     env_vecs_to_cat = np.load(os.path.join(generated_data_path,
    #                                  f'sample_{ep:05d}/env_cell_states.npy'))
    #     env_vecs = np.concatenate((env_vecs, env_vecs_to_cat))

    # PCA
    print('Starting PCA...')
    env_vecs_pcaed, env_vecs_scaled, pca_obj, scaler = \
        scale_then_pca_then_save(env_h, n_components_pca, save_path, "env",
                                 num_samples)
    above95explained = \
        plot_variance_expl_plot(pca_obj, n_components_pca, plot_save_path,
                                "env", num_samples)
    print("above95explained=%i" % int(above95explained))
    print('PCA finished.')

    # k-means clustering
    print('Starting clustering...')
    clustering_after_pca(env_h, above95explained, n_clusters, save_path,
                         "env", num_samples)
    print("Clustering finished.")
    
    # # tSNE
    print('Starting tSNE...')
    tsne_after_pca(env_h, above95explained, n_components_tsne,
                   save_path, "env", num_samples)
    print("tSNE finished.")

    print('Starting UMAP...')
    pca_for_umap = pca_for_tsne
    reducer = umap.UMAP()
    env_h_umap = reducer.fit_transform(pca_for_umap)
    np.save(save_path + 'env_h_umap_%i.npy' % num_samples, env_h_umap)
    print("tSNE finished.")

    print('Starting NMF...')
    nmf_then_save(env_h, n_components_nmf, save_path, "env", num_samples,
                  max_iter=hp.analysis.env_h.nmf_max_iter,
                  tol=hp.analysis.env_h.nmf_tol)
    print("NMF finished.")

if __name__ == "__main__":
    run()
