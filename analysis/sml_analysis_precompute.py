import pandas as pd
import numpy as np
from precomput_analysis_funcs import scale_then_pca_then_save, plot_variance_expl_plot, clustering_after_pca, tsne_after_pca, nmf_then_save
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
    num_samples = 20000 # number of generated samples to use
    num_epi_paths = 9  # Number of episode to plot paths through time for. Arrow plots.
    n_components_pca = 215
    n_components_tsne = 2
    n_components_nmf = 32
    n_clusters = 200
    path_epis = list(range(num_epi_paths))

    seed = 42  # for the tSNE algo

    # Prepare load and save dirs
    generated_data_path = args.generated_data_dir
    save_path = 'sml_analysis_precomp/'
    save_path = os.path.join(os.getcwd(), "analysis", save_path)
    plot_save_path = 'sml_plots'
    plot_save_path = os.path.join(os.getcwd(), "analysis", plot_save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)

    # env_space_w = 64./(2048.+64.)
    # agent_hx_space_w = 64./64.
    # action_space_w = 64./15.

    # Get vecs that were produced by the generative model
    print("Collecting SML data together...")
    env_hx = np.load(os.path.join(generated_data_path, 'sample_00000/env_hid_states.npy'))
    env_c = np.load(os.path.join(generated_data_path, 'sample_00000/env_cell_states.npy'))
    agnt_hx = np.load(os.path.join(generated_data_path, 'sample_00000/agent_hxs.npy'))
    agnt_lp = np.load(os.path.join(generated_data_path, 'sample_00000/agent_logprobs.npy'))
    z_g = np.load(os.path.join(generated_data_path, 'sample_00000/bottleneck_vec.npy'))
    z_g = [z_g[z_g.shape[0]//2:]] * env_c.shape[0]
    z_g = np.stack(z_g)
    sml_vecs = np.concatenate((env_hx,
                               env_c,
                               agnt_hx,
                               agnt_lp,
                               z_g), axis=1)

    sml_tplus1 = sml_vecs[1:]  # there is no diff vec for the last ts
    sml_t = sml_vecs[:-1]  # therefore we cut off the last ts for consistency
    sml_difference_vecs = sml_tplus1 - sml_t
    sml_vecs = sml_t

    for ep in range(1, num_samples):
        env_hx = np.load(os.path.join(generated_data_path,
                                     f'sample_{ep:05d}/env_hid_states.npy'))
        env_c = np.load(os.path.join(generated_data_path,
                                     f'sample_{ep:05d}/env_cell_states.npy'))
        agnt_hx = np.load(os.path.join(generated_data_path,
                                     f'sample_{ep:05d}/agent_hxs.npy'))
        agnt_lp = np.load(os.path.join(generated_data_path,
                                     f'sample_{ep:05d}/agent_logprobs.npy'))
        z_g = np.load(os.path.join(generated_data_path,
                                   f'sample_{ep:05d}/bottleneck_vec.npy'))
        z_g = [z_g[z_g.shape[0] // 2:]] * env_c.shape[0]
        z_g = np.stack(z_g)
        sml_vec_to_cat = np.concatenate((env_hx,
                                        env_c,
                                        agnt_hx,
                                        agnt_lp,
                                        z_g), axis=1)

        sml_vec_to_cat_tplus1 = sml_vec_to_cat[1:]  # there is no diff vec for the last ts
        sml_vec_to_cat_t = sml_vec_to_cat[:-1] # therefore we cut off the last ts for consistency
        sml_difference_vecs_to_cat = sml_vec_to_cat_tplus1 - sml_vec_to_cat_t
        sml_vec_to_cat = sml_vec_to_cat_t

        sml_vecs = np.concatenate((sml_vecs, sml_vec_to_cat))
        sml_difference_vecs = np.concatenate((sml_difference_vecs, sml_difference_vecs_to_cat))

    sml_vecs_dyn = np.concatenate((sml_vecs, sml_difference_vecs), axis=1)
    # # only hidden states, no cell state:
    # print("Collecting env data together...")
    # env_h = np.load(os.path.join(generated_data_path, 'sample_00000/env_cell_states.npy'))
    # for ep in range(1, num_samples):
    #     env_h_to_cat = np.load(os.path.join(generated_data_path,
    #                                  f'sample_{ep:05d}/env_cell_states.npy'))
    #     env_h = np.concatenate((env_h, env_h_to_cat))

    # PCA
    print('Starting PCA on raw sml data...')
    sml_vecs_raw_pcaed, sml_vecs_raw_scaled, pca_obj_raw, scaler_raw = \
        scale_then_pca_then_save(sml_vecs, n_components_pca, save_path, "sml_raw",
                                 num_samples)
    print('PCA on raw sml data finished.')

    print('Starting PCA on sml+dyn data...')
    sml_vecs_dyn_pcaed, sml_vecs_dyn_scaled, pca_obj_dyn, scaler_dyn = \
        scale_then_pca_then_save(sml_vecs_dyn, n_components_pca, save_path, "sml_dyn",
                                 num_samples)
    print('PCA on sml+dyn data finished.')

    print("Saving variance exlained plots")
    above95explained_raw = \
        plot_variance_expl_plot(pca_obj_raw, n_components_pca, plot_save_path,
                                "sml_raw", num_samples)

    above95explained_dyn = \
        plot_variance_expl_plot(pca_obj_dyn, n_components_pca, plot_save_path,
                                "sml_dyn", num_samples)

    # k-means clustering
    print('Starting clustering of raw SML vectors...')
    clustering_after_pca(sml_vecs, above95explained_raw, n_clusters, save_path,
                     "sml_raw", num_samples)
    print("Clustering of raw sml space finished.")

    print('Starting clustering of SML vectors + dynamics...')
    clustering_after_pca(sml_vecs_dyn, above95explained_dyn, n_clusters, save_path,
                     "sml_dyn", num_samples)
    print("Clustering finished.")

    # tSNE
    print('Starting tSNE...')
    tsne_after_pca(sml_vecs, above95explained_raw, n_components_tsne,
                    save_path, "sml_raw", num_samples)
    print("tSNE finished.")

    # print('Starting UMAP...')
    # pca_for_umap = pca_for_tsne
    # reducer = umap.UMAP()
    # env_h_umap = reducer.fit_transform(pca_for_umap)
    # np.save(save_path + 'env_h_umap_%i.npy' % num_samples, env_h_umap)
    # print("tSNE finished.")

    print('Starting NMF...')
    nmf_then_save(sml_vecs, n_components_nmf, save_path, "sml_raw", num_samples)
    print("NMF finished.")




if __name__ == "__main__":
    run()
