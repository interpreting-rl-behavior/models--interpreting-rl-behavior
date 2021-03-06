import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from precomput_analysis_funcs import \
    plot_variance_expl_plot, plot_cum_variance_expl_plot,\
    clustering_after_pca, tsne_after_pca, \
    nmf_then_save, nmf_crossvalidation, ica_then_save, \
    identify_outliers
import argparse
import os
from sklearn.decomposition import PCA, NMF, FastICA
import hyperparam_functions as hpf
from dimred_projector import HiddenStateDimensionalityReducer

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

    num_episodes = hp.analysis.agent_h.num_episodes  # number of episodes to make plots for
    num_generated_samples = hp.analysis.agent_h.num_generated_samples # number of generated samples to use
    num_epi_paths = hp.analysis.agent_h.num_epi_paths  # Number of episode to plot paths through time for. Arrow plots.
    n_components_pca = hp.analysis.agent_h.n_components_pca
    n_components_tsne = hp.analysis.agent_h.n_components_tsne
    n_components_ica = hp.analysis.agent_h.n_components_ica
    n_components_nmf = hp.analysis.agent_h.n_components_nmf
    n_clusters = hp.analysis.agent_h.n_clusters#40##100
    path_epis = list(range(num_epi_paths))

    # Prepare load and save dirs
    main_data_path = hp.data_dir
    generated_data_path = hp.generated_data_dir + hp.analysis.agent_h.informed_or_random_init
    save_path = 'hx_analysis_precomp/'
    save_path = os.path.join(os.getcwd(), "analysis", save_path)
    plot_save_path = 'hx_plots'
    plot_save_path = os.path.join(os.getcwd(), "analysis", plot_save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)

    # Get (true) hidden states that were generated by the agent interacting
    # with the real environment
    hx = np.load(os.path.join(main_data_path, 'episode_00000/hx.npy'))
    hx = hx[1:] # Cut 0th timestep
    for ep in range(1, num_episodes):
        print(ep)
        hx_to_cat = np.load(os.path.join(main_data_path,
                                         f'episode_{ep:05d}/hx.npy'))
        hx_to_cat = hx_to_cat[1:]  # Cut 0th timestep
        hx = np.concatenate((hx, hx_to_cat))

    # Get hidden states the were produced by the generative model
    gen_hx = np.load(os.path.join(generated_data_path, 'sample_00000/agent_hs.npy'))
    for ep in range(1, num_generated_samples):
        gen_hx_to_cat = np.load(os.path.join(generated_data_path,
                                         f'sample_{ep:05d}/agent_hs.npy'))
        gen_hx = np.concatenate((gen_hx, gen_hx_to_cat))

    # Preprocessing
    uncentred_unscaled_hx = hx
    uncentred_unscaled_gen_hx = gen_hx

    mu = np.mean(uncentred_unscaled_hx, axis=0)
    std = np.std(uncentred_unscaled_hx, axis=0)

    centred_scaled_hx = (uncentred_unscaled_hx - mu ) / std
    centred_scaled_gen_hx = (uncentred_unscaled_gen_hx - mu) / std

    # Outlier removal:
    # - Get pairwise distances between each point and its KNN.
    # - Calculate the median distances for each point. (because
    #   median will only get those that are far away from LOTS of points)
    # - Identify outliers using distances that have a median distance
    #   away from other points that is 2 sigma larger than the
    #   average such distance
    print("Starting outlier removal")
    outlier_mask = identify_outliers(centred_scaled_hx, save_path, max_k=hp.analysis.agent_h.outlier_max_k)
    outlier_fraction = outlier_mask.sum()/outlier_mask.shape[0]
    outlier_fraction_file_name = os.path.join(save_path, "hx_outlier_retention_fraction.npy")
    np.save(outlier_fraction_file_name, outlier_fraction)
    print(f"Outlier retention percentage is {outlier_fraction}")
    outlier_mask_file_name = os.path.join(save_path, f"hx_outlier_mask_{num_episodes}.npy")
    np.save(outlier_mask_file_name, outlier_mask)
    centred_scaled_hx_wo_outliers = centred_scaled_hx[outlier_mask]

    # PCA
    print('Starting PCA...')
    pca_obj = PCA(n_components=n_components_pca)
    pca_obj.fit(centred_scaled_hx_wo_outliers) # fit pca data without outliers
    centred_scaled_hx_pca = pca_obj.transform(centred_scaled_hx) # but still keep the outlier data for completeness and compatbility
    centred_scaled_gen_hx_projected = pca_obj.transform(centred_scaled_gen_hx)
    pc_variances = centred_scaled_hx_pca.var(axis=0)

    # For future reference the following returns an array of mostly True:
    # np.isclose(pca_obj.transform(centred_scaled_hx),
    #            centred_scaled_hx @ (pca_obj.components_.transpose()),
    #            atol=1e-3)

    print('PCA finished.')

    # Save PCs and the projections of each of the hx onto those PCs.
    np.save(save_path + 'hx_mu_%i.npy' % num_episodes, mu)
    np.save(save_path + 'hx_std_%i.npy' % num_episodes, std)
    np.save(save_path + 'hx_pca_%i.npy' % num_episodes, centred_scaled_hx_pca)
    np.save(save_path + 'pcomponents_%i.npy' % num_episodes, pca_obj.components_)
    np.save(save_path + 'pc_loading_variances_%i.npy' % num_episodes,
            pc_variances)
    np.save(save_path + 'pc_singular_values_%i.npy' % num_episodes,
            pca_obj.singular_values_)
    # And save the projections of the generated data onto the PCs of true data
    np.save(save_path + 'gen_hx_projected_real%i_gen%i.npy' % (num_episodes,
                                                        num_generated_samples),
           centred_scaled_gen_hx_projected)

    # Plot variance explained plot
    above95explained = \
       plot_variance_expl_plot(pca_obj, n_components_pca, plot_save_path,
                               "hx", num_episodes)
    above95explained = \
       plot_cum_variance_expl_plot(pca_obj, n_components_pca, plot_save_path,
                               "hx", num_episodes)

    # ICA
    print("Starting ICA")

    # # Dev only
    # from scipy import linalg
    # n_components = n_components_ica
    # n_samples = num_episodes
    # XT = uncentred_unscaled_hx.T
    # X_mean = XT.mean(axis=-1)
    # XT -= X_mean[:, np.newaxis]
    #
    # # Whitening and preprocessing by PCA
    # u, d, _ = linalg.svd(XT, full_matrices=False, check_finite=False) # u is the
    #
    # K = (u / d).T[:n_components]  # see (6.33) p.140
    # X1 = np.dot(K, XT)
    # # see (13.6) p.267 Here X1 is white and data
    # # in X has been projected onto a subspace by PCA
    # X1 *= np.sqrt(n_samples)
    #
    #
    # max_iter = hp.analysis.agent_h.ica_max_iter
    # tol = hp.analysis.agent_h.ica_tol
    # aux_name1, aux_name2 = "hx", num_episodes
    # model1 = FastICA(whiten=False, n_components=n_components_ica, random_state=0, max_iter=max_iter, tol=tol)
    # data_ica_whit = model1.fit_transform(whitened_data)
    # np.save(save_path + f'ica_unmixing_matrix_{aux_name1}_{aux_name2}.npy',
    #         model1.components_)
    # np.save(save_path + f'ica_mixing_matrix_{aux_name1}_{aux_name2}.npy',
    #         model1.mixing_)
    # np.save(save_path + f'ica_source_signals_{aux_name1}_{aux_name2}.npy',
    #         data_ica_whit)
    # directions_transformer = \
    #     HiddenStateDimensionalityReducer(hp,'ica',num_episodes,np.ndarray)
    # ica_recon = directions_transformer.transform(uncentred_unscaled_hx)
    #
    # model2 = FastICA(whiten=False, random_state=0, max_iter=max_iter, tol=tol)
    # data_ica_nowhit = model2.fit_transform(whitened_data)

    # End dev

    ica_then_save(centred_scaled_hx_pca, save_path,
                  "hx", num_episodes,
                  max_iter=hp.analysis.agent_h.ica_max_iter,
                  tol=hp.analysis.agent_h.ica_tol,
                  n_components_ica=hp.analysis.agent_h.n_components_ica,
                  outlier_mask=outlier_mask)
    # BEWARE: if your tol is too high, then this just won't converge even though your
    # input data may be perfectly fine. For instance, for me 0.02 converged but 0.01 didn't.
    print("ICA finished")

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

    # # NMF
    print('Starting NMF...')
    nmf_then_save(hx, n_components_nmf, save_path, "hx", num_episodes,
                 max_iter=hp.analysis.agent_h.nmf_max_iter,
                 tol=hp.analysis.agent_h.nmf_tol)
    # nmf_crossvalidation(hx, save_path, "hx", num_episodes)
    print("NMF finished.")




if __name__ == "__main__":
    run()
