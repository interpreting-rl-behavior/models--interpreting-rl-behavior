import pandas as pd
import numpy as np
from precomput_analysis_funcs import scale_then_pca_then_save, plot_variance_expl_plot, clustering_after_pca, \
    tsne_after_pca, clustering_after_tsne
import argparse
import os
import hyperparam_functions as hpf
from dimred_projector import HiddenStateDimensionalityReducer

pd.options.mode.chained_assignment = None  # default='warn'


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

    num_samples = hp.analysis.jacobian.num_samples  # number of generated samples to use
    n_components_pca = hp.analysis.jacobian.n_components_pca
    n_components_tsne = hp.analysis.jacobian.n_components_tsne
    n_clusters = hp.analysis.jacobian.n_clusters
    n_ic_directions = hp.analysis.agent_h.n_components_ica
    n_hx_dims = hp.gen_model.agent_hidden_size
    n_env_dims = hp.gen_model.deter_dim

    transformer = HiddenStateDimensionalityReducer(hp, 'ica', num_samples, data_type=np.ndarray)

    # Prepare load and save dirs
    generated_data_path = os.path.join(hp.generated_data_dir, 'informed_init')
    save_path = 'jacob_analysis_precomp/'
    save_path = os.path.join(os.getcwd(), "analysis", save_path)
    plot_save_path = 'jacob_plots'
    plot_save_path = os.path.join(os.getcwd(), "analysis", plot_save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)

    # Get hidden states the were produced by the generative model
    print("Collecting jacobian data together...")
    jacobians = []
    for sample_id in range(num_samples):
        print(sample_id)
        jacobian = collect_sample_jacobian(hp, generated_data_path, transformer, sample_id)
        jacobians.append(jacobian)
    jacobians = np.stack(jacobians)
    jacobians = jacobians.reshape(num_samples, (n_hx_dims + n_env_dims) * n_ic_directions)

    # PCA
    print('Starting PCA...')
    jacob_vecs_pcaed, jacob_vecs_scaled, pca_obj, scaler = \
        scale_then_pca_then_save(jacobians, n_components_pca, save_path, "jacob",
                                 num_samples)
    above95explained = \
        plot_variance_expl_plot(pca_obj, n_components_pca, plot_save_path,
                                "jacob", num_samples)
    print("above95explained=%i" % int(above95explained))
    print('PCA finished.')

    # clustering
    print('Starting clustering...')
    clustering_after_tsne(jacobians, max(above95explained, 64), 3, n_clusters, save_path,
                         "jacob", num_samples)
    print("Clustering finished.")

    # # tSNE
    print('Starting tSNE...')
    tsne_after_pca(jacobians, above95explained, n_components_tsne,
                   save_path, "jacob", num_samples)
    print("tSNE finished.")


def collect_sample_jacobian(hp, generated_data_path, transformer, sample_id):
    n_ic_directions = hp.analysis.agent_h.n_components_ica
    ts_id = hp.analysis.jacobian.grad_timestep
    jacob_vecs = []
    for direction_id in range(n_ic_directions):  # D
        hx_grad_vecs = np.load(os.path.join(generated_data_path,
                           f'sample_{sample_id:05d}/grad_hx_hx_direction_{direction_id}_ica.npy')) # (T x H)
        hx_grad_vec = hx_grad_vecs[ts_id]  # (H,)
        # hx_grad_vec = transformer.project_gradients(hx_grad_vec)  # (I,)

        env_grad_vecs = np.load(os.path.join(generated_data_path,
                           f'sample_{sample_id:05d}/grad_env_h_hx_direction_{direction_id}_ica.npy'))
        env_grad_vec = env_grad_vecs[ts_id]
        grad_vec = np.concatenate([hx_grad_vec, env_grad_vec])
        jacob_vecs.append(grad_vec)
    jacobian = np.stack(jacob_vecs, axis=-1)  # (I, D)
    return jacobian


if __name__ == "__main__":
    run()
