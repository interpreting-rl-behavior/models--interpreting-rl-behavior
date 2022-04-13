import pandas as pd
import numpy as np
from analysis.precomput_analysis_funcs import (
    scale_then_pca_then_save,
    plot_variance_expl_plot,
    clustering_after_pca,
    tsne_after_pca,
    nmf_then_save,
)
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn import datasets
import argparse
import os
import matplotlib.pyplot as plt
import hyperparam_functions as hpf
from dimred_projector import HiddenStateDimensionalityReducer

pd.options.mode.chained_assignment = None  # default='warn'


class xcorr_manager:
    def __init__(self):
        args = self.parse_args()
        self.hp = hpf.load_interp_configs(args.interpreting_params_name)

        # TODO change these hard coded hyperparams to use the config file (self.hp)
        self.num_samples_hx = (
            self.hp.analysis.agent_h.num_episodes
        )  # number of generated samples to use
        self.num_samples_hx_grad = 4000
        self.num_samples_env_h = 6000
        n_components_hx = 64
        n_components_env_h = 64
        self.ts_per_sample = 10
        self.direction_type = "ica"
        self.num_ica_components = self.hp.analysis.agent_h.n_components_ica
        self.hx_projector = HiddenStateDimensionalityReducer(
            self.hp, self.direction_type, self.num_samples_hx, data_type=np.ndarray
        )
        self.grad_projector = self.hx_projector.project_gradients

        hx_precomp_data_path = os.path.join("analysis", "hx_analysis_precomp")
        self.env_h_precomp_path = os.path.join("analysis", "env_analysis_precomp")
        env_components = np.load(
            f"{self.env_h_precomp_path}/pcomponents_env_{self.num_samples_env_h}.npy"
        )
        self.hx_pc_components = np.load(
            f"{hx_precomp_data_path}/pcomponents_{self.num_samples_hx}.npy"
        )
        self.hx_pc_variances = np.load(
            f"{hx_precomp_data_path}/pc_loading_variances_{self.num_samples_hx}.npy"
        )  # TODO put the extra
        self.hx_pc_components = self.hx_pc_components.transpose()
        self.hx_mu = np.load(hx_precomp_data_path + f"/hx_mu_{self.num_samples_hx}.npy")
        self.hx_std = np.load(
            hx_precomp_data_path + f"/hx_std_{self.num_samples_hx}.npy"
        )
        # self.grad_projector = self.project_gradients_into_pc_space

        if self.direction_type == "ica":
            ica_directions_path = os.path.join(
                hx_precomp_data_path,
                f"ica_unmixing_matrix_hx_{self.num_samples_hx}.npy",
            )
            self.unmix_mat = np.load(ica_directions_path)
            self.unmix_mat = self.unmix_mat.transpose()

            self.mix_mat = np.load(
                f"{hx_precomp_data_path}/ica_mixing_matrix_hx_{self.num_samples_hx}.npy"
            )
        #     # self.grad_projector = self.project_gradients_into_ica_space

        # Prepare load and save dirs
        self.generated_data_path = args.generated_data_dir
        self.save_path = "xcorr_matrices/"
        self.save_path = os.path.join(os.getcwd(), "analysis", self.save_path)
        os.makedirs(self.save_path, exist_ok=True)

    # def project_gradients_into_pc_space(self, grad_data):
    #     sigma = np.diag(self.hx_std)
    #     grad_data = grad_data.T  # So each column is a grad vector for a hx
    #     scaled_pc_comps = self.hx_pc_components.T @ sigma  # PCs calculated on X'=(X-mu)/sigma are scaled so it's like they were calculated on X
    #     projected_grads = scaled_pc_comps @ grad_data  # grads are projected onto the scaled PCs
    #     return projected_grads.T
    #
    # def project_gradients_into_ica_space(self, grad_data): # TODO fix
    #     sigma = np.diag(self.hx_std)
    #     grad_data = grad_data.T  # So each column is a grad vector for a hx
    #     scaled_pc_comps = self.hx_pc_components.T @ sigma  # PCs calculated on X'=(X-mu)/sigma are scaled so it's like they were calculated on X
    #     projected_grads_to_pc_space = scaled_pc_comps @ grad_data  # grads are projected onto the scaled PCs
    #     projected_grads_to_pc_space = projected_grads_to_pc_space[:self.num_ica_components, :]
    #     projected_grads_to_ic_space = projected_grads_to_pc_space.T @ self.mix_mat
    #     return projected_grads_to_ic_space

    def parse_args(self):
        parser = argparse.ArgumentParser(description="args for plotting")
        parser.add_argument("--agent_env_data_dir", type=str, default="data/")
        parser.add_argument(
            "--generated_data_dir",
            type=str,
            default="generative/rec_gen_mod_data/informed_init",
        )
        # TODO change the above args to just use the config file/interpreting params (self.hp)
        parser.add_argument("--interpreting_params_name", type=str, default="defaults")

        args = parser.parse_args()
        return args

    def plot_direction_xcausation_multi_timestep(self):

        # Get hidden states gradients that were produced by the generative model
        samples_dir_grads = []

        print("Collecting data together...")
        for ep in range(0, self.num_samples_hx_grad):
            sample_dir_grads = []
            for direction_id in range(self.num_ica_components):
                grads_name = (
                    f"grad_hx_hx_direction_{direction_id}_{self.direction_type}"
                )
                grads = np.load(
                    os.path.join(
                        self.generated_data_path, f"sample_{ep:05d}/{grads_name}.npy"
                    )
                )  # (T x hx)
                # project grads from hx_space into hx_direction_space
                dir_grads = self.grad_projector(grads)
                sample_dir_grads.append(dir_grads)  # forms each column of a slice
            stacked_sample_grads = np.stack(
                sample_dir_grads, axis=-1
            )  # forms the rows (by sticking columns together)
            samples_dir_grads.append(
                stacked_sample_grads
            )  # So the columns are 'grads_wrt, the rows are 'grads_of'
        stacked_samples_grads = np.stack(samples_dir_grads, axis=-1)

        # Clip grads so that they're not larger than 1 std above or below
        # grads_std = stacked_samples_grads.std(-1) / 2
        # grads_mean = stacked_samples_grads.mean(-1)
        # pos_std = np.stack([(grads_mean + grads_std)] * self.num_samples_hx_grad, axis=-1)
        # neg_std = np.stack([(grads_mean - grads_std)] * self.num_samples_hx_grad, axis=-1)
        # pos_grads_cond = stacked_samples_grads > pos_std
        # neg_grads_cond = stacked_samples_grads < neg_std
        # stacked_samples_grads = np.where(pos_grads_cond, pos_std, stacked_samples_grads)
        # stacked_samples_grads = np.where(neg_grads_cond, neg_std, stacked_samples_grads)

        # Average over samples
        stacked_samples_grads = stacked_samples_grads.mean(-1)
        print("Done collecting data together...")
        for ts in range(stacked_samples_grads.shape[0]):
            print("Timestep: %i " % ts)
            ts_slice = stacked_samples_grads[ts]
            ts_sub = self.hp.analysis.saliency.common_timesteps[0] - ts
            title_name = f"Cross-causation matrix: grads of hx_direction at t=0 (x-axis) \nwith respect to hx_directions at (t-{ts_sub}) (y-axis)"
            self.plot_heatmap(
                ts_slice,
                "xcaus_hx_t+%i" % int(ts),
                title_name,
                labels=None,
                type_of_xcorr="causation",
                clim=ts_slice.max() / 2,
            )

    def plot_direction_xcorrs_multi_timestep(self):

        # Collect env and hxs and put them into a list of arrays, where each
        # array is one sample.

        # Get hidden states the were produced by the generative model
        print("Collecting env data together...")
        samples_data_env = []
        samples_data_hx = []
        env_pca_data = np.load(
            self.env_h_precomp_path + f"/pca_data_env_{self.num_samples_env_h}.npy"
        )

        print("Collecting data together...")
        for ep in range(0, min(self.num_samples_hx, self.num_samples_env_h)):
            # Env
            start_ts = ep * self.ts_per_sample
            stop_ts = (ep * self.ts_per_sample) + self.ts_per_sample
            env_vecs = env_pca_data[start_ts:stop_ts]
            samples_data_env.append(env_vecs.transpose())

            # Agent hx # TODO convert to use the standardised grad transformer
            hx = np.load(
                os.path.join(self.generated_data_path, f"sample_{ep:05d}/agent_hs.npy")
            )
            hx_vecs = self.hx_projector.transform(hx)
            hx_vecs = hx_vecs.transpose()
            samples_data_hx.append(hx_vecs)
        print("Done collecting data together...")

        # For k in {-k, ... , -1, 0, 1, ... , k}
        K = 10
        m = 1e3
        for k in np.arange(0, K + 1, 1):
            ## Collect together the pairs of vectors that we'll be using to make the xcorr matrix
            xcorrs_hx = []
            xcorrs_env = []
            xcorrs_both = []
            for hx_sample, env_sample in zip(samples_data_hx, samples_data_env):
                # Clip samples so pathologically large PC loadings don't have as much influence
                hx_sample = np.clip(hx_sample, a_min=-m, a_max=m)
                env_sample = np.clip(env_sample, a_min=-m, a_max=m)
                if k == 0:
                    set_a_hx = hx_sample
                    set_a_env = env_sample
                    set_b_hx = hx_sample
                    set_b_env = env_sample
                else:
                    set_a_hx = hx_sample[:, :-k]
                    set_a_env = env_sample[:, :-k]
                    set_b_hx = hx_sample[:, k:]
                    set_b_env = env_sample[:, k:]
                set_a_hx_env = np.concatenate([set_a_hx, set_a_env])
                set_b_hx_env = np.concatenate([set_b_hx, set_b_env])

                ## Calculate the xcorr matrices
                n = set_a_hx.shape[0]  # num elements in set
                xcorr_hx = (set_a_hx @ set_b_hx.transpose()) / n
                xcorr_env = (set_a_env @ set_b_env.transpose()) / n
                xcorr_both = (set_a_hx_env @ set_b_hx_env.transpose()) / n

                xcorrs_hx.append(xcorr_hx)
                xcorrs_env.append(xcorr_env)
                xcorrs_both.append(xcorr_both)

            xcorrs_hx = np.stack(xcorrs_hx)
            xcorrs_env = np.stack(xcorrs_env)
            xcorrs_both = np.stack(xcorrs_both)

            xcorrs_hx = np.mean(xcorrs_hx, axis=0)
            xcorrs_env = np.mean(xcorrs_env, axis=0)
            xcorrs_both = np.mean(xcorrs_both, axis=0)

            ## Plot and save the xcorr matrices
            if k == 0:
                ordered_dist_mat, res_order, res_linkage = self.compute_serial_matrix(
                    squareform(pdist(xcorrs_hx)), "ward"
                )
                self.plot_dendrogram(res_linkage, f"dendrogram_{self.direction_type}")

            print("Plotting for k=%i" % int(k))
            title_name = (
                f"Cross-correlation matrix comparing t (y-axis) with t+{k} (x-axis)"
            )

            self.plot_heatmap(
                xcorrs_hx, "xcorr_hx_t+%i" % int(k), title_name, res_order
            )
            self.plot_heatmap(xcorrs_hx, "unordered_xcorr_hx_t+%i" % int(k), title_name)

    def plot_heatmap(
        self, matrix, name, plot_title, labels=None, type_of_xcorr="causation", clim=1.0, clim_low=None,
    ):

        plt.rcParams["figure.figsize"] = (10, 10)
        if labels is None:
            plt.imshow(matrix, cmap="seismic_r", interpolation="nearest")
            plt.xticks(ticks=np.arange(0, matrix.shape[0], 1.0))
            plt.yticks(ticks=np.arange(0, matrix.shape[0], 1.0))
        else:
            plt.imshow(matrix[:, labels], cmap="seismic_r", interpolation="nearest")
            plt.xticks(ticks=np.arange(0, matrix.shape[0], 1.0), labels=labels)
            plt.yticks(ticks=np.arange(0, matrix.shape[0], 1.0))
        plt.colorbar()
        if clim_low is None:
            plt.clim(-clim, clim)
        else:
            plt.clim(clim_low, clim)

        plt.tight_layout()
        plt.title(plot_title)
        plt.savefig(os.path.join(self.save_path, "%s.png" % name))
        plt.clf()
        plt.close()

    def plot_dendrogram(self, Z, name):
        plt.figure(figsize=(10, 10))
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("ICA direction name")
        plt.ylabel("Similarity of correlations with a timestep")
        dendrogram(
            Z,
            leaf_rotation=0.0,  # rotates the x axis labels
            leaf_font_size=13,  # font size for the x axis labels
        )
        plt.title(f"Dendrogram for ica direction")
        plt.savefig(os.path.join(self.save_path, "%s.png" % name))
        plt.clf()
        plt.close()

    def seriation(self, Z, N, cur_index):
        """
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
        """
        if cur_index < N:
            return [cur_index]
        else:
            left = int(Z[cur_index - N, 0])
            right = int(Z[cur_index - N, 1])
            return self.seriation(Z, N, left) + self.seriation(Z, N, right)

    def compute_serial_matrix(self, dist_mat, method="ward"):
        """
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
        """
        N = len(dist_mat)
        flat_dist_mat = squareform(dist_mat)
        res_linkage = linkage(flat_dist_mat, method=method)
        res_order = self.seriation(res_linkage, N, N + N - 2)
        seriated_dist = np.zeros((N, N))
        a, b = np.triu_indices(N, k=1)
        seriated_dist[a, b] = dist_mat[
            [res_order[i] for i in a], [res_order[j] for j in b]
        ]
        seriated_dist[b, a] = seriated_dist[a, b]

        return seriated_dist, res_order, res_linkage

    def plot_cosine_sim_heatmap(self):
        # Get IC directions in hx space
        hx_dim = self.hp.gen_model.agent_hidden_size
        id_mat = np.eye(hx_dim, hx_dim)
        id_mat_mod = (id_mat * self.hx_projector.hx_std) + self.hx_projector.hx_mu
        ic_directions_in_hx = self.hx_projector.transform(id_mat_mod)
        ic_directions_in_hx = ic_directions_in_hx / np.linalg.norm(ic_directions_in_hx, axis=0)
        num_ic_dirs = ic_directions_in_hx.shape[1]
        assert num_ic_dirs == self.hp.analysis.agent_h.n_components_ica

        # Calc and plot cosine similarities between each direction
        cosine_sim_mat = np.zeros((num_ic_dirs, num_ic_dirs))
        for direction_id_i in range(num_ic_dirs):
            dir_i = ic_directions_in_hx[:, direction_id_i]
            for direction_id_j in range(num_ic_dirs):
                dir_j = ic_directions_in_hx[:, direction_id_j]
                cos_sim_ij = cosine(dir_i, dir_j)
                cosine_sim_mat[direction_id_i, direction_id_j] = cos_sim_ij
        cosine_sim_mat = np.ones_like(cosine_sim_mat) - cosine_sim_mat
        self.plot_heatmap(
            cosine_sim_mat,
            "Cosine_sim_between_IC_directions_in_hx_space",
            "Cosine similarity between IC directions in hx space",
            clim=1.,
        )


if __name__ == "__main__":
    xcm = xcorr_manager()
    # xcm.plot_direction_xcausation_multi_timestep()
    # xcm.plot_direction_xcorrs_multi_timestep()
    xcm.plot_cosine_sim_heatmap()
