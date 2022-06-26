import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform, cosine
import argparse
import os
import matplotlib.pyplot as plt
import hyperparam_functions as hpf
from dimred_projector import HiddenStateDimensionalityReducer


pd.options.mode.chained_assignment = None  # default='warn'


class DynGradManager:
    def __init__(self):
        args = self.parse_args()
        self.hp = hpf.load_interp_configs(args.interpreting_params_name)

        self.num_samples_hx = (
            self.hp.analysis.agent_h.num_episodes
        )  # number of generated samples to use
        self.num_samples_hx_grad = 4000
        assert len(self.hp.analysis.saliency.common_timesteps) == 1
        self.saliency_ts = self.hp.analysis.saliency.common_timesteps[0]
        # self.num_samples_env_h = self.hp.analysis.env_h.num_samples
        self.num_samples_clusters = self.hp.analysis.combined_agent_env_hx.num_samples
        self.ts_per_sample = self.hp.analysis.saliency.num_sim_steps
        self.direction_type = self.hp.analysis.saliency.direction_type
        self.num_ica_components = self.hp.analysis.agent_h.n_components_ica
        self.hx_projector = HiddenStateDimensionalityReducer(
            self.hp, self.direction_type, self.num_samples_hx, data_type=np.ndarray
        )
        self.grad_projector = self.hx_projector.project_gradients

        hx_precomp_data_path = os.path.join("analysis", "hx_analysis_precomp")
        # clusters_precomp_data_path = os.path.join("analysis", "jacob_analysis_precomp")
        clusters_precomp_data_path = os.path.join(
            "analysis", "combined_agent_env_hx_analysis_precomp"
        )

        # self.env_h_precomp_path = os.path.join("analysis", "env_analysis_precomp")
        # env_components = np.load(
        #     f"{self.env_h_precomp_path}/pcomponents_env_{self.num_samples_env_h}.npy"
        # )
        self.cluster_ids = np.load(
            f"{clusters_precomp_data_path}/clusters_per_sample_{self.num_samples_clusters}.npy"
        )
        self.hx_pc_components = np.load(
            f"{hx_precomp_data_path}/pcomponents_{self.num_samples_hx}.npy"
        )
        self.hx_pc_variances = np.load(
            f"{hx_precomp_data_path}/pc_loading_variances_{self.num_samples_hx}.npy"
        )  # TODO put the extra
        # self.hx_pc_components = self.hx_pc_components.transpose()
        # self.hx_mu = np.load(hx_precomp_data_path + f"/hx_mu_{self.num_samples_hx}.npy")
        # self.hx_std = np.load(
        #     hx_precomp_data_path + f"/hx_std_{self.num_samples_hx}.npy"
        # )

        self.xcorr_data_path = os.path.join(
            os.getcwd(), "analysis", "cross_corr_and_causation_plot_data"
        )

        # Prepare load and save dirs
        self.generated_data_path = os.path.join(
            self.hp.generated_data_dir, self.hp.analysis.agent_h.informed_or_random_init
        )
        save_path = self.hp.analysis.dyn_grad_comparison.save_dir  # "xcorr_matrices/"
        save_path_data = self.hp.analysis.dyn_grad_comparison.save_dir_data
        self.save_path = os.path.join(os.getcwd(), "analysis", save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path_data = os.path.join(os.getcwd(), "analysis", save_path_data)
        os.makedirs(self.save_path_data, exist_ok=True)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="args for plotting")
        parser.add_argument("--interpreting_params_name", type=str, default="defaults")
        args = parser.parse_args()
        return args

    def plot_consistencies(self):

        # Get arrays for xcorr plots for all clusters and time differences
        K = self.ts_per_sample
        num_clusters = len(list(set(self.cluster_ids)))
        per_cluster_xcorrs = []
        for cluster in list(set(self.cluster_ids)):
            per_ts_xcorrs = []
            for k in np.arange(0, K + 1, 1):
                data_name = f"xcorr_hx_clust{cluster}_t+{int(k)}.npy"
                data_path = os.path.join(self.xcorr_data_path, data_name)
                data = np.load(data_path)
                per_ts_xcorrs.append(data)
            per_ts_xcorrs = np.stack(per_ts_xcorrs)
            per_cluster_xcorrs.append(per_ts_xcorrs)
        per_cluster_xcorrs = np.stack(
            per_cluster_xcorrs
        )  # (Clusters, Timesteps, IC_d, IC_d)
        per_cluster_xcorrs = np.flip(
            per_cluster_xcorrs, axis=1
        )  # Put time in the same order as jacobs

        # Get arrays for Jacobians for all clusters and time differences
        per_cluster_jacobs = self.collect_cluster_jacobians()

        # Get aligned xcorr and jacob timesteps
        num_ts = per_cluster_jacobs.shape[1]
        per_cluster_xcorrs = per_cluster_xcorrs[:, -num_ts:]

        print("boop")
        # Compare sign, create mask
        jacobs_pos = per_cluster_jacobs >= 0.0
        xcorrs_pos = per_cluster_xcorrs >= 0.0
        pos_match = jacobs_pos == xcorrs_pos
        print(pos_match.mean(axis=0).mean(axis=-1).mean(axis=-1))
        #
        # Make a heatmap for every cluster for every ts
        clims = [[-0.25, 0.25], [-0.8, 0.8], [-1.0, 1.0]]
        for c in range(len(list(set(self.cluster_ids)))):
            for t_diff in range(num_ts):
                matrix_list = [
                    per_cluster_jacobs[c, t_diff],
                    per_cluster_xcorrs[c, t_diff],
                    pos_match[c, t_diff],
                ]
                plot_title = (
                    f"Comparison between \n Jacobians of ICs w.r.t ICs {self.saliency_ts - t_diff} "
                    + f"timesteps in the past \n & cross-correlations in behaviour cluster{c}"
                    + f" between timestep t-{self.saliency_ts - t_diff} (y-axis) and t (x-axis)"
                )
                self.plot_heatmaps_side_by_side(
                    matrix_list,
                    self.save_path,
                    f"comparison_cluster{c}_tdiff{t_diff}",
                    clims,
                    plot_title,
                    [
                        "Jacobian",
                        "Cross correlation",
                        "Jacobian same sign as cross correlation",
                    ],
                )

        # Compare magnitude

        pass

    def plot_quant_over_time(self, data):
        raise NotImplementedError

    def plot_heatmaps_side_by_side(
        self, matrix_list, save_path, save_name, clims, plot_title, subplot_titles
    ):
        fig, ax = plt.subplots(1, len(matrix_list), figsize=(30, 10))
        fig.suptitle(plot_title)
        num_ics = matrix_list[0].shape[0]
        # plt.xticks(np.arange(0, num_ics, 1.0))
        # plt.yticks(np.arange(0, num_ics, 1.0))
        ticks_and_labels = list(range(0, num_ics))

        for i, matrix in enumerate(matrix_list):

            im = ax[i].imshow(matrix, cmap="seismic_r", interpolation="nearest")
            ax[i].set_title(subplot_titles[i])
            ax[i].set_xticks(ticks_and_labels, labels=ticks_and_labels)
            ax[i].set_yticks(ticks_and_labels, labels=ticks_and_labels)
            im.set_clim(clims[i][0], clims[i][1])
            plt.colorbar(im, ax=ax[i])

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "%s.png" % save_name))
        plt.clf()
        plt.close()
        pass

    def plot_heatmap(self, matrix, save_path, save_name, plot_title):
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.imshow(matrix, cmap="seismic_r", interpolation="nearest")
        plt.colorbar()
        plt.clim(-1.0, 1.0)
        plt.xticks(np.arange(0, matrix.shape[0], 2.0))
        plt.yticks(np.arange(0, matrix.shape[0], 2.0))
        plt.tight_layout()
        plt.title(plot_title)
        plt.savefig(os.path.join(save_path, "%s.png" % save_name))
        plt.clf()
        plt.close()

    def collect_sample_jacobian(self, sample_id):
        n_ic_directions = self.hp.analysis.agent_h.n_components_ica
        assert len(self.hp.analysis.saliency.common_timesteps) == 1
        ts_id = self.hp.analysis.saliency.common_timesteps[0] + 1
        jacob_vecs = []
        for direction_id in range(n_ic_directions):  # D
            hx_grad_vecs = np.load(
                os.path.join(
                    self.generated_data_path,
                    f"sample_{sample_id:05d}/grad_hx_hx_direction_{direction_id}_ica.npy",
                )
            )  # (T x H)
            hx_grad_vec = hx_grad_vecs[:ts_id]  # (H,)
            hx_grad_vec = self.grad_projector(hx_grad_vec)  # (I,)
            jacob_vecs.append(hx_grad_vec)
        jacobian = np.stack(jacob_vecs, axis=-1)  # (I, D)
        return jacobian

    def collect_cluster_jacobians(self):
        per_cluster_jacobs = []
        print("Collecting jacobs for...")
        for cluster in list(set(self.cluster_ids)):
            print(f"... cluster {cluster}")
            cluster_sample_ids = np.where(self.cluster_ids == cluster)[0]
            cluster_jacobs = []
            for sample_id in cluster_sample_ids:
                sample_jacobs = self.collect_sample_jacobian(sample_id)
                cluster_jacobs.append(sample_jacobs)
            cluster_jacobs = np.stack(cluster_jacobs, axis=0)
            cluster_jacobs = np.mean(cluster_jacobs, axis=0)
            per_cluster_jacobs.append(cluster_jacobs)
        per_cluster_jacobs = np.stack(per_cluster_jacobs)
        return per_cluster_jacobs


if __name__ == "__main__":
    dgm = DynGradManager()
    dgm.plot_consistencies()
