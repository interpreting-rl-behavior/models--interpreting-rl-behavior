import pandas as pd
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import imageio
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

    num_samples = hp.analysis.jacobian.num_samples #2000#20000  # number of episodes to make plots for. Needs to be
    # the same as the precomputed data you want to use
    plot_pca = True
    plot_3d_pca_all = False
    plot_clusters = True
    plot_3d_pca = True
    plot_tsne = True

    first_pc_ind = hp.analysis.jacobian.first_pc_ind
    second_pc_ind = hp.analysis.jacobian.second_pc_ind
    third_pc_ind = hp.analysis.jacobian.third_pc_ind

    precomputed_analysis_data_path = hp.analysis.jacobian.precomputed_analysis_data_path

    data = {}

    # Prepare load and save dirs
    generated_data_path = os.path.join(hp.generated_data_dir, 'informed_init')
    save_path = 'analysis/jacob_plots'


    # cluster identity
    jacobian_cluster = np.load(precomputed_analysis_data_path + \
                     'clusters_jacob_%i.npy' % num_samples)
    data['cluster_id'] = jacobian_cluster

    data = pd.DataFrame.from_dict(data)

    # Prepare for plotting
    plotting_variables = ['cluster_id']

    cluster_labels = list(range(max(jacobian_cluster)))
    cluster_cmap = sns.color_palette("husl", max(jacobian_cluster), as_cmap=True)

    plot_cmaps = {'cluster_id':  cluster_cmap}

    jacobian_pca = np.load(precomputed_analysis_data_path + \
                           'pca_data_jacob_%i.npy' % num_samples)

    data['pca_X'] = jacobian_pca[:, first_pc_ind]
    data['pca_Y'] = jacobian_pca[:, second_pc_ind]

    outlier_threshold = 20
    outlier_cond_pos = np.any(data[['pca_X', 'pca_Y']] > outlier_threshold, axis=1)
    outlier_cond_neg = np.any(data[['pca_X', 'pca_Y']] < -outlier_threshold, axis=1)
    outlier_cond = outlier_cond_pos | outlier_cond_neg
    data = data[~outlier_cond]

    # Plotting
    if plot_pca:
        print("Plotting PCAs")


        # Create grid of plots
        pca_alpha = 0.99
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(10., 10.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            print(col)
            ax = fig.add_subplot(1, 1, plot_idx)
            splot = plt.scatter(data['pca_X'],
                                data['pca_Y'],
                                c=data[col],
                                cmap=plot_cmaps[col],
                                s=1.0, alpha=pca_alpha)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1),borderaxespad=0)
            ax.set_frame_on(False)

        fig.tight_layout()
        fig.savefig(f'{save_path}/jacobian_pca_PC{first_pc_ind}vsPC{second_pc_ind}_epsd{num_samples}_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()


    if plot_3d_pca_all:
        print("Plotting PCs in 3D")
        data['pca_Z'] = jacobian_pca[:, third_pc_ind]
        now = time.strftime('%Y%m%d-%H%M%S')
        dir_name_3d = f"gifs_jacobian_pca_epsd{num_samples}_at{now}"
        dir_name_3d = os.path.join(save_path, dir_name_3d)
        if not (os.path.exists(dir_name_3d)):
            os.makedirs(dir_name_3d)

        for plot_var in plotting_variables:
            image_names = []
            for angle in np.arange(start=0, stop=360, step=10):
                fig = plt.figure(figsize=(11,11))
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(30, angle)
                plt.draw()
                p = ax.scatter(data['pca_X'], data['pca_Y'], data['pca_Z'],
                               s=1.0, c=data[plot_var],
                               cmap=plot_cmaps[plot_var])
                # fig.colorbar(p, fraction=0.023, pad=0.04)
                fig.tight_layout()
                image_name = f"{plot_var}_{angle}.png"
                image_name = f'{dir_name_3d}/{image_name}'
                image_names.append(image_name)
                fig.savefig(image_name)
                plt.close()

            gif_name = f"{plot_var}.gif"
            gif_name = os.path.join(dir_name_3d, gif_name)
            with imageio.get_writer(gif_name, mode='I', fps=4) as writer:
                for filename in image_names:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            # Remove files
            for filename in set(image_names):
                os.remove(filename)

    if plot_clusters:
        print("Plotting plots where cluster alone is highlighted")
        cluster_dir_name = \
            f'{save_path}/cluster_plots{num_samples}_at{time.strftime("%Y%m%d-%H%M%S")}'
        os.mkdir(cluster_dir_name)
        num_clusters = max(data['cluster_id'])
        for cluster_id in range(num_clusters):
            cluster_id_mask = data['cluster_id'] == cluster_id
            sizes = np.where(cluster_id_mask, 0.1, 0.05)
            fig = plt.figure()
            plt.scatter(data['pca_X'],
                        data['pca_Y'],
                        c=cluster_id_mask,
                        cmap='Set1_r',
                        s=sizes,
                        alpha=0.75)
            fig.tight_layout()
            fig.savefig(f'{cluster_dir_name}/cluster_{cluster_id}.png')
            plt.close()

    if plot_tsne:
        jacobian_tsne = np.load(precomputed_analysis_data_path + \
                         'tsne_jacob_%i.npy' % num_samples)
        print('Starting tSNE...')
        # _pca_for_tsne = PCA(n_components=64)
        # hx_tsne = TSNE(n_components=2, random_state=seed).fit_transform(hx)
        # print("tSNE finished.")
        data['tsne_X'] = jacobian_tsne[:, 0][~outlier_cond]
        data['tsne_Y'] = jacobian_tsne[:, 1][~outlier_cond]

        # Create grid of plots
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(10., 10.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            ax = fig.add_subplot(1, 1, plot_idx)
            splot = plt.scatter(data['tsne_X'],
                                data['tsne_Y'],
                                c=data[col],
                                cmap=plot_cmaps[col],
                                s=1.0, alpha=0.99)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1), borderaxespad=0)
            ax.set_frame_on(False)
        fig.tight_layout()
        fig.savefig(
            f'{save_path}/jacobian_tsne_epsd{num_samples}_at{time.strftime("%Y%m%d-%H%M%S")}.png')

        plt.close()

        fig.tight_layout()
        fig.savefig(
            f'{save_path}/jacobian_tsne_epsd{num_samples}_arrows_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()

# fig = plt.figure(figsize=(11,11))
# ax = fig.add_subplot(111, projection='3d')
# if angle is not None:
#     if type(angle)==tuple:
#         ax.view_init(angle[0], angle[1])
#     else:
#         ax.view_init(30, angle)
#     plt.draw()
# if colours is not None:
#     p = ax.scatter(pca_data[:,0], pca_data[:,1], pca_data[:,2], s=s, c=colours, cmap=cmap)
#     fig.colorbar(p, fraction=0.023, pad=0.04)
#
# else:
#     ax.scatter(pca_data[:,0], pca_data[:,1], pca_data[:,2], s=s)

if __name__ == "__main__":
    run()
