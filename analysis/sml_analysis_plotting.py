import pandas as pd
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import imageio

pd.options.mode.chained_assignment = None  # default='warn'

COINRUN_ACTIONS = {0: 'downleft', 1: 'left', 2: 'upleft', 3: 'down', 4: None, 5: 'up',
                   6: 'downright', 7: 'right', 8: 'upright', 9: None, 10: None, 11: None,
                   12: None, 13: None, 14: None}
def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--agent_env_data_dir', type=str,
        default="data")
    parser.add_argument(
        '--precomputed_analysis_data_path', type=str, default="analysis/sml_analysis_precomp")
    parser.add_argument(
        '--generated_data_dir', type=str,
        default='generative/recorded_informinit_gen_samples')
    parser.add_argument(
        '--presaved_data_path', type=str, default="/media/lee/DATA/DDocs/AI_neuro_work/assurance_project_stuff/data/precollected/")
    args = parser.parse_args()
    return args


def run():
    args = parse_args()
    num_samples = 20000  # number of episodes to make plots for. Needs to be
    # the same as the precomputed data you want to use
    plot_pca = True
    plot_3d_pca_all = False
    plot_clusters = True
    plot_tsne = True

    first_PC_ind = 0
    second_PC_ind = 1
    third_pc_ind = 2

    data = {}

    # Prepare load and save dirs
    generated_data_path = args.generated_data_dir
    save_path = 'analysis/sml_plots'

    # Load the non vector outputs

    ## Get rewards at each timestep and max rews for sample
    rews = np.load(os.path.join(generated_data_path, 'sample_00000/rews.npy'))[:-1]
    max_rews = np.load(os.path.join(generated_data_path, 'sample_00000/rews.npy'))[:-1]
    max_rews[:] = max_rews.max()
    for ep in range(1, num_samples):
        rews_to_cat = np.load(os.path.join(generated_data_path,
                                         f'sample_{ep:05d}/rews.npy'))[:-1]
        rews = np.concatenate((rews, rews_to_cat))

        rews_to_cat[:] = rews_to_cat.max()
        max_rews = np.concatenate((max_rews, rews_to_cat))
    data['reward'] = rews.squeeze()
    data['max_sample_reward'] = max_rews.squeeze()

    ## Get 'done' at each timestep and 'ever dones' (whether the sample has any dones)
    dones = np.load(os.path.join(generated_data_path, 'sample_00000/dones.npy'))[:-1]

    ever_dones =  np.load(os.path.join(generated_data_path, 'sample_00000/dones.npy'))[:-1]
    ever_dones[:] = ever_dones.max()
    for ep in range(1, num_samples):
        dones_to_cat = np.load(os.path.join(generated_data_path,
                                         f'sample_{ep:05d}/dones.npy'))[:-1]
        dones = np.concatenate((dones, dones_to_cat))

        dones_to_cat[:] = dones_to_cat.max()
        ever_dones = np.concatenate((ever_dones, dones_to_cat))
    data['done'] = dones.squeeze()
    data['ever_done'] = ever_dones.squeeze()

    ## Get value output at each timestep
    agent_values = np.load(os.path.join(generated_data_path, 'sample_00000/agent_values.npy'))[:-1]
    for ep in range(1, num_samples):
        agent_values_to_cat = np.load(os.path.join(generated_data_path,
                                         f'sample_{ep:05d}/agent_values.npy'))[:-1]
        agent_values = np.concatenate((agent_values, agent_values_to_cat))
    data['value'] = agent_values.squeeze()

    ## Assign an index vec to each sample that has the same length as the sample
    sample_length = np.load(os.path.join(generated_data_path, 'sample_00000/agent_values.npy')).shape[0]
    idx_vec = np.array([0] * sample_length)[:-1]
    for ep in range(1, num_samples):
        idx_vec_to_cat = np.array([ep] * sample_length)[:-1]
        idx_vec = np.concatenate((idx_vec, idx_vec_to_cat))
    data['sample_idx'] = idx_vec.squeeze()

    # Get log probs for actions produced by the generative model
    lp = np.load(os.path.join(generated_data_path, 'sample_00000/agent_logprobs.npy'))[:-1]
    for ep in range(1, num_samples):
        lp_to_cat = np.load(os.path.join(generated_data_path,
                                         f'sample_{ep:05d}/agent_logprobs.npy'))[:-1]
        lp = np.concatenate((lp, lp_to_cat))
    lp_max = np.argmax(lp, axis=1)
    data['argmax_action_log_prob'] = lp_max.squeeze()

    entropy = -1 * np.sum(np.exp(lp)*lp, axis=1)
    data['entropy'] = entropy.squeeze()
    del lp

    # Add extra columns for further analyses variables
    # -  % way through episode
    # -  episode_rewarded?
    # -  logprob max
    # -  entropy
    # -  value delta (not plotted currently)

    # nmf max factor
    sml_nmf = np.load(args.precomputed_analysis_data_path + \
                    '/nmf_sml_raw_%i.npy' % num_samples)
    nmf_max_factor = np.argmax(sml_nmf, axis=1)
    data['nmf_max_factor'] = nmf_max_factor.squeeze()

    # cluster identity
    sml_cluster_raw = np.load(args.precomputed_analysis_data_path + \
                     '/clusters_sml_raw_%i.npy' % num_samples)
    data['cluster_id_raw'] = sml_cluster_raw
    sml_cluster_dyn = np.load(args.precomputed_analysis_data_path + \
                     '/clusters_sml_dyn_%i.npy' % num_samples)
    data['cluster_id_dyn'] = sml_cluster_dyn

    data = pd.DataFrame.from_dict(data)


    # Prepare for plotting
    plotting_variables = ['entropy', 'argmax_action_log_prob',
                          'cluster_id_raw', 'cluster_id_dyn', 'nmf_max_factor',
                          'ever_done',
                          'value', 'max_sample_reward', 'reward',]

    action_labels = list(range(15))
    action_cmap = sns.color_palette("husl", max(action_labels), as_cmap=True)

    nmf_labels = list(range(max(nmf_max_factor)))
    nmf_cmap = sns.color_palette("Paired", max(nmf_max_factor), as_cmap=True)

    num_clusters = max([max(sml_cluster_raw), max(sml_cluster_dyn)])
    cluster_labels = list(range(num_clusters))
    cluster_cmap = sns.color_palette("husl", num_clusters, as_cmap=True)

    plot_cmaps = {'entropy':                 'winter',
                  'argmax_action_log_prob':   action_cmap,
                  'cluster_id_raw':           cluster_cmap,
                  'cluster_id_dyn':           cluster_cmap,
                  'nmf_max_factor':           nmf_cmap,
                  'done':                    'autumn_r',
                  'ever_done':               'autumn_r',
                  'value':                   'cool',
                  'max_sample_reward':       'cool',
                  'reward':                  'cool',}

    # Plotting
    if plot_pca:
        print("Plotting PCAs")
        sml_pca = np.load(args.precomputed_analysis_data_path + \
                         '/pca_data_sml_raw_%i.npy' % num_samples)

        data['pca_X'] = sml_pca[:, first_PC_ind]
        data['pca_Y'] = sml_pca[:, second_PC_ind]

        # Create grid of plots
        pca_alpha = 0.95
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(25., 18.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            print(col)
            ax = fig.add_subplot(3, 3, plot_idx)
            splot = plt.scatter(data['pca_X'],
                                data['pca_Y'],
                                c=data[col],
                                cmap=plot_cmaps[col],
                                s=0.05, alpha=pca_alpha)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1),borderaxespad=0)
            ax.set_frame_on(False)


        fig.tight_layout()
        fig.savefig(f'{save_path}/sml_pca_PC{first_PC_ind}vsPC{second_PC_ind}_epsd{num_samples}_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()

        # Now plot paths of individual samples, connecting points by arrows
        groups = [dfg for dfg in
                  data.groupby(by='sample_idx')[['pca_X', 'pca_Y']]]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 18.)
        for plot_idx in range(1, 13):
            ax = fig.add_subplot(3, 4, plot_idx)
            splot = plt.scatter(
                data['pca_X'],
                data['pca_Y'],
                c=data['cluster_id_dyn'],
                cmap=plot_cmaps['cluster_id_dyn'],
                s=0.05, alpha=pca_alpha)
            # for epi in path_epis:
            epi_data = groups[plot_idx-1][1]
            for i in range(len(epi_data) - 1):
                x1, y1 = epi_data.iloc[i][['pca_X', 'pca_Y']]
                x2, y2 = epi_data.iloc[i + 1][['pca_X', 'pca_Y']]
                dx, dy = x2 - x1, y2 - y1
                arrow = matplotlib.patches.FancyArrowPatch((x1, y1),
                                                           (x2, y2),
                                                           arrowstyle=matplotlib.patches.ArrowStyle.CurveB(
                                                               head_length=1.5,
                                                               head_width=2.0),
                                                           mutation_scale=1,
                                                           shrinkA=0.,
                                                           shrinkB=0.,
                                                           color='black')
                ax.add_patch(arrow)
                ax.set_frame_on(False)
            if plot_idx == 4:
                fig.colorbar(splot, fraction=0.023, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            f'{save_path}/sml_pca_epsd{num_samples}_arrows_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()

    if plot_3d_pca_all:
        print("Plotting PCs in 3D")
        data['pca_Z'] = sml_pca[:, third_pc_ind]
        now = time.strftime('%Y%m%d-%H%M%S')
        dir_name_3d = f"gifs_sml_pca_epsd{num_samples}_at{now}"
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
                               s=0.05, c=data[plot_var],
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
        print("raw")
        cluster_dir_name = \
            f'{save_path}/cluster_raw_plots{num_samples}_at{time.strftime("%Y%m%d-%H%M%S")}'
        os.mkdir(cluster_dir_name)
        num_clusters = max(data['cluster_id_raw'])
        for cluster_id in range(num_clusters):
            cluster_id_mask = data['cluster_id_raw'] == cluster_id
            sizes = np.where(cluster_id_mask, 0.1, 0.05)
            fig = plt.figure()
            plt.scatter(data['pca_X'],
                        data['pca_Y'],
                        c=cluster_id_mask,
                        cmap='Set1_r',
                        s=sizes,
                        alpha=0.75)
            fig.tight_layout()
            fig.savefig(f'{cluster_dir_name}/cluster_raw_{cluster_id}.png')
            plt.close()

        print("Dyn clusters")
        cluster_dir_name = \
            f'{save_path}/cluster_dyn_plots{num_samples}_at{time.strftime("%Y%m%d-%H%M%S")}'
        os.mkdir(cluster_dir_name)
        num_clusters = max(data['cluster_id_dyn'])
        for cluster_id in range(num_clusters):
            cluster_id_mask = data['cluster_id_dyn'] == cluster_id
            sizes = np.where(cluster_id_mask, 0.1, 0.05)
            fig = plt.figure()
            plt.scatter(data['pca_X'],
                        data['pca_Y'],
                        c=cluster_id_mask,
                        cmap='Set1_r',
                        s=sizes,
                        alpha=0.75)
            fig.tight_layout()
            fig.savefig(f'{cluster_dir_name}/cluster_dyn_{cluster_id}.png')
            plt.close()

    if plot_tsne:
        sml_tsne = np.load(args.precomputed_analysis_data_path + \
                         '/tsne_sml_raw_%i.npy' % num_samples)
        print('Starting tSNE...')
        # _pca_for_tsne = PCA(n_components=64)
        # hx_tsne = TSNE(n_components=2, random_state=seed).fit_transform(hx)
        # print("tSNE finished.")
        data['tsne_X'] = sml_tsne[:, 0]
        data['tsne_Y'] = sml_tsne[:, 1]

        # Create grid of plots
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 18.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            ax = fig.add_subplot(3, 3, plot_idx)
            splot = plt.scatter(data['tsne_X'],
                                data['tsne_Y'],
                                c=data[col],
                                cmap=plot_cmaps[col],
                                s=0.05, alpha=0.99)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1), borderaxespad=0)
            ax.set_frame_on(False)
        fig.tight_layout()
        fig.savefig(
            f'{save_path}/sml_tsne_epsd{num_samples}_at{time.strftime("%Y%m%d-%H%M%S")}.png')

        plt.close()

        # Now plot paths of individual episodes, connecting points by arrows
        paths_per_plot = 1
        groups = [dfg for dfg in
                  data.groupby(by='sample_idx')[['tsne_X', 'tsne_Y']]]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 15.)
        for plot_idx in range(1, 13):
            ax = fig.add_subplot(3, 4, plot_idx)
            splot = plt.scatter(
                data['tsne_X'],
                data['tsne_Y'],
                c=data['cluster_id_dyn'],
                cmap=plot_cmaps['cluster_id_dyn'],
                s=0.05, alpha=1.)
            epi_data = groups[plot_idx-1][1]
            for i in range(len(epi_data) - 1):
                x1, y1 = epi_data.iloc[i][['tsne_X', 'tsne_Y']]
                x2, y2 = epi_data.iloc[i + 1][['tsne_X', 'tsne_Y']]
                dx, dy = x2 - x1, y2 - y1
                arrow = matplotlib.patches.FancyArrowPatch((x1, y1),
                                                           (x2, y2),
                                                           arrowstyle=matplotlib.patches.ArrowStyle.CurveB(
                                                               head_length=1.5,
                                                               head_width=2.0),
                                                           mutation_scale=1,
                                                           shrinkA=0.,
                                                           shrinkB=0.,
                                                           color='black')
                ax.add_patch(arrow)
                ax.set_frame_on(False)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            if plot_idx==4:
                fig.colorbar(splot, fraction=0.023, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            f'{save_path}/sml_tsne_epsd{num_samples}_arrows_at{time.strftime("%Y%m%d-%H%M%S")}.png')
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
