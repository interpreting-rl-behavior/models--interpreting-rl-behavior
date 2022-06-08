"""Make sure you've run hidden_analysis_precompute.py before running this
because it generates data that this script uses."""

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


def bool_converter(str_or_bool):
    if type(str_or_bool) == str:
        if str_or_bool == 'False':
            return False
        elif str_or_bool == 'True':
            return True
        elif str_or_bool == '0.0':
            return False
        elif str_or_bool == '1.0':
            return True
        else:
            raise ValueError(f"{str_or_bool} is an invalid string for bool_converter")
    elif type(str_or_bool) == np.bool_ or type(str_or_bool) == bool:
        return str_or_bool
    elif type(str_or_bool) == np.int or type(str_or_bool) == int:
        return bool(str_or_bool)
    else:
        raise ValueError(f"{str_or_bool} is an invalid type for bool_converter. It is type {type(str_or_bool)}")


def run():
    args = parse_args()
    hp = hpf.load_interp_configs(args.interpreting_params_name)

    num_episodes = hp.analysis.agent_h.num_episodes #2000#100#4000#2000  # number of episodes to make plots for. Needs to be
    num_generated_samples = hp.analysis.agent_h.num_generated_samples
    # the same as the precomputed data you want to use
    plot_pca = True
    plot_3d_pca_all = True
    plot_gen_hx_pca = True
    plot_clusters = True
    plot_tsne = True

    first_PC_ind = 0
    second_PC_ind = 1
    third_PC_ind = 2

    # Prepare load and save dirs
    save_path = 'analysis/hx_plots'

    presaved_data_path = hp.analysis.presaved_data_path
    hx_presaved_filepath = presaved_data_path + "hxs_%i.npy" % num_episodes
    lp_presaved_filepath = presaved_data_path + "lp_%i.npy" % num_episodes

    # Load the non vector outputs
    lens = []
    main_data_path = hp.data_dir
    data = pd.read_csv(os.path.join(main_data_path, 'data_gen_model_00000.csv'))
    lens.append([data.__len__()])
    for ep in range(1, num_episodes):
        data_epi = pd.read_csv(os.path.join(main_data_path,
                                             f'data_gen_model_{ep:05d}.csv'))
        lens.append([data_epi.__len__()])
        data = data.append(data_epi)
    data = data.loc[data['episode'] < num_episodes]
    print('data shape', data.shape)
    #level_seed, episode, global_step, episode_step, done, reward, value, action

    bool_checker = lambda x: type(x) == np.bool_ or type(x) == bool
    if not all(map(bool_checker, data['done'])):
        data['done'] = pd.DataFrame(list(map(bool_converter, data['done'])))

    # Get hidden states
    if os.path.isfile(hx_presaved_filepath):
        # Load if already done before
        hx = np.load(hx_presaved_filepath)
        print("USING PRESAVED DATA PATH %s" % hx_presaved_filepath)
    else:
        # Collect them one by one
        hx = np.load(os.path.join(main_data_path, 'episode_00000/hx.npy'))
        hx = hx[1:]  # Cut 0th timestep
        lens[0].append(hx.__len__())

        for ep in range(1, num_episodes):
            hx_to_cat = np.load(os.path.join(main_data_path,
                                             f'episode_{ep:05d}/hx.npy'))
            hx_to_cat = hx_to_cat[1:]  # Cut 0th timestep
            lens[ep].append(hx_to_cat.__len__())
            hx = np.concatenate((hx, hx_to_cat))

    print(lens)
    print(all([x[0] == x[1] for x in lens]))
     
    if plot_gen_hx_pca:
        gen_hx_pca = np.load(hp.analysis.agent_h.precomputed_analysis_data_path + \
                             '/gen_hx_projected_real%i_gen%i.npy' % (num_episodes,
                                                                num_generated_samples))

    # Get log probs for actions
    if os.path.isfile(lp_presaved_filepath):
        # Load if already done before
        lp = np.load(lp_presaved_filepath)
    else:
        # Collect them one by one
        lp = np.load(os.path.join(main_data_path, 'episode_00000/lp.npy'))
        for ep in range(1,num_episodes):
            lp_to_cat = np.load(os.path.join(main_data_path,f'episode_{ep:05d}/lp.npy'))
            lp = np.concatenate((lp, lp_to_cat))
    lp_max = np.argmax(lp, axis=1)
    entropy = -1 * np.sum(np.exp(lp)*lp, axis=1)
    del lp

    # Add extra columns for further analyses variables
    # -  % way through episode
    # -  episode_rewarded?
    # -  logprob max
    # -  entropy
    # -  value delta (not plotted currently)

    ## % way through episode
    episode_step_groups = [dfg for dfg in
                           data.groupby(by='episode')['episode_step']]
    max_steps_per_epi = [np.max(np.array(group)[1]) for group in episode_step_groups]
    max_steps_per_epi_list = [[x] * int(x + 1) for x in max_steps_per_epi]
    max_steps_per_epi_list = [item for sublist in max_steps_per_epi_list for item in sublist] # flattens
    data['episode_max_steps'] = max_steps_per_epi_list
    data['% through episode'] = data['episode_step'] / data['episode_max_steps']

    ## episode rewarded?
    episode_rew_groups = [dfg for dfg in
                          data.groupby(by='episode')['reward']]
    epi_rewarded = []
    for i, gr in enumerate(episode_rew_groups):
        if np.any(gr[1]):
            rew_bool_list = [1] * int(max_steps_per_epi[i] + 1)
        else:
            rew_bool_list = [0] * int(max_steps_per_epi[i] + 1)
        epi_rewarded.extend(rew_bool_list)
    data['episode_rewarded'] = epi_rewarded

    ## max logprob
    data['argmax_action_log_prob'] = lp_max

    ## entropy
    data['entropy'] = entropy

    ## value delta
    episode_val_groups = [dfg for dfg in
                           data.groupby(by='episode')['value']]


    value_deltas = []
    for i, gr in enumerate(episode_val_groups):
        summand1, summand2 = gr[1].to_numpy(), gr[1].to_numpy()
        summand2 = np.roll(summand2, shift=1)
        delta = summand1 - summand2
        mask = delta != delta[np.argmax(np.abs(delta))] # removes largest, which is outlier
        delta = delta * mask
        delta = - np.log(delta)
        delta = list(delta)
        value_deltas.extend(delta)
    data['neg_log_value_delta'] = value_deltas

    # nmf max factor
    hx_nmf = np.load(hp.analysis.agent_h.precomputed_analysis_data_path + \
                     'nmf_hx_%i.npy' % num_episodes)
    nmf_max_factor = np.argmax(hx_nmf, axis=1)
    data['nmf_max_factor'] = nmf_max_factor
    # TODO i think we need to throw away the 0th hx in each episode. There
    #  are e.g. 47 hxs in episode 00000 but only 46 steps were counted. This is
    # because on the final timestep we exceptionally add the 'current' hx, not
    # just the previous hx. So since the 0th hx never actually makes  an
    # action or value estimate (we first pass the 0th hx to the GRU along with
    # the encoded obs input to get the current hx, which in turn is used to make
    # the action and value output), it makes sense that we get rid of the 0th
    # hx because it's uninformative and unused by values and action.

    # ica max factor
    hx_ica = np.load(hp.analysis.agent_h.precomputed_analysis_data_path + \
                     'ica_source_signals_hx_%i.npy' % num_episodes)
    ica_max_factor = np.argmax(hx_ica, axis=1)
    data['ica_max_factor'] = ica_max_factor

    # cluster identity
    hx_cluster = np.load(hp.analysis.agent_h.precomputed_analysis_data_path + \
                     'clusters_hx_%i.npy' % num_episodes)
    data['cluster_id'] = hx_cluster

    # Prepare for plotting
    plotting_variables = ['entropy', 'argmax_action_log_prob',  'action',
                          'cluster_id', 'nmf_max_factor', 'ica_max_factor',
                          'episode_max_steps', '% through episode', 'done',
                          'value', 'episode_rewarded', 'reward',]

    action_labels = list(range(15))
    action_cmap = sns.color_palette("husl", max(action_labels), as_cmap=True)

    nmf_labels = list(range(max(nmf_max_factor)))
    nmf_cmap = sns.color_palette("Paired", max(nmf_max_factor), as_cmap=True)

    cluster_labels = list(range(max(hx_cluster)))
    cluster_cmap = sns.color_palette("husl", max(hx_cluster), as_cmap=True)

    plot_cmaps = {'entropy':                 'winter',
                  'argmax_action_log_prob':   action_cmap,
                  'action':                   action_cmap,
                  'nmf_max_factor':           'hsv',
                  'cluster_id':               cluster_cmap,
                  'episode_max_steps':       'turbo', #hsv
                  '% through episode':       'hsv',#'brg',
                  'done':                    'autumn_r',
                  'value':                   'cool',
                  'ica_max_factor':          'hsv',
                  'episode_rewarded':        'cool',
                  'reward':                  'cool',}

    # Plotting
    if plot_pca:
        print("Plotting PCAs")
        hx_pca = np.load(hp.analysis.agent_h.precomputed_analysis_data_path + \
                         'hx_pca_%i.npy' % num_episodes)

        data['pca_X'] = hx_pca[:, first_PC_ind]
        data['pca_Y'] = hx_pca[:, second_PC_ind]

        # Create grid of plots
        pca_alpha = 0.95
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(25., 18.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            print(col)
            ax = fig.add_subplot(3, 4, plot_idx)
            splot = plt.scatter(data['pca_X'].loc[data['episode_step']!=0],
                                data['pca_Y'].loc[data['episode_step']!=0],
                                c=data[col].loc[data['episode_step']!=0],
                                cmap=plot_cmaps[col],
                                s=0.0005, alpha=pca_alpha)
            if plot_gen_hx_pca:
                splot = plt.scatter(
                    gen_hx_pca[:, first_PC_ind],
                    gen_hx_pca[:, second_PC_ind],
                    c='black',
                    s=0.05, alpha=0.9)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1),borderaxespad=0)
            ax.set_frame_on(False)


        fig.tight_layout()
        fig.savefig(f'{save_path}/agent_pca_PC{first_PC_ind}vsPC{second_PC_ind}_epsd{num_episodes}_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()

        # Now plot paths of individual episodes, connecting points by arrows
        groups = [dfg for dfg in
                  data.groupby(by='episode')[['pca_X', 'pca_Y']]]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 18.)
        for plot_idx in range(1, 13):
            ax = fig.add_subplot(3, 4, plot_idx)
            splot = plt.scatter(
                data['pca_X'].loc[data['episode_step']!=0],
                data['pca_Y'].loc[data['episode_step']!=0],
                c=data['% through episode'].loc[data['episode_step']!=0],
                cmap=plot_cmaps['% through episode'],
                s=0.0005, alpha=pca_alpha)
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
            f'{save_path}/agent_pca_arrows_PC{first_PC_ind}vsPC{second_PC_ind}_epsd{num_episodes}_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()

    if plot_3d_pca_all:
        data['pca_Z'] = hx_pca[:, third_PC_ind]
        now = time.strftime('%Y%m%d-%H%M%S')
        dir_name_3d = f"gifs_agent_pca_epsd{num_episodes}_at{now}"
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
                               s=0.005, c=data[plot_var],
                               cmap=plot_cmaps[plot_var])
                # fig.colorbar(p, fraction=0.023, pad=0.04)
                fig.tight_layout()
                image_name = f"{plot_var}_{angle}_PC{first_PC_ind}vsPC{second_PC_ind}vs{third_PC_ind}.png"
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
        cluster_dir_name = \
            f'{save_path}/cluster_plots{num_episodes}_at{time.strftime("%Y%m%d-%H%M%S")}'
        os.mkdir(cluster_dir_name)
        num_clusters = max(hx_cluster)
        for cluster_id in range(num_clusters):
            cluster_id_mask = data['cluster_id'] == cluster_id
            sizes = np.where(cluster_id_mask, 0.05, 0.005)
            fig = plt.figure()
            plt.scatter(data['pca_X'],
                        data['pca_Y'],
                        c=cluster_id_mask,
                        cmap='Set1_r',
                        s=sizes,
                        alpha=0.75)
            fig.tight_layout()
            fig.savefig(f'{cluster_dir_name}/cluster_{cluster_id}_PC{first_PC_ind}vsPC{second_PC_ind}.png')
            plt.close()

    if plot_tsne:
        hx_tsne = np.load(hp.analysis.agent_h.precomputed_analysis_data_path + \
                         'tsne_hx_%i.npy' % num_episodes)
        print('Starting tSNE...')
        # _pca_for_tsne = PCA(n_components=64)
        # hx_tsne = TSNE(n_components=2, random_state=seed).fit_transform(hx)
        # print("tSNE finished.")
        data['tsne_X'] = hx_tsne[:, 0]
        data['tsne_Y'] = hx_tsne[:, 1]

        # Create grid of plots
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 18.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            ax = fig.add_subplot(4, 3, plot_idx)
            splot = plt.scatter(data['tsne_X'].loc[data['episode_step']!=0],
                                data['tsne_Y'].loc[data['episode_step']!=0],
                                c=data[col].loc[data['episode_step']!=0],
                                cmap=plot_cmaps[col],
                                s=0.005, alpha=0.99)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1), borderaxespad=0)
            ax.set_frame_on(False)
        fig.tight_layout()
        fig.savefig(
            f'{save_path}/agent_tsne_epsd{num_episodes}_at{time.strftime("%Y%m%d-%H%M%S")}.png')

        plt.close()

        # Now plot paths of individual episodes, connecting points by arrows
        paths_per_plot = 1
        groups = [dfg for dfg in
                  data.groupby(by='episode')[['tsne_X', 'tsne_Y']]]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 15.)
        for plot_idx in range(1, 13):
            ax = fig.add_subplot(3, 4, plot_idx)
            splot = plt.scatter(
                data['tsne_X'].loc[data['episode_step']!=0],
                data['tsne_Y'].loc[data['episode_step']!=0],
                c=data['% through episode'].loc[data['episode_step']!=0],
                cmap=plot_cmaps['% through episode'],
                s=0.005, alpha=1.)
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
            f'{save_path}/agent_tsne_epsd{num_episodes}_arrows_at{time.strftime("%Y%m%d-%H%M%S")}.png')
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
