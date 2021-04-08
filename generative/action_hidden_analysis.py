import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

#TODO: think about t-SNE initialization 
# https://www.nature.com/articles/s41587-020-00809-z 
# https://jlmelville.github.io/smallvis/init.html

pd.options.mode.chained_assignment = None  # default='warn'

COINRUN_ACTIONS = {0: 'downleft', 1: 'left', 2: 'upleft', 3: 'down', 4: None, 5: 'up',
                   6: 'downright', 7: 'right', 8: 'upright', 9: None, 10: None, 11: None,
                   12: None, 13: None, 14: None}

# EPISODE_STRINGS = {v:str(v) for v in range(3431)}
def run():

    num_episodes = 3000  # number of episodes to make plots for
    seed = 42  # for the tSNE algo
    plot_pca = True
    plot_tsne = True
    # TODO args parser func
    presaved_data_path = "/media/lee/DATA/DDocs/AI_neuro_work/Assurance Project stuff/data/precollected/"
    hx_presaved_filepath = presaved_data_path + "hxs_%i.npy" % num_episodes
    lp_presaved_filepath = presaved_data_path + "lp_%i.npy" % num_episodes

    # Load the agent's output
    data = pd.read_csv(f'data/data_gen_model.csv')
    data = data.loc[data['episode'] < num_episodes]
    print('data shape', data.shape)
    #level_seed, episode, global_step, episode_step, done, reward, value, action

    # Prepare save dir
    save_path = 'analysis/hx_plots'
    os.makedirs(save_path, exist_ok=True)

    # Get hidden states
    if os.path.isfile(hx_presaved_filepath):
        # Load if already done before
        hx = np.load(hx_presaved_filepath)
    else:
        # Collect them one by one
        hx = np.load('data/episode0/hx.npy')
        for ep in range(1,num_episodes):
            hx_to_cat = np.load(f'data/episode{ep}/hx.npy')
            hx = np.concatenate((hx, hx_to_cat))
        # TODO save

    # Get log probs for actions
    if os.path.isfile(lp_presaved_filepath):
        # Load if already done before
        lp = np.load(lp_presaved_filepath)
    else:
        # Collect them one by one
        lp = np.load('data/episode0/lp.npy')
        for ep in range(1,num_episodes):
            lp_to_cat = np.load(f'data/episode{ep}/lp.npy')
            lp = np.concatenate((lp, lp_to_cat))
    lp_max = np.argmax(lp, axis=1)
    entropy = -1 * np.sum(np.exp(lp)*lp, axis=1)
    del lp

    # Add extra columns for further analyses variables
    # -  % way through episode
    # -  episode_rewarded?
    # -  done? # TODO in plotting
    # -  logprob max
    # -  entropy
    # -  value delta

    ## % way through episode
    episode_step_groups = [dfg for dfg in
                           data.groupby(by='episode')['episode_step']]
    max_steps_per_epi = [np.max(np.array(group)[1]) for group in episode_step_groups]
    max_steps_per_epi_list = [[x] * (x+1) for x in max_steps_per_epi]
    max_steps_per_epi_list = [item for sublist in max_steps_per_epi_list for item in sublist] # flattens
    data['episode_max_steps'] = max_steps_per_epi_list
    data['% through episode'] = data['episode_step'] / data['episode_max_steps']

    ## episode rewarded?
    episode_rew_groups = [dfg for dfg in
                          data.groupby(by='episode')['reward']]
    epi_rewarded = []
    for i, gr in enumerate(episode_rew_groups):
        if np.any(gr[1]):
            rew_bool_list = [1] * (max_steps_per_epi[i] + 1)
        else:
            rew_bool_list = [0] * (max_steps_per_epi[i] + 1)
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

    # Prepare for plotting
    plotting_variables = ['entropy', 'argmax_action_log_prob', 'action',
                          '% through episode', 'episode_max_steps',
                          'done',
                          'value', 'reward', 'episode_rewarded'] #TODO done

    plot_cmaps = {'entropy':                 'winter',
                  'argmax_action_log_prob':  'Paired_r',
                  'action':                  'tab20',
                  '% through episode':       'hsv',
                  'episode_max_steps':       'hsv',
                  'done':                    'autumn_r',
                  'value':                   'cool',
                  'reward':                  'cool',
                  'episode_rewarded':        'cool',}

    # Plotting
    if plot_pca:
        print('Starting PCA...')
        hx = StandardScaler().fit_transform(hx)
        pca = PCA(n_components=2)
        hx_pca = pca.fit_transform(hx)
        print('PCA finished.')
        data['pca_X'] = hx_pca[:, 0]
        data['pca_Y'] = hx_pca[:, 1]

        # TODO plot with arrows between consecutive points for the first n episodes
        # TODO remove outlier init state

        # Create grid of plots
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 18.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            ax = fig.add_subplot(3, 3, plot_idx)
            splot = plt.scatter(data['pca_X'], data['pca_Y'],
                            c=data[col],
                            cmap=plot_cmaps[col],
                            s=0.005, alpha=1.)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1),borderaxespad=0)
        fig.tight_layout()
        fig.savefig(f'{save_path}/agent_pca_epsd{num_episodes}_at{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.close()

    if plot_tsne:
        print('Starting tSNE...')
        _pca_for_tsne = PCA(n_components=64)
        hx_tsne = TSNE(n_components=2, random_state=seed).fit_transform(hx)
        print("tSNE finished.")
        data['tsne_X'] = hx_tsne[:, 0]
        data['tsne_Y'] = hx_tsne[:, 1]

        # TODO plot with arrows between consecutive points for the first n episodes

        # Create grid of plots
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.8, wspace=0.8)
        fig.set_size_inches(21., 18.)
        for plot_idx, col in enumerate(plotting_variables, start=1):
            ax = fig.add_subplot(3, 3, plot_idx)
            splot = plt.scatter(data['tsne_X'], data['tsne_Y'],
                                c=data[col],
                                cmap=plot_cmaps[col],
                                s=0.05, alpha=0.99)
            fig.colorbar(splot, fraction=0.023, pad=0.04)
            ax.legend(title=col, bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig.tight_layout()
        fig.savefig(
            f'{save_path}/agent_tsne_epsd{num_episodes}_at{time.strftime("%Y%m%d-%H%M%S")}.png')

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