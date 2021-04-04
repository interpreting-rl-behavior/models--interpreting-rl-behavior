import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
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
def run():
    # Set the number of episodes to make plots for (currently only starts from the first episode)
    num_episodes = 100
    # Set the random state for the TSNE algo
    seed = 42

    # Load the agent's output
    #action_df = pd.read_csv(f'generative/data/data_gen_model.csv')
    action_df = pd.read_csv(f'data/data_gen_model.csv')
    action_df = action_df.loc[action_df['episode'] < num_episodes]
    print('action_df shape', action_df.shape)
    #level_seed, episode, global_step, episode_step, done, reward, value, action
    #TODO: make 256 more columns and insert hidden data 
    #TODO: change name of df to something else 

    #save_path = 'generative/analysis/agent_tsne'
    save_path = 'analysis/agent_tsne'
    # Make directory for saving figures
    os.makedirs(save_path, exist_ok=True)

    hx = np.load('data/episode0/hx.npy')
    # print('hx shape', hx.shape)
    episodes = sorted(action_df['episode'].unique())
    # print('episodes', episodes)
    for ep in range(1,num_episodes):
        # Get the data for the current episode only
        # sub_df = action_df.loc[action_df['episode'] == ep]
        # Load the agent hidden state for this episode
        #hx = np.load(f'generative/data/episode{ep}/hx.npy')
        hx_to_cat = np.load(f'data/episode{ep}/hx.npy')
        # print('hxcat shape', hx_to_cat.shape)
        hx = np.concatenate((hx, hx_to_cat))
        # print('new shape', hx.shape)
        #some dimension by 256 
        # print(hx.shape)
        # Create TSNE embedding
        # possibly do PCA first and then TSNE 
        # hx_embedded = TSNE(n_components=2, random_state=seed).fit_transform(hx)
        # sub_df['X'] = hx_embedded[:,0]
        # sub_df['Y'] = hx_embedded[:,1]
        # # Map action indices to names
        # sub_df['action'] = sub_df['action'].map(COINRUN_ACTIONS)

        # # print(sub_df)

        # if i == num_episodes:
        #     # Don't create anymore plots
        #     break
    
    print('starting TSNE')
    hx_embedded = TSNE(n_components=2, random_state=seed).fit_transform(hx)
    print('hx embed shape', hx_embedded[:,0].shape)
    print('hx embed', hx_embedded[:,0])
    action_df['X'] = hx_embedded[:,0]
    action_df['Y'] = hx_embedded[:,1]
    # Map action indices to names
    action_df['action'] = action_df['action'].map(COINRUN_ACTIONS)

    # Create grid of plots
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.8, wspace=0.8)
    for plot_idx, col in zip([1,2,3,4], ['episode_step', 'action', 'value', 'reward']):
        ax = fig.add_subplot(2, 2, plot_idx)
        sns_plot = sns.scatterplot(x='X', y='Y', hue=col, data=action_df, ax=ax)
        ax.legend(title=col, bbox_to_anchor=(1.01, 1),borderaxespad=0)
    #TODO get title working
    # fig.suptitle(f't-SNE plot for episode {ep}')
    fig.tight_layout()
    fig.savefig(f'{save_path}/agent_tsne_ep{time.strftime("%Y%m%d-%H%M%S")}.png')

if __name__ == "__main__":
    run()