import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torchvision.io as tvio
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

#TODO: think about t-SNE initialization 
# https://www.nature.com/articles/s41587-020-00809-z 
# https://jlmelville.github.io/smallvis/init.html

pd.options.mode.chained_assignment = None  # default='warn'

def run():
    # Set the number of episodes to make plots for (currently only starts from the first episode)
    num_episodes = 100

    save_dir = 'viz_obs/'
    # Make directory for saving figures
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(0,num_episodes):
        print(ep)
        obs = torch.tensor(np.load(f'data/episode{ep}/ob.npy'))
        obs = obs.permute(0, 2,3,1)
        save_path = save_dir + str(ep) + '.mp4'
        tvio.write_video(save_path, obs, fps=15)

if __name__ == "__main__":
    run()