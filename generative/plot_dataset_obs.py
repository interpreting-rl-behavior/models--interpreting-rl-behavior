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


def run():
    # Set the number of episodes to make plots for starting from the
    # first episode
    num_episodes = 100

    # Make directory for saving figures
    save_dir = 'viz_obs/'
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(0,num_episodes):
        print(ep)
        obs = np.load(f'data/episode{ep}/ob.npy')
        obs = obs.transpose([0,2,3,1])
        save_path = save_dir + str(ep) + '.mp4'
        tvio.write_video(save_path, obs, fps=14)

if __name__ == "__main__":
    run()