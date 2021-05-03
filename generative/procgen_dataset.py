import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ProcgenDataset(Dataset):
    """Coinrun dataset."""

    def __init__(self, data_dir='generative/data/', total_seq_len=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.procgen_data = pd.read_csv(os.path.join(data_dir, 'data_gen_model.csv'))
        self.seq_len = total_seq_len
        self.dataset_len = len(self.procgen_data)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.procgen_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # TODO some way to ensure that the episode doesn't end during the
        #  initialisation sequence (but it's okay if it ends on the 0th
        #  timestep of the main simulated sequence).

        # TODO some way to ensure that the initialization sequence can start
        #  on the 0th timestep of the episode.

        # Gets frames starting at idx #TODO proper treatment of when the end of the dataset is sampled
        end_frame_idx = idx + self.seq_len
        frames = self.procgen_data.iloc[idx:end_frame_idx]

        episode_number = frames['episode'].iloc[0]
        initial_episode_step = frames['episode_step'].iloc[0]

        # Put columns into data_dict
        keys = list(frames.keys())
        data_keys = ['done', 'reward', 'value', 'action']
        data = frames.to_numpy().T.tolist()
        data_dict = {}
        for values, key in zip(data, keys):
            if key in data_keys:
                if end_frame_idx >= self.dataset_len:
                    print("Debugging: end_frame_idx >= self.dataset_len, which should happen only once per epoch")
                    # Fill with zeros if asking for data indices that are
                    # beyond the end of the dataset (and therefore don't exist)
                    zero_num = 0. if type(values[0])==float else 0
                    values.extend([zero_num] * (end_frame_idx - self.dataset_len))
                data_dict[key] = np.array(values)

        # Goes to end of episode and fills in the rest with zeros
        if np.any(data_dict['done']):
            segment_len = np.argmax(data_dict['done']) + 1
        else:
            segment_len = self.seq_len

        # Remove any info from the following episode that is not part
        # of the batch element.
        for key, value in data_dict.items(): # TODO consider just freezing the frame at the last frame instead of zeroing it out.
            if key == 'done':
                value[:segment_len-1] = 0. # -1 because the final frame of the
                # episode should be 'done'(==1)
                value[segment_len:] = 1.
            elif key == 'reward':
                value[segment_len:] = 0.
            elif key == 'value':
                value[segment_len:] = 0.
            data_dict[key] = np.stack(value)

        # Get obs, hx, and log probs and set anything after episode done to zero
        save_path = self.data_dir + 'episode' + str(episode_number)
        obs = np.load(save_path + '/ob.npy')
        hx  = np.load(save_path + '/hx.npy')
        lp  = np.load(save_path + '/lp.npy')

        episode_len = len(obs)
        ## observations
        if episode_len > initial_episode_step+self.seq_len:
            obs = obs[initial_episode_step:initial_episode_step+self.seq_len]
            obs = obs / 255.
        else:
            t, c, h, w = obs.shape
            zeros = np.zeros([self.seq_len, c, h, w])
            obs = obs[initial_episode_step:episode_len]
            zeros[0:episode_len-initial_episode_step] = obs
            obs = zeros / 255.

        ## hidden states
        if episode_len > initial_episode_step+self.seq_len:
            hx = hx[initial_episode_step:initial_episode_step+self.seq_len]
        else:
            t, d = hx.shape
            zeros = np.zeros([self.seq_len, d])
            hx = hx[initial_episode_step:episode_len]
            zeros[0:episode_len-initial_episode_step] = hx
            hx = zeros

        ## action log probs
        if episode_len > initial_episode_step+self.seq_len:
            lp = lp[initial_episode_step:initial_episode_step+self.seq_len]
        else:
            t, d = lp.shape
            zeros = np.zeros([self.seq_len, d])
            lp = lp[initial_episode_step:episode_len]
            zeros[0:episode_len-initial_episode_step] = lp
            lp = zeros

        # Add arrays to data_dict
        array_keys = ['obs', 'hx', 'act_log_probs']
        arrays = [obs, hx, lp]
        for k, v in zip(array_keys, arrays):
            data_dict[k] = v

        return data_dict
