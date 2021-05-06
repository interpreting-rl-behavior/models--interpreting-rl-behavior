import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ProcgenDataset(Dataset):
    """Coinrun dataset."""

    def __init__(self, data_dir='generative/data/', initializer_seq_len=None, total_seq_len=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.idx_to_epi_table = pd.read_csv(os.path.join(data_dir, 'idx_to_episode.csv')).iloc[0:999]
        # TODO undo above slicing
        self.seq_len = total_seq_len
        self.dataset_len = len(self.idx_to_epi_table)
        self.data_dir = data_dir
        self.init_seq_len = initializer_seq_len
        hx_size = 64
        obs_hw = 64
        obs_ch = 3
        act_space_size = 15
        self.null_element = {'done': np.zeros(self.seq_len),
                             'reward': np.zeros(self.seq_len),
                             'value': np.zeros(self.seq_len),
                             'action': np.zeros(self.seq_len),
                             'obs': np.zeros((self.seq_len, obs_ch, obs_hw, obs_hw)),
                             'hx': np.zeros((self.seq_len, hx_size)),
                             'act_log_probs': np.zeros((self.seq_len, act_space_size)),
                             }
        self.data_keys = list(self.null_element.keys())

    def __len__(self):
        # note: - self.seq_len to avoid sampling the end of the dataset where
        # the remaining sequence is too short.
        return len(self.idx_to_epi_table) - self.seq_len


    def __getitem__(self, idx):
        """It's kind of complicated, but it's as simple as I could make it so
        that it could fetch all the types of edge cases we should expect:
            - normal - init seq same as main seq and NOT close to end of epi
            - normal end - init seq same as main seq and epi ends during main seq
            - last epi - init seq or main seq extends beyond last epi
            - init seq 0 - init seq ends on final timestep of epi
            - init seq 1 - init seq ends 1 timestep before final timestep
            - init seq k - init seq ends k timesteps before final timestep
                and init seq k end - init seq ends k timesteps before final timestep
                and epi ends during main seq

        A note on nomenclature: 'element' refers to the batch element, which
        is what is output. 'Full segment' refers to the segment of the episode
        that is retrieved from storage."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Decide which episode to use.
        frames = self.idx_to_epi_table.iloc[idx:idx + self.seq_len]
        episode_number = frames['episode'].iloc[0]
        use_next_epi = frames['episode'].iloc[0] != \
                       frames['episode'].iloc[self.init_seq_len]
        episode_number = episode_number + use_next_epi

        # Get data
        csv_load_path = self.data_dir + 'data_gen_model_%i.csv' % episode_number
        data = pd.read_csv(csv_load_path)
        vecs_load_path = self.data_dir + 'episode' + str(episode_number)
        obs = np.load(vecs_load_path + '/ob.npy')
        hx  = np.load(vecs_load_path + '/hx.npy')
        lp  = np.load(vecs_load_path + '/lp.npy')

        # Now get the data segment and the initial index to place it on
        ## first get the starting timestep of the segment.
        full_segment_init_global_step = frames['global_step'].iloc[0]
        if use_next_epi:
            full_segment_init_step = 0
            element_first_step = np.argmin(-(frames['episode'] == episode_number)) - 1  # n.b. -1 because we want the last element of the init seq to be the 0th element of the simulated seq.
            if not element_first_step >= 0:
                print("boop")
                assert element_first_step >= 0
        else:
            full_segment_init_step = \
                data['episode_step'].loc[data['global_step'] == \
                                         full_segment_init_global_step]
            full_segment_init_step = int(full_segment_init_step)
            element_first_step = 0

        full_segment_len = sum(frames['episode'] == episode_number)
        full_segment_last_step = full_segment_init_step + full_segment_len
        element_last_step = element_first_step + full_segment_len

        ## Make data_dict and put the parts you want in the batch element.
        data = data[self.data_keys[:4]]
        data = data.to_numpy().T.tolist()
        vec_list = [obs, hx, lp]
        data.extend(vec_list)
        data_dict = {k: v for k, v in zip(self.data_keys, data)}

        ## Insert data into null vecs
        batch_ele = self.null_element.copy()
        for key, val in data_dict.items():
            assert len(list(range(element_first_step,
                                  element_last_step))) == \
                   len(list(range(full_segment_init_step,
                                  full_segment_last_step)))
            data_array = np.array(data_dict[key][full_segment_init_step:
                                                 full_segment_last_step])
            if key == 'obs':
                data_array = data_array / 255.
                # TODO consider copying the last frame after 'done' instead
                #  of leaving as all zeros.
            batch_ele[key][element_first_step:element_last_step] = data_array

        return batch_ele
