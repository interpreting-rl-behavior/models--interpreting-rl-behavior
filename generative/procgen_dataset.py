import torch

import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ProcgenDataset(Dataset):
    """Coinrun dataset."""

    def __init__(self, csv_file, total_seq_len, inp_seq_len):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.procgen_data = pd.read_csv(csv_file)
        self.seq_len = total_seq_len
        self.dataset_len = len(self.procgen_data)
        self.inp_seq_len = inp_seq_len

        # Consider modifying recording so that you save each observation frame
        # as a separate file and just save the filename in the dataset here.
        # This may be more memory efficient.

    def _nullify_row(self, row_to_nullify, template_row, episode_step):
        """Using template row, go through filling the rows either with
        zeros of their default values for that episode"""
        keys = list(row_to_nullify.keys())
        indice = row_to_nullify.index[0]
        for key in keys:
            if key in ['level_seed', 'episode',]:
                row_to_nullify.at[key] = template_row[key]
            if key in ['global_step']:
                row_to_nullify.at[key] = np.nan
            if key in ['episode_step']:
                row_to_nullify.at[key] = episode_step
            if key in ['done']:
                row_to_nullify.at[key] = 1.
            if key in ['reward', 'value', 'action', 'act_log_probs', 'hx']:
                row_to_nullify.at[key] = np.zeros_like(template_row[key])
            if key in ['obs']:
                row_to_nullify.at[key] = 1.

    def __len__(self):
        return len(self.procgen_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Gets frames starting at idx
        end_frame_idx = idx + self.seq_len
        if end_frame_idx >= self.dataset_len:
            idx = idx - np.random.randint(1000,3000)###
            # end_frame_idx = self.dataset_len
            print("oh dear")
            end_frame_idx = idx + self.seq_len###

        frames = self.procgen_data.iloc[idx:end_frame_idx]

        # Make columns into lists
        keys = list(frames.keys())
        data_keys = ['done', 'reward', 'value',
                       'act_log_probs', 'hx', 'obs']
        data = frames.to_numpy().T.tolist()
        data_dict = {}
        for values, key in zip(data, keys):
            if key in data_keys:
                data_dict[key] = np.array(values)

        # Goes to end of episode and fills in the rest with zeros
        if np.any(data_dict['done']):
            segment_len = np.argmax(data_dict['done']) + 1
            prune_len = self.seq_len - segment_len
        else:
            segment_len = self.seq_len
            prune_len = 0

        for key, value in data_dict.items():
            if key == 'done':
                value[:segment_len-1] = 0.
                value[segment_len:] = 1.
            if key == 'reward':
                value[segment_len:] = 0.
            if key == 'value':
                value[segment_len:] = 0.
            if key in ['act_log_probs', 'hx', 'obs']:
                value = list(value)
                value[:segment_len] = [np.load(f).squeeze() / 255.
                                       for f in value[:segment_len]]
                value[segment_len:] = [np.zeros_like(value[0])] * prune_len
            data_dict[key] = np.stack(value)

        return data_dict
