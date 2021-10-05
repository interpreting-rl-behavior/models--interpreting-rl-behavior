import os
import pandas as pd
import numpy as np

if __name__=='__main__':

    data_dir_base = "."#'/cluster/scratch/sharkeyl/'##
    data_dir = os.path.join(data_dir_base, 'data/')

    # Making dataset for generative model
    ## Init dataset
    column_names = ['level_seed',
                    'episode',
                    'global_step',
                    'episode_step',
                    'done',
                    'reward',
                    'value',
                    'action',]
    max_episodes = 50000
    data = pd.DataFrame(columns=['global_step', 'episode'])
    global_step_curr = 0
    for e in range(max_episodes):
        epi_filename = os.path.join(data_dir, f'data_gen_model_{e:05d}.csv')
        data_e = pd.read_csv(epi_filename)
        global_step_range = np.arange(global_step_curr,
                                      global_step_curr+len(data_e))
        data_e['global_step'] = global_step_range
        data_e.to_csv(epi_filename)
        data = data.append(data_e[['global_step', 'episode']])
        global_step_curr = global_step_curr + len(data_e)
    data['global_step'] = np.arange(len(data['global_step'])) # Set global step
    # so that each data step has a unique global step.
    data.to_csv(data_dir + f'idx_to_episode.csv', index=False)
