import os
import pandas as pd

if __name__=='__main__':

    # Make save dirs
    logdir_base = "data/"#'/cluster/scratch/sharkeyl/'##

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

    print("Combining datasets")
    data = pd.DataFrame(columns=column_names)
    for e in range(70000):#7):#
        epi_filename = logdir + f'data_gen_model_{e:05d}.csv'
        data_e = pd.read_csv(epi_filename)
        data = data.append(data_e)
        #os.remove(epi_filename)
    data.to_csv(logdir + f'data_gen_model.csv',
                index=False)