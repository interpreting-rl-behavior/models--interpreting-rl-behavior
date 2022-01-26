#!/bin/bash
set -o errexit

num_seeds=10000
#num_seeds=10

random_percent=$SLURM_ARRAY_TASK_ID

if [[ $1 = 'standard' ]]
then
    python run_coinrun.py --model_file $random_percent --start_level_seed 0 --num_seeds $num_seeds --random_percent 100
elif [[ $1 = 'joint' ]]
then
    python run_coinrun.py --model_file $random_percent --start_level_seed 0 --num_seeds $num_seeds --random_percent $random_percent --reset_mode "complete"
fi







# run(){
#     for random_percent in 0 2 4 6 8 10 12 14 16 18 20
#     do
# 	echo running model trained on random_percent $random_percent
#         python run_coinrun.py --model_file $random_percent --start_level_seed 0 --num_seeds $num_seeds --random_percent 100
#     done
# }
# 
# 
# run_iid(){
#     for random_percent in 0 2 4 6 8 10 12 14 16 18 20
#     do
# 	echo running model trained on random_percent $random_percent
#         python run_coinrun.py --model_file $random_percent --start_level_seed 0 --num_seeds $num_seeds --random_percent $random_percent --reset_mode "complete"
#     done
# }

