#!/bin/bash
set -o errexit

# gather metrics for figure 2
# load trained coinrun models and deploy them in the test environment.
# to do this without specifying the model_file every time, trained coinrun
# models must be stored in logs with exp_name 'freq-sweep-random-percent-$random_percent'
# write output metrics to csv files in ./experiments/results/

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
