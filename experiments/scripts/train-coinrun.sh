#!/bin/bash
set -o errexit

# This script trains on the coinrun environment, with the coin
# placed randomly $random_percent % of the time, and otherwise
# placed at the end of the level.

random_percent=$SLURM_ARRAY_TASK_ID
experiment_name="freq-sweep-random-percent-$random_percent"

let n_steps=80*10**6
n_checkpoints=4
n_threads=32 # 32 CPUs per GPU
wandb_tags="hpc large-model"


coinrun_opt="--env_name coinrun --random_percent $random_percent --val_env_name coinrun_aisc" # coinrun_aisc ignores random_percent arg

# include the option
#       --model_file auto
# when resuming a run from a saved checkpoint. This should load the latest model
# saved under $experiment_name

options="
	$coinrun_opt
	--use_wandb
	--param_name A100
	--distribution_mode hard
	--num_timesteps $n_steps
	--num_checkpoints $n_checkpoints
	--num_threads $n_threads
	--wandb_tags $wandb_tags
	--exp_name $experiment_name
        "


python train.py $options
