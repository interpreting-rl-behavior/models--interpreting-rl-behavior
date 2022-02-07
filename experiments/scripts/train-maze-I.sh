#!/bin/bash
set -o errexit

# This script trains on maze_aisc, where goal position
# is randomized within a region of size --rand-region


rand_region=$SLURM_ARRAY_TASK_ID
experiment_name="maze-I-sweep-rand-region-$rand_region"

let n_steps=80*10**6
n_checkpoints=4
n_threads=32 # 32 CPUs per GPU

wandb_tags="hpc large-model rand-region-sweep"
export WANDB_RUN_ID="maze-sweep-rand-region-$rand_region"


maze_opt="--env_name maze_aisc --rand_region $rand_region --val_env_name maze" # coinrun_aisc ignores random_percent arg

# include the option
#       --model_file auto
# when resuming a run from a saved checkpoint. This should load the latest model
# saved under $experiment_name

options="
	$maze_opt
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
