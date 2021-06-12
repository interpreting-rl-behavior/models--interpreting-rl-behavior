 Understanding RL agents using generative visualisation and differentiable environment simulation
===============

This README provides instructions for how to replicate the results in our paper. 

Overview of steps:

- Train agent on procgen task
- Record dataset of agent-environment rollouts
- Train generative model on recorded dataset
- Run analyses


To train the agent on coinrun:

> bsub -W 23:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python train.py --exp_name trainhx_1Mlvls --env_name coinrun --param_name hard-rec --num_levels 1000000 --distribution_mode hard --num_timesteps 200000000 --num_checkpoints 500

> bsub -W 71:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python train.py --exp_name trainhx_2Mlvls_cave_long --env_name caveflyer --param_name hard-rec --num_levels 2000000 --distribution_mode hard --num_timesteps 200000000 --num_checkpoints 300


This will save training data and a model in a directory in
> logs/procgen/coinrun/rec1M_reset/

Each training run has a unique seed, so each seed gets its own directory in the 
above folder.  

Then to plot the training curve for that training run:

> python plot_training_csv.py --datapath="/home/lee/Documents/AI_ML_neur_projects/aisc_project/train-procgen-pytorch/logs/procgen/coinrun/rec1M_reset/seed_7985_09-04-2021_22-01-54"

You can render your trained agent to see what its behaviour looks
like:

> python render.py --exp_name="trainhx_1Mlvls" --env_name="coinrun" --distribution_mode="hard" --param_name="hard-local-dev-rec" --device="cpu" --model_file="/home/lee/Documents/AI_ML_neur_projects/aisc_project/train-procgen-pytorch/logs/procgen/coinrun/trainhx_1Mlvls/seed_498_07-06-2021_23-26-27/model_80412672.pth"

> python render.py --exp_name="trainhx_1Mlvls_cave" --env_name="caveflyer" --distribution_mode="hard" --param_name="hard-local-dev-rec" --device="cpu" --vid_dir="/home/lee/Documents/AI_ML_neur_projects/aisc_project/train-procgen-pytorch/logs/procgen/caveflyer/trainhx_1Mlvls_cave/seed_4552_07-06-2021_23-28-43" --model_file="/home/lee/Documents/AI_ML_neur_projects/aisc_project/train-procgen-pytorch/logs/procgen/caveflyer/trainhx_1Mlvls_cave/seed_4552_07-06-2021_23-28-43/model_79233024.pth"""

Assuming your agent is behaviour as you'd like it to, now we can start 
interpreting it. 

To begin interpretation, we need to record a bunch of agent-environment 
rollouts in order to train the generative model:

>  bsub -W 47:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python record.py --exp_name recording_exp --env_name coinrun --param_name hard-rec --num_levels 1000000 --distribution_mode hard --num_checkpoints 200 --model_file="/cluster/home/sharkeyl/aisc_project/train-procgen-pytorch/logs/procgen/coinrun/rec1M_64dim/seed_498_07-06-2021_23-26-27/model_80412672.pth" --logdir="/cluster/scratch/sharkeyl/"

With this recorded data, we can start to train the generative model of 
agent-environment rollouts:

> bsub -W 71:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python train_gen_model.py --agent_file="/cluster/home/sharkeyl/aisc_project/train-procgen-pytorch/logs/procgen/coinrun/trainhx_1Mlvls/seed_498_07-06-2021_23-26-27/model_80412672.pth" --param_name=hard-rec --log_interval=10 --batch_size=28 --num_sim_steps=28 --num_initializing_steps=3 --save_interval=10000 --lr=1e-4 --env_name=coinrun --loss_scale_obs=1000.0 --loss_scale_hx=1.0 --loss_scale_reward=0.01 --loss_scale_done=0.1 --loss_scale_act_log_probs=0.00001 --loss_scale_gen_adv=0. --loss_scale_kl=1.0 --tgm_exp_name=trainable_hx --data_dir=/cluster/scratch/sharkeyl/data/
> 


