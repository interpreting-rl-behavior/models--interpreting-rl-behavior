 Understanding RL agents using generative visualisation and differentiable environment simulation
===============

This README provides instructions for how to replicate the results in our paper. 

Overview of steps:

- Train agent on procgen task
- Record dataset of real agent-environment rollouts
- Train generative model on recorded dataset of real agent-environment rollouts
- Run analyses on recorded dataset of real agent-environment rollouts
- Record dataset of simulated agent-environment rollouts from the generative model
- Run analyses on the recorded simulated rollouts. 

_N.B. During development, some of the below commands are prefixed with terms that
are for submitting jobs on ETHZ's clusters, which use IBM Spectrum LSF e.g.:_
> bsub -W 23:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" 
> python script.py

_These are kept here for convenience but will be removed in the published README.
You will need to ignore them when running locally._

All scripts should be run from the root dir.

To train the agent on coinrun:

> bsub -W 23:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" 
> python train.py --exp_name [agent_training_experiment_name] --env_name coinrun --param_name hard-rec --num_levels 1000000 --distribution_mode hard --num_timesteps 200000000 --num_checkpoints 500

This will save training data and a model in a directory in
> logs/procgen/coinrun/[your_experiment_name]/

Each training run has a unique seed, so each seed gets its own directory in the 
above folder like so:
> logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]

Then to plot the training curve for that training run:

> python plot_training_csv.py --datapath="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]"

You can render your trained agent to see what its behaviour looks
like:

> python render.py --exp_name=[agent_rendering_experiment_name] --env_name="coinrun" --distribution_mode="hard" --param_name="hard-local-dev-rec" --device="cpu" --model_file="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]/[agent_name].pth"

Assuming your agent is behaviour as you'd like it to, now we can start 
interpreting it. 

# Making recordings and training the generative model 

To begin interpretation, we need to record a bunch of agent-environment 
rollouts in order to train the generative model:

>  bsub -W 47:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" 
> python record.py --exp_name [recording_experiment_name] --env_name coinrun --param_name hard-rec --num_levels 1000000 --distribution_mode hard --num_checkpoints 200 --model_file="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]/[agent_name].pth" --logdir="[path_to_rollout_data_save_dir]"

Note that ``--logdir`` should have plenty of storage space (100's of GB).

With this recorded data, we can start to train the generative model on 
agent-environment rollouts:

> bsub -W 71:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" 
> python train_gen_model.py --agent_file="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]/[agent_name].pth" --param_name=hard-rec --log_interval=10 --batch_size=28 --num_sim_steps=28 --num_initializing_steps=3 --save_interval=10000 --lr=1e-4 --env_name=coinrun --loss_scale_obs=1000.0 --loss_scale_hx=1.0 --loss_scale_reward=0.01 --loss_scale_done=0.1 --loss_scale_act_log_probs=0.00001 --loss_scale_gen_adv=0. --loss_scale_kl=1.0 --tgm_exp_name=[generative_model_training_experiment_name] --data_dir=[path_to_real_rollout_data_save_dir]

That'll take a while to train (a few days/weeks). Once it's trained, we'll record some agent-
environment rollouts from the model. This will enable us to compare the 
simulations to the true rollouts and will help us understand our generative 
model (which includes the agent that we want to interpret) better. This is how
we record samples from the generative model:

> bsub -W 23:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" 
> python record_informinit_gen_samples.py --agent_file="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]/[agent_name].pth" --param_name=hard-rec --log_interval=10 --batch_size=16 --num_sim_steps=28 --num_initializing_steps=3 --save_interval=10000 --lr=1e-4 --env_name=coinrun --model_file="generative/results/[generative_model_training_experiment_name]/[date_time_of_gen_model_training]/[gen_model_name].pt" --data_dir=[path_to_real_rollout_data_save_dir]/data --data_save_dir=[path_to_sim_rollout_data_save_dir]/recorded_informinit_gen_samples

> bsub -W 23:59 -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" 
> python record_random_gen_samples.py --agent_file="logs/procgen/coinrun/[agent_training_experiment_name]/[agent_training_unique_seed]/[agent_name].pth" --param_name=hard-rec --log_interval=10 --batch_size=16 --num_sim_steps=28 --num_initializing_steps=3 --save_interval=10000 --lr=1e-4 --env_name=coinrun --model_file="generative/results/[generative_model_training_experiment_name]/[date_time_of_gen_model_training]/[gen_model_name].pt" --data_save_dir=[path_to_sim_rollout_data_save_dir]/recorded_randinit_gen_samples

Now we're ready to start some analysis. 

# Analysis

The generative model is a VAE, and therefore consists of an encoder and decoder.
The decoder is the part we want to interpret because it simulates agent-
environment rollouts. It will be informative, therefore, to get a picture of
what's going on inside the latent vector of the VAE, since this is the input
to the decoder. 

## Analysis of latent vector of VAE

In theory, the distribution of the VAE latent vector space is trained to be as close
as possible to a standard multivariate gaussian distribution (i.e. it is trained
to have a hyperspherical density function). In practice, however, the KL 
divergence never reaches zero so the distribution of the latent vector never
becomes a perfect hypersphere. We produce PCA and and tSNE plots of the VAE
latent vectors to observe the structure of the distribution. 

> bsub -W 23:59 -R "rusage[mem=32768]" 
> python latent_vec_analysis_precompute.py --agent_env_data_dir=[path_to_real_rollout_data_save_dir]/data --generated_data_dir_inf=[path_to_sim_rollout_data_save_dir]/recorded_informinit_gen_samples --generated_data_dir_rand=[path_to_sim_rollout_data_save_dir]/recorded_random_gen_samples
> 
> bsub -W 23:59 -R "rusage[mem=32768]" 
> python latent_vec_analysis_plotting.py --agent_env_data_dir=[path_to_real_rollout_data_save_dir]/data --precomputed_analysis_data_path=analysis/latent_vec_analysis_precomp/

## Analysis of agent's hidden state
We'll next analyse the agent's hidden state with a few dimensionality reduction
methods. First we precompute the dimensionality reduction analyses:
> python hidden_analysis_precompute.py --agent_env_data_dir="[path_to_real_rollout_data_save_dir]/data"

or on the cluster

> bsub -W 23:59 -R "rusage[mem=65536]" 
> python hidden_analysis_precompute.py --agent_env_data_dir=[path_to_real_rollout_data_save_dir]/data --generated_data_dir=[path_to_sim_rollout_data_save_dir]/recorded_informinit_gen_samples

with 10'000 episodes (not samples). Increase request for memory and compute time to cope with more episodes.  

which will save the analysis data in ``analysis/hx_analysis_precomp/``

Next we'll make some plots from the precomputed analyses of the agent's hidden
states:
> python hidden_analysis_plotting.py --agent_env_data_dir="[path_to_real_rollout_data_save_dir]/data" --precomputed_analysis_data_path="analysis/hx_analysis_precomp"

These depict what the agent is 'thinking' during many episodes, visualised
using several different dimensionality reduction and clustering methods. 

## Analysis of environment hidden states

> bsub -W 23:59 -R "rusage[mem=65536]" 
> python env_h_analysis_precompute.py --agent_env_data_dir=[path_to_real_rollout_data_save_dir]/data --generated_data_dir=[path_to_sim_rollout_data_save_dir]/recorded_informinit_gen_samples

with 20'000 samples of len 24.  Increase request for memory and compute time to cope with more samples.  

then

> bsub -W 23:59 -R "rusage[mem=32768]" 
> python env_h_analysis_plotting.py --agent_env_data_dir=[path_to_real_rollout_data_save_dir]/data --precomputed_analysis_data_path=analysis/env_analysis_precomp --generated_data_dir=[path_to_sim_rollout_data_save_dir]/recorded_informinit_gen_samples


## Analysis of SensoriMotorLoop space

> bsub -W 23:59 -R "rusage[mem=65536]" 
> python sml_analysis_precompute.py --agent_env_data_dir=[path_to_real_rollout_data_save_dir]/data --generated_data_dir=[path_to_sim_rollout_data_save_dir]/recorded_informinit_gen_samples

## Analysis of the prediction quality over time
We measure the mean squared error of each component of the generative model's loss, and see how it changes with the
number of simulation steps the generative model produces. To run this experiment and output a json file with the results, run:
> python loss_over_time_exp.py --exp_name demo2 --epochs 1 --batch_size 200 --agent_file=[your pth file] --device cpu --param_name hard-local-dev-rec --model_file=[your pt file]

Note that you may need to add arguments for the scaling factors of each loss component (e.g. --loss_scale_obs=1000.0 --loss_scale_hx=1.0). To create a line plot using the data from the above experiment, run:
> python analysis/plot_loss_over_time.py --presaved_data_path=generative/analysis/loss_over_time/[json output file above]"
