"""Makes a dataset for the generative model.

A note on timestep indexing and saving:

Each timestep's variables consist of the same variables as in the Deletang
diagram (augmented with rew and done info) and the PREVIOUS hidden state. We
use the previous hx because that is what is input to the agent at this timestep.
 But the agent uses the (current) hidden_state to produce the action. But we
 still save the (current) hidden_state at the end. This means that we need to
 be careful about indexing when visualising. The (current) hidden state should
 be aligned with the action and the obs at the current timestep. Just be careful
  when taking gradients that you know what you're actually taking grads wrt.
"""
import random

import numpy as np
import torch
import pandas as pd
import os, time, yaml, argparse
import shutil
import gym

from train import create_venv
from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from overlay_image import overlay_actions

import torchvision.io as tvio


if __name__=='__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name',         type=str, default = 'test', help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'coinrun', help='environment ID')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--num_levels',       type=int, default = int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode',type=str, default = 'easy', help='distribution mode for environment')
    parser.add_argument('--param_name',       type=str, default = 'easy-200', help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'gpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(1), help='number of checkpoints to store')
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--logdir', type=str, default='.') #todo does this work?

    args = parser.parse_args()

    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    # Hyperparameters
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    hyperparameters['n_envs'] = n_envs  # overwrite because can only record one
    # at a time.

    # Recording-specific hyperparams
    max_episodes = 50000
    secs_in_24h = 60*60*24
    max_time_recording = secs_in_24h * 1.8

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    # Environment
    print('INITIALIZAING ENVIRONMENTS...')
    env = create_venv(args, hyperparameters, is_valid=True)

    # Logger
    print('INITIALIZING LOGGER...')
    logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'RENDER_seed' + '_' + \
             str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir)

    # Model
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)

    # Storage
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    # Agent
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)
    checkpoint = torch.load(args.model_file, map_location=device)
    agent.policy.load_state_dict(checkpoint["model_state_dict"])
    agent.policy.action_noise = True # Only for recording data for gen model training
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.n_envs = n_envs

    # Make save dirs
    logdir_base = args.logdir
    logdir = os.path.join(logdir_base, 'data/')
    if not (os.path.exists(logdir_base)):
        os.makedirs(logdir_base)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

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

    ## Init logging-objects for recording loop
    data = []
    for i in range(n_envs):
        data.append(pd.DataFrame(columns=column_names))
    obs_lists = [[] for i in range(n_envs)]
    hx_lists = [[] for i in range(n_envs)]
    logprob_lists = [[] for i in range(n_envs)]

    # Init agent and env
    obs = agent.env.reset()
    hidden_state_prev = np.stack(
        [agent.policy.init_hx.clone().detach().cpu().numpy()] \
        * agent.n_envs)     # init with hx param
    prev_act = np.ones(agent.n_envs) * 4  # Because in Coinrun, 4==null_action.
    done = np.zeros(agent.n_envs)
    rew = np.zeros(agent.n_envs)

    # Timestep trackers
    global_steps = 0
    episode_steps = np.zeros_like(np.arange(n_envs))
    episode_number = np.array(np.arange(n_envs))
    episode_lens = np.zeros(max_episodes)

    # Check you're not overwriting
    dir_name = os.path.join(logdir, f'episode_{episode_number[0]:05d}')
    # if os.path.exists(dir_name):
    #     raise UserWarning("You are overwriting your previous data! Delete " + \
    #                       "or move your old dataset first.")
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)

    while True:
        epi_max = np.max(episode_number)
        print(f"Episode min|50%%|max: {np.min(episode_number)} | {np.median(episode_number)} | {np.max(episode_number)}")
        agent.policy.eval()

        # Step agent and environment
        act, log_prob_act, value, hidden_state = agent.predict_record(obs, hidden_state_prev, done)
        obs_next, rew_next, done_next, info = agent.env.step(act)
        #if done, append the final hidden state (even though it's never input to the
        # agent) and the last obs (in order to black it out, and it also is
        # never input to agent)

        # Store variables
        for i in range(n_envs):
            data[i] = data[i].append({'level_seed': info[i]['level_seed'], # TODO for some reason seed is misaligned for the first step of each episode except the 0th episode.
                                     'episode': episode_number[i],
                                     'global_step': global_steps,
                                     'episode_step': episode_steps[i],
                                     'done': done[i],
                                     'reward': rew[i],
                                     'value': value[i],
                                     'action': act[i],
                                     }, ignore_index=True)

            if done[i]:
                    black_obs = np.zeros_like(obs[i])
                    obs_lists[i].append(black_obs)
                    hx_lists[i].append(hidden_state_prev[i])
                    hx_lists[i].append(hidden_state[i])
                    logprob_lists[i].append(log_prob_act[i])
            else:
                obs_lists[i].append(obs[i])
                hx_lists[i].append(hidden_state_prev[i])
                logprob_lists[i].append(log_prob_act[i])

        # At end of episode
        if np.any(done):
            done_idxs = np.where(done)[0] # [0] is because np.where returns a tuple
            for idx in done_idxs:
                done_epi_idx = episode_number[idx]

                if episode_number[idx] < max_episodes: # save episode len
                    episode_lens[done_epi_idx] = episode_steps[idx] + 1

                data[idx].to_csv(os.path.join(logdir,
                    f'data_gen_model_{done_epi_idx:05d}.csv'),
                    index=False)

                # Make dirs for files
                dir_name = os.path.join(logdir, f'episode_{done_epi_idx:05d}')
                if not (os.path.exists(dir_name)):
                    os.makedirs(dir_name)

                # Stack arrays for this episode into one array
                obs_array = np.stack(obs_lists[idx]).squeeze()
                hx_array  = np.stack(hx_lists[idx]).squeeze()
                lp_array  = np.stack(logprob_lists[idx]).squeeze()

                # Prepare names for saving
                obs_name = dir_name + '/ob.npy'
                hx_name = dir_name + '/hx.npy'
                lp_name = dir_name + '/lp.npy'

                # Save stacked array
                np.save(obs_name, np.array(obs_array * 255, dtype=np.uint8))
                np.save(hx_name, hx_array)
                np.save(lp_name, lp_array)

                # Save vid
                ob = torch.tensor(obs_array * 255)
                ob = ob.permute(0, 2, 3, 1)
                ob = ob.clone().detach().type(torch.uint8)
                ob = ob.cpu().numpy()
                # Overlay a square with arrows showing the agent's actions
                actions = np.array(data[idx]['action'])
                ob = overlay_actions(ob, actions, size=16)
                save_str = os.path.join(logdir, f'sample_{done_epi_idx:05d}.mp4')
                tvio.write_video(save_str, ob, fps=14)

                # Reset things for the beginning of the next episode
                data[idx] = pd.DataFrame(columns=column_names)
                episode_number[idx] = epi_max + 1
                epi_max += 1
                episode_steps[idx] = 0

                obs_lists[idx] = []
                hx_lists[idx] = []
                logprob_lists[idx] = []

                # Reset hidden state
                hidden_state_prev[idx,:] = np.stack(
                    agent.policy.init_hx.clone().detach().cpu().numpy())


        # Increment for next step
        obs = obs_next
        rew = rew_next
        done = done_next
        hidden_state_prev = hidden_state
        global_steps += 1
        episode_steps += 1
        print("Episode number: ", episode_number)
        print("Episode len: ", episode_steps)
        print("Done:        ", done + 0)

        # End recording loop if max num recorded episodes OR time-limit has
        # been reached
        if (np.min(episode_number) > max_episodes):# or ((time.time() - start_time) > max_time_recording):
            break

    # Delete superfluous data episodes
    for e in range(max_episodes, np.max(episode_number)):
        print(f"Deleting superfluous episode {e}")
        superfluous_dir = os.path.join(logdir, f'episode_{e:05d}')
        superfluous_file = os.path.join(logdir, f'data_gen_model_{e:05d}.csv')
        superfluous_vid = os.path.join(logdir, f'sample_{e:05d}.mp4')
        if os.path.exists(superfluous_dir):
            shutil.rmtree(superfluous_dir)
            os.remove(superfluous_file)
            os.remove(superfluous_vid)

    # Combine data into one dataset
    print("Combining datasets")
    data = pd.DataFrame(columns=['global_step', 'episode'])
    list_of_ref_dfs = []

    max_global_step = 0
    for e in range(max_episodes):
        print(f"Creating indices for episode {e}")
        ref_df_e = pd.DataFrame(columns=['global_step', 'episode'])
        ref_df_e['global_step'] = \
            np.arange(max_global_step, max_global_step+episode_lens[e])
        ref_df_e['episode'] = (np.ones(int(episode_lens[e])) * e).astype(int)
        list_of_ref_dfs.append(ref_df_e)
        max_global_step = max_global_step + episode_lens[e]
    reference_df = pd.concat(list_of_ref_dfs)
    reference_df_name = os.path.join(logdir, f'idx_to_episode.csv')
    print("Saving idx csv...")
    reference_df.to_csv(reference_df_name, index=False)
    print("Saved")

    # Then go through all the individual episodes and fix the global step data
    for e in range(max_episodes):
        print(f"Fixing global step in episode {e}")
        epi_filename = os.path.join(logdir, f'data_gen_model_{e:05d}.csv')
        data_e = pd.read_csv(epi_filename)
        glob_steps_e = reference_df[reference_df['episode'] == e]['global_step']
        data_e['global_step'] = glob_steps_e
        data_e.to_csv(epi_filename)

    print("Done recording and processing data.")
