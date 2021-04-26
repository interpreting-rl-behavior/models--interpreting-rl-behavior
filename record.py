"""Makes a dataset for the generative model."""
import datetime

from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

import os, time, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch
import pandas as pd

if __name__=='__main__':
    start_time = time.time()
    secs_in_24h = 60*60*24

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

    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    #render parameters
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--logdir', type=str, default='generative/')

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

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    device = torch.device('cuda')

    # Environment
    print('INITIALIZAING ENVIRONMENTS...')
    def create_venv(args, hyperparameters, is_valid=False):
        venv = ProcgenEnv(num_envs=1,
                          env_name=args.env_name,
                          num_levels=0 if is_valid else args.num_levels,
                          start_level=0 if is_valid else args.start_level,
                          distribution_mode=args.distribution_mode,
                          use_backgrounds=False,
                          num_threads=1,)
        venv = VecExtractDictObs(venv, "rgb")
        normalize_rew = hyperparameters.get('normalize_rew', True)
        if normalize_rew:
            venv = VecNormalize(venv, ob=False) # normalizing returns, but not
            #the img frames
        venv = TransposeFrame(venv)
        venv = ScaledFloatFrame(venv)
        return venv

    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = 1
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

    data = pd.DataFrame(columns=column_names)

    ## Init params for training loop
    obs = agent.env.reset()
    hidden_state = np.zeros((agent.n_envs, agent.storage.hidden_state_size))
    done = np.zeros(agent.n_envs)

    global_steps = 0
    episode_steps = 0
    episode_number = 0

    max_episodes = 50


    ## Make dirs for files #TODO add some unique identifier so you don't end up with a bunch of partial episodes due to overwriting
    dir_name = logdir + 'episode' + str(episode_number)
    if os.path.exists(dir_name):
        raise UserWarning("You are overwriting your previous data! Delete " + \
                          "or move your old dataset first.")
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)

    obs_list = []
    hx_list = []
    logprob_list = []

    while True:
        agent.policy.eval()
        # for _ in range(agent.n_steps):

        # Step agent and environment
        act, log_prob_act, value, next_hidden_state = agent.predict_record(obs, hidden_state, done)
        next_obs, rew, done, info = agent.env.step(act)

        # Store non-array variables
        data = data.append({
            'level_seed': info[0]['level_seed'],
            'episode': episode_number,
            'global_step': global_steps,
            'episode_step': episode_steps,
            'done': done[0],
            'reward': rew[0],
            'value': value[0],
            'action': act[0],
        }, ignore_index=True)


        obs_list.append(obs)
        hx_list.append(hidden_state)
        logprob_list.append(log_prob_act)

        # Increment for next step
        obs = next_obs
        hidden_state = next_hidden_state
        global_steps += 1
        episode_steps += 1

        if done[0]:  # At end of episode
            data.to_csv(logdir + f'data_gen_model_{episode_number}.csv', index=False) #TODO change so that it doesn't get slower over time due to the growing size of the data csv. save each individually then combine once done.

            # Make dirs for files
            dir_name = os.path.join(logdir, 'episode' + str(episode_number))
            if not (os.path.exists(dir_name)):
                os.makedirs(dir_name)

            # Stack arrays for this episode into one array
            obs_array = np.stack(obs_list).squeeze()
            hx_array  = np.stack(hx_list).squeeze()
            lp_array  = np.stack(logprob_list).squeeze()

            # Prepare names for saving
            obs_name = dir_name + '/' + 'ob.npy'
            hx_name = dir_name + '/' + 'hx.npy'
            lp_name = dir_name + '/' + 'lp.npy'

            # Save stacked array
            np.save(obs_name, np.array(obs_array * 255, dtype=np.uint8))
            np.save(hx_name, hx_array)
            np.save(lp_name, lp_array)

            # Reset things for the beginning of the next episode
            data = pd.DataFrame(columns=column_names)
            print("Episode number: %i ;  Episode len: %i " % (episode_number, episode_steps))
            episode_number += 1
            episode_steps = 0

            obs_list = []
            hx_list = []
            logprob_list = []

            hidden_state = np.zeros_like(hidden_state) #New

        if (episode_number >= max_episodes) or \
                ((time.time() - start_time) > (secs_in_24h - 600)):
            break

        # _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)

    print("Combining datasets")
    data = pd.DataFrame(columns=column_names)
    for e in range(episode_number):
        epi_filename = logdir + f'data_gen_model_{e}.csv'
        data_e = pd.read_csv(epi_filename)
        data = data.append(data_e)
        os.remove(epi_filename)
    data.to_csv(logdir + f'data_gen_model.csv',
                index=False)