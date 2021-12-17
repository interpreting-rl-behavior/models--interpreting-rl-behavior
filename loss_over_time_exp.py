from common.env.procgen_wrappers import *
import util.logger as logger  # from common.logger import Logger
from util.parallel import DataParallel
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

import os, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch
from generative.generative_models import VAE
from generative.procgen_dataset import ProcgenDataset

from collections import deque
import torchvision.io as tvio
from datetime import datetime
import json


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test',
                        help='experiment name')
    parser.add_argument('--tgm_exp_name', type=str, default='test_tgm',
                        help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun',
                        help='environment ID')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train the generative model')
    parser.add_argument('--start_level', type=int, default=int(0),
                        help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(0),
                        help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy',
                        help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='hard-rec',
                        help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='gpu', required=False,
                        help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0),
                        required=False, help='visible device in CUDA')
    parser.add_argument('--num_timesteps', type=int, default=int(25000000),
                        help='number of training timesteps')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999),
                        help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40),
                        help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1),
                        help='number of checkpoints to store')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--agent_file', type=str)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_initializing_steps', type=int, default=8)
    parser.add_argument('--num_sim_steps', type=int, default=22)
    parser.add_argument('--layer_norm', type=int, default=0)

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    # Loss function hyperparams
    parser.add_argument('--loss_scale_obs', type=float, default=1.)
    parser.add_argument('--loss_scale_hx', type=float, default=1.)
    parser.add_argument('--loss_scale_reward', type=float, default=1.)
    parser.add_argument('--loss_scale_done', type=float, default=1.)
    parser.add_argument('--loss_scale_act_log_probs', type=float, default=1.)
    parser.add_argument('--loss_scale_gen_adv', type=float, default=1.)
    parser.add_argument('--loss_scale_kl', type=float, default=1.)


    # Set hyperparameters
    args = parser.parse_args()
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints
    batch_size = args.batch_size
    num_initializing_steps = args.num_initializing_steps
    num_sim_steps = args.num_sim_steps
    total_seq_len = num_initializing_steps + num_sim_steps - 1
    # minus one because the first simulated observation is the last
    # initializing context obs.

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    # Set up environment (Only used for initializing agent)
    print('INITIALIZING ENVIRONMENTS...')
    n_steps = 1#hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    env = create_venv(args, hyperparameters)

    # Make save dirs
    print('INITIALIZING LOGGER...')
    save_dir_base = 'analysis/'
    if not (os.path.exists(save_dir_base)):
        os.makedirs(save_dir_base)
    save_dir = os.path.join(save_dir_base, 'loss_over_time')
    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)

    gen_model_session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    # sess_dir = os.path.join(save_dir, gen_model_session_name)
    # if not (os.path.exists(sess_dir)):
    #     os.makedirs(sess_dir)

    # Logger
    logger.configure(dir=save_dir, format_strs=['csv', 'stdout'])


    # Set up agent
    print('INTIALIZING AGENT MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    ## Agent architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)

    ## Agent's discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_space_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_space_size)
    else:
        raise NotImplementedError
    policy.to(device)

    ## Agent's storage
    print('INITIALIZING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs,
                      device)

    ## And, finally, the agent itself
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints,
                  **hyperparameters)
    if args.agent_file is not None:
        logger.info("Loading agent from %s" % args.agent_file)
        checkpoint = torch.load(args.agent_file, map_location=device)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Done loading agent.")

    # Set up generative model
    ## Make dataset
    train_dataset = ProcgenDataset(args.data_dir,
                                   initializer_seq_len=num_initializing_steps,
                                   total_seq_len=total_seq_len,)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    ## Make or load generative model and optimizer
    gen_model = VAE(agent, device, num_initializing_steps, total_seq_len)
    gen_model = DataParallel(gen_model)
    gen_model = gen_model.to(device)
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr)

    if args.model_file is not None:
        checkpoint = torch.load(args.model_file, map_location=device)
        gen_model.load_state_dict(checkpoint['gen_model_state_dict'], device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info('Loaded generative model from {}.'.format(args.model_file))
    else:
        logger.info('Training generative model from scratch.')


    with torch.no_grad():
        losses = train(args, train_loader, optimizer, gen_model, device)

    with open(f"analysis/loss_over_time/{gen_model_session_name}.json", 'w') as f:
        json.dump(losses, f, indent=4)


def train(args, train_loader, optimizer, gen_model, device):

    # Set up logging queue objects
    loss_keys = ['obs', 'hx', 'done', 'reward', 'act_log_probs', 'total recon w/o KL']

    train_info_bufs = {k: [] for k in loss_keys}

    # Prepare for training cycle
    gen_model.train()

    # Training cycle
    for batch_idx, data in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}/{len(train_loader)}")
        if batch_idx == 500:
            break

        # Make all data into floats and put on the right device
        data = {k: v.to(device).float() for k, v in data.items()}

        # Get input data for generative model
        full_obs = data['obs']
        agent_h0 = data['hx'][:, -args.num_sim_steps, :]
        actions_all = data['action'][:, -args.num_sim_steps:]

        # Forward and backward pass and update generative model parameters
        optimizer.zero_grad()
        mu_c, logvar_c, mu_g, logvar_g, preds = gen_model(full_obs, agent_h0, actions_all,
                                                          use_true_h0=True,
                                                          use_true_actions=True)
        train_info_bufs = loss_function(args, preds, data, train_info_bufs, device)
    
    losses = {}
    for key in loss_keys:
        loss_tensor = torch.cat(train_info_bufs[key])
        losses[key + '_mean'] = torch.mean(loss_tensor, dim=0).tolist()
        losses[key + '_max'] = torch.max(loss_tensor,  dim=0).values.tolist()
        losses[key + '_min'] = torch.min(loss_tensor,  dim=0).values.tolist()

    return losses
       
def loss_function(args, preds, labels, train_info_bufs, device):

    loss_hyperparams = {'obs': args.loss_scale_obs,
                        'hx': args.loss_scale_hx,
                        'reward': args.loss_scale_reward,
                        'done': args.loss_scale_done,
                        'act_log_probs': args.loss_scale_act_log_probs}
    bs = args.batch_size
    loss = torch.zeros(bs, args.num_sim_steps).to(device)

    for key in train_info_bufs.keys():
        if key == 'total recon w/o KL': # Not using values for loss
            continue
        pred  = torch.stack(preds[key], dim=1).squeeze()
        label = labels[key].to(device).float().squeeze()
        label = label[:, -args.num_sim_steps:]

        # Get the mean value over all dims except for batch and num_sim_steps (dim 1)
        mean_dims = list(range(len(pred.shape)))[2:]
        # Get MSE for this key
        if mean_dims:  # ensures it doesn't mean for empty dims list
            key_loss = torch.mean((pred - label) ** 2, dim=mean_dims)
        else:
            key_loss = (pred - label) ** 2
        # scaled_loss = key_loss * loss_hyperparams[key]

        # Currently saves the scaled loss, not the raw loss
        train_info_bufs[key].append(key_loss)
        loss += key_loss

    train_info_bufs['total recon w/o KL'].append(loss)
    return train_info_bufs

def create_venv(args, hyperparameters, is_valid=False):
    venv = ProcgenEnv(num_envs=hyperparameters.get('n_envs', 256),
                      env_name=args.env_name,
                      num_levels=0 if is_valid else args.num_levels,
                      start_level=0 if is_valid else args.start_level,
                      distribution_mode=args.distribution_mode,
                      num_threads=args.num_threads)
    venv = VecExtractDictObs(venv, "rgb")
    normalize_rew = hyperparameters.get('normalize_rew', True)
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv

if __name__ == "__main__":
    run()