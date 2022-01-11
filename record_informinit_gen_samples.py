import numpy as np

from common.env.procgen_wrappers import *
import util.logger as logger  # from common.logger import Logger
from util.parallel import DataParallel
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from train import create_venv
from overlay_image import overlay_actions

import os, yaml, argparse
import gym
import random
import torch
from generative.generative_models import VAE
from generative.procgen_dataset import ProcgenDataset

from collections import deque
import torchvision.io as tvio
from datetime import datetime


def run():
    """This script `record_gen_samples.py' uses random vectors to generate a
    library of samples that we can manually sort through to identify samples
    with specific behaviours and features. It records the hx and env and
    other variables so that we can make target functions that optimize for
    those behaviours and features.

    It saves the obs, hx, env_hx, etc. in a unique folder for that sample.

    e.g. directory name 'generative/recorded_gen_samples/sample_00001' contains obs.npy, hx.npy, env_hx.npy

    and the directory 'generative/recorded_gen_samples' contains the videos
    of the samples. There will also be a manually managed csv file that marks
    each of the thousands of samples with binary markers if they contain certain
    behaviours.

    This script can also test that the environment model is independent of the agent used to train it.
    Simply set the cmd arg --manual_action to a direction you wish to set all actions to in the
    decoder. This will run the gen model but force the agent to take this action at each timestep.
    Note that you may wish to set the --data_save_dir to the place you would wish to output the
    video samples.
    """
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--data_save_dir', type=str, default='generative/recorded_informinit_gen_samples')
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_initializing_steps', type=int, default=3)
    parser.add_argument('--num_sim_steps', type=int, default=28)
    parser.add_argument('--layer_norm', type=int, default=0)
    parser.add_argument('--swap_directions_from', nargs='+')
    parser.add_argument('--swap_directions_to', nargs='+')
    parser.add_argument('--samples_to_record', type=int, default=200)
    parser.add_argument('--manual_action', type=str,
                        choices=['downleft', 'left', 'upleft', 'down', 'up', 'downright', 'right', 'upright'],
                        help='Action direction to manually set for all timesteps (upright, left, ...)')
    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

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

    if args.samples_to_record % batch_size > 0:
        raise ValueError("Samples to record should be an integer multiple of "+\
                         "batch size")
    num_epochs = args.samples_to_record // batch_size

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    n_steps = 1
    n_envs = hyperparameters.get('n_envs', 64)

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    # Set up environment (Only used for initializing agent)
    print('INITIALIZING ENVIRONMENTS...')
    env = create_venv(args, hyperparameters)

    # Make save dirs
    data_dir = args.data_save_dir
    if not (os.path.exists(data_dir)):
        os.makedirs(data_dir)

    # Logger
    print('INITIALIZING LOGGER...')
    logger.configure(dir=data_dir, format_strs=['csv', 'stdout'])

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

    # Make dataset
    train_dataset = ProcgenDataset(args.data_dir,
                                   initializer_seq_len=num_initializing_steps,
                                   total_seq_len=total_seq_len,)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    # Set up generative model
    ## Make or load generative model
    gen_model = VAE(agent, device, num_initializing_steps, total_seq_len)
    gen_model = DataParallel(gen_model)
    gen_model = gen_model.to(device)

    if args.model_file is not None:
        checkpoint = torch.load(args.model_file, map_location=device)
        gen_model.load_state_dict(checkpoint['gen_model_state_dict'], device)
        logger.info('Loaded generative model from {}.'.format(args.model_file))
    else:
        logger.info('Using untrained generative model.')

    # Training
    ## Epoch cycle (Train, Save visualized random samples, Demonstrate
    ##   reconstruction quality)
    for epoch in range(0, num_epochs):
        data = next(iter(train_loader))
        record_gen_samples(epoch, args, gen_model, batch_size,
                           agent, data, logger, data_dir, device)


def action_str_to_int(action):
    """
    Converts a string representing the action direction to the integer that takes that action
    in procgen.
    """

def record_gen_samples(epoch, args, gen_model, batch_size, agent, data, logger, data_dir, device):

    # Set up logging queue objects
    loss_keys = ['obs', 'hx', 'done', 'reward', 'act_log_probs', 'KL',
                 'total recon w/o KL']
    train_info_bufs = {k:deque(maxlen=100) for k in loss_keys}
    logger.info('Samples recorded: {}'.format(epoch * batch_size))

    # Prepare for training cycle
    gen_model.eval()

    coinrun_actions = {'downleft': 0, 'left': 1, 'upleft': 2, 'down': 3, 'noop':4, 'up': 5, 'downright': 6,
                       'right': 7, 'upright': 8}

    if args.manual_action:
        manual_action = coinrun_actions[args.manual_action]
        use_true_actions = True
    else:
        manual_action = None
        # Use the predicted actions from the gen model if not using manual actions
        use_true_actions = False

    # Recording cycle
    with torch.no_grad():

        # Make all data into floats and put on the right device
        data = {k: v.to(device).float() for k, v in data.items()}

        # Get input data for generative model
        full_obs = data['obs']
        agent_h0 = data['hx'][:, -args.num_sim_steps, :]
        actions_all = data['action'][:, -args.num_sim_steps:]

        if args.manual_action is not None:
            actions_all.fill_(manual_action)

        # If doing validation experiments that swap or collapse directions, put
        # the arguments into the right format.
        if args.swap_directions_from is not None:
            assert len(args.swap_directions_from) == \
                   len(args.swap_directions_to)
            from_dircs = []
            to_dircs = []
            # Convert from strings into the right type (int or None)
            for from_dirc, to_dirc in zip(args.swap_directions_from,
                                        args.swap_directions_to):
                if from_dirc == 'None':
                    from_dircs.append(None)
                else:
                    from_dircs.append(int(from_dirc))
                if to_dirc == 'None':
                    to_dircs.append(None)
                else:
                    to_dircs.append(int(to_dirc))
            swap_directions = [from_dircs, to_dircs]
        else:
            swap_directions = None
        # Forward pass of generative model
        _, _, _, _, preds = gen_model(full_obs, agent_h0, actions_all,
                                                          use_true_h0=False,
                                                          use_true_actions=use_true_actions,
                                                          swap_directions=swap_directions)
        # Both of these were false but for some reason it doesn't appear to be
        #  leading to the same distribution of hxs

        pred_obs = preds['obs']
        pred_rews = preds['reward']
        pred_dones = preds['done']
        pred_agent_hxs = preds['hx']
        pred_agent_logprobs = preds['act_log_probs']
        pred_agent_values = preds['values']
        env_rnn_states = preds['env_hx']
        bottleneck_vecs = preds['bottleneck_vec']

        # Stack samples into single tensors and convert to numpy arrays
        pred_obs = np.array(torch.stack(pred_obs, dim=1).cpu().numpy() * 255, dtype=np.uint8)
        pred_rews = torch.stack(pred_rews, dim=1).cpu().numpy()
        pred_dones = torch.stack(pred_dones, dim=1).cpu().numpy()
        pred_agent_hxs = torch.stack(pred_agent_hxs, dim=1).cpu().numpy()
        pred_agent_logprobs = torch.stack(pred_agent_logprobs, dim=1).cpu().numpy()
        pred_agent_values = torch.stack(pred_agent_values, dim=1).cpu().numpy()
        pred_env_hid_states = torch.stack(env_rnn_states[0], dim=1).cpu().numpy()
        pred_env_cell_states = torch.stack(env_rnn_states[1], dim=1).cpu().numpy()

        # no timesteps in latent vecs, so only cat not stack along time dim.
        bottleneck_vecs = torch.cat(bottleneck_vecs, dim=1).cpu().numpy()

        vars = [pred_obs, pred_rews, pred_dones, pred_agent_hxs,
                pred_agent_logprobs, pred_agent_values, pred_env_hid_states,
                pred_env_cell_states, bottleneck_vecs]
        var_names = ['obs', 'rews', 'dones', 'agent_hxs',
                     'agent_logprobs', 'agent_values', 'env_hid_states',
                     'env_cell_states', 'bottleneck_vec']

        # Recover the actions for use in the action overlay
        if manual_action is not None:
            actions = np.ones((args.batch_size, args.num_sim_steps)) * manual_action
        else:
            actions = np.argmax(pred_agent_logprobs, axis=-1)

        # Make dirs for these variables and save variables to dirs and save vid
        samples_so_far = epoch * batch_size
        new_sample_indices = range(samples_so_far, samples_so_far + batch_size)
        sample_dir_base = os.path.join(data_dir, 'sample_')
        for i, new_sample_idx in enumerate(new_sample_indices):
            sample_dir = sample_dir_base + f'{new_sample_idx:05d}'
            # Make dirs
            if not (os.path.exists(sample_dir)):
                os.makedirs(sample_dir)

            # Save variables
            for var, var_name in zip(vars, var_names):
                var_sample_name = os.path.join(sample_dir, var_name + '.npy')
                np.save(var_sample_name, var[i])

            # Save vid
            ob = torch.tensor(pred_obs[i])
            ob = ob.permute(0, 2, 3, 1)
            ob = ob.clone().detach().type(torch.uint8)
            ob = ob.cpu().numpy()
            # Overlay a square in the top right showing the agent's actions
            ob = overlay_actions(ob, actions[i], size=16)
            save_str = data_dir + '/sample_' + f'{new_sample_idx:05d}.mp4'
            tvio.write_video(save_str, ob, fps=14)

    # TODO merge this script with record_random_gen_samples but just use
    #  conditionals instead of having multiple scripts. Use git diff

def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def done_labels_to_mask(dones, num_unsqueezes=0):
    argmax_dones = torch.argmax(dones, dim=1)
    before_dones = torch.ones_like(dones)
    for batch, argmax_done in enumerate(argmax_dones):
        if argmax_done > 0:
            before_dones[batch, argmax_done + 1:] = 0

    # Applies unsqueeze enough times to produce a tensor of the same
    # order as the masked tensor. It can therefore be broadcast to the
    # same shape as the masked tensor
    unsqz_lastdim = lambda x: x.unsqueeze(dim=-1)
    for _ in range(num_unsqueezes):
        before_dones = unsqz_lastdim(before_dones)

    return before_dones


if __name__ == "__main__":
    run()