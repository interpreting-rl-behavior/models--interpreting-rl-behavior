from common.env.procgen_wrappers import *
# from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

import os, time, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch
from generative.generative_models import VAE
from generative.procgen_dataset import ProcgenDataset

from collections import deque
import torchvision.io as tvio
from torch.nn import functional as F
import util.logger as logger



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test',
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
    parser.add_argument('--param_name', type=str, default='easy-200',
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
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_recon_obs', type=int, default=8)
    parser.add_argument('--num_pred_steps', type=int, default=22)


    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    args = parser.parse_args()
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
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    # Environment # Not used, just for initializing agent.
    print('INITIALIZING ENVIRONMENTS...')

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
    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    env = create_venv(args, hyperparameters)

    # Logger
    print('INITIALIZING LOGGER...')
    logdir_base = 'generative/'
    if not (os.path.exists(logdir_base)):
        os.makedirs(logdir_base)
    logdir = logdir_base + 'results/'
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger.configure(dir=logdir, format_strs=['csv', 'stdout'])

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
    print('INITIALIZING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs,
                      device)

    # Agent
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)
    if args.agent_file is not None:
        logger.info("Loading agent from %s" % args.agent_file)
        checkpoint = torch.load(args.agent_file)
        agent.policy.load_state_dict(checkpoint["state_dict"])
        # agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Done loading agent.")
    # Generative model stuff:

    ## Make (fake) dataset
    train_loader = None
    batch_size = args.batch_size
    num_recon_obs = args.num_recon_obs
    num_pred_steps = args.num_pred_steps
    total_seq_len = num_recon_obs + num_pred_steps
    train_dataset = ProcgenDataset('generative/data/data_gen_model.csv',
                                   total_seq_len=total_seq_len,
                                   inp_seq_len=num_recon_obs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    # train_loader = get_fake_data_and_labels(num_obs=1000,
    #                                         num_inputs=num_recon_obs,
    #                                         total_seq_len=total_seq_len,
    #                                         obs_size=(3,64,64),
    #                                         act_space_size=15,
    #                                         agent_h_size=256,
    #                                         batch_size=batch_size)
    ## Make or load generative model
    gen_model = VAE(agent, device, num_recon_obs, num_pred_steps)
    gen_model = gen_model.to(device)

    if args.model_file is not None:
        gen_model.load_from_file(args.model_file, device)
        logger.info('Loaded generative model from {}.'.format(args.model_file))
    else:
        logger.info('Training generative model from scratch.')

    # Training
    ## Epoch cycle (Train, Validate, Save samples)
    for epoch in range(0, args.epochs + 1):
        train(epoch, args, train_loader, gen_model, agent, logger, logdir,
              device)
        # test(epoch) # TODO validation step
        if epoch % 10 == 0:
            with torch.no_grad():
                samples = torch.randn(20, 256).to(device)
                samples = torch.stack(gen_model.decoder(samples)[0], dim=1)
                for b in range(batch_size):
                    sample = samples[b].permute(0, 2, 3, 1)
                    sample = sample - torch.min(sample)
                    sample = sample / torch.max(sample) * 255
                    sample = sample.clone().detach().type(torch.uint8).cpu().numpy()
                    save_str = 'generative/results/sample_' + str(epoch) + '_' + str(
                        b) + '.mp4'
                    tvio.write_video(save_str, sample, fps=8)
            # TODO save target sequences and compare to predicted sequences

def loss_function(preds, labels, mu, logvar, device):
    """ Calculates the difference between predicted and actual:
        - observation
        - agent's recurrent hidden states
        - agent's logprobs
        - rewards
        - 'done' status

        If this is insufficient to produce high quality samples, then we'll
        add the attentive mask described in Rupprecht et al. (2019). And if
        _that_ is still insufficient, then we'll look into adding a GAN
        discriminator and loss term.
      """

    # Mean Squared Error
    mses = []
    for key in preds.keys():
        pred  = torch.stack(preds[key], dim=1).squeeze()
        label = labels[key].to(device).float()
        mse = F.mse_loss(pred, label) # TODO test whether MSE or MAbsE is better (I think the VQ-VAE2 paper suggested MAE was better)
        mses.append(mse)

    mse = sum(mses)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse + kl_divergence

def train(epoch, args, train_loader, gen_model, agent, logger, log_dir, device):

    # Set up logging objects
    train_info_buf = deque(maxlen=100)
    logger.info('Start training epoch {}'.format(epoch))

    # Prepare for training cycle
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr) # ensure that the agent params don't update
    gen_model.train()


    # Training cycle
    for batch_idx, data in enumerate(train_loader):

        data = {k: v.to(device).float() for k,v in data.items()}

        # Get input data for generative model
        obs = data['obs'][:,0:train_loader.dataset.inp_seq_len]#gets first few timesteps
        agent_h0 = data['hx'][:,0,:]

        # Forward and backward pass and upate generative model parameters
        optimizer.zero_grad()
        mu, logvar, preds = gen_model(obs, agent_h0)
        loss = loss_function(preds, data, mu, logvar, device)
        for p in gen_model.decoder.agent.policy.parameters():
            if p.grad is not None:  # freeze agent parameters but not model's.
                p.grad.data = torch.zeros_like(p.grad.data)
        loss.backward()
        optimizer.step()

        # Logging and saving info
        train_info_buf.append(loss.item())
        if batch_idx % args.log_interval == 0:
            loss.item()
            logger.logkv('epoch', epoch)
            logger.logkv('batches', batch_idx)
            logger.logkv('loss',
                         safe_mean([loss for loss in train_info_buf]))
            logger.dumpkvs()

        # Saving model
        if batch_idx % args.save_interval == 0:
            model_path = os.path.join(
                log_dir, 'model_epoch{}_batch{}.pt'.format(epoch, batch_idx))
            torch.save({'gen_model_state_dict': gen_model.state_dict()},
                       model_path)
            logger.info('Generative model saved to {}'.format(model_path))

def get_fake_data_and_labels(num_obs, num_inputs, total_seq_len, obs_size, act_space_size, agent_h_size, batch_size):
    """
    Notes from proposal doc:
    Data:
        K-sized minibatch where each element contains:
            (J*observations from timestep T-J to T-1) and
            initial recurrent states at timestep T-J, where:
                J = the number of observations that we're feeding into the VAE
                T = the 0th timestep that we're actually predicting (i.e. the first timestep in the future)
                So we're feeding observations from timestep (T-J):(T-1) (inclusive) into the VAE
                And we also want the VAE to produce initial hidden states at timestep T-J
            Action log probabilities (vector)
            Action (integer)
            Agent value function output (scalar)
            Reward at time t (scalar)
            Timestep (integer)
            Episode number (integer)
            ‘Done’ (boolean/integer)
            Level seed (will help us reinstantiate the same levels later if we want to study them.
    Labels:
        Reconstruction part:
            the observations (same as data)
            The initial hidden states (same as data)
        Prediction part:
            k*(L*observations from timestep T to T+L))
    ############################################################
    Args:
        num_obs: total number of observations in the dataset
        total_seq_len: the number of observations in each sequence. (=J+L in the docstring)
        obs_size: (tuple) - size of each observation (I think for coinrun we will use 64x64x3?)
        act_space_size: number of actions available (15 for coinrun?)
        batch_size: the number of VAE inputs in each batch
    Returns:
        batches of data of size batch_size
    """

    num_batches = num_obs // total_seq_len // batch_size
    obs_seq_size = (batch_size, total_seq_len,) + obs_size  # (B, T, C, H, W)

    data_and_labels = []
    for batch in range(num_batches):
        actions, timestep, episode, done, lvl_seed = \
            [torch.randint(low=0, high=act_space_size,
                           size=(batch_size, total_seq_len)) for _ in range(5)]
        values, reward = \
            [torch.randn(batch_size, total_seq_len) for _ in range(2)]
        obs = torch.rand(obs_seq_size)
        rec_h_state = torch.randn(batch_size, total_seq_len, agent_h_size)
        # Assumes that we have 1 action for every observation in our data.
        action_log_probs = torch.randn(batch_size, total_seq_len,act_space_size)

        # Since enumerating trainloader returns batch_idx, (data,_), we make this a tuple.
        labels = {
            'actions': actions, 'values': values, 'reward': reward, 'timestep': timestep,
            'episode': episode, 'done': done, 'lvl_seed': lvl_seed, 'obs': obs, 'rec_h_state': rec_h_state,
            'action_log_probs': action_log_probs
        }

        # Just get first `num_inputs' elements of sequence for input to the VAE
        loss_keys = ['reward', 'obs', 'rec_h_state', 'action_log_probs']
        data = {}
        for key in loss_keys:
            data.update({key: labels[key][:,0:num_inputs]})

        data_and_labels.append((data, labels))
    return data_and_labels

def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

if __name__ == "__main__":
    run()