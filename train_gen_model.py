from common.env.procgen_wrappers import *
import util.logger as logger  # from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

import os, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch
from generative.generative_models import VAE, Discriminator
from generative.procgen_dataset import ProcgenDataset

from collections import deque
import torchvision.io as tvio
from datetime import datetime


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test',
                        help='experiment name')
    parser.add_argument('--tgm_exp_name', type=str, default='test',
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
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_recon_obs', type=int, default=8)
    parser.add_argument('--num_pred_steps', type=int, default=22)

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    # Hyperparameters
    args = parser.parse_args()
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    batch_size = args.batch_size
    num_recon_obs = args.num_recon_obs
    num_pred_steps = args.num_pred_steps
    total_seq_len = num_recon_obs + num_pred_steps

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
    n_steps = 1#hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    env = create_venv(args, hyperparameters)

    # Logger
    print('INITIALIZING LOGGER...')
    logdir_base = 'generative/'
    if not (os.path.exists(logdir_base)):
        os.makedirs(logdir_base)
    resdir = logdir_base + 'results/'
    if not (os.path.exists(resdir)):
        os.makedirs(resdir)
    resdir = resdir + '/' + args.tgm_exp_name
    if not (os.path.exists(resdir)):
        os.makedirs(resdir)

    # TODO put exp name as main dir then this as subdir
    gen_model_session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_dir = os.path.join(resdir, gen_model_session_name)
    if not (os.path.exists(sess_dir)):
        os.makedirs(sess_dir)
    logger.configure(dir=sess_dir, format_strs=['csv', 'stdout'])
    reconpred_dir = os.path.join(sess_dir, 'recons_v_preds')
    if not (os.path.exists(reconpred_dir)):
        os.makedirs(reconpred_dir)


    # Model
    print('INTIALIZING AGENT MODEL...')
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
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints,
                  **hyperparameters)
    if args.agent_file is not None:
        logger.info("Loading agent from %s" % args.agent_file)
        checkpoint = torch.load(args.agent_file, map_location=device)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Done loading agent.")

    # Generative model stuff:

    ## Make dataset
    train_dataset = ProcgenDataset('generative/data/data_gen_model.csv',
                                   total_seq_len=total_seq_len,
                                   inp_seq_len=num_recon_obs)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    ## Make or load generative model and optimizer
    gen_model = VAE(agent, device, num_recon_obs, num_pred_steps)
    discrim   = Discriminator(device)
    gen_model = gen_model.to(device)
    optimizer = torch.optim.Adam(list(gen_model.parameters()) + \
                                 list(discrim.parameters()), lr=args.lr)

    if args.model_file is not None:
        checkpoint = torch.load(args.model_file, map_location=device)
        gen_model.load_state_dict(checkpoint['gen_model_state_dict'], device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        discrim.load_state_dict(checkpoint['discrim_state_dict'], device)
        logger.info('Loaded generative model from {}.'.format(args.model_file))
    else:
        logger.info('Training generative model from scratch.')


    # Training
    ## Epoch cycle (Train, Validate, Save samples)
    for epoch in range(0, args.epochs + 1):

        train(epoch, args, train_loader, optimizer, gen_model, agent, discrim,
              logger, sess_dir, device)

        # Visualise random samples
        if epoch % 10 == 0:
            with torch.no_grad():
                viz_batch_size = 20
                vae_latent_size = 128
                samples = torch.randn(viz_batch_size, vae_latent_size).to(device)
                samples = torch.stack(gen_model.decoder(samples,
                                                        true_actions=None)[0], dim=1)
                for b in range(viz_batch_size):
                    sample = samples[b].permute(0, 2, 3, 1)
                    sample = sample * 255
                    sample = sample.clone().detach().type(torch.uint8).cpu().numpy()
                    save_str = sess_dir + '/sample_' + str(
                        epoch) + '_' + str(epoch) + '_' + str(
                        b) + '.mp4'
                    tvio.write_video(save_str, sample, fps=14)

        # Demonstrate reconsruction and prediction quality by comparing preds
        # with ground truth (the closes thing a VAE gets to validation because
        # there are no labels).
        demo_recon_quality(epoch, args, train_loader, optimizer, gen_model, agent, logger, sess_dir,
              device)


def loss_function(preds, labels, mu, logvar, train_info_bufs, discrim, device):
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

    loss_hyperparams = {'obs': 7.0,
                        'hx': 0.7,
                        'reward': 0.2,
                        'done': 0.2,
                        'act_log_probs': 0.2,
                        'discrim': 1.0}
    batch_size = labels['reward'].shape[0]

    # Reconstruction loss
    losses = []
    for key in preds.keys():
        #if key in ['obs']: # for debugging only
        pred  = torch.stack(preds[key], dim=1).squeeze()
        label = labels[key].to(device).float().squeeze()
        if key == 'obs':
            # # Calculate a mask to exclude loss on 'zero' observations
            # mask = torch.ones_like(labels['done']) - labels['done'] # excludes done timesteps
            # for b, argmin in enumerate(torch.argmax(labels['done'], dim=1)):
            #     mask[b, argmin] = 1. # unexcludes the first 'done' timestep
            # mask = mask.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) # so it can be broadcast to same shape as loss sum

            # Calculate loss
            label = label / 255.
            loss = torch.abs(pred - label)
            # loss = loss * mask
            loss = torch.mean(loss)  # Mean Absolute Error
        else:
            loss = torch.mean(torch.abs(pred - label))  # Mean Absolute Error

        #mse = F.mse_loss(pred, label) # TODO test whether MSE or MAbsE is better (I think the VQ-VAE2 paper suggested MAE was better)
        train_info_bufs[key].append(loss.item())
        loss = loss * loss_hyperparams[key]
        losses.append(loss)

    loss = sum(losses)
    train_info_bufs['total recon w/o KL'].append(loss.item())

    # Do adversarial loss

    ## Make labels
    discrim_labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)], dim=0).to(device)
    pred_obs = torch.stack(preds['obs'], dim=0).permute([1,0,2,3,4])
    discrim_inps = torch.cat([pred_obs, labels['obs']], dim=0)

    ## Feed observations into discriminator and get loss
    discrim_preds = discrim(discrim_inps)
    fake_data_preds = discrim_preds[:batch_size]
    real_data_preds = discrim_preds[batch_size:]

    discrim_loss = torch.mean(torch.log(real_data_preds)) + \
                   torch.mean(torch.log(torch.ones_like(fake_data_preds) - \
                                        fake_data_preds))
    # discrim_loss = discrim_labels * (-1*torch.log(discrim_preds)) + \
    #                 (torch.ones_like(discrim_labels) - discrim_labels) * \
    #                 (-1*torch.log(torch.ones_like(discrim_preds) - discrim_preds))
    discrim_loss = discrim_loss.mean()

    ## Log the discriminator loss
    train_info_bufs['Discrim'].append(discrim_loss.item())

    discrim_loss = discrim_loss * loss_hyperparams[key]

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(),
                                     dim=1) # Sum along latent size dim.
    kl_divergence = torch.mean(kl_divergence)  # Mean along batch dim
    train_info_bufs['KL'].append(kl_divergence.item())

    return loss + kl_divergence - discrim_loss, train_info_bufs

def train(epoch, args, train_loader, optimizer, gen_model, agent, discrim, logger, save_dir, device):
    # Set up logging objects
    loss_keys = ['obs', 'hx', 'done', 'reward', 'act_log_probs', 'KL', 'total recon w/o KL', 'Discrim']
    train_info_bufs = {k:deque(maxlen=100) for k in loss_keys}
    logger.info('Start training epoch {}'.format(epoch))

    # Prepare for training cycle
    gen_model.train()

    # Training cycle
    for batch_idx, data in enumerate(train_loader):

        data = {k: v.to(device).float() for k, v in data.items()}

        # Get input data for generative model (only taking inp_seq_len timesteps)
        obs = data['obs'][:, 0:train_loader.dataset.inp_seq_len]
        agent_hx = data['hx'][:, 0:train_loader.dataset.inp_seq_len]
        actions_all = data['action']

        # Forward and backward pass and upate generative model parameters
        optimizer.zero_grad()
        mu, logvar, preds = gen_model(obs, agent_hx, actions_all,
                                      use_true_actions=True)
        loss, train_info_bufs = loss_function(preds, data, mu, logvar, train_info_bufs, discrim, device)
        for p in gen_model.decoder.agent.policy.parameters():
            if p.grad is not None:  # freeze agent parameters but not model's.
                p.grad.data = torch.zeros_like(p.grad.data)
        loss.backward()
        optimizer.step()

        # Logging and saving info
        if batch_idx % args.log_interval == 0:
            loss.item()
            logger.logkv('epoch', epoch)
            logger.logkv('batches', batch_idx)
            for key in loss_keys:
                logger.logkv('loss/%s' % key,
                             safe_mean([loss for loss in train_info_bufs[key]]))
            logger.dumpkvs()

        # Saving model
        if batch_idx % args.save_interval == 0:
            model_path = os.path.join(
                save_dir,
                'model_epoch{}_batch{}.pt'.format(epoch, batch_idx))
            torch.save({'gen_model_state_dict': gen_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'discrim_state_dict': discrim.state_dict()},
                       model_path)
            logger.info('Generative model saved to {}'.format(model_path))

        # viz for debugging only
        if batch_idx % 1000 == 0 or (epoch < 2 and batch_idx % 200 == 0):
            with torch.no_grad():
                viz_batch_size = 20
                vae_latent_size = 128
                samples = torch.randn(viz_batch_size, vae_latent_size).to(
                    device)
                samples = torch.stack(gen_model.decoder(samples,
                                                        true_actions=None)[0], dim=1)
                for b in range(viz_batch_size):
                    sample = samples[b].permute(0, 2, 3, 1)
                    sample = sample * 255
                    sample = sample.clone().detach().type(
                        torch.uint8).cpu().numpy()
                    save_str = save_dir + '/sample_' + str(
                        epoch) + '_' + str(batch_idx) + '_' + str(
                        b) + '.mp4'
                    tvio.write_video(save_str, sample, fps=14)


def demo_recon_quality(epoch, args, train_loader, optimizer, gen_model, agent, logger,
          save_dir, device):
    # Set up logging objects
    logger.info('Demonstrating reconstruction and prediction quality')

    # Prepare for training cycle
    gen_model.train()

    # Plot both the inputs and their reconstruction/prediction for one batch
    for batch_idx, data in enumerate(train_loader):
        if batch_idx > 0: # Just do one batch
            break
        data = {k: v.to(device).float() for k,v in data.items()}

        # Get input data for generative model (only taking inp_seq_len
        # timesteps)
        full_obs = data['obs']
        inp_obs = full_obs[:, 0:train_loader.dataset.inp_seq_len]
        agent_hx = data['hx'][:, 0:train_loader.dataset.inp_seq_len]
        actions_all = data['action']

        # Forward pass to get predicted observations
        optimizer.zero_grad()
        mu, logvar, preds = gen_model(inp_obs, agent_hx, actions_all,
                                      use_true_actions=True)

        with torch.no_grad():
            pred_obs = torch.stack(preds['obs'], dim=1).squeeze()

            viz_batch_size = 20
            viz_batch_size = min(int(pred_obs.shape[0]), viz_batch_size)

            for b in range(viz_batch_size):
                pred_ob = pred_obs[b].permute(0, 2, 3, 1)
                full_ob = full_obs[b].permute(0, 2, 3, 1)

                pred_ob = pred_ob * 255  # full_ob is already in [0,255] ) Z^+

                pred_ob = pred_ob.clone().detach().type(torch.uint8).cpu().numpy()
                full_ob = full_ob.clone().detach().type(torch.uint8).cpu().numpy()

                # Join the prediction and the true observation side-by-side
                combined_ob = np.concatenate([pred_ob, full_ob], axis=2)

                # Save vid
                save_str = save_dir + '/recons_v_preds' + '/sample_' + str(epoch) + '_' + str(batch_idx) + '_' + str(
                    b) + '.mp4'
                tvio.write_video(save_str, combined_ob, fps=14)


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

if __name__ == "__main__":
    run()