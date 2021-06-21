from common.env.procgen_wrappers import *
import util.logger as logger  # from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from train import create_venv

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
    print('INITIALIZING LOGGER...')
    logdir_base = 'generative/'
    if not (os.path.exists(logdir_base)):
        os.makedirs(logdir_base)
    resdir = logdir_base + 'results/'
    if not (os.path.exists(resdir)):
        os.makedirs(resdir)
    resdir = resdir + args.tgm_exp_name
    if not (os.path.exists(resdir)):
        os.makedirs(resdir)
    gen_model_session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_dir = os.path.join(resdir, gen_model_session_name)
    if not (os.path.exists(sess_dir)):
        os.makedirs(sess_dir)

    # Logger
    logger.configure(dir=sess_dir, format_strs=['csv', 'stdout'])
    reconpred_dir = os.path.join(sess_dir, 'recons_v_preds')
    if not (os.path.exists(reconpred_dir)):
        os.makedirs(reconpred_dir)

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
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
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
                                               num_workers=2)

    ## Make or load generative model and optimizer
    gen_model = VAE(agent, device, num_initializing_steps, total_seq_len)

    gen_model = gen_model.to(device)
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr)

    if args.model_file is not None:
        checkpoint = torch.load(args.model_file, map_location=device)
        gen_model.load_state_dict(checkpoint['gen_model_state_dict'], device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info('Loaded generative model from {}.'.format(args.model_file))
    else:
        logger.info('Training generative model from scratch.')

    # Training
    ## Epoch cycle (Train, Save visualized random samples, Demonstrate
    ##     reconstruction quality)
    for epoch in range(0, args.epochs + 1):

        train(epoch, args, train_loader, optimizer, gen_model, agent,
              logger, sess_dir, device)

        # Save visualized random samples
        if epoch % 10 == 0 and epoch >= 1:
            with torch.no_grad():
                viz_batch_size = 20
                vae_latent_size = 128
                samples = torch.randn(viz_batch_size, vae_latent_size)
                samples = samples.to(device)
                z_c, z_g = torch.split(samples, split_size_or_sections=64, dim=1)
                samples = gen_model.decoder(z_c, z_g, true_actions=None)[0]
                samples = torch.stack(samples, dim=1)
                for b in range(viz_batch_size):
                    sample = samples[b].permute(0, 2, 3, 1)
                    sample = sample * 255
                    sample = sample.clone().detach().type(torch.uint8)
                    sample = sample.cpu().numpy()
                    save_str = sess_dir + '/generated_sample_' + str(epoch) + '_' + str(b) + '.mp4'
                    tvio.write_video(save_str, sample, fps=14)

        # Demonstrate reconsruction and prediction quality by comparing preds
        # with ground truth (the closes thing a VAE gets to validation because
        # there are no labels).
        demo_recon_quality(args, epoch, train_loader, optimizer, gen_model,
                           logger, sess_dir, device)


def train(epoch, args, train_loader, optimizer, gen_model, agent, logger, save_dir, device):

    # Set up logging queue objects
    loss_keys = ['obs', 'hx', 'done', 'reward', 'act_log_probs', 'KL',
                 'total recon w/o KL']
    train_info_bufs = {k:deque(maxlen=100) for k in loss_keys}
    logger.info('Start training epoch {}'.format(epoch))

    # Prepare for training cycle
    gen_model.train()

    # Training cycle
    for batch_idx, data in enumerate(train_loader):

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
        loss, train_info_bufs = loss_function(args, preds, data, mu_c, logvar_c, mu_g, logvar_g,
                                              train_info_bufs, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 0.001)
        for p in gen_model.decoder.agent.policy.parameters():
            if p.grad is not None:  # freeze agent parameters but not model's.
                p.grad.data = torch.zeros_like(p.grad.data)
        optimizer.step()
        #TODO sort out hyperparameters (incl archi) - put them all in one namespace.

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
            torch.save(
                {'gen_model_state_dict': gen_model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                model_path)
            logger.info('Generative model saved to {}'.format(model_path))

        # Visualize the predictions compared with the ground truth
        if batch_idx % 2000 == 0 or (epoch < 1 and batch_idx % 1000 == 0):

            with torch.no_grad():
                pred_obs = torch.stack(preds['obs'], dim=1).squeeze()

                viz_batch_size = 20
                viz_batch_size = min(int(pred_obs.shape[0]), viz_batch_size)

                for b in range(viz_batch_size):
                    # Put channel dim to the end
                    pred_ob = pred_obs[b].permute(0, 2, 3, 1)
                    full_ob = full_obs[b].permute(0, 2, 3, 1)

                    # Make predictions and ground truth into right format for
                    #  video saving
                    pred_ob = pred_ob * 255
                    full_ob = full_ob * 255

                    pred_ob = pred_ob.clone().detach().type(torch.uint8)
                    pred_ob = pred_ob.cpu().numpy()
                    full_ob = full_ob.clone().detach().type(torch.uint8)
                    full_ob = full_ob.cpu().numpy()[-args.num_sim_steps:]

                    # Join the prediction and the true observation side-by-side
                    combined_ob = np.concatenate([pred_ob, full_ob], axis=2)

                    # Save vid
                    save_str = save_dir + '/recons_v_preds' + '/sample_' + \
                               str(epoch) + '_' + str(batch_idx) + '_' + \
                               str(b) + '.mp4'
                    tvio.write_video(save_str, combined_ob, fps=14)


def loss_function(args, preds, labels, mu_c, logvar_c, mu_g, logvar_g, train_info_bufs, device):
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

    loss_hyperparams = {'obs': args.loss_scale_obs,
                        'hx': args.loss_scale_hx,
                        'reward': args.loss_scale_reward,
                        'done': args.loss_scale_done,
                        'act_log_probs': args.loss_scale_act_log_probs}

    dones = labels['done'][:, -args.num_sim_steps:]
    before_dones = done_labels_to_mask(dones)

    # Reconstruction loss
    losses = []
    for key in preds.keys():
        if key == 'values': # Not using values for loss
            continue
        pred  = torch.stack(preds[key], dim=1).squeeze()

        label = labels[key].to(device).float().squeeze()
        label = label[:, -args.num_sim_steps:]

        # Calculate masks for hx & logprobs
        if key in ['hx', 'act_log_probs']:
            num_dims = len(label.shape) - 2
            # subtract 2 is because batch and time dim are already present in the
            # mask

            unsqz_lastdim = lambda x: x.unsqueeze(dim=-1)
            mask = before_dones
            for d in range(num_dims):
                # Applies unsqueeze enough times to produce a tensor of the same
                # order as the loss tensor. It can therefore be broadcast to the
                # same shape as the loss tensor
                mask = unsqz_lastdim(mask)

            # Calculate loss
            # loss = torch.mean(torch.abs(pred - label))  # Mean Absolute Error
            loss = (pred - label) * mask
            loss = torch.mean(loss ** 2)
        else:
            # Calculate loss
            # loss = torch.mean(torch.abs(pred - label))  # Mean Absolute Error
            loss = (pred - label)
            loss = torch.mean(loss ** 2)


        #mse = F.mse_loss(pred, label) # TODO test whether MSE or MAbsE is better (I think the VQ-VAE2 paper suggested MAE was better)
        train_info_bufs[key].append(loss.item())
        loss = loss * loss_hyperparams[key]
        losses.append(loss)

    loss = sum(losses)
    train_info_bufs['total recon w/o KL'].append(loss.item())

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    mu = torch.cat([mu_c, mu_g], dim=1)
    logvar = torch.cat([logvar_c, logvar_g], dim=1)
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(),
                                     dim=1) # Mean(sum?) along latent size dim.
    kl_divergence = torch.mean(kl_divergence)  # Mean along batch dim
    train_info_bufs['KL'].append(kl_divergence.item())
    kl_divergence *= args.loss_scale_kl

    return loss + kl_divergence, train_info_bufs

#DONE fioverwriting of samples every epoch (something to do with names): Decided that it's better to keep it the way it is and just get rid of it for online publication. It's also nice to be able to see which runs I'd left running based on num files in dir. # TODO remove this comment when you're happy with the decision.

# TODO check ur iin the right train/test mode

def demo_recon_quality(args, epoch, train_loader, optimizer, gen_model, logger,
          save_dir, device, use_true_h0=False, use_true_actions=True):

    # Set up logging objects
    logger.info('Demonstrating reconstruction and prediction quality')

    # Prepare for training cycle
    gen_model.train()

    # Plot both the inputs and their reconstruction/prediction for one batch
    for batch_idx, data in enumerate(train_loader):
        if batch_idx > 0: # Just do one batch
            break
        data = {k: v.to(device).float() for k,v in data.items()}

        # Get input data for generative model
        full_obs = data['obs']
        agent_h0 = data['hx'][:, -args.num_sim_steps, :]
        actions_all = data['action'][:, -args.num_sim_steps:]

        # Forward pass to get predicted observations
        optimizer.zero_grad()
        mu_c, logvar_c, mu_g, logvar_g, preds = gen_model(full_obs, agent_h0, actions_all,
                                      use_true_h0, use_true_actions)

        with torch.no_grad():
            pred_obs = torch.stack(preds['obs'], dim=1).squeeze()
            pred_dones = torch.stack(preds['done'], dim=1).squeeze()

            viz_batch_size = 20
            viz_batch_size = min(int(pred_obs.shape[0]), viz_batch_size)

            for b in range(viz_batch_size):
                # Put channel dim to the end
                pred_ob = pred_obs[b].permute(0, 2, 3, 1)
                full_ob = full_obs[b].permute(0, 2, 3, 1)

                # Make predictions and ground truth into right format for
                #  video saving
                pred_ob = pred_ob * 255
                full_ob = full_ob * 255

                pred_ob = pred_ob.clone().detach().type(torch.uint8)
                pred_ob = pred_ob.cpu().numpy()
                full_ob = full_ob.clone().detach().type(torch.uint8)
                full_ob = full_ob.cpu().numpy()[-args.num_sim_steps:]

                # Join the prediction and the true observation side-by-side
                combined_ob = np.concatenate([pred_ob, full_ob], axis=2)

                # Save vid
                save_str = save_dir + '/recons_v_preds' + '/demo_' + \
                           str(epoch) + '_' + str(batch_idx) + '_' + \
                           str(b) + '.mp4'
                tvio.write_video(save_str, combined_ob, fps=14)


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