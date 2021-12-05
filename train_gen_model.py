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
from generative.generative_models import AgentEnvironmentSimulator
from generative.procgen_dataset import ProcgenDataset
from overlay_image import overlay_actions

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
    parser.add_argument('--num_init_steps', type=int, default=8)
    parser.add_argument('--num_sim_steps', type=int, default=22)
    parser.add_argument('--layer_norm', type=int, default=0)

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    # Loss function hyperparams
    parser.add_argument('--loss_scale_ims', type=float, default=1.)
    parser.add_argument('--loss_scale_hx', type=float, default=1.)
    parser.add_argument('--loss_scale_reward', type=float, default=1.)
    parser.add_argument('--loss_scale_terminal', type=float, default=1.)
    parser.add_argument('--loss_scale_act_log_probs', type=float, default=1.)
    parser.add_argument('--loss_scale_gen_adv', type=float, default=1.)
    parser.add_argument('--loss_scale_kl', type=float, default=1.)


    # Collect hyperparams from arguments
    args = parser.parse_args()
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints
    batch_size = args.batch_size
    num_init_steps = args.num_init_steps
    num_sim_steps = args.num_sim_steps
    num_steps_full = num_init_steps + num_sim_steps - 1
    # minus one because the first simulated image is the last
    # initializing context ims.

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    print('[LOADING AGENT HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        agent_hyperparams = yaml.safe_load(f)[param_name]
    for key, value in agent_hyperparams.items():
        print(key, ':', value)

    n_steps = 1
    n_envs = agent_hyperparams.get('n_envs', 64)

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    # Set up environment (Only used for initializing agent)
    print('INITIALIZING ENVIRONMENTS...')
    env = create_venv(args, agent_hyperparams)

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
    architecture = agent_hyperparams.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    ## Agent architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)

    ## Agent's discrete action space
    recurrent = agent_hyperparams.get('recurrent', False)
    action_size = action_space.n
    if isinstance(action_space, gym.spaces.Discrete):
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
    algo = agent_hyperparams.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints,
                  **agent_hyperparams)
    if args.agent_file is not None:
        logger.info("Loading agent from %s" % args.agent_file)
        checkpoint = torch.load(args.agent_file, map_location=device)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Done loading agent.")

    # Set up generative model
    ## Make dataset
    train_dataset = ProcgenDataset(args.data_dir,
                                   initializer_seq_len=num_init_steps,
                                   num_steps_full=num_steps_full,)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=2)

    gen_model_hyperparams = Namespace(
        num_init_steps=num_init_steps,
        num_steps_full=num_steps_full,
        num_sim_steps=num_sim_steps,
        stoch_discrete=32,
        stoch_dim=32,
        env_h_stoch_size=32 * 32,
        agent_hidden_size=64,
        action_dim=action_size,
        deter_dim=512,  # 2048
        initializer_rnn_hidden_size=256,
        layer_norm=True,
        hidden_dim=1000,
        image_channels=3,
        cnn_depth=48,
        reward_decoder_layers=4,
        terminal_decoder_layers=4,
        kl_weight=0.1,
        kl_balance=0.8,
    )

    ## Make or load generative model and optimizer
    # gen_model = VAE(agent, device, num_initializing_steps, total_seq_len)
    gen_model = AgentEnvironmentSimulator(agent, device, gen_model_hyperparams)
    gen_model = gen_model.to(device)
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr)

    if args.model_file is not None:
        logger.info("Loading generative model from %s" % args.model_file)
        checkpoint = torch.load(args.model_file, map_location=device)
        gen_model.load_state_dict(checkpoint['gen_model_state_dict'], device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info('Loaded generative model from {}.'.format(args.model_file))
    else:
        logger.info('Training generative model from scratch.')

    # Training cycle (Train, Save visualized random samples, Demonstrate
    #  reconstruction quality)
    for epoch in range(0, args.epochs + 1):
        train(epoch, args, train_loader, optimizer, gen_model, agent,
              logger, sess_dir, device)


def train(epoch, args, train_loader, optimizer, gen_model, agent, logger, save_dir, device):

    # Set up logging queue objects
    loss_keys = ['ims', 'hx', 'terminal', 'reward', 'act_log_probs', 'KL',
                 'total recon w/o KL']
    train_info_bufs = {k:deque(maxlen=100) for k in loss_keys}
    logger.info('Start training epoch {}'.format(epoch))

    # Prepare for training cycle
    gen_model.train()

    # Training cycle
    for batch_idx, data in enumerate(train_loader):

        # Make all data into floats and put on the right device
        data = {k: v.to(device).float() for k, v in data.items()}
        data = {k: torch.swapaxes(v, 0, 1) for k, v in data.items()}  # (B, T, :...) --> (T, B, :...)

        # Forward and backward pass and update generative model parameters
        optimizer.zero_grad()
        (loss_model, priors, posts, samples, features, env_states,
        env_state, metrics_list, tensors_list, pred_actions_1hot, pred_agent_hs) = \
            gen_model(data=data, use_true_actions=True, imagine=False)

        loss = torch.mean(torch.sum(loss_model, dim=0))  # sum over T, mean over B
        # TODO confirm that masking of losses works as required
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 100.)
        for p in gen_model.agent_env_stepper.agent.policy.parameters():
            if p.grad is not None:  # freeze agent parameters but not model's.
                p.grad.data = torch.zeros_like(p.grad.data)
        optimizer.step()

        # Logging and saving info
        if batch_idx % args.log_interval == 0:
            loss.item()
            logger.logkv('epoch', epoch)
            logger.logkv('batches', batch_idx)
            logger.logkv('loss total', loss.item())
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
        pred_images = torch.cat(
            [tensors_list[t]['image_rec'] for t in range(args.num_sim_steps)],
            dim=0)

        preds = {'ims': pred_images, 'actions': pred_actions_1hot}
        if (epoch > 1 and batch_idx % 20000 == 0) or (epoch < 1 and batch_idx % 5000 == 0):
        # if batch_idx % 10000 == 0 or (epoch < 1 and batch_idx % 20000 == 0):
            visualize(args, epoch, train_loader, optimizer, gen_model,
                               logger, batch_idx=batch_idx, save_dir=save_dir,
                               device=device, data=data, preds=preds,
                               use_true_actions=True, save_root='sample')

        # Demo recon quality without using true images
        if (epoch > 1 and batch_idx % 20000 == 0) or (epoch < 1 and batch_idx % 5000 == 0):
            visualize(args, epoch, train_loader, optimizer, gen_model,
                               logger, batch_idx=batch_idx, save_dir=save_dir,
                               device=device, data=None, preds=None,
                               use_true_actions=True, save_root='demo_true_acts')
            visualize(args, epoch, train_loader, optimizer, gen_model,
                               logger, batch_idx=batch_idx, save_dir=save_dir,
                               device=device, data=None, preds=None,
                               use_true_actions=False, save_root='demo_sim_acts')


def compute_kl_div_categorical(p_logits, q_logits, num_categ_distribs=32, num_categs=32):

    # episilon = 1e-7
    ts = p_logits.shape[0]
    b = p_logits.shape[1]
    p_logits = p_logits.view(ts, b, num_categ_distribs, num_categs)
    q_logits = q_logits.view(ts, b, num_categ_distribs, num_categs)

    probs_p = torch.softmax(p_logits, dim=3)
    probs_q = torch.softmax(q_logits, dim=3)
    kl_div = torch.sum(probs_p * torch.log(probs_p / probs_q), dim=[0,2,3]) # sum over t and categoricals
    kl_div = torch.mean(kl_div) # mean over batch
    return kl_div

def loss_function(args, preds, labels, train_info_bufs, device):

    recon_loss_hyperparams = {'ims': args.loss_scale_ims,
                              'hx': args.loss_scale_hx,
                              'reward': args.loss_scale_reward,
                              'terminal': args.loss_scale_terminal,
                              'act_log_probs': args.loss_scale_act_log_probs
                              }
    KL_loss_hyperparams = {
        'pp_KL': 1.0,
        'KL_alpha': 0.8,
    }

    terminals = labels['terminal'][:, -args.num_sim_steps:]
    before_terminals = terminal_labels_to_mask(terminals)

    # Reconstruction loss
    recon_losses = []
    for key in recon_loss_hyperparams.keys():
        pred  = torch.stack(preds[key], dim=1).squeeze()

        label = labels[key].to(device).float().squeeze()
        label = label[:, -args.num_sim_steps:]

        # Calculate masks for hx & logprobs
        if key in ['hx', 'act_log_probs']:
            num_dims = len(label.shape) - 2
            # subtract 2 is because batch and time dim are already present in the
            # mask

            unsqz_lastdim = lambda x: x.unsqueeze(dim=-1)
            mask = before_terminals
            for d in range(num_dims):
                # Applies unsqueeze enough times to produce a tensor of the same
                # order as the loss tensor. It can therefore be broadcast to the
                # same shape as the loss tensor
                mask = unsqz_lastdim(mask)

            # Calculate loss
            loss = (pred - label) * mask
            loss = torch.mean(loss ** 2)
        else:
            # Calculate loss
            loss = (pred - label)
            loss = torch.mean(loss ** 2)

        train_info_bufs[key].append(loss.item())
        loss = loss * recon_loss_hyperparams[key]
        recon_losses.append(loss)

    recon_loss = sum(recon_losses)
    train_info_bufs['total recon w/o KL'].append(recon_loss.item())

    # KL term
    alpha = KL_loss_hyperparams['KL_alpha']

    q = preds['env_h_stoch_logits']  # approx posterior
    p = preds['pred_env_h_stoch_logits']  # prior
    q = torch.stack(q)
    p = torch.stack(p)

    kl_divergence_q = compute_kl_div_categorical(q.detach(), p) * alpha
    kl_divergence_p = compute_kl_div_categorical(q, p.detach()) * (1-alpha)
    kl_divergence = kl_divergence_q + kl_divergence_p
    train_info_bufs['KL'].append(kl_divergence.item())

    return recon_loss + kl_divergence, train_info_bufs

def visualize(args, epoch, train_loader, optimizer, gen_model, logger,
              batch_idx, save_dir, device, data=None, preds=None, use_true_actions=True, save_root=''):

    logger.info('Demonstrating reconstruction and prediction quality')

    gen_model.train()
    action_size = gen_model.agent_env_stepper.agent.env.action_space.n

    if data is None:
        # Get a single batch from the train_loader
        for batch_idx_new, data in enumerate(train_loader):
            if batch_idx_new > 0:
                break
        # Make all data into floats, put on the right device, and swap B and T axes
        data = {k: v.to(device).float() for k, v in data.items()}
        data = {k: torch.swapaxes(v, 0, 1) for k, v in data.items()}  # (B, T, :...) --> (T, B, :...)

    # Get input data for generative model
    full_ims = data['ims']
    true_actions_inds = data['action'][-args.num_sim_steps:]

    # Forward pass to get predictions if not already done
    if preds is None:
        optimizer.zero_grad()
        (loss_model, priors, posts, samples, features, env_states,
         env_state, metrics_list, tensors_list, pred_actions_1hot, pred_agent_hs) = \
            gen_model(data=data,
                      use_true_actions=use_true_actions,
                      imagine=True,
                      modal_sampling=True)
        # Extract predictions from model output
        pred_images = torch.cat(
            [tensors_list[t]['image_rec'] for t in range(args.num_sim_steps)],
            dim=0)
    else:
        pred_images, pred_actions_1hot = preds['ims'], preds['actions']

    # Establish the right settings for visualisation
    viz_batch_size = min(int(pred_images.shape[1]), 20)
    pred_actions_inds = torch.argmax(pred_actions_1hot, dim=2)

    # (T, B) --> (B, T)
    pred_actions_inds = pred_actions_inds.permute(1, 0)
    true_actions_inds = true_actions_inds.permute(1, 0)

    true_actions_inds = true_actions_inds.clone().cpu().numpy()
    pred_actions_inds = pred_actions_inds.cpu().detach().numpy()

    if use_true_actions:
        viz_actions_inds = true_actions_inds
    else:
        viz_actions_inds = pred_actions_inds

    with torch.no_grad():
        # (T,B,C,H,W) --> (B,T,H,W,C)
        pred_images = pred_images.permute(1,0,3,4,2)
        full_ims = full_ims.permute(1,0,3,4,2)

        for b in range(viz_batch_size):
            pred_im = pred_images[b]
            full_im = full_ims[b]

            # Make predictions and ground truth into right format for
            #  video saving
            pred_im = pred_im * 255
            full_im = full_im * 255

            pred_im = torch.clip(pred_im, 0, 255)

            pred_im = pred_im.clone().detach().type(torch.uint8).cpu().numpy()
            pred_im = overlay_actions(pred_im, viz_actions_inds[b], size=16)

            full_im = full_im.clone().detach().type(torch.uint8).cpu().numpy()
            full_im = full_im[-args.num_sim_steps:]
            full_im = overlay_actions(full_im, true_actions_inds[b], size=16)


            # Join the prediction and the true image side-by-side
            combined_im = np.concatenate([pred_im, full_im], axis=2)

            # Save vid
            save_str = save_dir + \
                       f'/recons_v_preds/{save_root}_' + \
                       f'{epoch:02d}_{batch_idx:06d}_{b:03d}.mp4'
            tvio.write_video(save_str, combined_im, fps=14)


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



class Namespace:
    """
    Because they're nicer to work with than dictionaries
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    run()