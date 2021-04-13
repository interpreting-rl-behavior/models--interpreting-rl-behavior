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


class LatentSpaceExperiment():
    def __init__(self, args):
        """
        A class for experiments that involve sampling from a VAE latent space.

        Its purpose is to have all the infrastructure necessary for
        running experiments that involve generating samples from the latent
        space of the VAE. It therefore accommodates the following experiments:
          - TargetFunctionExperiments
          - LatentSpaceInterpolationExperiment
          - LatentSpaceStructureExplorationExperiment

        It is necessary because there is a lot of bumf involved in setting up
        the generative model. Setting up the generative model requires:
          - Setting a bunch of hyperparams
          - Creating dummy environment and storage objects for the
              instantiation of the agent
          - Instantiating the agent that will be used in the decoder
          - Any infrastructure for loading any of the above
        """
        super(LatentSpaceExperiment, self).__init__()

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

        #
        # # TODO put exp name as main dir then this as subdir
        # gen_model_session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        # sess_dir = os.path.join(resdir, gen_model_session_name)
        # if not (os.path.exists(sess_dir)):
        #     os.makedirs(sess_dir)
        # logger.configure(dir=sess_dir, format_strs=['csv', 'stdout'])
        # reconpred_dir = os.path.join(sess_dir, 'recons_v_preds')
        # if not (os.path.exists(reconpred_dir)):
        #     os.makedirs(reconpred_dir)


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

        if args.model_file is not None:
            checkpoint = torch.load(args.model_file, map_location=device)
            gen_model.load_state_dict(checkpoint['gen_model_state_dict'], device)
            discrim.load_state_dict(checkpoint['discrim_state_dict'], device)
            logger.info('Loaded generative model from {}.'.format(args.model_file))
        else:
            logger.info('Using an UNTRAINED generative model')

        self.args = args
        self.gen_model = gen_model
        self.agent = agent
        self.train_loader = train_loader
        # self.sess_dir = sess_dir
        self.device = device
        self.logger = logger

    def demo_recon_quality(self, epoch, args, train_loader, optimizer, gen_model, agent, logger,
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