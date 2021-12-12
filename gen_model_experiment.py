import os, yaml, argparse
import random
import torchvision.io as tvio
import util.logger as logger
from common.env.procgen_wrappers import *
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from train import create_venv
from generative.generative_models import AgentEnvironmentSimulator
from generative.procgen_dataset import ProcgenDataset
from overlay_image import overlay_actions, overlay_box_var

from util.namespace import Namespace
from datetime import datetime



class GenerativeModelExperiment():
    def __init__(self):
        """ # TODO generalize docstring
        A class for experiments that involve sampling from a VAE latent space.

        Its purpose is to have all the infrastructure necessary for
        running experiments that involve generating samples from the latent
        space of the VAE. It therefore accommodates the following experiments:
          -
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
        super(GenerativeModelExperiment, self).__init__()

        args = self.parse_the_args()

        # Collect hyperparams from arguments
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

        # Device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        if args.device == 'gpu':
            device = torch.device('cuda')
        elif args.device == 'cpu':
            device = torch.device('cpu')

        # Set up environment (Only used for initializing agent)
        print('INITIALIZING ENVIRONMENTS...')
        n_steps = 1
        n_envs = agent_hyperparams.get('n_envs', 64)
        env = create_venv(args, agent_hyperparams)

        # Make save dirs
        print('INITIALIZING LOGGER...')
        log_dir_base = args.log_dir_base
        # log_dir_base = 'generative/'   # for training
        # log_dir_base = 'experiments/'  # for experiments
        if not (os.path.exists(log_dir_base)):
            os.makedirs(log_dir_base)
        resdir = log_dir_base + 'results/'
        if not (os.path.exists(resdir)):
            os.makedirs(resdir)
        resdir = resdir + args.gen_mod_exp_name
        if not (os.path.exists(resdir)):
            os.makedirs(resdir)
        gen_model_session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess_dir = os.path.join(resdir, gen_model_session_name)
        if not (os.path.exists(sess_dir)):
            os.makedirs(sess_dir)

        if log_dir_base == 'generative/':
            # i.e. during gen_model training
            reconpred_dir = os.path.join(sess_dir, 'recons_v_preds')
            if not (os.path.exists(reconpred_dir)):
                os.makedirs(reconpred_dir)

        # Logger
        logger.configure(dir=sess_dir, format_strs=['csv', 'stdout'])

        # Set up agent
        print('INITIALIZING AGENT MODEL...')
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
                                       num_steps_full=num_steps_full, )
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=2)

        ## Make or load generative model and optimizer
        gen_model_hyperparams = Namespace( # TODO maybe put all configs like this and CLI args in a single configs file
            num_init_steps=num_init_steps,
            num_steps_full=num_steps_full,
            num_sim_steps=num_sim_steps,
            stoch_discrete=32,
            stoch_dim=32,
            env_h_stoch_size=32 * 32,
            agent_hidden_size=64,
            action_dim=action_size, #TODO go through making action_dim and _size consistent
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
            vae_kl_weight=1.,
            sample_vec_size=256,
        )

        gen_model = AgentEnvironmentSimulator(agent, device, gen_model_hyperparams)
        gen_model = gen_model.to(device)
        optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr)

        if args.model_file is not None:
            logger.info("Loading generative model from %s" % args.model_file)
            checkpoint = torch.load(args.model_file, map_location=device)
            gen_model.load_state_dict(checkpoint['gen_model_state_dict'],
                                      device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(
                'Loaded generative model from {}.'.format(args.model_file))
        else:
            logger.info('Using an UNTRAINED generative model')

        self.args = args
        self.gen_model = gen_model
        self.agent = agent
        self.train_loader = train_loader
        self.resdir = resdir
        self.sess_dir = sess_dir
        self.data_save_dir = args.data_dir
        self.device = device
        self.logger = logger
        self.optimizer = optimizer

    def parse_the_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_name', type=str, default='test',
                            help='experiment name')  # TODO can we get rid of this?
        parser.add_argument('--gen_mod_exp_name', type=str, default='test_tgm',
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
        parser.add_argument('--log_dir_base', type=str, default='generative/')
        # log_dir_base = 'generative/'   # for training
        # log_dir_base = 'experiments/'  # for experiments
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
        parser.add_argument('--loss_scale_act_log_probs', type=float,
                            default=1.)
        parser.add_argument('--loss_scale_gen_adv', type=float, default=1.)
        parser.add_argument('--loss_scale_kl', type=float, default=1.)

        # Collect hyperparams from arguments
        args = parser.parse_args()
        return args

    def get_single_batch(self):
        # Get a single batch from the train_loader
        for batch_idx_new, data in enumerate(self.train_loader):
            if batch_idx_new > 0:
                break
        # Make all data into floats, put on the right device, and swap B and T axes
        data = {k: v.to(self.device).float() for k, v in
                data.items()}
        data = {k: torch.swapaxes(v, 0, 1) for k, v in
                data.items()}  # (B, T, :...) --> (T, B, :...)
        return data

    def postprocess_preds(self, preds):
        #### The ones needed for viz
        pred_images = preds['ims']
        pred_terminals = preds['terminal']
        pred_rews = preds['reward']
        pred_actions_1hot = preds['action']

        # TODO Extract all vars so that you can use this method in both viz and record

        # Establish the right settings for visualisation
        pred_actions_inds = torch.argmax(pred_actions_1hot, dim=2)
        pred_actions_inds = pred_actions_inds.permute(1, 0)
        pred_actions_inds = pred_actions_inds.cpu().detach().numpy()

        # (T,B,C,H,W) --> (B,T,H,W,C)
        pred_images = pred_images.permute(1, 0, 3, 4, 2)
        # (T,B,...) --> (B,T,...)
        pred_terminals = pred_terminals.permute(1, 0)
        pred_rews = pred_rews.permute(1, 0)


        ### The ones not needed by viz (but all are used by record
        #'act_log_prob''value''hx''sample_vec', 'env_h'
        pred_act_log_prob = preds['act_log_prob'].transpose(0, 1)
        pred_value = preds['value'].transpose(0, 1)
        pred_agent_h = preds['hx'].transpose(0, 1)
        pred_sample_vec = preds['sample_vec']
        pred_env_h = preds['env_h'].transpose(0, 1)

        return pred_images, pred_terminals, pred_rews, pred_actions_1hot, \
               pred_actions_inds, \
               pred_act_log_prob, pred_value, pred_agent_h, pred_sample_vec,\
               pred_env_h

    def extract_from_data(self, data):
        full_ims = data['ims']
        full_ims = full_ims[-self.args.num_sim_steps:]
        full_ims = full_ims.permute(1, 0, 3, 4, 2)

        true_actions_inds = data['action'][-self.args.num_sim_steps:]
        true_actions_inds = true_actions_inds.permute(1, 0)

        true_terminals = data['terminal'][-self.args.num_sim_steps:]
        true_terminals = true_terminals.permute(1, 0)

        true_rews = data['reward'][-self.args.num_sim_steps:]
        true_rews = true_rews.permute(1, 0)
        return full_ims, true_actions_inds, true_terminals, true_rews

    def record(self, informed_initialization=True):


        # Within some recording loop
        samples_so_far = epoch * batch_size
        new_sample_indices = range(samples_so_far, samples_so_far + batch_size)

    def save_preds(self, preds, new_sample_indices_range, manual_action):
        pred_obs = preds['obs']
        pred_rews = preds['reward']
        pred_dones = preds['done']
        pred_agent_hxs = preds['hx']
        pred_agent_logprobs = preds['act_log_probs']
        pred_agent_values = preds['values']
        pred_env_states = preds['env_h']
        sample_latent_vecs = preds['latent_vecs_c_and_g']

        # Stack samples into single tensors and convert to numpy arrays
        pred_obs = np.array(torch.stack(pred_obs, dim=1).cpu().numpy() * 255, dtype=np.uint8)
        pred_rews = torch.stack(pred_rews, dim=1).cpu().numpy()
        pred_dones = torch.stack(pred_dones, dim=1).cpu().numpy()
        pred_agent_hxs = torch.stack(pred_agent_hxs, dim=1).cpu().numpy()
        pred_agent_logprobs = torch.stack(pred_agent_logprobs, dim=1).cpu().numpy()
        pred_agent_values = torch.stack(pred_agent_values, dim=1).cpu().numpy()
        pred_env_states = torch.stack(pred_env_states, dim=1).cpu().numpy()

        # no timesteps in latent vecs, so only cat not stack along time dim.
        sample_latent_vecs = torch.cat(sample_latent_vecs, dim=1).cpu().numpy()

        vars = [pred_obs, pred_rews, pred_dones, pred_agent_hxs,
                pred_agent_logprobs, pred_agent_values, pred_env_states, sample_latent_vecs]
        var_names = ['obs', 'rews', 'dones', 'agent_hxs',
                     'agent_logprobs', 'agent_values', 'env_hid_states',
                     'env_cell_states', 'latent_vec']

        # # Recover the actions for use in the action overlay
        # if manual_action is not None:
        #     actions = np.ones((args.batch_size, args.num_sim_steps)) * manual_action
        # else:
        #     actions = np.argmax(pred_agent_logprobs, axis=-1)

        # Make dirs for these variables and save variables to dirs and save vid
        sample_dir_base = os.path.join(self.data_save_dir, 'sample_')
        for i, new_sample_idx in enumerate(new_sample_indices_range):
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

    def visualize(self,
                  epoch,
                  batch_idx,
                  data=None,
                  preds=None,
                  use_true_actions=True,
                  save_root=''):

        self.logger.info('Demonstrating reconstruction and prediction quality')

        self.gen_model.train()

        if data is None:
            data = self.get_single_batch()

        # Get labels and swap B and T axes
        full_ims, true_actions_inds, true_terminals, true_rews = self.extract_from_data(data)

        # Forward pass to get predictions if not already done
        if preds is None:
            self.optimizer.zero_grad()
            (   loss_model,
                loss_vae_kl_divergence,
                loss_agent_h0,
                priors,  # tensor(T,B,2S)
                posts,  # tensor(T,B,2S)
                samples,  # tensor(T,B,S)
                features,  # tensor(T,B,D+S)
                env_states,
                (env_h, env_z),
                metrics_list,
                tensors_list,
                preds,
            ) = \
                self.gen_model(data=data,
                          use_true_actions=use_true_actions,
                          imagine=True,
                          modal_sampling=True)

        pred_images, pred_terminals, pred_rews, pred_actions_1hot, \
        pred_actions_inds, _, _, _, _, _  = \
            self.postprocess_preds(preds)
        true_actions_inds = true_actions_inds.clone().cpu().numpy()

        if use_true_actions:
            viz_actions_inds = true_actions_inds
        else:
            viz_actions_inds = pred_actions_inds

        viz_batch_size = min(int(pred_images.shape[0]), 20)

        with torch.no_grad():
            for b in range(viz_batch_size):
                pred_im = pred_images[b]
                full_im = full_ims[b]

                # Overlay Done and Reward
                pred_im = overlay_box_var(pred_im, pred_terminals[b], 'left', max=1.)
                pred_im = overlay_box_var(pred_im, pred_rews[b], 'right', max=10.)
                full_im = overlay_box_var(full_im, true_terminals[b], 'left', max=1.)
                full_im = overlay_box_var(full_im, true_rews[b], 'right', max=10.)

                # Make predictions and ground truth into right format for
                #  video saving
                pred_im = pred_im * 255
                full_im = full_im * 255
                pred_im = torch.clip(pred_im, 0,
                                     255)  # TODO check that this is no longer necessary

                pred_im = pred_im.clone().detach().type(torch.uint8).cpu().numpy()
                pred_im = overlay_actions(pred_im, viz_actions_inds[b], size=16)

                full_im = full_im.clone().detach().type(
                    torch.uint8).cpu().numpy()
                full_im = overlay_actions(full_im, true_actions_inds[b], size=16)

                # Join the prediction and the true image side-by-side
                combined_im = np.concatenate([pred_im, full_im], axis=2)

                # Save vid
                save_str = self.sess_dir + \
                           f'/recons_v_preds/{save_root}_' + \
                           f'{epoch:02d}_{batch_idx:06d}_{b:03d}.mp4'
                tvio.write_video(save_str, combined_im, fps=14)

    def visualize_single(self,
                                  epoch,
                                  batch_idx,
                                  data=None,
                                  preds=None,
                                  latent_vec=None,
                                  informed_init=False,
                                  use_true_actions=False,
                                  save_root='',
                                  batch_size=20):

        self.logger.info('Demonstrating samples from latent vecs')

        self.gen_model.train()
        self.optimizer.zero_grad()
        viz_batch_size = batch_size

        # TODO systematize the below conjunction
        if preds is None and latent_vec is None:
            if informed_init:  # encoder --> latent_vec --> decoder --> viz preds
                if data is None:
                    data = self.get_single_batch()

                _, true_actions_inds, _, _ = self.extract_from_data(data)

                (loss_model,
                 loss_vae_kl_divergence,
                 loss_agent_h0,
                 priors,  # tensor(T,B,2S)
                 posts,  # tensor(T,B,2S)
                 samples,  # tensor(T,B,S)
                 features,  # tensor(T,B,D+S)
                 env_states,
                 (env_h, env_z),
                 metrics_list,
                 tensors_list,
                 preds,
                 ) = \
                    self.gen_model(data=data,
                                   use_true_actions=use_true_actions,
                                   imagine=True,
                                   modal_sampling=True)
            else:  # random latent_vec --> decoder
                latent_vec = torch.randn(viz_batch_size,
                                        self.gen_model.sample_vec_size,
                                        device=self.device)

                (   loss_model,
                    loss_agent_aux_init,
                    priors,  # tensor(T,B,2S)
                    posts,  # tensor(T,B,2S)
                    samples,  # tensor(T,B,S)
                    features,  # tensor(T,B,D+S)
                    env_states,
                    (env_h, env_z),
                    metrics_list,
                    tensors_list,
                    preds,
                ) = \
                    self.gen_model.vae_decode(
                        latent_vec,
                        data=None,
                        true_actions_1hot=None,
                        use_true_actions=False,
                        true_agent_h0=None,
                        use_true_agent_h0=False,
                        imagine=True,
                        calc_loss=False,
                        modal_sampling=True,
                        retain_grads=True, )

        # Use the latent vec given in the arguments
        elif preds is None and latent_vec is not None:
            # (already run encoder) latent_vec --> decoder --> viz preds
            (loss_model,
             loss_agent_aux_init,
             priors,  # tensor(T,B,2S)
             posts,  # tensor(T,B,2S)
             samples,  # tensor(T,B,S)
             features,  # tensor(T,B,D+S)
             env_states,
             (env_h, env_z),
             metrics_list,
             tensors_list,
             preds,
             ) = \
                self.gen_model.vae_decode(
                    latent_vec,
                    data=None,
                    true_actions_1hot=None,
                    use_true_actions=False,
                    true_agent_h0=None,
                    use_true_agent_h0=False,
                    imagine=True,
                    calc_loss=False,
                    modal_sampling=True,
                    retain_grads=True)

        # Just visualise the given preds
        elif preds is not None and latent_vec is None:
            # Dont make a new preds dict by running a gen model decoder
            # just --> viz preds
            pass

        pred_images, pred_terminals, pred_rews, pred_actions_1hot, \
        pred_actions_inds, _, _, _, _, _ = self.postprocess_preds(preds)
        viz_batch_size = min(int(pred_images.shape[0]), 20)

        if use_true_actions:  # N.b. We only ever have true actions with informed init
            viz_actions_inds = true_actions_inds.clone().cpu().numpy()
        else:
            viz_actions_inds = pred_actions_inds#.cpu().detach().numpy()

        with torch.no_grad():
            for b in range(viz_batch_size):
                pred_im = pred_images[b]

                # Overlay Done and Reward
                pred_im = overlay_box_var(pred_im, pred_terminals[b], 'left', max=1.)
                pred_im = overlay_box_var(pred_im, pred_rews[b], 'right', max=10.)

                # Make predictions and ground truth into right format for
                #  video saving
                pred_im = pred_im * 255
                pred_im = torch.clip(pred_im, 0,
                                     255)  # TODO check that this is no longer necessary

                pred_im = pred_im.clone().detach().type(torch.uint8).cpu().numpy()
                pred_im = overlay_actions(pred_im, viz_actions_inds[b], size=16)

                # Save vid
                save_str = self.sess_dir + \
                           f'/recons_v_preds/{save_root}_' + \
                           f'{epoch:02d}_{batch_idx:06d}_{b:03d}.mp4'
                tvio.write_video(save_str, pred_im, fps=14)

    def get_swap_directions(self):
        if self.args.swap_directions_from is not None:
            assert len(self.args.swap_directions_from) == \
                   len(self.args.swap_directions_to)
            from_dircs = []
            to_dircs = []
            # Convert from strings into the right type (int or None)
            for from_dirc, to_dirc in zip(self.args.swap_directions_from,
                                          self.args.swap_directions_to):
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
        return swap_directions
