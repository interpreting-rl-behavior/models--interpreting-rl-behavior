from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
import torchvision.io as tvio

import os, time, yaml, argparse
import gym
from procgen import ProcgenGym3Env
import random
import torch

from gym3 import ViewerWrapper, VideoRecorderWrapper, ToBaselinesVecEnv

if __name__=='__main__':
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
    parser.add_argument('--tps', type=int, default=15, help="env fps")
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str)

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

    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    ############
    ## DEVICE ##
    ############
    if args.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')
    n_envs = 1

    def create_venv_render(args, hyperparameters, is_valid=False):
        venv = ProcgenGym3Env(num=n_envs,
                          env_name=args.env_name,
                          num_levels=0 if is_valid else args.num_levels,
                          start_level=0 if is_valid else args.start_level,
                          distribution_mode=args.distribution_mode,
                          num_threads=1,
                          use_backgrounds=False,
                          render_mode="rgb_array")
        venv = ViewerWrapper(venv, tps=args.tps, info_key="rgb")
        if args.vid_dir is not None:
            venv = VideoRecorderWrapper(venv, directory=args.vid_dir,
                                        info_key="rgb", fps=args.tps)
        venv = ToBaselinesVecEnv(venv)
        venv = VecExtractDictObs(venv, "rgb")
        normalize_rew = hyperparameters.get('normalize_rew', True)
        if normalize_rew:
            venv = VecNormalize(venv, ob=False) # normalizing returns, but not
            #the img frames
        venv = TransposeFrame(venv)
        venv = ScaledFloatFrame(venv)

        return venv
    n_steps = hyperparameters.get('n_steps', 256)

    #env = create_venv(args, hyperparameters)
    #env_valid = create_venv(args, hyperparameters, is_valid=True)
    env = create_venv_render(args, hyperparameters, is_valid=True)

    ############
    ## LOGGER ##
    ############
    print('INITIALIZAING LOGGER...')
    logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'RENDER_seed' + '_' + \
             str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir)

    ###########
    ## MODEL ##
    ###########
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

    #############
    ## STORAGE ##
    #############
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    #storage_valid = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)

    agent.policy.load_state_dict(torch.load(args.model_file, map_location=device)["model_state_dict"])
    agent.n_envs = n_envs

    ##############
    ## TRAINING ##
    ##############

    obs = agent.env.reset()
    hidden_state = np.stack(
        [agent.policy.init_hx.clone().detach().cpu().numpy()] * agent.n_envs)
    done = np.zeros(agent.n_envs)

    all_obs = []
    num_episodes_to_render = 100
    episode = 0
    for e in range(num_episodes_to_render):
    #while True:
        agent.policy.eval()
        for _ in range(agent.n_steps):
            all_obs.append(obs)
            act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
            next_obs, rew, done, info = agent.env.step(act)
            agent.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state
            if np.any(done):
                hidden_state[done] = \
                    agent.policy.init_hx.clone().detach().cpu().numpy()
                # Save vid of obs
                all_obs = np.squeeze(np.stack(all_obs, 1))
                all_obs = all_obs.transpose([0, 2, 3, 1])
                all_obs = all_obs * 255
                # sample = sample.clone().detach().type(torch.uint8)
                # sample = sample.cpu().numpy()
                save_str = 'logs/rendered_what_agent_sees_%i.mp4' % episode
                full_resvid_names = [f for f in os.listdir('logs/') \
                                     if
                                     os.path.isfile(os.path.join('logs/', f))]
                tvio.write_video(save_str, all_obs, fps=14)
                all_obs = []
                episode += 1
        _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)
        agent.storage.store_last(obs, hidden_state, last_val)
        agent.storage.compute_estimates(agent.gamma, agent.lmbda, agent.use_gae,
                                       agent.normalize_adv)
