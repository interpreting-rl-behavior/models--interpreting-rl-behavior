from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
import custom_envs
from custom_envs.wrappers import ResizeObservationVec

import os, time, yaml, argparse
import gym
from gym.wrappers.time_limit import TimeLimit
import gym3
from procgen import ProcgenEnv
import random
import torch


def create_venv(args, hyperparameters, is_valid=False):
    try:
        # Load Env from custom_envs
        env_cls = getattr(custom_envs, args.env_name)
    except AttributeError:
        # Assume Procgen Environment
        venv = ProcgenEnv(num_envs=hyperparameters.get('n_envs', 256),
                        env_name=args.env_name,
                        num_levels=0 if is_valid else args.num_levels,
                        start_level=0 if is_valid else args.start_level,
                        distribution_mode=args.distribution_mode,
                        use_backgrounds=False,
                        num_threads=args.num_threads)
        venv = VecExtractDictObs(venv, "rgb")
    else:
        def init_env(env_cls=env_cls, hyperparameters=hyperparameters):
            env = env_cls()
            env = TimeLimit(env, max_episode_steps=hyperparameters.get('n_steps', 1000))
            return env
        # Note that we can't use register if we want subprocessing (because we'd then need to
        # register on each subprocess which isn't trivial without altering gym)
        # gym.envs.register(
        #     id=f"{args.env_name}-v0",
        #     # entry_point='gym.envs.classic_control:MountainCarEnv',
        #     entry_point=f"custom_envs:{args.env_name}",
        #     max_episode_steps=hyperparameters.get('n_steps', 1000),  # MountainCar-v0 uses 200
        #     reward_threshold=-110.0,
        # )
        # gym.make(args.env_name)
        venv = gym3.vectorize_gym(
            num=hyperparameters.get('n_envs', 256),
            # env_fn=lambda: env_cls(),
            env_fn=init_env,
            render_mode="rgb_array",
            # env_kwargs={"id": "MountainCar-v0"},
            # env_kwargs={"id": f"{args.env_name}-v0"},
            env_kwargs={"env_cls": env_cls, "hyperparameters": hyperparameters},
            # use_subproc=False,
            seed=args.seed
        )
        venv = gym3.ToBaselinesVecEnv(venv)
        venv = ResizeObservationVec(venv, (64, 64))

    normalize_rew = hyperparameters.get('normalize_rew', True)
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


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
    parser.add_argument('--num_timesteps',    type=int, default = int(25000000), help = 'number of training timesteps')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(1), help='number of checkpoints to store')
    parser.add_argument('--model_file',       type=str)
    parser.add_argument('--num_threads',      type=int, default=8)
    args = parser.parse_args()

    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    num_timesteps = args.num_timesteps
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

    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)

    ############
    ## DEVICE ##
    ############
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')
    env = create_venv(args, hyperparameters)
    env_valid = create_venv(args, hyperparameters, is_valid=True)

    ############
    ## LOGGER ##
    ############
    print('INITIALIZING LOGGER...')
    logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
             str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    print(f'Logging to {logdir}')
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
        model = ImpalaModel(in_channels=in_channels, out_features=64)

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_space_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_space_size)
    else:
        raise NotImplementedError
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    storage_valid = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints,
                  env_valid, storage_valid,  **hyperparameters)
    if args.model_file is not None:
        print("Loading agent from %s" % args.model_file)
        checkpoint = torch.load(args.model_file)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    # import time
    # start = time.time()
    # for i in range(1000):
    #     act = np.array([env.action_space.sample() for _ in range(env.num_envs)])
    #     env.step(act)
    # print("taken:", time.time() - start)
    agent.train(num_timesteps)
