from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

import os, time, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch

try:
    import wandb
except ImportError:
    pass


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'test', help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'coinrun', help='environment ID')
    parser.add_argument('--val_env_name',     type=str, default = None, help='optional validation environment ID')
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
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--use_wandb',        action="store_true")

    parser.add_argument('--wandb_tags',       type=str, nargs='+')


    parser.add_argument('--random_percent',   type=int, default=0, help='COINRUN: percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--key_penalty',   type=int, default=0, help='HEIST_AISC: Penalty for picking up keys (divided by 10)')
    parser.add_argument('--step_penalty',   type=int, default=0, help='HEIST_AISC: Time penalty per step (divided by 1000)')
    parser.add_argument('--rand_region',   type=int, default=0, help='MAZE: size of region (in upper left corner) in which goal is sampled.')


    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    val_env_name = args.val_env_name if args.val_env_name else args.env_name
    start_level = args.start_level
    start_level_val = random.randint(0, 9999)
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    gpu_device = args.gpu_device
    num_timesteps = int(args.num_timesteps)
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    if args.start_level == start_level_val:
        raise ValueError("Seeds for training and validation envs are equal.")

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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')

    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 256)

    def create_venv(args, hyperparameters, is_valid=False):
        venv = ProcgenEnv(num_envs=n_envs,
                          env_name=val_env_name if is_valid else env_name,
                          num_levels=0 if is_valid else args.num_levels,
                          start_level=start_level_val if is_valid else args.start_level,
                          distribution_mode=args.distribution_mode,
                          num_threads=args.num_threads,
                          random_percent=args.random_percent,
                          step_penalty=args.step_penalty,
                          key_penalty=args.key_penalty,
                          rand_region=args.rand_region)
        venv = VecExtractDictObs(venv, "rgb")
        normalize_rew = hyperparameters.get('normalize_rew', True)
        if normalize_rew:
            venv = VecNormalize(venv, ob=False) # normalizing returns, but not
            #the img frames
        venv = TransposeFrame(venv)
        venv = ScaledFloatFrame(venv)
        return venv

    env = create_venv(args, hyperparameters)
    env_valid = create_venv(args, hyperparameters, is_valid=True)




    ############
    ## LOGGER ##
    ############
    def listdir(path):
        return [os.path.join(path, d) for d in os.listdir(path)]

    def get_latest_model(model_dir):
        """given model_dir with files named model_n.pth where n is an integer,
        return the filename with largest n"""
        steps = [int(filename[6:-4]) for filename in os.listdir(model_dir) if filename.startswith("model_")]
        return list(os.listdir(model_dir))[np.argmax(steps)]

    print('INITIALIZING LOGGER...')

    logdir = os.path.join('logs', 'train', env_name, exp_name)
    if args.model_file == "auto":  # try to figure out which file to load
        logdirs_with_model = [d for d in listdir(logdir) if any(['model' in filename for filename in os.listdir(d)])] 
        if len(logdirs_with_model) > 1:
            raise ValueError("Received args.model_file = 'auto', but there are multiple experiments"
                                f" with saved models under experiment_name {exp_name}.")
        elif len(logdirs_with_model) == 0:
            raise ValueError("Received args.model_file = 'auto', but there are"
                                f" no saved models under experiment_name {exp_name}.")
        model_dir = logdirs_with_model[0]
        args.model_file = os.path.join(model_dir, get_latest_model(model_dir))
        logdir = model_dir # reuse logdir
    else:
        run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{seed}'
        logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    print(f'Logging to {logdir}')
    if args.use_wandb:
        cfg = vars(args)
        cfg.update(hyperparameters)
        wb_resume = "allow" if args.model_file is None else "must"
        wandb.init(project="objective-robustness", config=cfg, tags=args.wandb_tags, resume=wb_resume)
    logger = Logger(n_envs, logdir, use_wandb=args.use_wandb)

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
    agent = AGENT(env, policy, logger, storage, device,
                  num_checkpoints,
                  env_valid=env_valid, 
                  storage_valid=storage_valid,  
                  **hyperparameters)
    if args.model_file is not None:
        print("Loading agent from %s" % args.model_file)
        checkpoint = torch.load(args.model_file)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)
