"""
Before running this experiment, two important steps must be undertaken:
1. record several episodes of an agent using record.py. If you wish to record an agent taking
a specific action A, then modify the "act" variable inside the while True loop to act=np.array([A])
after it is first declared. A should be an integer representing a coinrun action (as defined in
target_func_exps.TargetFunction.COINRUN_ACTIONS)
2. After recording an agent, modify the "act" variable to
act=torch.tensor([A]*len(act)).to(self.device) after it is declared in ppo.predict_STE. This will
ensure that when a ppo instance is loaded in this file to be used in the decoder, the agent will
always take action A.
"""
import os, yaml, argparse
import gym
import random
import torch
from datetime import datetime

from common import set_global_seeds, set_global_log_levels
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from generative.procgen_dataset import ProcgenDataset
from generative.generative_models import VAE
from train import create_venv
from train_gen_model import demo_recon_quality
import util.logger as logger


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test',
                        help='experiment name')
    parser.add_argument('--tgm_exp_name', type=str, default='test_tgm',
                        help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun',
                        help='environment ID')
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
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999),
                        help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40),
                        help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1),
                        help='number of checkpoints to store')
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--agent_file', type=str)
    parser.add_argument('--data_dir', type=str, default='generative/data/')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_initializing_steps', type=int, default=8)
    parser.add_argument('--num_sim_steps', type=int, default=22)

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

    # Set up environment (Only used for initializing agent)
    print('INITIALIZING ENVIRONMENTS...')
    n_steps = 1#hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
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

    # Load the agent that produced the recordings in data_dir
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
                                               num_workers=0)

    ## Make or load generative model and optimizer
    gen_model = VAE(agent, device, num_initializing_steps, total_seq_len)

    gen_model = gen_model.to(device)
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr)

    # Load the generative model that was trained with a good agent
    checkpoint = torch.load(args.model_file, map_location=device)
    gen_model.load_state_dict(checkpoint['gen_model_state_dict'], device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info('Loaded generative model from {}.'.format(args.model_file))

    # Demonstrate reconsruction and prediction quality by comparing preds
    # with ground truth (the closes thing a VAE gets to validation because
    # there are no labels).
    demo_recon_quality(args, 0, train_loader, optimizer, gen_model,
                        logger, sess_dir, device, use_true_h0=False, use_true_actions=False)

if __name__ == "__main__":
    run()