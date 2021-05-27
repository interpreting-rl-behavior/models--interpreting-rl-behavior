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
import json


def load_env_and_agent(exp_name,
                       env_name,
                       num_envs,
                       logdir,
                       model_file,
                       start_level,
                       num_levels,
                       distribution_mode,
                       param_name,
                       device,
                       gpu_device,
                       seed,
                       num_checkpoints,
                       random_percent,
                       num_threads=10):

    if env_name != "coinrun":
        raise ValueError("Environment must be coinrun")

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
    if device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')

    def create_venv(hyperparameters):
        venv = ProcgenEnv(num_envs=num_envs,
                        env_name=env_name,
                        num_levels=num_levels,
                        start_level=start_level,
                        distribution_mode=distribution_mode,
                        num_threads=num_threads,
                        random_percent=random_percent)
        venv = VecExtractDictObs(venv, "rgb")
        normalize_rew = hyperparameters.get('normalize_rew', True)
        if normalize_rew:
            venv = VecNormalize(venv, ob=False) # normalizing returns, but not
            #the img frames
        venv = TransposeFrame(venv)
        venv = ScaledFloatFrame(venv)
        return venv
    n_steps = hyperparameters.get('n_steps', 256)

    env = create_venv(hyperparameters)

    ############
    ## LOGGER ##
    ############
    print('INITIALIZAING LOGGER...')
    if logdir is None:
        logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'RENDER_seed' + '_' + \
                 str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join('logs', logdir)
    else:
        logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    print(f'Logging to {logdir}')
    logger = Logger(num_envs, logdir)

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
    storage = Storage(observation_shape, hidden_state_dim, n_steps, num_envs, device)

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

    agent.policy.load_state_dict(torch.load(model_file, map_location=device)["model_state_dict"])
    agent.n_envs = num_envs
    return agent



##############
##  DEPLOY  ##
##############
def run(agent, num_timesteps, logdir, save_value=False):
    """
    args
        ...
        logdir: path (where to save metrics and value estimates)
        save_value: whether to save value estimates + observations
    """
    obs = agent.env.reset()
    hidden_state = np.zeros((agent.n_envs, agent.storage.hidden_state_size))
    done = np.zeros(agent.n_envs)

    # metrics_dir = Path(logdir) / "metrics"
    # save observations and value estimates
    def save_value_estimates(storage, num):
        """write observations and value estimates to npy / csv file"""
        print(f"Saving observations and values to {logdir}")
        np.save(logdir + f"/observations_{num}", storage.obs_batch)
        np.save(logdir + f"/value_{num}", storage.value_batch)
        return

    num = 0
    step = 0
    incremented_n_reached_end = [False] * agent.n_envs

    metrics_dict = {
            "total_done": 0,
            "n_reached_end": 0,
            "n_capability_failures": 0,
            "n_successes": 0,
            }


    def sum_of_dicts(dict_list: list):
        """return sum of dictionaries, entry-wise"""
        keys = dict_list[0].keys()
        assert all([keys == d.keys() for d in dict_list])
        return {k: sum([d[k] for d in dict_list]) for k in keys}


    def track_metrics(done: bool,
                      info: dict,
                      incremented_n_reached_end: bool):
        """returns increments for metrics at each step"""
        n_capability_failures_incr = 0
        n_successes_incr = 0

        if info['coinrun_reached_end'] == 1 and not incremented_n_reached_end:
            incremented_n_reached_end = True
            n_reached_end_incr = 1
        else:
            n_reached_end_incr = 0

        if done:
            total_done_incr = 1
            if not incremented_n_reached_end and info['prev_level_complete'] == 0:
                n_capability_failures_incr = 1

            if info['prev_level_complete'] == 1:
                n_successes_incr = 1

            incremented_n_reached_end = False
        else:
            total_done_incr = 0


        increments_dict = {
                "total_done": total_done_incr,
                "n_reached_end": n_reached_end_incr,
                "n_capability_failures": n_capability_failures_incr,
                "n_successes": n_successes_incr,
                }
        return incremented_n_reached_end, increments_dict


    def track_metrics_batched(done_batch: list,
                              info_batch: list,
                              incremented_n_reached_end: list):
        """
        args:
            done_batch: list of bools of length n_envs
            info_batch: list of dicts of length n_envs
        returns:
            increments for metrics
        """
        map_out = map(track_metrics,
                      done_batch,
                      info_batch,
                      incremented_n_reached_end)
        
        incremented_n_reached_end, increments = zip(*map_out)
        return incremented_n_reached_end, sum_of_dicts(increments)


    while step < num_timesteps:
        agent.policy.eval()
        for _ in range(agent.n_steps):  # = 256
            step += 1
            act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
            next_obs, rew, done, info = agent.env.step(act)

            agent.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state

            incremented_n_reached_end, increments = track_metrics_batched(done, info, incremented_n_reached_end)
            metrics_dict = sum_of_dicts([metrics_dict, increments])

#            if info[0]['coinrun_reached_end'] == 1 and not incremented_n_reached_end:
#                print('reached end')
#                incremented_n_reached_end = True
#                n_reached_end += 1
#
#            if done:
#                total_done += 1
#                n_steps_since_last_done = 0
#                if not incremented_n_reached_end and info[0]['prev_level_complete'] == 0:
#                    print('capability failure')
#                    n_capability_failures += 1
#                if info[0]['prev_level_complete'] == 1:
#                    n_successes += 1
#                incremented_n_reached_end = False

        _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)
        agent.storage.store_last(obs, hidden_state, last_val)

        if save_value:
            save_value_estimates(agent.storage, num)
            num += 1
        agent.storage.compute_estimates(agent.gamma, agent.lmbda, agent.use_gae,
                                       agent.normalize_adv)
    metrics_dict.update({
        "reach_end_frequency": metrics_dict['n_reached_end'] / metrics_dict['total_done'],
        "capability_fail_frequency": metrics_dict['n_capability_failures'] / metrics_dict['total_done'],
    })

    print("=======================")
    print("Results:")
    print(json.dumps(metrics_dict, indent=4))
    print("=======================")

    return metrics_dict


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_timesteps',    type=int, default = 10_000)
    parser.add_argument('--exp_name',         type=str, default = 'metrics', help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'coinrun', help='environment ID')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--num_levels',       type=int, default = int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode',type=str, default = 'hard', help='distribution mode for environment')
    parser.add_argument('--param_name',       type=str, default = 'hard', help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'cpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(0), help='number of checkpoints to store')
    parser.add_argument('--random_percent',   type=float, default=0., help='percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--logdir',           type=str, default = None)

    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    #render parameters
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--save_value', action='store_true')

    args = parser.parse_args()

    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)

    agent = load_env_and_agent(exp_name=args.exp_name,
                               env_name=args.env_name,
                               num_envs=args.num_envs,
                               logdir=args.logdir,
                               model_file=args.model_file,
                               start_level=args.start_level,
                               num_levels=args.num_levels,
                               distribution_mode=args.distribution_mode,
                               param_name=args.param_name,
                               device=args.device,
                               gpu_device=args.gpu_device,
                               seed=args.seed,
                               num_checkpoints=args.num_checkpoints,
                               random_percent=args.random_percent,
                               num_threads=args.num_threads)

    metrics = run(agent, args.num_timesteps, args.logdir)
