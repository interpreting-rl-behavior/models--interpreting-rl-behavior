from numpy.lib.npyio import save
from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

from pathlib import Path
import os, time, yaml
import gym
from procgen import ProcgenEnv
import torch
import csv
import numpy as np


def load_env_and_agent(exp_name,
                       env_name,
                       num_envs,
                       model_file,
                       start_level,
                       num_levels,
                       distribution_mode="hard",
                       param_name="hard",
                       device="cpu",
                       gpu_device=0,
                       random_percent=0,
                       logdir=None,
                       num_threads=10):

    if env_name != "coinrun":
        raise ValueError("Environment must be coinrun")

    ####################
    ## HYPERPARAMETERS #
    ####################
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]

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
    def create_venv(hyperparameters):
        venv = ProcgenEnv(num_envs=num_envs,
                        env_name=env_name,
                        num_levels=num_levels,
                        start_level=int(start_level),
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
    logger = Logger(num_envs, "/dev/null")

    ###########
    ## MODEL ##
    ###########
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
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, num_envs, device)

    ###########
    ## AGENT ##
    ###########
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, 0, **hyperparameters)

    agent.policy.load_state_dict(torch.load(model_file, map_location=device)["model_state_dict"])
    agent.n_envs = num_envs
    return agent


def load_episode(exp_name, level_seed, **kwargs):
    """Load a single coinrun level with fixed seed. Same level layout after reset
    logdir is just for agent logs."""
    return load_env_and_agent(
        exp_name=exp_name,
        env_name="coinrun",
        num_envs=1,
        num_levels=1,
        start_level=level_seed,
        num_threads=1,
        **kwargs)


##############
##  DEPLOY  ##
##############

def run_env(
    exp_name,
    level_seed,
    logfile=None,
    reset_mode="inv_coin", 
    max_num_timesteps=10_000,
    save_value=False,
    **kwargs):
    """
    Runs one coinrun level.
    Reset modes:
        - inv_coin returns when agent gets the inv coin OR finishes the level
        - complete returns when the agent finishes the level
        - off resets only when max_num_timesteps is reached (repeating always the same level)
    
    returns level metrics. If logfile (csv) is supplied, metrics are also
    appended there.
    """
    if save_value:
        raise NotImplementedError

    if logfile is not None:
        append_to_csv = True

    agent = load_episode(exp_name, level_seed, **kwargs)
    
    obs = agent.env.reset()
    hidden_state = np.zeros((agent.n_envs, agent.storage.hidden_state_size))
    done = np.zeros(agent.n_envs)


    def log_to_csv(metrics):
        """write metrics to csv"""
        if not metrics:
            return
        column_names = ["seed", "steps", "rand_coin", "coin_collected", "inv_coin_collected", "died", "timed_out"]
        metrics = [int(m) for m in metrics]
        if append_to_csv:
            with open(logfile, "a") as f:
                w = csv.writer(f)
                if f.tell() == 0: # write header first
                    w.writerow(column_names)
                w.writerow(metrics)


    def log_metrics(done: bool, info: dict):
        """
        When run complete, log metrics in the 
        following format:
        seed, steps, randomize_goal, collected_coin, collected_inv_coin, died, timed_out
        """
        metrics = None
        if done:
            keys = ["prev_level_seed", "prev_level/total_steps", "prev_level/randomize_goal", "prev_level_complete", "prev_level/invisible_coin_collected"]
            metrics = [info[key] for key in keys]
            if info["prev_level_complete"]:
                metrics.extend([False, False])
            else:
                timed_out = info["prev_level/total_steps"] > 999
                metrics.extend([not timed_out, timed_out])
        elif info["invisible_coin_collected"]:
            keys = ["level_seed", "total_steps", "randomize_goal"]
            metrics = [info[key] for key in keys]
            metrics.extend([-1, True, -1, -1])
        else:
            raise
        log_to_csv(metrics)
        return metrics


    def check_if_break(done: bool, info: dict):
        if reset_mode == "inv_coin":
            return done or info["invisible_coin_collected"]
        elif reset_mode == "complete":
            return done
        elif reset_mode == "off":
            return False
        else:
            raise ValueError("Reset mode must be one of inv_coin, complete, off."
                             f"Instead got {reset_mode}")

    step = 0
    while step < max_num_timesteps:
        agent.policy.eval()
        for _ in range(agent.n_steps):  # = 256
            step += 1
            act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
            next_obs, rew, done, info = agent.env.step(act)

            agent.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state

            if check_if_break(done[0], info[0]):
                log_metrics(done[0], info[0])
                return
    return

