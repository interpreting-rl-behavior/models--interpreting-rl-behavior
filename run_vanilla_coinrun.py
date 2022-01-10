from common.env.procgen_wrappers import *
from common import set_global_seeds, set_global_log_levels

from pathlib import Path
import os, time, argparse
from procgen import ProcgenEnv
import random
from tqdm import tqdm
import config
import numpy as np

from run_utils import run_env

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'metrics', help='experiment name')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--device',           type=str, default = 'cpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--agent_seed',       type=int, default = random.randint(0,999999), help='Seed for pytorch')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--logdir',           type=str, default = None)
    parser.add_argument('--start_level_seed', type=int, default = 0)
    parser.add_argument('--num_seeds',        type=int, default = 10)

    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    #render parameters
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--save_value', action='store_true')

    args = parser.parse_args()

    set_global_seeds(args.agent_seed)
    set_global_log_levels(args.log_level)

    vanilla_env_seeds = np.arange(args.num_seeds) + args.start_level_seed
    metrics = []

    logpath = config.results_dir + "vanilla-coinrun/"
    if not (os.path.exists(logpath)):
        os.makedirs(logpath)

    #logfile = logpath + f"agent_seed_{args.agent_seed}__date_" + time.strftime("%d-%m-%Y_%H-%M-%S.csv")
    logfile = logpath + "metrics.csv"
    print(f"Saving metrics to {logfile}.")
    print("Running vanilla environment...")
    for env_seed in tqdm(vanilla_env_seeds):
        run_env(exp_name=args.exp_name,
            logfile=logfile,
            model_file=args.model_file,
            level_seed=env_seed,
            device=args.device,
            gpu_device=args.gpu_device,
            random_percent=0,
            reset_mode="inv_coin")
