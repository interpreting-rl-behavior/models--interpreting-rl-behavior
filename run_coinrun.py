from common.env.procgen_wrappers import *
from common import set_global_seeds, set_global_log_levels
import os, argparse
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
    parser.add_argument('--random_percent',   type=int, default = 0)
    parser.add_argument('--seed_file',        type=str, help="path to text file with env seeds to run on.")
    parser.add_argument('--reset_mode',       type=str, default="inv_coin", help="Reset modes:"
                                                            "- inv_coin returns when agent gets the inv coin OR finishes the level"
                                                            "- complete returns when the agent finishes the level")

    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    #render parameters
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str, help="Can be either a path to a model file, or an "
                                       "integer. Integer is interpreted as random_percent in training")

    args = parser.parse_args()

    # Seeds
    set_global_seeds(args.agent_seed)
    set_global_log_levels(args.log_level)

    if args.seed_file:
        print(f"Loading env seeds from {args.seed_file}")
        with open(args.seed_file, 'r') as f:
            seeds = f.read()
        seeds = [int(s) for s in seeds.split()]
    else:
        print(f"Running on env seeds {args.start_level_seed} to {args.start_level_seed + args.num_seeds}.")
        seeds = np.arange(args.num_seeds) + args.start_level_seed

    # Model file
    def get_model_path(random_percent):
        """return path of saved model trained with random_percent"""
        assert random_percent in range(101)
        logpath = "./logs" if config.on_cluster else "./hpc-logs"
        logpath = os.path.join(logpath, f"train/coinrun/freq-sweep-random-percent-{random_percent}")
        run = list(os.listdir(logpath))[0]
        return os.path.join(logpath, run, "model_80084992.pth")

    datestr = time.strftime("%Y-%m-%d_%H:%M:%S")
    logpath = os.path.join(config.results_dir, f"test_rand_percent_{args.random_percent}")
    try:
        path_to_model_file = get_model_path(int(args.model_file))
        logpath  = os.path.join(logpath, f"train_rand_percent_{args.model_file}") 
    except (ValueError, AssertionError):
        path_to_model_file = args.model_file
        logpath = os.path.join(logpath, f"unkown_model__" + datestr)

    os.makedirs(logpath, exist_ok=True)
    with open(os.path.join(logpath, "metadata.txt"), "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + f", modelfile {path_to_model_file}\n")

    logfile = os.path.join(logpath, f"metrics_agent_seed_{args.agent_seed}.csv")
    print(f"Saving metrics to {logfile}.")
    print(f"Running coinrun with random_percent={args.random_percent}...")
    for env_seed in tqdm(seeds, disable=True):
        run_env(exp_name=args.exp_name,
            logfile=logfile,
            model_file=path_to_model_file,
            level_seed=env_seed,
            device=args.device,
            gpu_device=args.gpu_device,
            random_percent=args.random_percent,
            reset_mode=args.reset_mode)
