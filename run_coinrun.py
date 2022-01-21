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

    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    #render parameters
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str, help="Can be either a path to a model file, or an "
                                       "integer. Integer is interpreted as random_percent in training")
    parser.add_argument('--save_value', action='store_true')


    args = parser.parse_args()

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
    metrics = []

    def get_model_path(random_percent):
        """saved model trained with random_percent"""
        assert random_percent in [0,1,2,3,6,11]
        base_path = "../results/random_percent/"
        if random_percent > 0:
            # correct for off by one error during training
            return base_path + f"random_percent_{random_percent-1}/model_200015872.pth"
        else:
            return "../model-files/coinrun.pth"
    
    try:
        model_file = get_model_path(int(args.model_file))
    except (ValueError, AssertionError):
        model_file = args.model_file

    if args.random_percent == 0:
        logpath = config.results_dir + "vanilla-coinrun/"
    elif args.random_percent == 100:
        logpath = config.results_dir + "modified-coinrun/"
    else:
        logpath = config.results_dir + f"coinrun-random_percent_{args.random_percent}/"

    assert int(args.model_file) in range(12)  # TODO allow for arbitrary model_file
    logpath += f"model_rand_percent_{args.model_file}/"

    if not (os.path.exists(logpath)):
        os.makedirs(logpath)

    #logfile = logpath + f"agent_seed_{args.agent_seed}__date_" + time.strftime("%d-%m-%Y_%H-%M-%S.csv")
    logfile = logpath + "metrics.csv"
    print(f"Saving metrics to {logfile}.")
    print(f"Running coinrun with random_percent={args.random_percent}...")
    for env_seed in tqdm(seeds):
        run_env(exp_name=args.exp_name,
            logfile=logfile,
            model_file=model_file,
            level_seed=env_seed,
            device=args.device,
            gpu_device=args.gpu_device,
            random_percent=args.random_percent,
            reset_mode="inv_coin")
