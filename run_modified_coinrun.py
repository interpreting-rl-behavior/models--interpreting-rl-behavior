from pathlib import Path
import os, time, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch
import json
import pandas as pd
import csv
from tqdm import tqdm
import config

from common import set_global_seeds, set_global_log_levels
from run_utils import run_env
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed_file',        type=str, help="filename of seed file")
parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
parser.add_argument('--device',           type=str, default = 'cpu', required = False, help='whether to use gpu')
parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
parser.add_argument('--agent_seed',       type=int, default = random.randint(0,9999), help='Seed for pytorch')
parser.add_argument('--logdir',           type=str, default = None)

parser.add_argument('--num_threads', type=int, default=8)
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--vid_dir', type=str, default=None)
parser.add_argument('--model_file', type=str)
parser.add_argument('--save_value', action='store_true')

args = parser.parse_args()

set_global_seeds(args.agent_seed)

with open(args.seed_file, 'r') as f:
    seeds = f.read()
seeds = [int(s) for s in seeds.split()]

logpath = config.results_dir + "modified-coinrun/"
if not (os.path.exists(logpath)):
    os.makedirs(logpath)

logfile = logpath + f"agent_seed_{args.agent_seed}__date_" + time.strftime("%d-%m-%Y_%H-%M-%S.csv")
print(f"Saving metrics to {logfile}.")
print("Running modified environment...")
for env_seed in tqdm(seeds):
    run_env(exp_name="blublbub-expname",
        logfile=logfile,
        model_file=args.model_file,
        level_seed=env_seed,
        distribution_mode="hard",
        param_name="hard",
        device=args.device,
        gpu_device=args.gpu_device,
        random_percent=100,
        reset_mode="complete")
