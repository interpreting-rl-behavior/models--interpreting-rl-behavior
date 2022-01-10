from torch._C import Value
from test import load_env_and_agent, run
import argparse
from common import set_global_seeds, set_global_log_levels
import numpy as np
from pathlib import Path
import csv

if __name__=='__main__':
    raise NotImplementedError("I made changes to test.py, so now this script needs to be overhauled")


    parser = argparse.ArgumentParser()
    parser.add_argument('--random_percent_model_dir', type=str, default=None,
                help="directory with saved coinrun models trained on "
                     "environments with coin position randomized "
                     "0, 1, 2, 5, and 10 percent of the time.")

    parser.add_argument('--num_levels_model_dir', type=str, default=None,
                help="directory with saved coinrun models trained on "
                     "environments with different numbers of "
                     "distinct levels.")

    parser.add_argument('--results_dir', type=str, default=None)

    parser.add_argument('--num_timesteps',    type=int, default = 10_000)
    parser.add_argument('--exp_name',         type=str, default = 'compute_metrics', help='experiment name')
    parser.add_argument('--start_level',      type=int, default = np.random.randint(0, 10**9), help='start-level for environment')
    parser.add_argument('--distribution_mode',type=str, default = 'hard', help='distribution mode for environment')
    parser.add_argument('--param_name',       type=str, default = 'hard', help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'cpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--seed',             type=int, default = np.random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--logdir',           type=str, default = None)

    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--num_envs', type=int, default=1)

    args = parser.parse_args()

    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)


    # deploy saved models in --random_percent_model_dir and compute
    # how often the models navigate to the end of the level instead of getting
    # the coin

    def random_percent_ablation():
        def get_agent_path(random_percent):
            """return path of saved agent trained on env with coin randomized
            random_percent of the time"""
            model_path = Path(args.random_percent_model_dir)
            model_path = model_path / f"random_percent_{random_percent}" / "model_200015872.pth"
            return model_path

        if args.results_dir:
            results_dir = Path(args.results_dir)
        else:
            results_dir = Path(args.random_percent_model_dir)

        for random_percent in [0, 1, 2, 5, 10]:
            model_path = get_agent_path(random_percent)
            print(f"Loading agent trained on distribution random_percent_{random_percent}")
            print(f"Loading from {model_path}...")
            print()

            agent = load_env_and_agent(exp_name=args.exp_name,
                                    env_name="coinrun",
                                    num_envs=args.num_envs,
                                    logdir=args.logdir,
                                    model_file=model_path,
                                    start_level=args.start_level,
                                    num_levels=0, # this means start_level is meaningless (level seeds are drawn randomly)
                                    distribution_mode=args.distribution_mode,
                                    param_name=args.param_name,
                                    device=args.device,
                                    gpu_device=args.gpu_device,
                                    seed=args.seed,
                                    num_checkpoints=0,
                                    random_percent=100,
                                    num_threads=args.num_threads)

            print()
            print("Running...")
            results = run(agent, args.num_timesteps, args.logdir)
            results.update({"random_percent": random_percent})

            results_file = str(results_dir / "results.csv")
            print()
            print(f"Saving results to {results_file}")
            print()
            # write results to csv
            if random_percent == 0:
                with open(results_file, "w") as f:
                    w = csv.DictWriter(f, results.keys())
                    w.writeheader()
                    w.writerow(results)
            else:
                with open(results_file, "a") as f:
                    w = csv.DictWriter(f, results.keys())
                    w.writerow(results)
            
            
    def num_levels_ablation():
        def get_agent_path(num_levels):
            model_path = Path(args.num_levels_model_dir)
            model_path = model_path / f"nr_levels_{num_levels}" / "model_200015872.pth"
            return model_path
        
        if args.results_dir:
            results_dir = Path(args.results_dir)
        else:
            results_dir = Path(args.num_levels_model_dir)

        for num_levels in [100, 316, 1000, 3160, 10_000, 31_600, 100_000]:
            model_path = get_agent_path(num_levels)
            print(f"Loading agent trained on distribution nr_levels_{num_levels}")
            print(f"Loading from {model_path}...")
            print()

            agent = load_env_and_agent(exp_name=args.exp_name,
                                    env_name="coinrun",
                                    num_envs=args.num_envs,
                                    logdir=args.logdir,
                                    model_file=model_path,
                                    start_level=args.start_level,
                                    num_levels=0,
                                    distribution_mode=args.distribution_mode,
                                    param_name=args.param_name,
                                    device=args.device,
                                    gpu_device=args.gpu_device,
                                    seed=args.seed,
                                    num_checkpoints=0,
                                    random_percent=100,
                                    num_threads=args.num_threads)

            print()
            print("Running...")

            results = run(agent, args.num_timesteps, args.logdir)
            results.update({"num_levels": num_levels})
            results_file = str(results_dir / "results.csv")

            print()
            print(f"Saving results to {results_file}")
            print()

            # write results to csv
            if num_levels == 100:
                with open(results_file, "w") as f:
                    w = csv.DictWriter(f, results.keys())
                    w.writeheader()
                    w.writerow(results)
            else:
                with open(results_file, "a") as f:
                    w = csv.DictWriter(f, results.keys())
                    w.writerow(results)

    if args.random_percent_model_dir:
        random_percent_ablation()
    
    if args.num_levels_model_dir:
        num_levels_ablation()





