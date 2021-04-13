"""
An experiment that does interpolation in the latent space.

In this experiment, we take two randomly chosen VAE latent space vectors and
we generate samples corresponding to the points along the line between them.

The purpose of this experiment is to show that the latent space has captured
realistic environment dynamics throughout the latent space.

"""
from common.env.procgen_wrappers import *
# from common.logger import Logger
# from common.storage import Storage
# from common.model import NatureModel, ImpalaModel
# from common.policy import CategoricalPolicy
# from common import set_global_seeds, set_global_log_levels

import os, argparse #time, yaml,
# import gym
# from procgen import ProcgenEnv
import random
import torch
#import pandas as pd
from latent_space_experiment import LatentSpaceExperiment
from datetime import datetime
import torchvision.io as tvio

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test',
                        help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun',
                        help='environment ID')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train the generative model')
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
    parser.add_argument('--num_timesteps', type=int, default=int(25000000),
                        help='number of training timesteps')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999),
                        help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40),
                        help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1),
                        help='number of checkpoints to store')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--agent_file', type=str)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_recon_obs', type=int, default=8)
    parser.add_argument('--num_pred_steps', type=int, default=22)

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    # Set up args and exp
    args = parser.parse_args()
    iexp = LatentSpaceExperiment(args)  # for 'Interpolation Experiment'

    # Prepare save dirs
    experiment_type = 'latent_interpolation'
    logdir_base = 'experiments/'
    if not (os.path.exists(logdir_base)):
        os.makedirs(logdir_base)
    resdir = logdir_base + 'results/'
    if not (os.path.exists(resdir)):
        os.makedirs(resdir)
    resdir = resdir + experiment_type
    if not (os.path.exists(resdir)):
        os.makedirs(resdir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get latent size
    latent_size = 128

    # Choose two vecs
    vec1 = torch.normal(0., 1., size=[latent_size]).to(iexp.device)
    vec2 = torch.normal(0., 1., size=[latent_size]).to(iexp.device)

    # Choose number of points along line
    num_interpol_points = 9

    # Get equally spaced points along line
    vecs = []
    interval = 1/num_interpol_points
    for frac in np.arange(0, 1+interval, interval):
        new_vec = (1-frac)*vec1 + frac*vec2
        vecs.append(new_vec)

    # Stack them along batch dimension
    vecs = torch.stack(vecs, dim=0)
    samples = torch.stack(iexp.gen_model.decoder(vecs,
                                                 true_actions=None)[0], dim=1)

    vids = []
    for i in range(num_interpol_points):
        sample = samples[i].permute(0, 2, 3, 1)
        sample = sample * 255
        sample = sample.clone().detach().type(torch.uint8).cpu().numpy()
        vids.append(sample)

    # Join the prediction and the true observation side-by-side
    combined_ob = np.concatenate(vids, axis=2)

    # Save vid
    save_str = resdir + '/' + current_time + '.mp4'
    tvio.write_video(save_str, combined_ob, fps=14)

