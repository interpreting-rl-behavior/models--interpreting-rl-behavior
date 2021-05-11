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
import pandas as pd
from latent_space_experiment import LatentSpaceExperiment
from datetime import datetime
import torchvision.io as tvio

class TargetFunction():
    def __init__(self, target_function_type, timesteps=None, sim_len=30, device='cuda'):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(TargetFunction, self).__init__()
        self.device = device
        if timesteps is None:
            self.timesteps = list(range(0, sim_len))
        else:
            self.timesteps = timesteps
        COINRUN_ACTIONS = {0: 'downleft', 1: 'left', 2: 'upleft',
                           3: 'down', 4: None, 5: 'up',
                           6: 'downright', 7: 'right', 8: 'upright',
                           9: None, 10: None, 11: None,
                           12: None, 13: None, 14: None}
        if target_function_type == 'left_action_seq':
            self.loss_func = self.left_action_seq_target_function
            self.target_actions = 1
            self.min_loss = 1e-3
            self.num_its = 50

        elif target_function_type == 'value_increase':
            raise NotImplementedError
        elif target_function_type == 'value_decrease':
            raise NotImplementedError
        elif target_function_type == 'high_value':
            raise NotImplementedError
        elif target_function_type == 'low_value':
            raise NotImplementedError

    def left_action_seq_target_function(self, preds_dict):
        preds = preds_dict['act_log_probs']
        preds = torch.stack(preds, dim=1).squeeze()
        seq_len = preds.shape[1]  # Number of ts in sequence
        action_space_dim = preds.shape[2]
        target_actions = 1

        # Make a target log prob that is simply slightly higher than
        # the current prediction.

        target_log_probs = preds.clone().detach().cpu().numpy()
        print(target_log_probs[:, self.timesteps, target_actions].sum())
        target_log_probs[:, self.timesteps, target_actions] += 0.001
        target_log_probs = torch.tensor(target_log_probs, device=self.device)

        # Calculate the difference between the target log probs and the pred
        diff = torch.abs(target_log_probs - preds)
        loss_sum = diff.sum()

        # Calculate the cumulative distribution of the samples' losses and
        # find the top quartile boundary
        diff_cum_df = torch.cumsum(diff.sum(dim=[1,2]), dim=0)
        top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
        loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

        return loss_sum, loss_info_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_function_type', type=str)
    parser.add_argument('--ls_exp_name', type=str, default='target_func_exp',
                        help='type of latent space experiment')
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
    parser.add_argument('--data_dir', type=str, default='generative/data/') # TODO change to test data directory.
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_initializing_steps', type=int, default=8)
    parser.add_argument('--num_sim_steps', type=int, default=22)

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    # Set up args and exp
    args = parser.parse_args()
    tfe = LatentSpaceExperiment(args)  # for 'Target Function Experiment'

    # Prepare save dirs
    targ_func_type_dir = os.path.join(tfe.resdir, args.target_function_type)
    if not (os.path.exists(targ_func_type_dir)):
        os.makedirs(targ_func_type_dir)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_time = session_name
    sess_dir = os.path.join(targ_func_type_dir, session_name)
    if not (os.path.exists(sess_dir)):
        os.makedirs(sess_dir)

    # Make desired target function
    target_func = TargetFunction(target_function_type=args.target_function_type,
                                 sim_len=args.num_sim_steps)

    # Get random starting vectors, which we will optimize
    viz_batch_size = 20
    vae_latent_size = 128
    z_c_size = 64
    z_g_size = 64
    num_epochs = 1

    # remove grads from generative model
    tfe.gen_model.requires_grad = False

    for epoch in range(num_epochs):
        sample_vecs = torch.randn(viz_batch_size, vae_latent_size).to(
            tfe.device)
        sample_vecs = torch.nn.Parameter(sample_vecs)
        sample_vecs.requires_grad = True

        # Set up optimizer for samples
        targ_func_opt = torch.optim.SGD(params=[sample_vecs], momentum=0.3,
                                        lr=1e-4)

        # Start target func optimization loop
        run_target_func_loop = True
        iteration_count = 0
        while run_target_func_loop:
            pred_obs, pred_rews, pred_dones, pred_agent_hs, \
            pred_agent_logprobs, pred_agent_values = \
                tfe.gen_model.decoder(z_c=sample_vecs[:,0:z_c_size],
                                      z_g=sample_vecs[:,z_c_size:z_c_size+z_g_size],
                                      true_actions=None,
                                      true_h0=None)
            preds_dict = {'obs': pred_obs,
                          'hx': pred_agent_hs,
                          'reward': pred_rews,
                          'done': pred_dones,
                          'act_log_probs': pred_agent_logprobs,
                          'value': pred_agent_values}

            # Calculate Target function loss
            target_func_loss, loss_info = target_func.loss_func(preds_dict)
            print("Iteration %i target function loss: %f" % (iteration_count,
                                                             float(target_func_loss.item())))
            # Decide whether to stop the target func optimization
            if loss_info['top_quartile_loss'] < target_func.min_loss or \
               iteration_count > target_func.num_its:
                run_target_func_loop = False

            # Get gradient and step the optimizer
            target_func_loss.backward()
            targ_func_opt.step()

            iteration_count += 1

        # Visualize the optimized latent vectors
        obs = torch.stack(pred_obs, dim=1).squeeze()
        for b in range(viz_batch_size):
            sample = obs[b].permute(0, 2, 3, 1)
            sample = sample * 255
            sample = sample.clone().detach().type(
                torch.uint8).cpu().numpy()
            save_str = sess_dir + '/' + args.target_function_type + \
                       'sample_' + str(epoch) + '_' + str(b) + '_' + str(b) + \
                       '.mp4'
            tvio.write_video(save_str, sample, fps=14)


