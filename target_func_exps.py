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
import matplotlib.pyplot as plt

class TargetFunction():
    def __init__(self, args, sim_len=30, device='cuda'):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(TargetFunction, self).__init__()
        self.device = device
        print("Target function type: %s" % args.target_function_type)
        self.coinrun_actions = {0: 'downleft', 1: 'left', 2: 'upleft',
                                3: 'down', 4: None, 5: 'up',
                                6: 'downright', 7: 'right', 8: 'upright',
                                9: None, 10: None, 11: None,
                                12: None, 13: None, 14: None}
        # self.coinrun_actions = {k: v for k, v in self.coinrun_actions.items()
        #                         if v is not None}

        # Set default settings (particular target funcs will modify some of
        #  these)
        # TODO add decaying 'temperature' setting that forces samples to be
        #  different
        self.lr = 1e-2
        value_lr = 1e-2
        self.min_loss = 1e-3
        self.num_its = 30000
        num_its_hx = 10000
        self.num_epochs = 1
        self.time_of_jump = min([15, sim_len//2])
        self.origin_attraction_scale = 0.4#0.01
        self.targ_func_loss_scale = 1000.
        self.directions_scale = 0.05
        self.timesteps = list(range(0, sim_len))
        self.distance_threshold = 1.3
        self.target_function_type = args.target_function_type
        num_episodes_precomputed = 2000 # hardcoded for dev

        self.grad_norm = 100.
        value_grad_norm = 10.
        num_its_value = 100000
        self.optimized_quantity = []


        # Set settings for specific target functions
        if self.target_function_type == 'action':
            self.loss_func = self.action_target_function
            self.num_epochs = 15 #len(self.coinrun_actions)
            self.timesteps = (0,1,2)
            self.lr = 1e-1
            self.increment = 10.0
            self.targ_func_loss_scale = 1.
            self.optimized_quantity_name = 'Logit of action minus logit of action with largest logit'
        elif self.target_function_type == 'value_increase':
            self.loss_func = self.value_incr_or_decr_target_function
            self.increment = 1.0
            self.lr = value_lr
            self.targ_func_loss_scale = 1.
            self.grad_norm = value_grad_norm
            self.num_its = num_its_value
            self.optimized_quantity_name = 'Difference between values in 1st and 2nd half of sequence'
        elif self.target_function_type == 'value_decrease':
            self.loss_func = self.value_incr_or_decr_target_function
            self.increment = 1.0 * -1. # because decrease
            self.lr = value_lr
            self.targ_func_loss_scale = 1.
            self.grad_norm = value_grad_norm
            self.num_its = num_its_value
            self.optimized_quantity_name = 'Difference between values in 1st and 2nd half of sequence'
        elif self.target_function_type == 'high_value':
            self.loss_func = self.value_high_or_low_target_function
            self.increment = 1.0
            #self.timesteps = (0,)
            self.lr = value_lr
            self.targ_func_loss_scale = 1.
            self.grad_norm = value_grad_norm
            self.num_its = num_its_value
            self.optimized_quantity_name = 'Mean value during sequence'
        elif self.target_function_type == 'low_value':
            self.loss_func = self.value_high_or_low_target_function
            self.increment = 1.0 * -1. # because decrease
            self.lr = value_lr
            self.targ_func_loss_scale = 1.
            self.grad_norm = value_grad_norm
            self.num_its = num_its_value
            self.optimized_quantity_name = 'Mean value during sequence'
        elif self.target_function_type == 'increase_hx_neuron':
            self.loss_func = self.hx_neuron_target_function
            self.num_epochs = 64
            self.increment = 1.0
            self.timesteps = (0,)
            self.lr = 1e-0
            self.num_its = num_its_hx
            self.optimized_quantity_name = 'Neuron activation'
        elif self.target_function_type == 'decrease_hx_neuron':
            self.loss_func = self.hx_neuron_target_function
            self.num_epochs = 64
            self.increment = 1.0 * -1. # because decrease
            self.timesteps = (0,)
            self.lr = 1e-0
            self.num_its = num_its_hx
            self.optimized_quantity_name = 'Neuron activation'
        elif self.target_function_type == 'increase_hx_direction_pca':
            # self.num_its = 400
            self.loss_func = self.hx_direction_target_function
            directions = np.load(args.precomputed_analysis_dir + \
                                      '/pcomponents_%i.npy' %
                                      num_episodes_precomputed)
            self.num_epochs = directions.shape[0]
            self.timesteps = (0,)
            directions = [directions.copy()
                               for _ in range(len(self.timesteps))]
            self.directions = np.stack(directions, axis=0)
            self.increment = 1.0
            self.lr = 1e-1
            self.optimized_quantity_name = 'Inner product between PC and hidden state'
        elif self.target_function_type == 'decrease_hx_direction_pca':
            self.loss_func = self.hx_direction_target_function
            directions = np.load(args.precomputed_analysis_dir + \
                                      '/pcomponents_%i.npy' %
                                      num_episodes_precomputed)
            self.num_epochs = directions.shape[0]
            self.timesteps = (0,)
            directions = [directions.copy()
                               for _ in range(len(self.timesteps))]
            self.directions = np.stack(directions, axis=0)
            self.increment = 1.0 * -1. # because decrease
            self.lr = 1e-1
            self.optimized_quantity_name = 'Inner product between PC and hidden state'
        elif self.target_function_type == 'increase_hx_direction_nmf':
            self.num_its = 400
            self.loss_func = self.hx_direction_target_function
            directions = np.load(args.precomputed_analysis_dir + \
                                      '/nmf_components_%i.npy' %
                                      num_episodes_precomputed)
            self.num_epochs = directions.shape[0]
            self.timesteps = (0,)
            directions = [directions.copy()
                               for _ in range(len(self.timesteps))]
            self.directions = np.stack(directions, axis=0)
            self.increment = 1.0
            self.lr = 1e-1
            self.optimized_quantity_name = 'Inner product between NMF factor and hidden state'
        elif self.target_function_type == 'decrease_hx_direction_nmf':
            self.loss_func = self.hx_direction_target_function
            directions = np.load(args.precomputed_analysis_dir + \
                                      '/nmf_components_%i.npy' %
                                      num_episodes_precomputed)
            self.num_epochs = directions.shape[0]
            self.timesteps = (0,)
            directions = [directions.copy()
                               for _ in range(len(self.timesteps))]
            self.directions = np.stack(directions, axis=0)
            self.increment = 1.0 * -1. # because decrease
            self.lr = 1e-1
            self.optimized_quantity_name = 'Inner product between NMF factor and hidden state'

    def action_target_function(self, preds_dict, epoch):
        preds = preds_dict['act_log_probs']
        sample_vecs = preds_dict['sample_vecs']
        preds = torch.stack(preds, dim=1).squeeze()

        # Make a target log prob that is simply slightly higher than
        # the current prediction.
        target_action_idx = epoch
        target_log_probs = preds.clone().detach().cpu().numpy()
        argmaxes = target_log_probs[:, self.timesteps].argmax(axis=2)
        opt_quant = \
            target_log_probs[:, self.timesteps, target_action_idx].mean() - \
            target_log_probs[:, self.timesteps, argmaxes].mean()
        self.optimized_quantity.append(opt_quant)
        print(opt_quant)

        target_log_probs[:, self.timesteps, target_action_idx] += \
            self.increment #* 2
        #target_log_probs[:, self.timesteps] -= self.increment
        target_log_probs = torch.tensor(target_log_probs, device=self.device)

        # Calculate the difference between the target log probs and the pred
        diff = torch.abs(target_log_probs - preds)
        loss_sum = diff.mean() * self.targ_func_loss_scale

        # Calculate the cumulative distribution of the samples' losses and
        # find the top quartile boundary
        diff_cum_df = torch.cumsum(diff.sum(dim=[1,2]), dim=0)
        top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
        loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

        # Add l2 loss on size of sample vector to attract the vector toward the
        # origin
        sample_l2_loss = self.origin_attraction_scale*(sample_vecs ** 2).mean()
        print("TargFunc loss: %f ; vec l2 loss: %f " % (loss_sum, sample_l2_loss))
        loss_sum = loss_sum + sample_l2_loss

        return loss_sum, loss_info_dict

    def value_incr_or_decr_target_function(self, preds_dict, epoch):
        preds = preds_dict['value']
        sample_vecs = preds_dict['sample_vecs']
        preds = torch.stack(preds, dim=1).squeeze()

        # Make a target that is simply slightly higher than
        # the current prediction.
        target_values = preds.clone().detach().cpu().numpy()
        print(target_values[:, :self.time_of_jump].mean())
        print(target_values[:, self.time_of_jump:].mean())
        opt_quant = target_values[:, self.time_of_jump:].mean() - \
                    target_values[:, :self.time_of_jump].mean()
        self.optimized_quantity.append(opt_quant)
        base_increments = np.arange(start=-1, stop=1,
                                    step=(2/target_values.shape[1]))
        target_values += base_increments * self.increment
        # target_values[:, :self.time_of_jump] -= self.increment
        # target_values[:, self.time_of_jump:] += self.increment
        target_values = torch.tensor(target_values, device=self.device)

        # Calculate the difference between the target and the pred
        diff = torch.abs(target_values - preds)
        loss_sum = diff.mean() * self.targ_func_loss_scale

        # Calculate the cumulative distribution of the samples' losses and
        # find the top quartile boundary
        diff_cum_df = torch.cumsum(diff.sum(dim=[1]), dim=0)
        top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
        loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

        # Add l2 loss on size of sample vector to attract the vector toward the
        # origin
        sample_l2_loss = self.origin_attraction_scale*(sample_vecs ** 2).mean()
        print("TargFunc loss: %f ; vec l2 loss: %f " % (loss_sum, sample_l2_loss))
        loss_sum = loss_sum + sample_l2_loss

        return loss_sum, loss_info_dict

    def value_high_or_low_target_function(self, preds_dict, epoch):
        preds = preds_dict['value']
        sample_vecs = preds_dict['sample_vecs']
        preds = torch.stack(preds, dim=1).squeeze()

        # Make a target that is simply slightly higher than
        # the current prediction.
        target_values = preds.clone().detach().cpu().numpy()
        print(target_values.mean())
        self.optimized_quantity.append(target_values.mean())
        target_values += self.increment
        target_values = torch.tensor(target_values, device=self.device)

        # Calculate the difference between the target and the pred
        diff = torch.abs(target_values - preds)
        loss_sum = diff.mean() * self.targ_func_loss_scale


        # Calculate the cumulative distribution of the samples' losses and
        # find the top quartile boundary
        diff_cum_df = torch.cumsum(diff.sum(dim=[1]), dim=0)
        top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
        loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

        # Add l2 loss on size of sample vector to attract the vector toward the
        # origin
        sample_l2_loss = self.origin_attraction_scale*(sample_vecs ** 2).mean()
        print("TargFunc loss: %f ; vec l2 loss: %f " % (loss_sum, sample_l2_loss))
        loss_sum = loss_sum + sample_l2_loss

        return loss_sum, loss_info_dict

    def hx_neuron_target_function(self, preds_dict, epoch):
        preds = preds_dict['hx']
        sample_vecs = preds_dict['sample_vecs']
        preds = torch.stack(preds, dim=1).squeeze()
        neuron_optimized = epoch

        # Make a target that is simply slightly higher than
        # the current prediction.

        target_hx = preds.clone().detach().cpu().numpy()
        print(target_hx[:, self.timesteps, neuron_optimized].mean())
        self.optimized_quantity.append(target_hx[:, self.timesteps, neuron_optimized].mean())
        target_hx[:, self.timesteps, neuron_optimized] += self.increment
        target_hx = torch.tensor(target_hx, device=self.device)

        # Calculate the difference between the target and the pred
        diff = torch.abs(target_hx - preds)
        loss_sum = diff.mean() * self.targ_func_loss_scale

        # Calculate the cumulative distribution of the samples' losses and
        # find the top quartile boundary
        diff_cum_df = torch.cumsum(diff.sum(dim=[1,2]), dim=0)
        top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
        loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

        # Add l2 loss on size of sample vector to attract the vector toward the
        # origin
        sample_l2_loss = self.origin_attraction_scale*(sample_vecs ** 2).mean()
        print("TargFunc loss: %f ; vec l2 loss: %f " % (loss_sum, sample_l2_loss))
        loss_sum = loss_sum + sample_l2_loss

        return loss_sum, loss_info_dict

    def hx_direction_target_function(self, preds_dict, epoch):
        preds = preds_dict['hx']
        sample_vecs = preds_dict['sample_vecs']
        preds = torch.stack(preds, dim=1).squeeze()
        directions = self.directions[:, epoch]
        # pred_magnitude = np.linalg.norm(preds[:, self.timesteps], axis=1)
        # directions_magnitude = np.linalg.norm(directions, axis=1)
        # direc_scales = pred_magnitude/directions_magnitude

        # Make a target that is the direction of the target than
        # the current prediction.
        target_hx = preds.clone().detach().cpu().numpy()
        opt_quant = np.inner(target_hx[:, self.timesteps], directions).mean()
        print(opt_quant)
        self.optimized_quantity.append(opt_quant)
        target_hx[:, self.timesteps] += (directions * self.directions_scale)
        target_hx = torch.tensor(target_hx, device=self.device)

        # Calculate the difference between the target and the pred
        diff = torch.abs(target_hx - preds)
        loss_sum = diff.mean() * self.targ_func_loss_scale

        # Calculate the cumulative distribution of the samples' losses and
        # find the top quartile boundary
        diff_cum_df = torch.cumsum(diff.sum(dim=[1,2]), dim=0)
        top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
        loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

        # Add l2 loss on size of sample vector to attract the vector toward the
        # origin
        sample_l2_loss = self.origin_attraction_scale*(sample_vecs ** 2).mean()
        print("TargFunc loss: %f ; vec l2 loss: %f " % (loss_sum, sample_l2_loss))
        loss_sum = loss_sum + sample_l2_loss

        return loss_sum, loss_info_dict












if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_function_type', type=str)
    parser.add_argument('--directions_path', type=str)
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
    parser.add_argument('--precomputed_analysis_dir', type=str,
                        default='analysis/hx_analysis_precomp/') # TODO change to test data directory.

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

    # Device
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

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
    target_func = TargetFunction(args=args,
                                 sim_len=args.num_sim_steps,
                                 device=device)

    # Set some hyperparams
    viz_batch_size = 9
    vae_latent_size = 128
    z_c_size = 64
    z_g_size = 64

    # remove grads from generative model
    tfe.gen_model.requires_grad = False

    for epoch in range(target_func.num_epochs):
        print("Epoch: " + str(epoch))
        # Get random starting vectors, which we will optimize
        sample_vecs = torch.randn(viz_batch_size, vae_latent_size).to(
            tfe.device)
        sample_vecs = torch.nn.Parameter(sample_vecs)
        start_sample_vecs = sample_vecs.detach().clone()
        sample_vecs.requires_grad = True

        # Set up optimizer for samples
        targ_func_opt = torch.optim.SGD(params=[sample_vecs], momentum=0.3,
                                        lr=target_func.lr,
                                        nesterov=True) # TODO try alternatives
        # targ_func_opt = torch.optim.Adam(params=[sample_vecs], lr=target_func.lr)

        # Start target func optimization loop
        run_target_func_loop = True
        iteration_count = 0
        while run_target_func_loop:

            pred_obs, pred_rews, pred_dones, pred_agent_hs, \
            pred_agent_logprobs, pred_agent_values = \
                tfe.gen_model.decoder(z_c=sample_vecs[:,0:z_c_size],
                                      z_g=sample_vecs[:,z_c_size:z_c_size + \
                                                                 z_g_size],
                                      true_actions=None,
                                      true_h0=None)
            preds_dict = {'obs': pred_obs,
                          'hx': pred_agent_hs,
                          'reward': pred_rews,
                          'done': pred_dones,
                          'act_log_probs': pred_agent_logprobs,
                          'value': pred_agent_values,
                          'sample_vecs': sample_vecs}

            # Calculate Target function loss
            target_func_loss, loss_info = target_func.loss_func(preds_dict,
                                                                epoch)
            print("Iteration %i target function loss: %f" % (iteration_count,
                                                             float(target_func_loss.item())))
            # Decide whether to stop the target func optimization
            pairwise_dists = np.linalg.norm(sample_vecs[:,None,:].detach().clone().cpu().numpy() - \
                                            sample_vecs[None,:,:].detach().clone().cpu().numpy(), axis=-1)
            print("Distances between samples: %f" % pairwise_dists.mean())

            # Decide whether to stop loop
            if loss_info['top_quartile_loss'] < target_func.min_loss or \
               iteration_count > target_func.num_its or \
               pairwise_dists.mean() < target_func.distance_threshold:
                run_target_func_loop = False

            # Get gradient and step the optimizer
            target_func_loss.backward()
            print(torch.abs(sample_vecs.grad).max())

            torch.nn.utils.clip_grad_norm_(sample_vecs,
                                           target_func.grad_norm, norm_type=2.0)
            targ_func_opt.step()
            print("Total distance: %f" % \
                  ((sample_vecs - start_sample_vecs)**2).sum().sqrt())
            print("Prenorm grad mean mag: %f" % torch.abs(sample_vecs.grad).mean())

            # Prepare for the next step
            targ_func_opt.zero_grad()
            iteration_count += 1

        # Save results
        sample_vec_save_str = sess_dir + '/sample_vecs_' + \
                              args.target_function_type + \
                              '_' + str(epoch) + '.npy'
        np.save(sample_vec_save_str,
                sample_vecs.clone().detach().cpu().numpy())
        opt_quant_save_str = sess_dir + '/opt_quants_' + \
                              args.target_function_type + \
                              '_' + str(epoch) + '.npy'
        np.save(opt_quant_save_str,
                np.array(target_func.optimized_quantity))
        # plot optimized quantities over time
        plt.plot(range(len(target_func.optimized_quantity)),
                 target_func.optimized_quantity)
        opt_q_plot_str = sess_dir + '/plot_opt_quants_' + \
                              args.target_function_type + \
                              '_' + str(epoch) + '.png'
        plt.xlabel("Optimization iterations")
        plt.ylabel(target_func.optimized_quantity_name)
        plt.savefig(opt_q_plot_str)
        plt.close()
        target_func.optimized_quantity = []



        # Visualize the optimized latent vectors
        obs = torch.stack(pred_obs, dim=1).squeeze()
        overlay_vids = []
        for b in range(viz_batch_size):
            sample = obs[b].permute(0, 2, 3, 1)
            overlay_vids.append(sample)
            sample = sample * 255
            sample = sample.clone().detach().type(
                torch.uint8).cpu().numpy()
            save_str = sess_dir + '/' + args.target_function_type + \
                       '_' + str(epoch) + '_' + str(b) + \
                       '.mp4'
            tvio.write_video(save_str, sample, fps=14)

        # Make another vid with all samples overlaid to help find commonalities
        # between vids
        overlaid_vids = torch.stack(overlay_vids).sum(dim=0) / viz_batch_size
        overlaid_vids = overlaid_vids * 255
        overlaid_vids = overlaid_vids.clone().detach().type(
                                         torch.uint8).cpu().numpy()
        save_str = sess_dir + '/overlay_' + args.target_function_type + \
                   '_' + str(epoch) + \
                   '.mp4'
        tvio.write_video(save_str, overlaid_vids, fps=14)

        # Make another vid that sets 9 samples in a grid
        grid_size = 9 #4 #16
        sqrt_grid_size = int(np.sqrt(grid_size))
        grid_vids = overlay_vids[0:grid_size]
        grid_vids = [(vid * 255).clone().detach().type(torch.uint8).cpu().numpy()
                     for vid in grid_vids]
        grid_rows = [grid_vids[sqrt_grid_size*i:sqrt_grid_size*(i+1)]
                     for i in range(sqrt_grid_size)]
        grid_rows = [np.concatenate(row, axis=2) for row in grid_rows]
        grid = np.concatenate(grid_rows, axis=1)
        save_str = sess_dir + '/grid_' + args.target_function_type + \
                   '_' + str(epoch) + \
                   '.mp4'
        tvio.write_video(save_str, grid, fps=14)



