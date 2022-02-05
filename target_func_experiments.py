from common.env.procgen_wrappers import *
import os, argparse
import random
import torch
from gen_model_experiment import GenerativeModelExperiment
from generative.rssm.functions import safe_normalize_with_grad
from datetime import datetime
import torchvision.io as tvio
import matplotlib.pyplot as plt

class TargetFuncExperiment(GenerativeModelExperiment):
    """Inherits everything from GenerativeModelExperiment but several methods
    for running and recording target function experiments.
    """
    def __init__(self):
        super(TargetFuncExperiment, self).__init__()

        # Set some hyperparams
        self.target_func_batch_size = self.hp.analysis.target_func.batch_size
        self.gen_model.num_sim_steps = self.hp.analysis.saliency.num_sim_steps
        is_square_number = lambda x: np.sqrt(x) % 1 == 0
        assert is_square_number(self.target_func_batch_size)
        self.bottleneck_vec_size = self.hp.gen_model.bottleneck_vec_size
        self.perturbation_scale = self.hp.analysis.saliency.perturbation_scale #0.0001  #was0.01# 2e-2
        self.num_epochs = self.hp.analysis.target_func.num_epochs

        # Whether to use rand init vectors or informed init vectors
        if True:
            self.informed_initialization = False
            self.recording_data_save_dir = self.recording_data_save_dir_rand_init
        else:
            self.informed_initialization = True
            self.recording_data_save_dir = self.recording_data_save_dir_informed_init

        # remove grads from generative model (we'll add them to the bottleneck
        # vectors later)
        self.gen_model.requires_grad = False
        self.target_func_types = self.hp.analysis.target_func.func_type

        ## Automatically make hx_direction names so you don't have to type them
        ## manually # TODO account for 'increasing' 'decreasing' and different direction and neuron indices
        if 'action' in self.target_func_types:
            self.make_indexed_target_func_types(base_type='action',
                                                indices=self.hp.analysis.target_func.action_ids)
        if 'hx_direction' in self.target_func_types:
            self.make_indexed_target_func_types(base_type='hx_direction',
                                                indices=self.hp.analysis.target_func.direction_ids,
                                                front_modifiers=self.hp.analysis.target_func.func_incr_or_decr)
        if 'hx_neuron' in self.target_func_types:
            self.make_indexed_target_func_types(base_type='hx_neuron',
                                                indices=self.hp.analysis.target_func.direction_ids,
                                                front_modifiers=self.hp.analysis.target_func.func_incr_or_decr)

        if 'value' in self.target_func_types:
            rm_ind = self.target_func_types.index('value')
            self.target_func_types.pop(rm_ind)
            for incr_or_decr in self.hp.analysis.target_func.func_incr_or_decr:
                target_func_name = incr_or_decr + '_value'
                self.target_func_types.append(target_func_name)


        ## Get the sample IDs for the samples that we'll use for informed init target funcs
        if 'to' in self.hp.analysis.target_func.sample_ids:
            self.target_func_sample_ids = range(int(self.hp.analysis.target_func.sample_ids[0]), int(self.hp.analysis.target_func.sample_ids[2]))
        else:
            self.target_func_sample_ids = self.hp.analysis.target_func.sample_ids

    def make_indexed_target_func_types(self, base_type, indices, front_modifiers=None,
                                       back_modifiers=None):
        """"For instance, if you want to make target functions for all actions
        you use base_type="action" and indices=['0', 'to', '15']"""
        if base_type in self.target_func_types:
            rm_ind = self.target_func_types.index(base_type)
            self.target_func_types.pop(rm_ind)
            assert base_type not in self.target_func_types

            if 'to' in indices:
                indices = range(
                    int(indices[0]),
                    int(indices[2]))

            self.target_func_indices = indices
            new_names_list = []
            for id in self.target_func_indices:
                target_func_name_id = f'{base_type}_{id}'
                
                if front_modifiers is not None and back_modifiers is not None:
                    for front_mod in front_modifiers:
                        new_target_func_name = front_mod + '_' + target_func_name_id
                        for back_mod in back_modifiers:
                            new_target_func_name = target_func_name_id + '_' + back_mod
                            new_names_list.append(new_target_func_name)
                elif front_modifiers is None and back_modifiers is not None:
                    for back_mod in back_modifiers:
                        new_target_func_name = target_func_name_id + '_' + back_mod
                        new_names_list.append(new_target_func_name)
                elif front_modifiers is not None and back_modifiers is None:
                    for front_mod in front_modifiers:
                        new_target_func_name = front_mod + '_' + target_func_name_id
                        new_names_list.append(new_target_func_name)
                elif front_modifiers is None and back_modifiers is None:
                    new_names_list.append(target_func_name_id)
            self.target_func_types.extend(new_names_list)




    def run_target_func_recording_loop(self):
        """Runs the different target func experiments consecutively."""
        # Prepare for recording cycle
        self.gen_model.eval()
        tfe.gen_model.requires_grad = False

        # Iterate over target func types
        for target_func_type in self.target_func_types:
            print(f"Target function type: {target_func_type}")

            # Make desired target function
            target_func = TargetFunction(target_func_type=target_func_type,
                                         hyperparams=self.hp,
                                         device=self.device)

            for epoch in range(self.num_epochs):
                print("Epoch: " + str(epoch))

                bottleneck_vecs = self.get_bottleneck_vecs(sample_id=epoch)
                bottleneck_vecs = torch.nn.Parameter(bottleneck_vecs,
                                                     requires_grad=True) # I don't know why this needs to be called again, twice in a row, since it was called in get_bottleneck_vecs already, but whatever.

                # Set up optimizer for samples
                target_func_opt = torch.optim.SGD(params=[bottleneck_vecs],
                                                momentum=0.2,
                                                lr=target_func.lr,
                                                nesterov=True)  # TODO try alternatives
                # targ_func_opt = torch.optim.Adam(params=[bottleneck_vecs], lr=target_func.lr)

                self.optimize_target_func(bottleneck_vecs, target_func, target_func_opt)
    #TODO go through changing bottleneck_vecs to vec or vicever
    def optimize_target_func(self, bottleneck_vecs, target_func, target_func_opt):
        """Run the forwardbackwardpass+update in a loop"""
        # Start target func optimization loop
        run_target_func_loop = True
        iteration_count = 0
        start_bottleneck_vecs = bottleneck_vecs.clone().detach()
        while run_target_func_loop:

            preds_dict, loss_info = self.forward_backward_update(bottleneck_vecs, target_func, target_func_opt, iteration_count)

            # Decide whether to stop the target func optimization
            pairwise_dists = np.linalg.norm(bottleneck_vecs[:, None,
                                            :].detach().clone().cpu().numpy() - \
                                            bottleneck_vecs[None, :,
                                            :].detach().clone().cpu().numpy(),
                                            axis=-1)
            print(
                "Distances between samples: %f" % pairwise_dists.mean())

            # Decide whether to stop loop
            # percentile = np.percentile(target_func.loss_record, 75)
            if loss_info['top_quartile_loss'] < target_func.min_loss or \
                    iteration_count > target_func.num_its or \
                    pairwise_dists.mean() < target_func.distance_threshold:
                run_target_func_loop = False

            print("Total distance: %f" % \
                  ((bottleneck_vecs - start_bottleneck_vecs) ** 2).sum().sqrt())
            print("\n")

            # Normalize vector to hypersphere
            norm = torch.norm(bottleneck_vecs, dim=1)
            bottleneck_vecs = bottleneck_vecs / norm.unsqueeze(dim=1)

            # Replace the param of the optimizer with the new, normalized vec
            bottleneck_vecs = torch.nn.Parameter(bottleneck_vecs,
                                                 requires_grad=True)
            target_func_opt.param_groups[0]['params'][0] = bottleneck_vecs

            # Prepare for the next step
            target_func_opt.zero_grad()
            iteration_count += 1
        return preds_dict

    def forward_backward_update(self, bottleneck_vec, target_func,
                                target_func_opt, iteration_count):
        """"""
        (loss_dict_no_grad,
         loss_model,
         loss_agent_aux_init,
         priors,  # tensor(T,B,2S)
         posts,  # tensor(T,B,2S)
         samples,  # tensor(T,B,S)
         features,  # tensor(T,B,D+S)
         env_states,
         (env_h, env_z),
         metrics_list,
         tensors_list,
         preds_dict,
         unstacked_preds_dict,
                     ) = \
            self.gen_model.ae_decode(
                bottleneck_vec,
                data=None,
                true_actions_1hot=None,
                use_true_actions=False,
                true_agent_h0=None,
                use_true_agent_h0=False,
                imagine=True,
                calc_loss=False,
                modal_sampling=True,
                retain_grads=True, )

        # Calculate target_func loss
        target_func_loss, loss_info = \
            target_func.loss_func(unstacked_preds_dict)
        print("Iteration %i target function loss: %f" % (
            iteration_count,
            float(target_func_loss.item())))
        target_func.loss_record.append(target_func_loss)

        # Get gradient and step the optimizer
        target_func_loss.backward()

        print("Biggest grad: %f" % torch.abs(
            bottleneck_vec.grad).max().item())
        print("Prenorm grad mean mag: %f" % torch.abs(
            bottleneck_vec.grad).mean())

        torch.nn.utils.clip_grad_norm_(bottleneck_vec,
                                       target_func.grad_norm,
                                       norm_type=2.0)
        target_func_opt.step()

        return preds_dict, loss_info

    def save_results(self, bottleneck_vecs, pred_obs, target_func):
        """Save the optimized target functions"""

        # Save results
        bottleneck_vec_save_str = self.sess_dir + '/bottleneck_vecs_' + \
                                  target_func.target_function_type + '.npy'
        np.save(bottleneck_vec_save_str,
                bottleneck_vecs.clone().detach().cpu().numpy())
        opt_quant_save_str = self.sess_dir + '/opt_quants_' + \
                             target_func.target_function_type + '.npy'
        np.save(opt_quant_save_str,
                np.array(target_func.optimized_quantity))
        # plot optimized quantities over time
        plt.plot(range(len(target_func.optimized_quantity)),
                 target_func.optimized_quantity)
        opt_q_plot_str = self.sess_dir + '/plot_opt_quants_' + \
                         target_func.target_function_type + '.png'
        plt.xlabel("Optimization iterations")
        plt.ylabel(target_func.optimized_quantity_name)
        plt.savefig(opt_q_plot_str)
        plt.close()

        # Visualize the optimized latent vectors
        obs = torch.stack(pred_obs, dim=1).squeeze()
        overlay_vids = []
        for b in range(self.hp.analysis.target_func.batch_size):
            sample = obs[b].permute(0, 2, 3, 1)
            overlay_vids.append(sample)
            sample = sample * 255
            sample = sample.clone().detach().type(
                torch.uint8).cpu().numpy()
            save_str = self.sess_dir + '/' + target_func.target_function_type + '_' + str(b) + \
                       '.mp4'
            tvio.write_video(save_str, sample, fps=14)

        # Make another vid with all samples overlaid to help find commonalities
        # between vids
        overlaid_vids = torch.stack(overlay_vids).sum(
            dim=0) / self.hp.analysis.target_func.batch_size
        overlaid_vids = overlaid_vids * 255
        overlaid_vids = overlaid_vids.clone().detach().type(
            torch.uint8).cpu().numpy()
        save_str = self.sess_dir + '/overlay_' + target_func.target_function_type + '.mp4'
        tvio.write_video(save_str, overlaid_vids, fps=14)

        # Make another vid that sets 9 samples in a grid
        grid_size = 9  # 4 #16
        sqrt_grid_size = int(np.sqrt(grid_size))
        grid_vids = overlay_vids[0:grid_size]
        grid_vids = [
            (vid * 255).clone().detach().type(torch.uint8).cpu().numpy()
            for vid in grid_vids]
        grid_rows = [
            grid_vids[sqrt_grid_size * i:sqrt_grid_size * (i + 1)]
            for i in range(sqrt_grid_size)]
        grid_rows = [np.concatenate(row, axis=2) for row in grid_rows]
        grid = np.concatenate(grid_rows, axis=1)
        save_str = self.sess_dir + '/grid_' + target_func.target_function_type + '.mp4'
        tvio.write_video(save_str, grid, fps=14)

    def get_bottleneck_vecs(self, sample_id):

        if not self.informed_initialization:
            # Get random starting vectors, which we will optimize
            bottleneck_vecs = torch.randn(self.target_func_batch_size,
                                          self.bottleneck_vec_size,
                                          requires_grad=True,
                                          device=self.device)
            bottleneck_vecs = torch.nn.Parameter(bottleneck_vecs,
                                          requires_grad=True)
            # bottleneck_vecs.requires_grad = True
        else:
            sample_dir = os.path.join(self.recording_data_save_dir,
                                      f'sample_{int(sample_id):05d}')
            bottleneck_vec_path = os.path.join(sample_dir, 'bottleneck_vec.npy')
            bottleneck_vecs = np.load(bottleneck_vec_path)
            bottleneck_vecs = np.stack(
                [bottleneck_vecs] * self.target_func_batch_size)
            bottleneck_vecs = torch.tensor(bottleneck_vecs, device=self.device)
            bottleneck_vecs = torch.nn.Parameter(bottleneck_vecs)
            bottleneck_vecs.requires_grad = True
            perturbation = torch.randn_like(
                bottleneck_vecs) * self.perturbation_scale
            perturbation[0,
            :] = 0.  # So 0th batch is the unperturbed trajectory
            bottleneck_vecs = bottleneck_vecs + perturbation

        bottleneck_vecs = safe_normalize_with_grad(bottleneck_vecs)

        return bottleneck_vecs












class TargetFunction():
    def __init__(self, target_func_type,
                                         hyperparams,
                                         device='cuda'):
        """
        """
        super(TargetFunction, self).__init__()
        self.target_function_type = target_func_type
        self.hp = hyperparams
        self.device = device
        print("Target function type: %s" % self.target_function_type )
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
        self.sim_len = self.hp.analysis.target_func.num_sim_steps
        self.lr = self.hp.analysis.target_func.lr
        value_lr = self.hp.analysis.target_func.value_lr
        self.min_loss = self.hp.analysis.target_func.min_loss
        self.num_its = self.hp.analysis.target_func.num_its
        num_its_hx = self.hp.analysis.target_func.num_its_hx
        num_its_action = self.hp.analysis.target_func.num_its_action
        num_its_direction = self.hp.analysis.target_func.num_its_direction
        num_its_value = self.hp.analysis.target_func.num_its_value

        self.num_epochs = self.hp.analysis.target_func.num_epochs
        self.time_of_jump = min([15, self.sim_len//2])
        self.targ_func_loss_scale = self.hp.analysis.target_func.targ_func_loss_scale
        self.directions_scale = self.hp.analysis.target_func.directions_scale
        self.timesteps = list(range(0, self.sim_len))
        self.distance_threshold = self.hp.analysis.target_func.distance_threshold
        hx_timesteps = (4,)
        directions_timesteps = (4,)
        num_episodes_precomputed = self.hp.analysis.target_func.num_samples_precomputed #4000 # hardcoded for dev
        self.num_episodes_precomputed = self.hp.analysis.target_func.num_samples_precomputed

        self.grad_norm = self.hp.analysis.target_func.grad_norm
        value_grad_norm = self.hp.analysis.target_func.value_grad_norm
        self.optimized_quantity = []
        self.loss_record = []


        # Set settings for specific target functions
        if 'action' in self.target_function_type:
            self.action_id = self.target_function_type.split('_')[-1]
            self.num_epochs = 1
            self.timesteps = (0,1,2)
            self.lr = 1e-1
            self.increment = 1.5
            self.targ_func_loss_scale = 10.
            self.num_its = num_its_action
            self.loss_func = self.make_action_target_function(action_id=self.action_id,
                                                              timesteps=self.timesteps)
            self.optimized_quantity_name = 'Logit of action minus logit of action with largest logit'
        elif 'hx_neuron' in self.target_function_type:
            self.nrn_id = self.target_function_type.split('_')[-1]
            self.num_epochs = 64
            self.increment = 1.0
            self.timesteps = hx_timesteps
            self.loss_func = self.make_hx_neuron_target_function(self.nrn_id,
                                                                 self.timesteps)
        elif 'hx_direction' in self.target_function_type:
            self.direction_id = self.target_function_type.split('_')[-1]
            self.num_epochs = 64
            self.increment = 1.0
            self.timesteps = hx_timesteps
            self.loss_func = self.make_hx_direction_target_function(self.direction_id,
                                                                 self.timesteps)
        elif self.target_function_type == 'increase_value_delta':
            self.loss_func = self.value_incr_or_decr_target_function
            self.increment = 1.0
            self.lr = value_lr
            self.targ_func_loss_scale = 1.
            self.grad_norm = value_grad_norm
            self.num_its = num_its_value
            self.optimized_quantity_name = 'Difference between values in 1st and 2nd half of sequence'
        elif self.target_function_type == 'decrease_value_delta':
            self.loss_func = self.value_incr_or_decr_target_function
            self.increment = 1.0 * -1. # because decrease
            self.lr = value_lr
            self.targ_func_loss_scale = 1.
            self.grad_norm = value_grad_norm
            self.num_its = num_its_value
            self.optimized_quantity_name = 'Difference between values in 1st and 2nd half of sequence'
        elif self.target_function_type == 'increase_value':
            self.loss_func = self.value_high_or_low_target_function
            self.increment = 1.0
            #self.timesteps = (0,)
            self.lr = value_lr
            self.targ_func_loss_scale = 1.
            self.grad_norm = value_grad_norm
            self.num_its = num_its_value
            self.optimized_quantity_name = 'Mean value during sequence'
        elif self.target_function_type == 'decrease_value':
            self.loss_func = self.value_high_or_low_target_function
            self.increment = 1.0 * -1. # because decrease
            self.lr = value_lr
            self.targ_func_loss_scale = 1.
            self.grad_norm = value_grad_norm
            self.num_its = num_its_value
            self.optimized_quantity_name = 'Mean value during sequence'
        elif self.target_function_type == 'increase_hx_neuron':
            self.nrn_id = self.target_function_type.split('_')[-1]


            self.num_epochs = 64
            self.increment = 1.0
            self.timesteps = hx_timesteps
            self.loss_func = self.make_hx_neuron_target_function(self.nrn_id,
                                                                 self.timesteps)
            self.lr = 1e-0
            self.num_its = num_its_hx
            self.optimized_quantity_name = 'Neuron activation'
        elif self.target_function_type == 'decrease_hx_neuron':
            self.loss_func = self.hx_neuron_target_function
            self.num_epochs = 64
            self.increment = 1.0 * -1. # because decrease
            self.timesteps = hx_timesteps
            self.lr = 1e-0
            self.num_its = num_its_hx
            self.optimized_quantity_name = 'Neuron activation'
        elif self.target_function_type == 'increase_hx_direction_pca':
            self.loss_func = self.hx_direction_target_function
            self.increment = 1.0
            self.lr = 1e-1
            self.num_its = num_its_direction
            self.optimized_quantity_name = 'Inner product between PC and hidden state'
        elif self.target_function_type == 'decrease_hx_direction_pca':
            self.loss_func = self.hx_direction_target_function
            directions = np.load(self.hp.analysis.agent_h.precomputed_analysis_data_path + \
                                      '/pcomponents_%i.npy' %
                                      num_episodes_precomputed)
            self.num_epochs = directions.shape[0]
            self.timesteps = directions_timesteps
            directions = [directions.copy()
                               for _ in range(len(self.timesteps))]
            self.directions = np.stack(directions, axis=0)
            # self.increment = 1.0 * -1. # because decrease
            self.directions_scale = 0.05 * -1  # because decrease
            self.num_its = num_its_direction
            self.lr = 1e-1
            self.optimized_quantity_name = 'Inner product between PC and hidden state'
        elif self.target_function_type == 'increase_hx_direction_nmf':
            self.loss_func = self.hx_direction_target_function
            directions = np.load(self.hp.analysis.agent_h.precomputed_analysis_data_path + \
                                      '/nmf_components_%i.npy' %
                                      num_episodes_precomputed)
            self.num_epochs = directions.shape[0]
            self.timesteps = directions_timesteps
            directions = [directions.copy()
                               for _ in range(len(self.timesteps))]
            self.directions = np.stack(directions, axis=0)
            self.increment = 1.0
            self.lr = 1e-1
            self.num_its = num_its_direction
            self.optimized_quantity_name = 'Inner product between NMF factor and hidden state'
        elif self.target_function_type == 'decrease_hx_direction_nmf':
            self.loss_func = self.hx_direction_target_function
            directions = np.load(self.hp.analysis.agent_h.precomputed_analysis_data_path + \
                                      '/nmf_components_%i.npy' %
                                      num_episodes_precomputed)
            self.num_epochs = directions.shape[0]
            self.timesteps = directions_timesteps
            directions = [directions.copy()
                               for _ in range(len(self.timesteps))]
            self.directions = np.stack(directions, axis=0)
            # self.increment = 1.0 * -1. # because decrease
            self.directions_scale = 0.05 * -1  # because decrease
            self.lr = 1e-1
            self.num_its = num_its_direction
            self.optimized_quantity_name = 'Inner product between NMF factor and hidden state'
        elif self.target_function_type == 'hx_location_as_cluster_mean':
            self.loss_func = self.hx_location_target_function
            directions = np.load(self.hp.analysis.agent_h.precomputed_analysis_data_path + \
                                      '/cluster_means_%i.npy' %
                                      num_episodes_precomputed)
            # clusters_to_viz = (15, )
            self.num_epochs = directions.shape[0]
            self.timesteps = directions_timesteps
            directions = [directions.copy()
                          for _ in range(len(self.timesteps))]
            self.directions = np.stack(directions, axis=0)
            self.increment = 1.0
            self.lr = 1e-2
            self.num_its = 90000
            self.targ_func_loss_scale = 15.
            self.optimized_quantity_name = 'Distance of hx from target hx'






    def make_action_target_function(self, action_id, timesteps):
        optimized_quantity = self.optimized_quantity
        increment = self.increment
        device = self.device
        target_func_loss_scale = self.targ_func_loss_scale
        target_action_idx = int(action_id)

        def action_target_function(preds_dict):
            preds = preds_dict['act_log_prob']
            preds = torch.stack(preds, dim=1).squeeze()

            # Make a target log prob that is simply slightly higher than
            # the current prediction.
            target_log_probs = preds.clone().detach().cpu().numpy()
            argmaxes = target_log_probs[:, timesteps].argmax(axis=2)
            logitmaxes = target_log_probs[:, timesteps].max(axis=2)
            total_num_acts = np.product(np.array(argmaxes.shape))
            fraction_correct = (argmaxes == target_action_idx).sum() / total_num_acts
            opt_quant = fraction_correct
            logitlogitmax = \
                (target_log_probs[:, timesteps, target_action_idx] -
                 logitmaxes).mean()
            optimized_quantity.append(opt_quant)
            print("fraction correct: %f" % opt_quant)
            print("logit-maxlogit: %f" % logitlogitmax)

            target_log_probs[:, timesteps, target_action_idx] += \
                increment * 10
            target_log_probs[:, timesteps] -= increment * 3
            target_log_probs = torch.tensor(target_log_probs, device=device)

            # Calculate the difference between the target log probs and the pred
            diff = torch.abs(target_log_probs - preds)
            loss_sum = diff.mean() * target_func_loss_scale

            # Calculate the cumulative distribution of the samples' losses and
            # find the top quartile boundary
            diff_cum_df = torch.cumsum(diff.sum(dim=[1,2]), dim=0)
            top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
            loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

            print("TargFunc loss: %f " % loss_sum)

            return loss_sum, loss_info_dict
        return action_target_function

    def make_hx_neuron_target_function(self, nrn_id, timesteps):
        optimized_quantity = self.optimized_quantity
        increment = self.increment
        device = self.device
        target_func_loss_scale = self.targ_func_loss_scale
        nrn_id = int(nrn_id)

        def hx_neuron_target_function(preds_dict):
            preds = preds_dict['hx']
            bottleneck_vecs = preds_dict['bottleneck_vec']
            preds = torch.stack(preds, dim=1).squeeze()
            neuron_optimized = nrn_id

            # Make a target that is simply slightly higher than
            # the current prediction.

            target_hx = preds.clone().detach().cpu().numpy()
            print(f"Neuron values: {target_hx[:, timesteps, neuron_optimized].mean()}")
            optimized_quantity.append(
                target_hx[:, timesteps, neuron_optimized].mean())
            target_hx[:, timesteps, neuron_optimized] += increment
            target_hx = torch.tensor(target_hx, device=device)

            # Calculate the difference between the target and the pred
            diff = torch.abs(target_hx - preds)
            loss_sum = diff.mean() * target_func_loss_scale

            # Calculate the cumulative distribution of the samples' losses and
            # find the top quartile boundary
            diff_cum_df = torch.cumsum(diff.sum(dim=[1, 2]), dim=0)
            top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
            loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

            print("TargFunc loss: %f " % loss_sum)
            return loss_sum, loss_info_dict
        return hx_neuron_target_function

    def make_hx_direction_target_function(self, direction_id, timesteps):
        optimized_quantity = self.optimized_quantity
        directions_scale = self.hp.analysis.target_func.directions_scale
        increment = self.increment
        device = self.device
        target_func_loss_scale = self.targ_func_loss_scale
        direction_id = int(direction_id)
        directions = np.load(
            self.hp.analysis.agent_h.precomputed_analysis_data_path + \
            '/pcomponents_%i.npy' %
            self.num_episodes_precomputed)
        self.timesteps = timesteps
        directions = [directions.copy()
                      for _ in range(len(self.timesteps))]
        directions = np.stack(directions, axis=0)
        def hx_direction_target_function(preds_dict):
            preds = preds_dict['hx']
            bottleneck_vecs = preds_dict['bottleneck_vec']
            preds = torch.stack(preds, dim=1).squeeze()
            directions_f = directions[:, direction_id]
            # pred_magnitude = np.linalg.norm(preds[:, timesteps], axis=1)
            # directions_magnitude = np.linalg.norm(directions, axis=1)
            # direc_scales = pred_magnitude/directions_magnitude

            # Make a target that is more in the direction of the goal direction than
            # the current prediction.
            target_hx = preds.clone().detach().cpu().numpy()
            opt_quant = np.inner(target_hx[:, timesteps],
                                 directions_f).mean()
            print(opt_quant)
            optimized_quantity.append(opt_quant)
            print("Opt quant: %f" % opt_quant)
            target_hx[:, timesteps] += (directions_f * directions_scale)
            target_hx = torch.tensor(target_hx, device=device)

            # Calculate the difference between the target and the pred
            diff = torch.abs(target_hx - preds)
            loss_sum = diff.mean() * target_func_loss_scale

            # Calculate the cumulative distribution of the samples' losses and
            # find the top quartile boundary
            diff_cum_df = torch.cumsum(diff.sum(dim=[1, 2]), dim=0)
            top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
            loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

            print("TargFunc loss: %f " % loss_sum)
            return loss_sum, loss_info_dict
        return hx_direction_target_function

    def value_incr_or_decr_target_function(self, preds_dict):
        preds = preds_dict['value']
        bottleneck_vecs = preds_dict['bottleneck_vecs']
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

        print("TargFunc loss: %f " % loss_sum)

        return loss_sum, loss_info_dict

    def value_high_or_low_target_function(self, preds_dict):
        preds = preds_dict['value']
        bottleneck_vecs = preds_dict['bottleneck_vec']
        preds = torch.stack(preds, dim=1).squeeze()

        # Make a target that is simply slightly higher than
        # the current prediction.
        target_values = preds.clone().detach().cpu().numpy()
        print(f"Target values: {target_values.mean()}")
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

        print("TargFunc loss: %f " % loss_sum)

        return loss_sum, loss_info_dict

    def hx_neuron_target_function(self, preds_dict, epoch):
        preds = preds_dict['hx']
        bottleneck_vecs = preds_dict['bottleneck_vec']
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

        print("TargFunc loss: %f " % loss_sum)

        return loss_sum, loss_info_dict

    def hx_direction_target_function(self, preds_dict, epoch):
        preds = preds_dict['hx']
        bottleneck_vecs = preds_dict['bottleneck_vec']
        preds = torch.stack(preds, dim=1).squeeze()
        directions = self.directions[:, epoch]
        # pred_magnitude = np.linalg.norm(preds[:, self.timesteps], axis=1)
        # directions_magnitude = np.linalg.norm(directions, axis=1)
        # direc_scales = pred_magnitude/directions_magnitude

        # Make a target that is more in the direction of the goal direction than
        # the current prediction.
        target_hx = preds.clone().detach().cpu().numpy()
        opt_quant = np.inner(target_hx[:, self.timesteps], directions).mean()
        print(opt_quant)
        self.optimized_quantity.append(opt_quant)
        print("Opt quant: %f" % opt_quant)
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

        print("TargFunc loss: %f " % loss_sum)

        return loss_sum, loss_info_dict

    def hx_location_target_function(self, preds_dict, epoch):
        preds = preds_dict['hx']
        bottleneck_vecs = preds_dict['bottleneck_vec']
        preds = torch.stack(preds, dim=1).squeeze()
        directions = self.directions[:, epoch]
        directions = torch.tensor(directions, device=preds.device)

        # Make a target that is the direction of the target than
        # the current prediction.
        # target_hx = preds.clone().detach()#.cpu().numpy()
        # target_hx[:, self.timesteps] += (directions * self.directions_scale)
        # target_hx = torch.tensor(target_hx, device=self.device)

        # Calculate the difference between the target and the pred
        diff = (preds[:, self.timesteps] - directions)**2
        loss_sum = diff.sum(axis=2).mean() * self.targ_func_loss_scale
        opt_quant = loss_sum.item()#.clone().detach()
        print("Loss (distance): %f" % torch.sqrt(loss_sum))
        self.optimized_quantity.append(opt_quant)

        # Calculate the cumulative distribution of the samples' losses and
        # find the top quartile boundary
        diff_cum_df = torch.cumsum(diff.sum(dim=[1, 2]), dim=0)
        top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
        loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

        print("TargFunc loss: %f " % loss_sum)

        return loss_sum, loss_info_dict




if __name__=='__main__':

    tfe = TargetFuncExperiment()
    tfe.run_target_func_recording_loop()


    # # Make desired target function
    # target_func = TargetFunction(args=args,
    #                              sim_len=args.num_sim_steps,
    #                              device=device)
    #
    # # Set some hyperparams
    # viz_batch_size = 9
    # bottleneck_vec_size = 128
    # z_c_size = 64
    # z_g_size = 64
    #
    # # remove grads from generative model
    # tfe.gen_model.requires_grad = False
    #
    # for epoch in range(target_func.num_epochs):
    #     print("Epoch: " + str(epoch))
    #     # Get random starting vectors, which we will optimize
    #     bottleneck_vecs = torch.randn(viz_batch_size, bottleneck_vec_size).to(
    #         tfe.device)
    #     bottleneck_vecs = torch.nn.Parameter(bottleneck_vecs)
    #     start_bottleneck_vecs = bottleneck_vecs.detach().clone()
    #     bottleneck_vecs.requires_grad = True
    #
    #     # Set up optimizer for samples
    #     targ_func_opt = torch.optim.SGD(params=[bottleneck_vecs], momentum=0.3,
    #                                     lr=target_func.lr,
    #                                     nesterov=True) # TODO try alternatives
    #     # targ_func_opt = torch.optim.Adam(params=[bottleneck_vecs], lr=target_func.lr)
    #
    #     # Start target func optimization loop
    #     run_target_func_loop = True
    #     iteration_count = 0
    #     while run_target_func_loop:
    #
    #         pred_obs, pred_rews, pred_dones, pred_agent_hs, \
    #         pred_agent_logprobs, pred_agent_values, pred_env_hs = \
    #             tfe.gen_model.decoder(z_c=bottleneck_vecs[:,0:z_c_size],
    #                                   z_g=bottleneck_vecs[:,z_c_size:z_c_size + \
    #                                                              z_g_size],
    #                                   true_actions=None,
    #                                   true_h0=None)
    #         preds_dict = {'obs': pred_obs,
    #                       'hx': pred_agent_hs,
    #                       'reward': pred_rews,
    #                       'done': pred_dones,
    #                       'act_log_prob': pred_agent_logprobs,
    #                       'value': pred_agent_values,
    #                       'bottleneck_vec': bottleneck_vecs,
    #                       'env_hx': pred_env_hs[0],
    #                       'env_cell_state': pred_env_hs[1]}
    #
    #         # Calculate Target function loss
    #         target_func_loss, loss_info = target_func.loss_func(preds_dict,
    #                                                             epoch)
    #         print("Iteration %i target function loss: %f" % (iteration_count,
    #                                                          float(target_func_loss.item())))
    #         # Decide whether to stop the target func optimization
    #         pairwise_dists = np.linalg.norm(bottleneck_vecs[:,None,:].detach().clone().cpu().numpy() - \
    #                                         bottleneck_vecs[None,:,:].detach().clone().cpu().numpy(), axis=-1)
    #         print("Distances between samples: %f" % pairwise_dists.mean())
    #
    #         # Decide whether to stop loop
    #         if loss_info['top_quartile_loss'] < target_func.min_loss or \
    #            iteration_count > target_func.num_its or \
    #            pairwise_dists.mean() < target_func.distance_threshold:
    #             run_target_func_loop = False
    #
    #         # Get gradient and step the optimizer
    #         target_func_loss.backward()
    #         print("Biggest grad: %f" % torch.abs(bottleneck_vecs.grad).max().item())
    #         print("Prenorm grad mean mag: %f" % torch.abs(bottleneck_vecs.grad).mean())
    #
    #         torch.nn.utils.clip_grad_norm_(bottleneck_vecs,
    #                                        target_func.grad_norm, norm_type=2.0)
    #         targ_func_opt.step()
    #         print("Total distance: %f" % \
    #               ((bottleneck_vecs - start_bottleneck_vecs)**2).sum().sqrt())
    #         print("\n")
    #
    #         # Prepare for the next step
    #         targ_func_opt.zero_grad()
    #         iteration_count += 1
    #
    #     # Save results
    #     bottleneck_vec_save_str = sess_dir + '/bottleneck_vecs_' + \
    #                           args.target_function_type + \
    #                           '_' + str(epoch) + '.npy'
    #     np.save(bottleneck_vec_save_str,
    #             bottleneck_vecs.clone().detach().cpu().numpy())
    #     opt_quant_save_str = sess_dir + '/opt_quants_' + \
    #                           args.target_function_type + \
    #                           '_' + str(epoch) + '.npy'
    #     np.save(opt_quant_save_str,
    #             np.array(target_func.optimized_quantity))
    #     # plot optimized quantities over time
    #     plt.plot(range(len(target_func.optimized_quantity)),
    #              target_func.optimized_quantity)
    #     opt_q_plot_str = sess_dir + '/plot_opt_quants_' + \
    #                           args.target_function_type + \
    #                           '_' + str(epoch) + '.png'
    #     plt.xlabel("Optimization iterations")
    #     plt.ylabel(target_func.optimized_quantity_name)
    #     plt.savefig(opt_q_plot_str)
    #     plt.close()
    #
    #     # Visualize the optimized latent vectors
    #     obs = torch.stack(pred_obs, dim=1).squeeze()
    #     overlay_vids = []
    #     for b in range(viz_batch_size):
    #         sample = obs[b].permute(0, 2, 3, 1)
    #         overlay_vids.append(sample)
    #         sample = sample * 255
    #         sample = sample.clone().detach().type(
    #             torch.uint8).cpu().numpy()
    #         save_str = sess_dir + '/' + args.target_function_type + \
    #                    '_' + str(epoch) + '_' + str(b) + \
    #                    '.mp4'
    #         tvio.write_video(save_str, sample, fps=14)
    #
    #     # Make another vid with all samples overlaid to help find commonalities
    #     # between vids
    #     overlaid_vids = torch.stack(overlay_vids).sum(dim=0) / viz_batch_size
    #     overlaid_vids = overlaid_vids * 255
    #     overlaid_vids = overlaid_vids.clone().detach().type(
    #                                      torch.uint8).cpu().numpy()
    #     save_str = sess_dir + '/overlay_' + args.target_function_type + \
    #                '_' + str(epoch) + \
    #                '.mp4'
    #     tvio.write_video(save_str, overlaid_vids, fps=14)
    #
    #     # Make another vid that sets 9 samples in a grid
    #     grid_size = 9 #4 #16
    #     sqrt_grid_size = int(np.sqrt(grid_size))
    #     grid_vids = overlay_vids[0:grid_size]
    #     grid_vids = [(vid * 255).clone().detach().type(torch.uint8).cpu().numpy()
    #                  for vid in grid_vids]
    #     grid_rows = [grid_vids[sqrt_grid_size*i:sqrt_grid_size*(i+1)]
    #                  for i in range(sqrt_grid_size)]
    #     grid_rows = [np.concatenate(row, axis=2) for row in grid_rows]
    #     grid = np.concatenate(grid_rows, axis=1)
    #     save_str = sess_dir + '/grid_' + args.target_function_type + \
    #                '_' + str(epoch) + \
    #                '.mp4'
    #     tvio.write_video(save_str, grid, fps=14)
    #
    #
    #     target_func.optimized_quantity = []

