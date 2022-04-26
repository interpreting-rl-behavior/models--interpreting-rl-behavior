from common.env.procgen_wrappers import *
import os, argparse
import random
import torch
from gen_model_experiment import GenerativeModelExperiment
from generative.rssm.functions import safe_normalize_with_grad
from datetime import datetime
import torchvision.io as tvio
import matplotlib.pyplot as plt
from target_functions import TargetFunction

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
        self.previous_bottleneck_vecs = None
        self.previous_metric = None

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

        if 'value_delta' in self.target_func_types:
            rm_ind = self.target_func_types.index('value_delta')
            self.target_func_types.pop(rm_ind)
            for incr_or_decr in self.hp.analysis.target_func.func_incr_or_decr:
                target_func_name = incr_or_decr + '_value_delta'
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

    def rejection_step(self, current_vecs, proposal_vecs, current_criterion, new_criterion, target_func_opt):
        temp = 100. * target_func_opt.param_groups[0]['lr']
        score_condition = new_criterion < current_criterion
        acceptance_prob = torch.where(score_condition,
                                      torch.ones(self.target_func_batch_size, device=self.device),
                                      torch.exp(-(new_criterion - current_criterion) / temp))
        threshold = torch.rand(self.target_func_batch_size, device=self.device)
        accept_cond = acceptance_prob > threshold
        accept_cond = torch.stack([accept_cond] * proposal_vecs.shape[1], dim=1)
        returned_vecs = torch.where(accept_cond, proposal_vecs, current_vecs)
        return returned_vecs

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
                                                nesterov=True)

                self.optimize_target_func(bottleneck_vecs, target_func, target_func_opt)

    def optimize_target_func(self, bottleneck_vecs, target_func, target_func_opt):
        """Run the forwardbackwardpass+update in a loop"""
        # Start target func optimization loop
        run_target_func_loop = True
        iteration_count = 0
        initial_lr = new_lr = target_func.lr
        start_bottleneck_vecs = bottleneck_vecs.clone().detach()
        prev_criterion = None
        while run_target_func_loop:

            prev_bottleneck_vecs = bottleneck_vecs.clone().detach() # TODO modularise this code
            # bottleneck_vecs = bottleneck_vecs + (0.0001 * new_lr * torch.randn_like(bottleneck_vecs, device=self.device))
            # norm = torch.norm(bottleneck_vecs, dim=1)
            # norm = norm + torch.ones_like(norm, device=self.device) * 1e-7
            # bottleneck_vecs = bottleneck_vecs / norm.unsqueeze(dim=1)

            # Replace the param of the optimizer with the new, normalized vec
            # bottleneck_vecs = torch.nn.Parameter(bottleneck_vecs,
            #                                      requires_grad=True)
            # target_func_opt.param_groups[0]['params'][0] = bottleneck_vecs


            preds_dict, losses, opt_proxy = self.forward_backward_update(bottleneck_vecs, target_func, target_func_opt, iteration_count)

            if target_func.target_function_type in ['action']:
                criterion = losses
            else:
                criterion = opt_proxy
            if prev_criterion is None:
                prev_criterion = criterion

            bottleneck_vecs = self.rejection_step(prev_bottleneck_vecs, bottleneck_vecs, prev_criterion, criterion, target_func_opt)

            # Simulated annealing (Cool temperature/update learning rate & add noise)
            new_lr = initial_lr * (1 - (iteration_count / target_func.num_its))
            target_func_opt.param_groups[0]['lr'] = new_lr
            print("Temp/LR: %f" % target_func_opt.param_groups[0]['lr'])



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
            print("\n")

            # Decide whether to stop the target func optimization
            pairwise_dists = np.linalg.norm(bottleneck_vecs[:, None,
                                            :].detach().clone().cpu().numpy() - \
                                            bottleneck_vecs[None, :,
                                            :].detach().clone().cpu().numpy(),
                                            axis=-1)
            print("Distances between samples: %f" % pairwise_dists.mean())
            print(bottleneck_vecs.shape, start_bottleneck_vecs.shape)
            print("Total distance: %f" % \
                  ((bottleneck_vecs - start_bottleneck_vecs) ** 2).sum().sqrt())

            if iteration_count >= target_func.num_its or \
                    pairwise_dists.mean() < target_func.distance_threshold:
                run_target_func_loop = False

                sample_root = f'sample_{target_func.target_function_type}'
                self.visualize_single(iteration_count,
                         iteration_count,
                         data=None,
                         preds=None,
                          bottleneck_vec=bottleneck_vecs, use_true_actions=False,
                          save_dir=self.sess_dir,
                          save_root=sample_root, batch_size=self.target_func_batch_size, numbering_scheme="ebi",
                          samples_so_far=0)
                self.plot_opt_proxy(target_func)
                self.plot_loss(target_func)
                self.save_results(bottleneck_vecs, target_func)


            # dev
            if iteration_count % 5000 == 0:
                sample_root = f'sample_{target_func.target_function_type}'
                self.visualize_single(iteration_count,
                                      iteration_count,
                                      data=None,
                                      preds=None,
                                      bottleneck_vec=bottleneck_vecs, use_true_actions=False,
                                      save_dir=self.sess_dir,
                                      save_root=sample_root, batch_size=self.target_func_batch_size, numbering_scheme="ebi",
                                      samples_so_far=0)
                self.plot_opt_proxy(target_func)
                self.plot_loss(target_func)
                self.save_results(bottleneck_vecs, target_func)

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
                retain_grads=True,
                env_grads=True, )

        # Calculate target_func loss
        target_func_losses, opt_proxy = \
            target_func.loss_func(unstacked_preds_dict)
        mean_loss = target_func_losses.mean()
        print("Iteration %i target function loss: %f" % (
            iteration_count,
            float(mean_loss.item())))

        # Get gradient and step the optimizer
        mean_loss.backward()

        print("Biggest grad: %f" % torch.abs(
            bottleneck_vec.grad).max().item())
        print("Prenorm grad mean mag: %f" % torch.abs(
            bottleneck_vec.grad).mean())

        target_func_opt.step()

        return preds_dict, target_func_losses, opt_proxy

    def plot_opt_proxy(self, target_func):
        opt_prox_record = torch.stack(target_func.optimized_proxy_record, dim=1)
        opt_prox_record = opt_prox_record.clone().detach().cpu().numpy()
        # plot optimized quantities over time
        for b in range(opt_prox_record.shape[0]):
            plt.plot(range(len(target_func.optimized_proxy_record)),
                     opt_prox_record[b])
        opt_q_plot_str = self.sess_dir + '/plot_opt_proxy_' + \
                         target_func.target_function_type + '.png'
        plt.xlabel("Iterations")
        plt.ylabel(target_func.optimized_proxy_record_name)
        plt.legend(range(opt_prox_record.shape[0]))
        plt.savefig(opt_q_plot_str)
        plt.close()

        # Mean only
        plt.plot(range(len(target_func.optimized_proxy_record)),
                 opt_prox_record.mean(0))
        opt_q_plot_str = self.sess_dir + '/mean_plot_opt_proxy_' + \
                         target_func.target_function_type + '.png'
        plt.xlabel("Iterations")
        plt.ylabel(target_func.optimized_proxy_record_name)
        plt.savefig(opt_q_plot_str)
        plt.close()

    def plot_loss(self, target_func):
        loss_record = np.stack(target_func.loss_record, axis=1)
        # plot optimized quantities over time
        for b in range(loss_record.shape[0]):
            plt.plot(range(len(target_func.loss_record)),
                     loss_record[b])
        opt_q_plot_str = self.sess_dir + '/plot_loss_' + \
                         target_func.target_function_type + '.png'
        plt.xlabel("Iterations")
        plt.ylabel("Loss") #TODO loss name
        plt.legend(range(loss_record.shape[0]))
        plt.savefig(opt_q_plot_str)
        plt.close()

        # Mean only
        plt.plot(range(len(target_func.loss_record)),
                 loss_record.mean(0))
        opt_q_plot_str = self.sess_dir + '/mean_plot_loss_' + \
                         target_func.target_function_type + '.png'
        plt.xlabel("Iterations")
        plt.ylabel("Loss") #TODO loss name
        plt.savefig(opt_q_plot_str)
        plt.close()

    def save_results(self, bottleneck_vecs, target_func):
        """Save the optimized target functions"""
        opt_prox_record = torch.stack(target_func.optimized_proxy_record, dim=1)
        opt_prox_record = opt_prox_record.clone().detach().cpu().numpy()
        loss_record = np.stack(target_func.loss_record, axis=1)

        # Save results
        bottleneck_vec_save_str = self.sess_dir + '/bottleneck_vecs_' + \
                                  target_func.target_function_type + '.npy'
        np.save(bottleneck_vec_save_str,
                bottleneck_vecs.clone().detach().cpu().numpy())

        opt_quant_save_str = self.sess_dir + '/opt_proxy_' + \
                             target_func.target_function_type + '.npy'
        np.save(opt_quant_save_str,
                opt_prox_record)

        loss_save_str = self.sess_dir + '/loss_' + \
                             target_func.target_function_type + '.npy'
        np.save(loss_save_str, loss_record)


    def lsave_results(self, bottleneck_vecs, pred_obs, target_func):
        """Save the optimized target functions"""

        # Save results
        bottleneck_vec_save_str = self.sess_dir + '/bottleneck_vecs_' + \
                                  target_func.target_function_type + '.npy'
        np.save(bottleneck_vec_save_str,
                bottleneck_vecs.clone().detach().cpu().numpy())
        opt_quant_save_str = self.sess_dir + '/opt_quants_' + \
                             target_func.target_function_type + '.npy'
        np.save(opt_quant_save_str,
                np.array(target_func.optimized_proxy))
        # plot optimized quantities over time
        plt.plot(range(len(target_func.optimized_proxy)),
                 target_func.optimized_proxy)
        opt_q_plot_str = self.sess_dir + '/plot_opt_quants_' + \
                         target_func.target_function_type + '.png'
        plt.xlabel("Optimization iterations")
        plt.ylabel(target_func.optimized_proxy_name)
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

