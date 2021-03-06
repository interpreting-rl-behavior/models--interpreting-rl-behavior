from gen_model_experiment import GenerativeModelExperiment
import os
import torch
from generative.rssm.functions import safe_normalize
import numpy as np
from generative.procgen_dataset import ProcgenDataset
from overlay_image import overlay_actions, overlay_box_var
import torchvision.io as tvio



class RecordingExperiment(GenerativeModelExperiment):
    """Inherits everything from GenerativeModelExperiment but adds a recording
    method.

    If the CLI arg for 'record_informed_init' is True, then it records samples
    that have been initialized using real data.

    If the CLI arg is False, then it records samples that use a random
    bottleneck vector to initialize.

    It doesn't record saliency stuff. That goes on in the saliency experiments
    script. It doesn't happen here because we use the bottleneck vectors that
    we record here in order to initialize the samples that we record the
    saliency on. This means we can record multiple different types of saliency
    function on the same sample.
    """
    def __init__(self):
        super(RecordingExperiment, self).__init__()
        if self.hp.record.rand_init:
            self.informed_initialization = False
            self.recording_data_save_dir = self.recording_data_save_dir_rand_init
        else:
            self.informed_initialization = True
            self.recording_data_save_dir = self.recording_data_save_dir_informed_init
        self.video_dir = os.path.join(self.recording_data_save_dir, 'videos')
        os.makedirs(self.video_dir, exist_ok=True)
        self.gen_model.num_sim_steps = self.hp.analysis.saliency.num_sim_steps
        self.max_samples = self.hp.record.max_samples

    def run_recording_loop(self):
        # TODO manual actions

        # Prepare for recording cycle
        self.gen_model.train()
        samples_so_far = 0
        # Recording cycle
        for batch_idx, data in enumerate(self.train_loader):
            self.record(data, batch_idx, samples_so_far,
                        informed_initialization=self.informed_initialization )
            samples_so_far += self.hp.gen_model.batch_size
            print(samples_so_far)
            if samples_so_far >= self.max_samples:
                break
        print("Finished recording.")
        # TODO swap directions (will want to record several similar samples:
        #  one with same init but with and without direction swapping.
        #  These will each need different names)

    def record(self, data, batch_idx, samples_so_far,
               informed_initialization=True, name_root='sample'):

        if informed_initialization:
            # Within some recording loop
            # Make all data into floats and put on the right device
            data = self.preprocess_data_dict(data)

            # Forward pass generative model parameters
            self.optimizer.zero_grad()
            (_,
             _,
             _,
             _,
             priors,
             posts,
             samples,
             features,
             env_states,
             env_state,
             metrics_list,
             tensors_list,
             preds_dict,
             unstacked_preds_dict,
             ) = \
                self.gen_model(data=data,
                               use_true_actions=False,
                               use_true_agent_h0=False,
                               imagine=True,
                               calc_loss=False,
                               modal_sampling=True)
        else:
            bottleneck_vec = torch.randn(self.hp.gen_model.batch_size,
                                         self.gen_model.bottleneck_vec_size,
                                         device=self.device)
            bottleneck_vec = safe_normalize(bottleneck_vec)

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

        # Logging and saving info
        if batch_idx % self.hp.gen_model.log_interval == 0:
            self.logger.logkv('batches', batch_idx)
            self.logger.dumpkvs()
        batch_size = self.hp.gen_model.batch_size
        new_sample_indices = range(samples_so_far,
                                   samples_so_far + batch_size)

        self.save_preds(preds_dict, new_sample_indices)

    def save_preds(self, preds, new_sample_indices_range, manual_actions=None,
                   name_root='sample'):

        pred_images, pred_terminals, pred_rews, pred_actions_1hot, \
        pred_actions_inds, \
        pred_act_log_prob, pred_value, pred_agent_h, pred_bottleneck_vec, \
        pred_env_h = self.postprocess_preds(preds)

        # Stack samples into single tensors and convert to numpy arrays
        pred_images = np.array(pred_images.detach().cpu().numpy() * 255,
                            dtype=np.uint8)
        pred_rews = pred_rews.detach().cpu().numpy()
        pred_terminals = pred_terminals.detach().cpu().numpy()
        pred_agent_h = pred_agent_h.detach().cpu().numpy()
        pred_act_log_prob = pred_act_log_prob.detach().cpu().numpy()
        pred_value = pred_value.detach().cpu().numpy()
        pred_env_h = pred_env_h.detach().cpu().numpy()

        # no timesteps in latent vecs, so only cat not stack along time dim.
        pred_bottleneck_vec = pred_bottleneck_vec.detach().cpu().numpy()

        vars = [pred_images, pred_rews, pred_terminals, pred_agent_h,
                pred_act_log_prob, pred_value, pred_env_h,
                pred_bottleneck_vec]
        var_names = ['ims', 'rews', 'dones', 'agent_hs',
                     'agent_logprobs', 'agent_values', 'env_hs',
                     'bottleneck_vec', ]

        # Recover the actions for use in the action overlay
        if manual_actions is not None:
            actions = np.ones(
                (self.hp.gen_model.batch_size, self.hp.analysis.saliency.num_sim_steps)) * manual_actions
        else:
            actions = np.argmax(pred_act_log_prob, axis=-1)
            # Maybe if you ever use this, use pred_actions_inds instead

        # Make dirs for these variables and save variables to dirs and save vid
        sample_dir_base = os.path.join(self.recording_data_save_dir,
                                       name_root + '_')
        for i, new_sample_idx in enumerate(new_sample_indices_range):
            sample_dir = sample_dir_base + f'{new_sample_idx:05d}'
            # Make dirs
            if not (os.path.exists(sample_dir)):
                os.makedirs(sample_dir)

            # Save variables
            for var, var_name in zip(vars, var_names):
                var_sample_name = os.path.join(sample_dir, var_name + '.npy')
                np.save(var_sample_name, var[i])

        # Save vid
        new_sample_indices_range = list(new_sample_indices_range)
        samples_so_far = new_sample_indices_range[0]
        b = len(new_sample_indices_range)
        self.visualize_single(
            0, batch_idx=samples_so_far, data=None, preds=preds,
            bottleneck_vec=None, use_true_actions=False,
            save_dir=self.video_dir,
            save_root='sample', batch_size=b, numbering_scheme="n",
            samples_so_far=samples_so_far)

    def record_manual_actions_patterned(self):

        custom_acts_save_dir = './generative/rec_gen_mod_data/manual_actions/'
        os.makedirs(custom_acts_save_dir, exist_ok=True)

        # Set custom batch size and custom data loader
        custom_bs = 9
        num_manual_actions_samples = 10
        num_init_steps = self.hp.gen_model.num_init_steps
        num_sim_steps = self.hp.analysis.saliency.num_sim_steps_manual_actions
        self.gen_model.num_sim_steps = self.hp.analysis.saliency.num_sim_steps_manual_actions
        num_steps_full = num_init_steps + num_sim_steps - 1
        train_dataset = ProcgenDataset(self.hp.data_dir,
                                       initializer_seq_len=num_init_steps,
                                       num_steps_full=num_steps_full)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=custom_bs,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=2)
        for sample_idx in range(num_manual_actions_samples):
            # Make dataset
            data = self.get_single_batch(train_loader)

            ## Modify actions
            clockwise_actions = [2,1,0,5,4,3,8,7,6]
            dim_inds = torch.tensor(clockwise_actions).to(self.device)
            sim_actions = torch.stack([dim_inds] * num_sim_steps)
            manual_actions = torch.ones_like(data['action']) * 4. # null action
            manual_actions[-num_sim_steps:] = sim_actions
            data['action'] = manual_actions

            ## Copy 0th batch onto other batches so all are the same
            keys_not_action = [k for k in data.keys() if k != 'action']
            for key in keys_not_action:
                template_tensor = data[key][:,0]
                data[key] = torch.stack([template_tensor] * custom_bs, dim=1)

            # Pass data through gen model using manual actions
            self.optimizer.zero_grad()
            (_,
             _,
             _,
             _,
             priors,
             posts,
             samples,
             features,
             env_states,
             env_state,
             metrics_list,
             tensors_list,
             preds_dict,
             unstacked_preds_dict,
             ) = \
                self.gen_model(data=data,
                               use_true_actions=True,
                               use_true_agent_h0=False,
                               imagine=True,
                               calc_loss=False,
                               modal_sampling=True)


            # Save tensors and video
            pred_images, _, _, _, \
            _, _, _, _, pred_bottleneck_vec, \
            _ = self.postprocess_preds(preds_dict)
            pred_images = np.array(pred_images.detach().cpu().numpy() * 255,
                                   dtype=np.uint8)
            pred_bottleneck_vec = pred_bottleneck_vec.detach().cpu().numpy()
            vars = [pred_images, pred_bottleneck_vec]
            var_names = ['ims', 'bottleneck_vec']

            # Save variables
            for var, var_name in zip(vars, var_names):
                for batch_idx in range(0,custom_bs):
                    var_sample_name = os.path.join(custom_acts_save_dir, var_name + f'_{sample_idx:05d}.npy')
                    np.save(var_sample_name, var[batch_idx])


            self.visualize_single(
                epoch=0, batch_idx=sample_idx, data=data, preds=preds_dict,
                bottleneck_vec=None, use_true_actions=True,
                save_dir=custom_acts_save_dir,
                save_root='sample', batch_size=custom_bs, numbering_scheme="ebi",
                samples_so_far=0)

            self.save_grid_video(preds_dict, data, custom_acts_save_dir, sample_idx)

    def save_grid_video(self, preds, data, save_dir, idx, use_true_actions=True):

        viz_batch_size = preds['ims'].shape[1]
        is_square_number = lambda x: np.sqrt(x) % 1 == 0
        assert is_square_number(viz_batch_size)

        pred_images, pred_terminals, pred_rews, pred_actions_1hot, \
        pred_actions_inds, _, _, _, _, _ = self.postprocess_preds(preds)
        # viz_batch_size = min(int(pred_images.shape[0]), 20)

        if use_true_actions:  # N.b. We only ever have true actions with informed init
            true_actions_inds = data['action'][-self.gen_model.num_sim_steps:]
            true_actions_inds = true_actions_inds.permute(1, 0)
            viz_actions_inds = true_actions_inds.clone().cpu().numpy()
        else:
            viz_actions_inds = pred_actions_inds  # .cpu().detach().numpy()

        prepped_vid_tensors = []
        for b in range(viz_batch_size):
            print(b)
            pred_im = pred_images[b]

            # Overlay Done and Reward
            pred_im = overlay_box_var(pred_im, pred_terminals[b], 'left',
                                      max=1.)
            pred_im = overlay_box_var(pred_im, pred_rews[b], 'right',
                                      max=10.)

            # Make predictions and ground truth into right format for
            #  video saving
            pred_im = pred_im * 255
            pred_im = torch.clip(pred_im, 0, 255)
            pred_im = pred_im.clone().detach().type(
                torch.uint8).cpu().numpy()
            pred_im = overlay_actions(pred_im, viz_actions_inds[b], size=16)

            prepped_vid_tensors.append(pred_im)

        # Put vids into grid
        grid_patches_hw = np.sqrt(viz_batch_size).astype(np.int8)
        top_column_inds = np.arange(0, viz_batch_size, grid_patches_hw).astype(np.int8)
        top_column_inds = list(top_column_inds)
        grid_vid_columns = [
            np.concatenate(prepped_vid_tensors[i:i+grid_patches_hw], axis=1)
            for i in top_column_inds]
        grid_vid = np.concatenate(grid_vid_columns, axis=2)


        # Save vid
        save_str = os.path.join(
            save_dir,
            f'grid_manual_actions_{idx}.mp4')
        tvio.write_video(save_str, grid_vid, fps=14)

if __name__ == "__main__":
    recording_exp = RecordingExperiment()
    recording_exp.run_recording_loop()
    # recording_exp.record_manual_actions_patterned()

# Keeping the below because it has template code for manual actions and
# direction swapping
#
# def run(): #TODO update docstring
#     """This script `record_gen_samples.py' uses random vectors to generate a
#     library of samples that we can manually sort through to identify samples
#     with specific behaviours and features. It records the hx and env and
#     other variables so that we can make target functions that optimize for
#     those behaviours and features.
#
#     It saves the obs, hx, env_hx, etc. in a unique folder for that sample.
#
#     e.g. directory name 'generative/rec_gen_mod_data/sample_00001' contains obs.npy, hx.npy, env_hx.npy
#
#     and the directory 'generative/recorded_gen_samples' contains the videos
#     of the samples. There will also be a manually managed csv file that marks
#     each of the thousands of samples with binary markers if they contain certain
#     behaviours.
#
#     This script can also test that the environment model is independent of the agent used to train it.
#     Simply set the cmd arg --manual_action to a direction you wish to set all actions to in the
#     decoder. This will run the gen model but force the agent to take this action at each timestep.
#     Note that you may wish to set the --data_save_dir to the place you would wish to output the
#     video samples.
#     """
#     recording_experiment = GenerativeModelExperiment()
#     recording_experiment.run_recording_loop()
#
# def action_str_to_int(action):
#     """
#     Converts a string representing the action direction to the integer that takes that action
#     in procgen.
#     """
#
# def record_gen_samples(epoch, args, gen_model, batch_size, agent, data, logger, data_dir, device):
#
#     # Set up logging queue objects
#     loss_keys = ['obs', 'hx', 'done', 'reward', 'act_log_probs', 'KL',
#                  'total recon w/o KL']
#     train_info_bufs = {k:deque(maxlen=100) for k in loss_keys}
#     logger.info('Samples recorded: {}'.format(epoch * batch_size))
#
#     # Prepare for recording cycle
#     gen_model.eval()
#
#     coinrun_actions = {'downleft': 0, 'left': 1, 'upleft': 2, 'down': 3, 'up': 5, 'downright': 6,
#                        'right': 7, 'upright': 8}
#
#     if args.manual_action:
#         manual_action = coinrun_actions[args.manual_action]
#         use_true_actions = True
#     else:
#         manual_action = None
#         # Use the predicted actions from the gen model if not using manual actions
#         use_true_actions = False
#
#     # Recording cycle
#     with torch.no_grad():
#
#         # Make all data into floats and put on the right device
#         data = {k: v.to(device).float() for k, v in data.items()}
#
#         # Get input data for generative model
#         full_obs = data['obs']
#         agent_h0 = data['hx'][:, -args.num_sim_steps, :]
#         actions_all = data['action'][:, -args.num_sim_steps:]
#
#         if args.manual_action is not None:
#             actions_all.fill_(manual_action)
#
#         # If doing validation experiments that swap or collapse directions, put
#         # the arguments into the right format.
#         if args.swap_directions_from is not None:
#             assert len(args.swap_directions_from) == \
#                    len(args.swap_directions_to)
#             from_dircs = []
#             to_dircs = []
#             # Convert from strings into the right type (int or None)
#             for from_dirc, to_dirc in zip(args.swap_directions_from,
#                                         args.swap_directions_to):
#                 if from_dirc == 'None':
#                     from_dircs.append(None)
#                 else:
#                     from_dircs.append(int(from_dirc))
#                 if to_dirc == 'None':
#                     to_dircs.append(None)
#                 else:
#                     to_dircs.append(int(to_dirc))
#             swap_directions = [from_dircs, to_dircs]
#         else:
#             swap_directions = None
#         # Forward pass of generative model
#         _, _, _, _, preds = gen_model(full_obs, agent_h0, actions_all,
#                                                           use_true_h0=False,
#                                                           use_true_actions=use_true_actions,
#                                                           swap_directions=swap_directions)
#         # Both of these were false but for some reason it doesn't appear to be
#         #  leading to the same distribution of hxs
#
#         pred_obs = preds['obs']
#         pred_rews = preds['reward']
#         pred_dones = preds['done']
#         pred_agent_hxs = preds['hx']
#         pred_agent_logprobs = preds['act_log_probs']
#         pred_agent_values = preds['values']
#         env_rnn_states = preds['env_hx']
#         bottleneck_vecs = preds['bottleneck_vecs']
#
#         # Stack samples into single tensors and convert to numpy arrays
#         pred_obs = np.array(torch.stack(pred_obs, dim=1).cpu().numpy() * 255, dtype=np.uint8)
#         pred_rews = torch.stack(pred_rews, dim=1).cpu().numpy()
#         pred_dones = torch.stack(pred_dones, dim=1).cpu().numpy()
#         pred_agent_hxs = torch.stack(pred_agent_hxs, dim=1).cpu().numpy()
#         pred_agent_logprobs = torch.stack(pred_agent_logprobs, dim=1).cpu().numpy()
#         pred_agent_values = torch.stack(pred_agent_values, dim=1).cpu().numpy()
#         pred_env_hid_states = torch.stack(env_rnn_states[0], dim=1).cpu().numpy()
#         pred_env_cell_states = torch.stack(env_rnn_states[1], dim=1).cpu().numpy()
#
#         # no timesteps in latent vecs, so only cat not stack along time dim.
#         bottleneck_vecs = torch.cat(bottleneck_vecs, dim=1).cpu().numpy()
#
#         vars = [pred_obs, pred_rews, pred_dones, pred_agent_hxs,
#                 pred_agent_logprobs, pred_agent_values, pred_env_hid_states,
#                 pred_env_cell_states, bottleneck_vecs]
#         var_names = ['obs', 'rews', 'dones', 'agent_hxs',
#                      'agent_logprobs', 'agent_values', 'env_hid_states',
#                      'env_cell_states', 'bottleneck_vec']
#
#         # Recover the actions for use in the action overlay
#         if manual_action is not None:
#             actions = np.ones((args.batch_size, args.num_sim_steps)) * manual_action
#         else:
#             actions = np.argmax(pred_agent_logprobs, axis=-1)
#
#         # Make dirs for these variables and save variables to dirs and save vid
#         samples_so_far = epoch * batch_size
#         new_sample_indices = range(samples_so_far, samples_so_far + batch_size)
#         sample_dir_base = os.path.join(data_dir, 'sample_')
#         for i, new_sample_idx in enumerate(new_sample_indices):
#             sample_dir = sample_dir_base + f'{new_sample_idx:05d}'
#             # Make dirs
#             if not (os.path.exists(sample_dir)):
#                 os.makedirs(sample_dir)
#
#             # Save variables
#             for var, var_name in zip(vars, var_names):
#                 var_sample_name = os.path.join(sample_dir, var_name + '.npy')
#                 np.save(var_sample_name, var[i])
#
#             # Save vid
#             ob = torch.tensor(pred_obs[i])
#             ob = ob.permute(0, 2, 3, 1)
#             ob = ob.clone().detach().type(torch.uint8)
#             ob = ob.cpu().numpy()
#             # Overlay a square in the top right showing the agent's actions
#             ob = overlay_actions(ob, actions[i], size=16)
#             save_str = data_dir + '/sample_' + f'{new_sample_idx:05d}.mp4'
#             tvio.write_video(save_str, ob, fps=14)
#
#     # TODO merge this script with record_random_gen_samples but just use
#     #  conditionals instead of having multiple scripts. Use git diff
#
# def safe_mean(xs):
#     return np.nan if len(xs) == 0 else np.mean(xs)
#
# def done_labels_to_mask(dones, num_unsqueezes=0):
#     argmax_dones = torch.argmax(dones, dim=1)
#     before_dones = torch.ones_like(dones)
#     for batch, argmax_done in enumerate(argmax_dones):
#         if argmax_done > 0:
#             before_dones[batch, argmax_done + 1:] = 0
#
#     # Applies unsqueeze enough times to produce a tensor of the same
#     # order as the masked tensor. It can therefore be broadcast to the
#     # same shape as the masked tensor
#     unsqz_lastdim = lambda x: x.unsqueeze(dim=-1)
#     for _ in range(num_unsqueezes):
#         before_dones = unsqz_lastdim(before_dones)
#
#     return before_dones
#
#
# if __name__ == "__main__":
#     run()

