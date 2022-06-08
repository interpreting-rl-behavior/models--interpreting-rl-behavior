from common.env.procgen_wrappers import *
import os, argparse
import torch
from gen_model_experiment import GenerativeModelExperiment
from generative.rssm.functions import safe_normalize
from overlay_image import overlay_actions
import torchvision as tv
import torchvision.io as tvio
from PIL import Image
from joblib import dump, load
from dimred_projector import HiddenStateDimensionalityReducer


class SaliencyExperiment(GenerativeModelExperiment):
    """Inherits everything from GenerativeModelExperiment but several methods
    for running and recording saliency experiments.

    It takes bottleneck vectors recorded during record_gen_samples.py and then
    recreates those samples. It then runs saliency functions on those samples.
    """
    def __init__(self):
        super(SaliencyExperiment, self).__init__()

        # Set some hyperparams
        self.saliency_batch_size = self.hp.analysis.saliency.batch_size
        self.gen_model.num_sim_steps = self.hp.analysis.saliency.num_sim_steps
        is_square_number = lambda x: np.sqrt(x) % 1 == 0
        assert is_square_number(self.saliency_batch_size)
        self.bottleneck_vec_size = self.hp.gen_model.bottleneck_vec_size
        self.perturbation_scale = self.hp.analysis.saliency.perturbation_scale #0.0001  #was0.01# 2e-2
        self.direction_type = self.hp.analysis.saliency.direction_type
        # Whether to use rand init vectors or informed init vectors
        if False:
            self.informed_initialization = False
            self.recording_data_save_dir = self.recording_data_save_dir_rand_init
        else:
            self.informed_initialization = True
            self.recording_data_save_dir = self.recording_data_save_dir_informed_init
        self.video_dir = os.path.join(self.recording_data_save_dir, 'videos')
        os.makedirs(self.video_dir, exist_ok=True)
        self.demo_savedir = self.hp.analysis.saliency.demo_savedir
        os.makedirs(self.demo_savedir, exist_ok=True)

        if 'to' in self.hp.analysis.saliency.difference_demo_sample_ids:
            self.difference_demo_sample_ids = range(
                int(self.hp.analysis.saliency.difference_demo_sample_ids[0]),
                int(self.hp.analysis.saliency.difference_demo_sample_ids[2]))
        else:
            self.difference_demo_sample_ids = self.hp.analysis.saliency.direction_ids

        # remove grads from generative model (we'll add them to the bottleneck
        # vectors later)
        self.gen_model.requires_grad = False

        # Determine what to iterate over (func types, samples)
        self.combine_samples_not_iterate = self.hp.analysis.saliency.combine_samples_not_iterate
        self.saliency_func_types = self.hp.analysis.saliency.func_type

        ## Automatically make hx_direction names so you don't have to type them
        ## manually
        if 'hx_direction' in self.saliency_func_types:
            rm_ind = self.saliency_func_types.index('hx_direction')
            self.saliency_func_types.pop(rm_ind)
            assert 'hx_direction' not in self.saliency_func_types

            if 'to' in self.hp.analysis.saliency.direction_ids:
                self.saliency_direction_ids = range(
                    int(self.hp.analysis.saliency.direction_ids[0]),
                    int(self.hp.analysis.saliency.direction_ids[2]))
            else:
                self.saliency_direction_ids = self.hp.analysis.saliency.direction_ids

            for direction_id in self.saliency_direction_ids:
                direction_name = f'hx_direction_{direction_id}_{self.direction_type}'
                self.saliency_func_types.append(direction_name)

        ## Get the sample IDs for the samples that we'll use to make saliency maps
        if 'to' in self.hp.analysis.saliency.sample_ids:
            self.saliency_sample_ids = range(int(self.hp.analysis.saliency.sample_ids[0]), int(self.hp.analysis.saliency.sample_ids[2]))
        else:
            self.saliency_sample_ids = self.hp.analysis.saliency.sample_ids

        # Load up the desired samples and their bottleneck vecs

    def run_saliency_recording_loop(self):
        # TODO manual actions

        # Prepare for recording cycle
        self.gen_model.eval()

        # Iterate over saliency func types and (possibly) over samples
        for saliency_func_type in self.saliency_func_types:
            print(f"Saliency function type: {saliency_func_type}")
            saliency_func = SaliencyFunction(saliency_func_type=saliency_func_type,
                                             hyperparams=self.hp,
                                             device=self.device)

            if self.combine_samples_not_iterate:
                print("Combining samples: " + str(self.saliency_sample_ids))
                bottleneck_vec_name = f"sample_{str(self.saliency_sample_ids[0:3])}"
                bottleneck_vecs = self.combine_bottleneck_vecs(self.saliency_sample_ids)
                self.run_saliency_mapping_and_save(bottleneck_vecs,
                                              bottleneck_vec_name,
                                              saliency_func)

            else:
                for sample_id in self.saliency_sample_ids:
                    print("Sample ID: " + str(sample_id))
                    bottleneck_vec_name = f"sample_{int(sample_id):05d}"
                    bottleneck_vecs = self.get_bottleneck_vecs(sample_id)
                    self.run_saliency_mapping_and_save(bottleneck_vecs,
                                                  bottleneck_vec_name,
                                                  saliency_func)


            # TODO swap directions

    def forward_backward_pass(self, bottleneck_vec, saliency_func, env_grads=True):
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
                env_grads=env_grads)

        # TODO go through making preds vs preds_dict consist
        # for tensor in preds_dict.values():
        #     tensor.retain_grad()
        # preds_dict = {k: v.transpose(0, 1) for k, v in preds_dict.items()}

        # Calculate saliency function loss
        saliency_func_loss = saliency_func.loss_func(unstacked_preds_dict)

        # Get gradient and step the optimizer
        saliency_func_loss.backward(retain_graph=True)

        # Collect into one tensor each
        grads_keys = ['ims', 'hx', 'env_h']
        grads_dict = {}
        for key in grads_keys:
            tensors = unstacked_preds_dict[key]
            grads_initial = [tensor.grad for tensor in tensors]
            null_grad = torch.zeros_like(tensors[0])
            grads = []
            for grad in grads_initial:
                if grad is not None:
                    grads.append(grad)
                else:
                    grads.append(null_grad)
            grads_dict[key] = torch.stack(grads, dim=1).mean(dim=0) # Mean batch dim

        return preds_dict, grads_dict

    def run_saliency_mapping_and_save(self, bottleneck_vecs, bottleneck_vec_name,
                                      saliency_func):
        # Forward and backward pass
        preds_dict, grads_dict = self.forward_backward_pass(bottleneck_vecs,
                                                            saliency_func)

        saliency_func_type = saliency_func.saliency_func_type
        timesteps = saliency_func.timesteps
        savedir = os.path.join(self.recording_data_save_dir,
                               bottleneck_vec_name)
        # Save results
        self.save_results(preds_dict, grads_dict, bottleneck_vec_name,
                     saliency_func_type, timesteps, savedir, savedir)

    def save_results(self, preds_dict, grads_dict, bottleneck_vec_name,
                     saliency_func_type,
                     timesteps, savedir, video_dir, difference_demo=False):

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for key in grads_dict.keys():
            save_str = os.path.join(savedir,
                                    f'grad_{key}_{saliency_func_type}.npy')
            np.save(save_str, grads_dict[key].clone().detach().cpu().numpy())

        # Visualize the optimized latent vectors
        sample_ims = preds_dict['ims'].mean(1).permute(0, 2, 3, 1)
        ims_grad = grads_dict['ims'].permute(0, 2, 3, 1)
        ims_grad = ims_grad.mean(3).unsqueeze(dim=3)  # mean over channel dim

        # Set actions to be those in the unperturbed sample
        actions = preds_dict['action'][:,0].argmax(dim=1).cpu().detach().numpy()

        # Scale according to typical grad sizes for each timestep
        # ims_grad = ims_grad / torch.abs(ims_grad).mean([1, 2]).unsqueeze(
        #     -1).unsqueeze(-1)
        ims_grad = ims_grad / torch.abs(ims_grad).mean()
        blurrer = tv.transforms.GaussianBlur(
            kernel_size=self.hp.analysis.saliency.gaussian_kernel_size,
            sigma=self.hp.analysis.saliency.sigma)#(5, sigma=(5., 6.))
        ims_grad = blurrer(ims_grad.squeeze()).unsqueeze(-1)

        pos_grads = ims_grad.where(ims_grad > 0., torch.zeros_like(ims_grad))
        neg_grads = ims_grad.where(ims_grad < 0.,
                                   torch.zeros_like(ims_grad)).abs()

        # Make a couple of copies of the original im for later
        sample_ims_faint = sample_ims.clone().detach() * 0.2
        sample_ims_faint = sample_ims_faint.mean(3)
        sample_ims_faint = torch.stack([sample_ims_faint] * 3, dim=-1)
        sample_ims_faint = sample_ims_faint * 255
        sample_ims_faint = sample_ims_faint.clone().detach().type(
            torch.uint8).cpu().numpy()

        sample_ims_copy = sample_ims.clone().detach()
        # Colour a patch green so that we know then the gradient is being taken from
        sample_ims_copy[timesteps, 5:11, 18:30, 1] = 1.
        sample_ims_copy = sample_ims_copy * 255
        sample_ims_copy = sample_ims_copy.clone().detach().type(
            torch.uint8).cpu().numpy()
        sample_ims_copy = overlay_actions(sample_ims_copy, actions, size=16)

        # Make the gradient video and save as uint8
        grad_vid = np.zeros_like(sample_ims_copy)
        pos_grads = pos_grads * 0.2 * 255
        neg_grads = neg_grads * 0.2 * 255
        grad_vid[:, :, :, 2] = pos_grads.squeeze().clone().detach().type(
            torch.uint8).cpu().numpy()
        grad_vid[:, :, :, 0] = neg_grads.squeeze().clone().detach().type(
            torch.uint8).cpu().numpy()
        grad_vid = grad_vid + sample_ims_faint
        grad_vid_name = os.path.join(savedir,
                                     f'grad_processed_ims_{saliency_func_type}.npy')
        np.save(grad_vid_name, grad_vid)

        # Save a side-by-side vid
        # Join the prediction and the true image side-by-side
        combined_vid = np.concatenate([sample_ims_copy, grad_vid], axis=2)

        # Save vid
        # combined_vid = combined_vid.clone().detach().type(torch.uint8).cpu().numpy()
        combo_vid_name = f'{bottleneck_vec_name}_saliency_{saliency_func_type}.mp4'
        combo_vid_name = os.path.join(video_dir,
                                      combo_vid_name)
        tvio.write_video(combo_vid_name, combined_vid, fps=14)

        if difference_demo:
            num_frames = sample_ims_copy.shape[0]
            for i in range(num_frames):
                # Save ims from standard vid
                sample_im_name = f'{bottleneck_vec_name}_saliency_{saliency_func_type}_{int(i):03d}.png'
                sample_im_path = os.path.join(video_dir, sample_im_name)
                sample_im = sample_ims_copy[i]
                sample_im = Image.fromarray(sample_im)
                sample_im.save(sample_im_path)

                # Save ims of grads
                grads_im_name = f'{bottleneck_vec_name}_saliency_{saliency_func_type}_grads_{int(i):03d}.png'
                grads_im_path = os.path.join(video_dir, grads_im_name)
                grads_im = grad_vid[i]
                grads_im = Image.fromarray(grads_im)
                grads_im.save(grads_im_path)





    def change_blur_in_saved_ims(self):
        max_samples = 300
        saliency_func_types = ['value', 'action']
        saliency_func_types.extend(['hx_direction_%d' % i for i in range(0, 9)])
        timesteps_used_in_saliency_exp = self.hp.analysis.saliency.common_timesteps

        for sample_idx in range(max_samples):
            for saliency_func_type in saliency_func_types:
                sample_name = f"sample_{int(sample_idx):05d}"
                load_dir = os.path.join(self.recording_data_save_dir,
                                       sample_name)
                sample_im = np.load(os.path.join(load_dir, f'ims.npy'))
                ims_grad_name = os.path.join(load_dir,
                                        f'grad_ims_{saliency_func_type}.npy')
                ims_grad = np.load(ims_grad_name)
                ims_grad = torch.tensor(ims_grad)
                sample_im = torch.tensor(sample_im)
                ims_grad = ims_grad.permute(0, 2, 3, 1)
                ims_grad = ims_grad.mean(3).unsqueeze(
                    dim=3)  # mean over channel dim

                # Scale according to typical grad sizes for each timestep

                # ims_grad = ims_grad / torch.abs(ims_grad).mean([1, 2]).unsqueeze(
                #     -1).unsqueeze(-1)


                ims_grad = ims_grad / torch.abs(ims_grad).mean()
                # blurrer = tv.transforms.GaussianBlur(5, sigma=(5., 6.))
                blurrer = tv.transforms.GaussianBlur(3, sigma=1.)#(1., 1.1))

                ims_grad = blurrer(ims_grad.squeeze()).unsqueeze(-1)

                pos_grads = ims_grad.where(ims_grad > 0.,
                                           torch.zeros_like(ims_grad))
                neg_grads = ims_grad.where(ims_grad < 0.,
                                           torch.zeros_like(ims_grad)).abs()

                # Make a couple of copies of the original im for later
                sample_ims_faint = sample_im.clone().detach() * 0.2
                sample_ims_faint = sample_ims_faint.mean(3)
                sample_ims_faint = torch.stack([sample_ims_faint] * 3, dim=-1)
                # sample_ims_faint = sample_ims_faint * 255
                sample_ims_faint = sample_ims_faint.clone().detach().type(
                    torch.uint8).cpu().numpy()
                #
                sample_ims_copy = sample_im.clone().detach()
                #
                # # Make the gradient video and save as uint8
                grad_vid = np.zeros_like(sample_ims_copy)
                pos_grads = pos_grads * 0.2 * 255
                neg_grads = neg_grads * 0.2 * 255
                grad_vid[:, :, :, 2] = pos_grads.squeeze().clone().detach().type(
                    torch.uint8).cpu().numpy()
                grad_vid[:, :, :, 0] = neg_grads.squeeze().clone().detach().type(
                    torch.uint8).cpu().numpy()
                grad_vid = grad_vid + sample_ims_faint
                # grad_vid = grad_vid.to(torch.uint8)
                grad_vid = grad_vid.astype(np.int8)
                grad_vid_name = os.path.join(load_dir,
                                             f'grad_processed_ims_{saliency_func_type}.npy')
                np.save(grad_vid_name, grad_vid)

                # Save a side-by-side vid
                # Join the prediction and the true image side-by-side
                combined_vid = np.concatenate([sample_ims_copy, grad_vid], axis=2)

                # Save vid
                # combined_vid = combined_vid.clone().detach().type(torch.uint8).cpu().numpy()
                combo_vid_name = f'{sample_name}_saliency_{saliency_func_type}.mp4'
                combo_vid_name = os.path.join(self.video_dir,
                                              combo_vid_name)
                tvio.write_video(combo_vid_name, combined_vid, fps=14)

    def get_bottleneck_vecs(self, sample_id):
        sample_dir = os.path.join(self.recording_data_save_dir,
                                  f'sample_{int(sample_id):05d}')
        bottleneck_vec_path = os.path.join(sample_dir, 'bottleneck_vec.npy')
        bottleneck_vecs = np.load(bottleneck_vec_path)
        bottleneck_vecs = np.stack([bottleneck_vecs] * self.saliency_batch_size)
        bottleneck_vecs = torch.tensor(bottleneck_vecs, device=self.device)
        bottleneck_vecs = torch.nn.Parameter(bottleneck_vecs)
        bottleneck_vecs.requires_grad = True
        perturbation = torch.randn_like(bottleneck_vecs) * self.perturbation_scale
        perturbation[0, :] = 0.  # So 0th batch is the unperturbed trajectory
        bottleneck_vecs = bottleneck_vecs + perturbation
        bottleneck_vecs = safe_normalize(bottleneck_vecs)
        return bottleneck_vecs

    def combine_bottleneck_vecs(self, sample_ids):
        sample_dirs = [os.path.join(self.recording_data_save_dir,
                                    f'sample_{int(sample_id):05d}')
                       for sample_id in sample_ids]
        bottleneck_vec_paths = [os.path.join(sample_dir, 'bottleneck_vec.npy')
                                for sample_dir in sample_dirs]
        bottleneck_vecs = [np.load(bottleneck_vec_path)
                           for bottleneck_vec_path in
                           bottleneck_vec_paths]  # Collect vecs from samples together
        bottleneck_vecs = np.stack(bottleneck_vecs)
        bottleneck_vecs = np.mean(bottleneck_vecs, axis=0)  # Take their mean
        bottleneck_vecs = np.stack([
                                       bottleneck_vecs] * self.saliency_batch_size)  # Create copies of the mean sample vec
        bottleneck_vecs = torch.tensor(bottleneck_vecs, device=self.device)
        bottleneck_vecs = torch.nn.Parameter(bottleneck_vecs)
        # Normalize first so that adding the perturbation below doesn't cause
        # major changes in direction, which might happen if, for instance, the
        # mean of two bottleneck vecs was close to 0 in a particular dimension
        bottleneck_vecs = safe_normalize(bottleneck_vecs)
        bottleneck_vecs.requires_grad = True

        # Add a slight perturbation to the mean sample vecs
        perturbation = torch.randn_like(bottleneck_vecs) * self.perturbation_scale
        perturbation[0, :] = 0.  # So 0th batch is the unperturbed trajectory
        bottleneck_vecs = bottleneck_vecs + perturbation
        bottleneck_vecs = safe_normalize(bottleneck_vecs)
        return bottleneck_vecs

    def run_demo_saliency_differences(self, ):

        self.gen_model.eval()

        # Iterate over saliency func types and (possibly) over samples
        saliency_func_type = 'value'
        print(f"Saliency function type: {saliency_func_type}")
        saliency_func = SaliencyFunction(saliency_func_type=saliency_func_type,
                                         hyperparams=self.hp,
                                         device=self.device)

        for sample_id in self.difference_demo_sample_ids:
            print("Sample ID: " + str(sample_id))
            bottleneck_vec_name = f"sample_{int(sample_id):05d}"
            savedir = os.path.join(self.demo_savedir,
                                   bottleneck_vec_name)
            video_dir = savedir
            bottleneck_vecs = self.get_bottleneck_vecs(sample_id)

            # Forward and backward pass WITH ENV GRADS
            preds_dict, grads_dict = self.forward_backward_pass(bottleneck_vecs,
                                                                saliency_func,
                                                                env_grads=True)

            saliency_func_type = saliency_func.saliency_func_type
            timesteps = saliency_func.timesteps
            # Save results
            self.save_results(preds_dict, grads_dict, bottleneck_vec_name,
                         saliency_func_type, timesteps, savedir, video_dir, difference_demo=True)

            # Forward and backward pass WITHOUT ENV GRADS
            preds_dict_wo, grads_dict_wo = self.forward_backward_pass(bottleneck_vecs,
                                                                saliency_func,
                                                                env_grads=False)

            saliency_func_type = saliency_func_type + '_without_env_grads'
            # Save results
            self.save_results(preds_dict_wo, grads_dict_wo, bottleneck_vec_name,
                         saliency_func_type, timesteps, savedir, video_dir, difference_demo=True)



class SaliencyFunction():
    def __init__(self, saliency_func_type, hyperparams, device='cuda'):
        """
        """
        super(SaliencyFunction, self).__init__()
        self.device = device
        self.hp = hyperparams
        self.direction_type = self.hp.analysis.saliency.direction_type

        self.saliency_func_type = saliency_func_type
        self.coinrun_actions = {0: 'downleft', 1: 'left', 2: 'upleft',
                                3: 'down', 4: None, 5: 'up',
                                6: 'downright', 7: 'right', 8: 'upright',
                                9: None, 10: None, 11: None,
                                12: None, 13: None, 14: None}

        num_analysis_samples = self.hp.analysis.agent_h.num_episodes
        directions_transformer = \
            HiddenStateDimensionalityReducer(self.hp,
                                             self.direction_type,
                                             num_analysis_samples)

        # Set settings for specific saliency functions
        common_timesteps = self.hp.analysis.saliency.common_timesteps
        common_timesteps = tuple(common_timesteps)
        if self.saliency_func_type == 'action':
            self.loss_func = self.action_saliency_loss_function
            self.timesteps = common_timesteps
        elif self.saliency_func_type == 'leftwards':
            self.loss_func = self.action_leftwards_saliency_loss_function
            self.timesteps = common_timesteps
        elif self.saliency_func_type == 'jumping_up':
            self.loss_func = self.action_jumping_up_saliency_loss_function
            self.timesteps = common_timesteps
        elif self.saliency_func_type == 'jumping_right':
            self.loss_func = self.action_jumping_right_saliency_loss_function
            self.timesteps = common_timesteps
        elif self.saliency_func_type == 'value':
            self.loss_func = self.value_saliency_loss_function
            self.timesteps = common_timesteps
        elif self.saliency_func_type == 'value_delta':
            self.loss_func = self.value_delta_saliency_loss_function
        elif 'hx_direction_' in self.saliency_func_type:
            self.timesteps = common_timesteps
            direction_id = int(''.join(filter(str.isdigit, self.saliency_func_type)))
            self.directions_transformer = directions_transformer
            self.loss_func = self.make_direction_saliency_function(
                direction_id,
                self.timesteps,
                self.directions_transformer)

    def make_direction_saliency_function(self,
                                         direction_id,
                                         timesteps,
                                         directions_transformer
                                         ):

        def hx_direction_saliency_loss_function(preds_dict):
            preds = preds_dict['hx']
            preds = torch.stack(preds, dim=1)
            preds = preds[:, timesteps, :]
            print(preds.shape)
            # Scale and project hx onto direction
            preds = directions_transformer.transform(preds)
            print(preds.shape)
            # Then just pick the direction we want to take the saliency of
            preds = preds[:, :, direction_id]

            #     # Then subtract the past from the present to get the delta for this
            #     # direction at the previous timestep
            #     preds = preds[:, 1] - preds[:, 0]

            loss_sum = preds.mean()
            return loss_sum

        return hx_direction_saliency_loss_function

    def action_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_prob']   # TODO it shouldn't be this because if the logprob is negative, then the gradients will be the wrong way round.
        preds = torch.stack(preds, dim=1)
        preds = preds[:, self.timesteps].max(dim=2)[0].squeeze()
        loss_sum = preds.mean()
        return loss_sum

    def action_leftwards_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_prob']
        preds = torch.stack(preds, dim=1)
        preds = preds[:,self.timesteps,:][:,:,(0,1,2)].squeeze()
        loss_sum = preds.mean()
        return loss_sum

    def action_jumping_up_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_prob']
        preds = torch.stack(preds, dim=1)
        preds = preds[:,self.timesteps,(5,)].squeeze()
        loss_sum = preds.mean()
        return loss_sum

    def action_jumping_right_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_prob']
        preds = torch.stack(preds, dim=1)
        preds = preds[:,self.timesteps,(8,)].squeeze()
        loss_sum = preds.mean()
        return loss_sum

    def value_delta_saliency_loss_function(self, preds_dict):
        preds = preds_dict['value']
        preds = torch.stack(preds, dim=1)
        losses = []
        for t in range(preds.shape[1]-1):
            pred_t_plus_1 = preds[:,t+1]
            pred_t = preds[:,t]
            losses.append(pred_t_plus_1 - pred_t)
        losses = torch.stack(losses)
        loss_sum = losses.mean()

        return loss_sum

    def value_saliency_loss_function(self, preds_dict):
        preds = preds_dict['value']
        preds = torch.stack(preds, dim=1)
        preds = preds[:, self.timesteps]
        loss_sum = preds.mean()

        return loss_sum



if __name__ == "__main__":
    saliency_exp = SaliencyExperiment()
    saliency_exp.run_demo_saliency_differences()
    saliency_exp.run_saliency_recording_loop()
    #saliency_exp.change_blur_in_saved_ims()
