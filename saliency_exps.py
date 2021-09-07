from common.env.procgen_wrappers import *
import os, argparse
import random
import torch
from gen_model_experiment import GenerativeModelExperiment
from datetime import datetime
import torchvision as tv
import torchvision.io as tvio
import matplotlib.pyplot as plt


class SaliencyFunction():
    def __init__(self, args, saliency_func_type, device='cuda'):
        """
        """
        super(SaliencyFunction, self).__init__()
        self.device = device

        self.saliency_func_type = saliency_func_type
        self.coinrun_actions = {0: 'downleft', 1: 'left', 2: 'upleft',
                                3: 'down', 4: None, 5: 'up',
                                6: 'downright', 7: 'right', 8: 'upright',
                                9: None, 10: None, 11: None,
                                12: None, 13: None, 14: None}

        # Get arrays necessary for projecting hx onto dim-reduced directions
        hx_analysis_dir = os.path.join('analysis', 'hx_analysis_precomp')
        directions_path = os.path.join(hx_analysis_dir, 'pcomponents_1000.npy')
        hx_mu_path = os.path.join(hx_analysis_dir, 'hx_mu_1000.npy')
        hx_std_path = os.path.join(hx_analysis_dir, 'hx_std_1000.npy')
        self.directions = torch.tensor(np.load(directions_path)).to(device).requires_grad_()
        self.directions = self.directions.transpose(0, 1)
        self.hx_mu = torch.tensor(np.load(hx_mu_path)).to(device).requires_grad_()
        self.hx_std = torch.tensor(np.load(hx_std_path)).to(device).requires_grad_()

        # Set settings for specific target functions
        common_timesteps = (8,)#tuple(range(0,28))
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
        elif self.saliency_func_type == 'hx_direction':
            self.loss_func = self.hx_direction_saliency_loss_function
            self.timesteps = common_timesteps
            self.direction_idx = 2

    def action_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_probs']
        preds = torch.stack(preds, dim=1)
        preds = preds[:, self.timesteps].max(dim=2)[0].squeeze()
        loss_sum = preds.mean()
        return loss_sum

    def action_leftwards_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_probs']
        preds = torch.stack(preds, dim=1)
        preds = preds[:,self.timesteps,:][:,:,(0,1,2)].squeeze()
        loss_sum = preds.mean()
        return loss_sum

    def action_jumping_up_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_probs']
        preds = torch.stack(preds, dim=1)
        preds = preds[:,self.timesteps,(5,)].squeeze()
        loss_sum = preds.mean()
        return loss_sum

    def action_jumping_right_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_probs']
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

    def hx_direction_saliency_loss_function(self, preds_dict):
        preds = preds_dict['hx']
        preds = torch.stack(preds, dim=1)
        preds = preds[:, self.timesteps, :]

        # Scale and project hx onto direction
        preds = (preds - self.hx_mu) / self.hx_std
        preds = preds @ (self.directions)

        # Then just pick the direction we want to take the saliency of
        preds = preds[:,:,self.direction_idx]

        loss_sum = preds.mean()
        return loss_sum

def combine_sample_latent_vecs(sample_ids):
    sample_dirs = [os.path.join(args.generated_data_dir,
                              f'sample_{int(sample_id):05d}')
                   for sample_id in sample_ids]
    sample_latent_vec_paths = [os.path.join(sample_dir, 'latent_vec.npy')
                               for sample_dir in sample_dirs]
    sample_vecs = [np.load(sample_latent_vec_path)
                   for sample_latent_vec_path in sample_latent_vec_paths]  # Collect vecs from samples together
    sample_vecs = np.stack(sample_vecs)
    sample_vecs = np.mean(sample_vecs, axis=0)  # Take their mean
    sample_vecs = np.stack([sample_vecs] * args.batch_size)  # Create copies of the mean sample vec
    sample_vecs = torch.tensor(sample_vecs, device=device)
    sample_vecs = torch.nn.Parameter(sample_vecs)
    sample_vecs.requires_grad = True

    # Add a slight perturbation to the mean sample vecs
    perturbation = torch.randn_like(sample_vecs) * perturbation_scale
    perturbation[0, :] = 0.  # So 0th batch is the unperturbed trajectory
    latent_vecs = sample_vecs + perturbation
    return latent_vecs

def get_sample_latent_vecs(sample_id):
    sample_dir = os.path.join(args.generated_data_dir,
                              f'sample_{int(sample_id):05d}')
    sample_latent_vec_path = os.path.join(sample_dir, 'latent_vec.npy')
    sample_vecs = np.load(sample_latent_vec_path)
    sample_vecs = np.stack([sample_vecs] * args.batch_size)
    sample_vecs = torch.tensor(sample_vecs, device=device)
    sample_vecs = torch.nn.Parameter(sample_vecs)
    sample_vecs.requires_grad = True
    perturbation = torch.randn_like(sample_vecs) * perturbation_scale
    perturbation[0, :] = 0.  # So 0th batch is the unperturbed trajectory
    latent_vecs = sample_vecs + perturbation
    return latent_vecs

def forward_backward_pass(sample_vecs, sfe, saliency_func):
    pred_obs, pred_rews, pred_dones, pred_agent_hs, \
    pred_agent_logprobs, pred_agent_values, pred_env_hs = \
        sfe.gen_model.decoder(z_c=sample_vecs[:, 0:z_c_size],
                              z_g=sample_vecs[:, z_c_size:z_c_size + \
                                                          z_g_size],
                              true_actions=None,
                              true_h0=None,
                              retain_grads=True)
    preds_dict = {'obs': pred_obs,
                  'hx': pred_agent_hs,
                  'reward': pred_rews,
                  'done': pred_dones,
                  'act_log_probs': pred_agent_logprobs,
                  'value': pred_agent_values,
                  'sample_vecs': sample_vecs,
                  'env_hx': pred_env_hs[0],
                  'env_cell_state': pred_env_hs[1]}

    # Calculate Target function loss
    saliency_func_loss = saliency_func.loss_func(preds_dict)

    # Get gradient and step the optimizer
    saliency_func_loss.backward()

    # Collect into one tensor each
    grads_keys = ['obs', 'hx', 'env_hx', 'env_cell_state', ]
    grads_dict = {}
    for key in grads_keys:
        tensors = preds_dict[key]
        preds_dict[key] = torch.stack(tensors, dim=1).squeeze()
        grads_initial = [tensor.grad for tensor in tensors]
        null_grad = torch.zeros_like(tensors[0])
        grads = []
        for grad in grads_initial:
            if grad is not None:
                grads.append(grad)
            else:
                grads.append(null_grad)
        grads_dict[key] = torch.stack(grads, dim=1).mean(dim=0)

    return preds_dict, grads_dict


def save_results(preds_dict, grads_dict, latent_vec_name, saliency_func_type):
    savedir = os.path.join(args.generated_data_dir, latent_vec_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for key in grads_dict.keys():
        save_str = os.path.join(savedir, f'grad_{key}_{saliency_func_type}.npy')
        np.save(save_str, grads_dict[key].clone().detach().cpu().numpy())

    # Visualize the optimized latent vectors
    sample_obs = preds_dict['obs'].mean(0).permute(0, 2, 3, 1)
    obs_grad = grads_dict['obs'].permute(0, 2, 3, 1)
    obs_grad = obs_grad.mean(3).unsqueeze(dim=3)  # mean over channel dim

    # Scale according to typical grad sizes for each timestep
    # obs_grad = obs_grad / torch.abs(obs_grad).mean([1, 2]).unsqueeze(
    #     -1).unsqueeze(-1)
    obs_grad = obs_grad / torch.abs(obs_grad).mean()
    blurrer = tv.transforms.GaussianBlur(5, sigma=(5., 6.))
    obs_grad = blurrer(obs_grad.squeeze()).unsqueeze(-1)

    pos_grads = obs_grad.where(obs_grad > 0., torch.zeros_like(obs_grad))
    neg_grads = obs_grad.where(obs_grad < 0., torch.zeros_like(obs_grad)).abs()

    # Make a couple of copies of the original observation for later
    sample_obs_faint = sample_obs.clone().detach() * 0.2
    sample_obs_faint = sample_obs_faint.mean(3)
    sample_obs_faint = torch.stack([sample_obs_faint] * 3, dim=-1)
    sample_obs_faint = sample_obs_faint * 255
    sample_obs_faint = sample_obs_faint.clone().detach().type(
        torch.uint8).cpu().numpy()

    sample_obs_copy = sample_obs.clone().detach() * 255
    sample_obs_copy = sample_obs_copy.clone().detach().type(
        torch.uint8).cpu().numpy()

    # Make the gradient video and save as uint8
    grad_vid = np.zeros_like(sample_obs_copy)
    pos_grads = pos_grads * 0.2 * 255
    neg_grads = neg_grads * 0.2 * 255
    grad_vid[:, :, :, 2] = pos_grads.squeeze().clone().detach().type(
        torch.uint8).cpu().numpy()
    grad_vid[:, :, :, 0] = neg_grads.squeeze().clone().detach().type(
        torch.uint8).cpu().numpy()
    grad_vid = grad_vid + sample_obs_faint
    grad_vid_name = os.path.join(savedir,
                                 f'grad_processed_obs_{saliency_func_type}.npy')
    np.save(grad_vid_name, grad_vid)

    # Save a side-by-side vid
    # Join the prediction and the true observation side-by-side
    combined_vid = np.concatenate([sample_obs_copy, grad_vid], axis=2)

    # Save vid
    # combined_vid = combined_vid.clone().detach().type(torch.uint8).cpu().numpy()
    combo_vid_name = f'{latent_vec_name}_saliency_{saliency_func_type}.mp4'
    combo_vid_name = os.path.join(args.generated_data_dir, combo_vid_name)
    tvio.write_video(combo_vid_name, combined_vid, fps=13)

def run_saliency_mapping_and_save(latent_vecs, latent_vec_name, sfe, saliency_func):
    # Forward and backward pass
    preds_dict, grads_dict = forward_backward_pass(latent_vecs,
                                                   sfe, saliency_func)
    saliency_func_type = saliency_func.saliency_func_type
    # Save results
    save_results(preds_dict, grads_dict, latent_vec_name, saliency_func_type)

if __name__=='__main__':
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument('--saliency_func_type', nargs='+')
        parser.add_argument('--gen_mod_exp_type', type=str, default='saliency_exp',
                            help='type of generative model experiment')
        parser.add_argument('--exp_name', type=str, default='test',
                            help='experiment name')
        parser.add_argument('--env_name', type=str, default='coinrun',
                            help='environment ID')
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
        parser.add_argument('--data_dir', type=str, default='data/') # TODO change to test data directory.
        parser.add_argument(
            '--generated_data_dir', type=str,
            default='generative/recorded_informinit_gen_samples')
        parser.add_argument('--save_interval', type=int, default=100)
        parser.add_argument('--log_interval', type=int, default=100)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--sample_ids', nargs='+', default=[])
        parser.add_argument('--num_initializing_steps', type=int, default=8)
        parser.add_argument('--num_sim_steps', type=int, default=22)

        # multi threading
        parser.add_argument('--num_threads', type=int, default=8)

        parser.add_argument('--combine_samples_not_iterate', dest='combine_samples_not_iterate', action='store_true')
        parser.set_defaults(combine_samples_not_iterate=False)
        # Set up args and exp
        args = parser.parse_args()
    sfe = GenerativeModelExperiment(args)  # for 'Saliency Function Experiment'

    # Device
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    # Set some hyperparams
    saliency_batch_size = 9
    vae_latent_size = 128
    z_c_size = 64
    z_g_size = 64
    perturbation_scale = 0.01# 2e-2

    # remove grads from generative model
    sfe.gen_model.requires_grad = False

    # Determine what to iterate over (func types, samples)
    combine_samples_not_iterate = args.combine_samples_not_iterate
    saliency_func_types = args.saliency_func_type
    ## Get the sample IDs for the samples that we'll use to make saliency maps
    if 'to' in args.sample_ids:
        sample_ids = range(int(args.sample_ids[0]), int(args.sample_ids[2]))
    else:
        sample_ids = args.sample_ids

    # Iterate over saliency func types and (possibly) over samples
    for saliency_func_type in saliency_func_types:
        print(f"Saliency function type: {saliency_func_type}")
        saliency_func = SaliencyFunction(args=args,
                                       saliency_func_type=saliency_func_type,
                                       device=device)

        if combine_samples_not_iterate:
            print("Combining samples: " + str(sample_ids))
            latent_vec_name = f"sample_{str(sample_ids[0:3])}"
            latent_vecs = combine_sample_latent_vecs(sample_ids)
            run_saliency_mapping_and_save(latent_vecs, latent_vec_name,
                                          sfe, saliency_func)

        else:
            for sample_id in sample_ids:
                print("Sample ID: " + str(sample_id))
                latent_vec_name = f"sample_{int(sample_id):05d}"
                sample_vecs = get_sample_latent_vecs(sample_id)
                run_saliency_mapping_and_save(sample_vecs, latent_vec_name,
                                              sfe, saliency_func)


