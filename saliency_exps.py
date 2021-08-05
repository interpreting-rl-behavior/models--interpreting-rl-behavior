from common.env.procgen_wrappers import *
import os, argparse
import random
import torch
from gen_model_experiment import GenerativeModelExperiment
from datetime import datetime
import torchvision.io as tvio
import matplotlib.pyplot as plt


class SaliencyFunction():
    def __init__(self, args, sim_len=30, device='cuda'):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(SaliencyFunction, self).__init__()
        self.device = device

        self.saliency_func_type = args.saliency_func_type
        self.coinrun_actions = {0: 'downleft', 1: 'left', 2: 'upleft',
                                3: 'down', 4: None, 5: 'up',
                                6: 'downright', 7: 'right', 8: 'upright',
                                9: None, 10: None, 11: None,
                                12: None, 13: None, 14: None}
        # Set settings for specific target functions
        if self.saliency_func_type == 'action':
            self.loss_func = self.action_saliency_loss_function
        elif self.saliency_func_type == 'leftwards':
            self.loss_func = self.action_leftwards_saliency_loss_function
        elif self.saliency_func_type == 'value':
            self.loss_func = self.value_saliency_loss_function
        elif self.saliency_func_type == 'value_delta':
            self.loss_func = self.value_delta_saliency_loss_function

    def action_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_probs']
        preds = torch.stack(preds, dim=1)
        preds = preds.max(dim=0)[0].squeeze() # TODO fix this - it isn't selecting for
        loss_sum = preds.mean()
        return loss_sum

    def action_leftwards_saliency_loss_function(self, preds_dict):
        preds = preds_dict['act_log_probs']
        preds = torch.stack(preds, dim=1)
        preds = preds[:,:,(0,1,2)].squeeze()
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
        loss_sum = preds.mean()

        return loss_sum

def get_sample_vecs(sample_id):
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
    sample_vecs = sample_vecs + perturbation
    return sample_vecs, sample_dir

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
        grads = [tensor.grad for tensor in tensors]
        grads_dict[key] = torch.stack(grads, dim=1).mean(dim=0)

    return preds_dict, grads_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saliency_func_type', type=str)
    parser.add_argument('--gen_mod_exp_type', type=str, default='saliency_exp',
                        help='type of latent space experiment')
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

    # Set up args and exp
    args = parser.parse_args()
    sfe = GenerativeModelExperiment(args)  # for 'Saliency Function Experiment'

    # Device
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    # Make desired target function
    saliency_func = SaliencyFunction(args=args,
                                   sim_len=args.num_sim_steps,
                                   device=device)

    # Set some hyperparams
    saliency_batch_size = 9
    vae_latent_size = 128
    z_c_size = 64
    z_g_size = 64
    perturbation_scale = 5e-2

    # remove grads from generative model
    sfe.gen_model.requires_grad = False

    for sample_id in args.sample_ids:
        print("Sample ID: " + str(sample_id))

        # Load latent vectors and perturb them
        sample_vecs, sample_dir = get_sample_vecs(sample_id)

        # Forward and backward pass
        preds_dict, grads_dict = forward_backward_pass(sample_vecs,
                                                      sfe, saliency_func)

        # Save results
        for key in grads_dict.keys():
            save_str = os.path.join(sample_dir, f'grad_{key}.npy')
            np.save(save_str, grads_dict[key].clone().detach().cpu().numpy())

        # Visualize the optimized latent vectors
        sample_obs = preds_dict['obs'].mean(0).permute(0, 2, 3, 1)
        obs_grad = grads_dict['obs'].permute(0, 2, 3, 1)
        obs_grad = obs_grad.mean(3).unsqueeze(dim=3)  # mean over channel dim

        # Scale according to typical grad sizes for each timestep
        obs_grad = obs_grad / torch.abs(obs_grad).mean([1,2]).unsqueeze(-1).unsqueeze(-1)
        pos_grads = obs_grad.where(obs_grad > 0., torch.zeros_like(obs_grad))
        neg_grads = obs_grad.where(obs_grad < 0., torch.zeros_like(obs_grad)).abs()

        sample_obs_copy = sample_obs.clone().detach() * 255
        sample_obs_copy = sample_obs_copy.clone().detach().type(torch.uint8).cpu().numpy()
        sample_obs[:,:,:,2] = sample_obs[:,:,:,2] + pos_grads.squeeze() * 0.1
        sample_obs[:,:,:,0] = sample_obs[:,:,:,0] + neg_grads.squeeze() * 0.1
        vid = torch.clip(sample_obs, min=0, max=1)
        vid = vid * 255

        vid = vid.clone().detach().type(torch.uint8).cpu().numpy()
        vid_name = f'sample_{int(sample_id):05d}' + '_saliency_' + args.saliency_func_type + '.mp4'
        save_str_specific = os.path.join(sample_dir, vid_name)
        tvio.write_video(save_str_specific, vid, fps=7)
        save_str_generic = os.path.join(args.generated_data_dir, vid_name)
        tvio.write_video(save_str_generic, vid, fps=7)

        print("Boop")




