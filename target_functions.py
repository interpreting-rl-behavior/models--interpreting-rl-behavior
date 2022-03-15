import torch
import numpy as np
from dimred_projector import HiddenStateDimensionalityReducer


class TargetFunction():
    def __init__(self,
                 target_func_type,
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
        self.sim_len = self.hp.analysis.target_func.num_sim_steps
        self.lr = self.hp.analysis.target_func.lr
        self.min_loss = self.hp.analysis.target_func.min_loss
        self.num_its = self.hp.analysis.target_func.num_its
        self.increment = 1.0
        if 'decrease' in self.target_function_type:
            self.increment *= -1.
        self.num_epochs = self.hp.analysis.target_func.num_epochs
        self.targ_func_loss_scale = self.hp.analysis.target_func.targ_func_loss_scale
        self.directions_scale = self.hp.analysis.target_func.directions_scale
        self.timesteps = list(range(0, self.sim_len))
        hx_timesteps = self.hp.analysis.target_func.hx_timesteps
        hx_target_func_loss_scale = self.hp.analysis.target_func.hx_target_func_loss_scale
        self.num_episodes_precomputed = self.hp.analysis.target_func.num_samples_precomputed
        self.distance_threshold = self.hp.analysis.target_func.distance_threshold
        self.optimized_proxy_record = []
        self.loss_record = []

        # Set settings for specific target functions
        if 'action' in self.target_function_type:
            self.action_id = self.target_function_type.split('_')[-1]
            self.timesteps = self.timesteps
            self.loss_func = self.make_action_target_function(action_id=self.action_id,
                                                              timesteps=self.timesteps)
            self.optimized_proxy_record_name = 'Target action logit - max action logit'
        elif 'hx_neuron' in self.target_function_type:
            self.nrn_id = self.target_function_type.split('_')[-1]
            self.timesteps = hx_timesteps
            self.target_func_loss_scale = hx_target_func_loss_scale
            self.loss_func = self.make_hx_neuron_target_function(self.nrn_id,
                                                                 self.timesteps)
            self.optimized_proxy_record_name = 'Activity of hx neuron %s' % self.nrn_id
        elif 'hx_direction' in self.target_function_type:
            self.direction_id = self.target_function_type.split('_')[-1]
            self.timesteps = hx_timesteps
            self.loss_func = self.make_hx_direction_target_function(self.direction_id,
                                                                 self.timesteps)
            self.optimized_proxy_record_name = 'Activity of hx direction %s' % self.direction_id
        elif self.target_function_type == 'increase_value_delta':
            self.time_of_jump = self.sim_len // 2
            self.loss_func = self.value_delta_incr_or_decr_target_function
            self.optimized_proxy_record_name = 'Difference between values in 1st and 2nd half of sequence'
        elif self.target_function_type == 'decrease_value_delta':
            self.time_of_jump = self.sim_len // 2
            self.loss_func = self.value_delta_incr_or_decr_target_function
            self.optimized_proxy_record_name = 'Difference between values in 1st and 2nd half of sequence'
        elif self.target_function_type == 'increase_value':
            self.loss_func = self.value_high_or_low_target_function
            self.optimized_proxy_record_name = 'Mean value during sequence'
        elif self.target_function_type == 'decrease_value':
            self.loss_func = self.value_high_or_low_target_function
            self.optimized_proxy_record_name = 'Mean value during sequence'

    def make_action_target_function(self, action_id, timesteps):
        optimized_proxy_record = self.optimized_proxy_record
        loss_record = self.loss_record
        target_action_idx = int(action_id)
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        device = self.device

        # TODO individuate loss for batch eles
        def action_target_function(preds_dict):
            preds = preds_dict['act_log_prob']
            preds = torch.stack(preds, dim=1).squeeze()
            preds = preds[:, timesteps]

            # Get loss per batch
            b, t, a = preds.shape
            preds_reshaped = preds.reshape(b*t, a)
            labels = torch.ones(b*t) * target_action_idx
            labels = labels.long().to(self.device)
            losses = ce_loss(preds_reshaped, labels)#torch.tensor([ce_loss(preds_reshaped.unsqueeze(1)[i], labels.unsqueeze(1)[i]) for i in range(b*t)])
            losses = losses.reshape(b, t)  # Loss per batch per timestep
            losses = losses.mean(1)  #  Loss per batch

            target_log_probs = preds.clone().detach()
            logitmaxes, argmaxes = target_log_probs.max(dim=2)
            total_num_acts = torch.prod(torch.tensor(argmaxes.shape)).to(device)
            fraction_correct = (argmaxes == target_action_idx).sum(1) / t
            logitlogitmax = \
                (target_log_probs[:, :, target_action_idx] -
                 logitmaxes).mean(1)

            optimized_proxy = logitlogitmax
            loss_record.append(losses.clone().detach().cpu().numpy())
            optimized_proxy_record.append(optimized_proxy)
            print("fraction correct: %f" % fraction_correct.mean())
            print("logit-maxlogit: %f" % logitlogitmax.mean())

            return losses, optimized_proxy

        return action_target_function

    def make_hx_neuron_target_function(self, nrn_id, timesteps):
        optimized_proxy_record = self.optimized_proxy_record
        increment = self.increment
        device = self.device
        target_func_loss_scale = self.targ_func_loss_scale
        nrn_id = int(nrn_id)

        def hx_neuron_target_function(preds_dict): # TODO individuate loss for batch eles
            preds = preds_dict['hx']
            bottleneck_vecs = preds_dict['bottleneck_vec']
            preds = torch.stack(preds, dim=1).squeeze()
            neuron_optimized = nrn_id

            # Make a target that is simply slightly higher than
            # the current prediction.

            target_hx = preds.clone().detach().cpu().numpy()
            print(f"Neuron values: {target_hx[:, timesteps, neuron_optimized].mean()}")
            optimized_proxy_record.append(
                target_hx[:, timesteps, neuron_optimized].mean())
            target_hx[:, timesteps, neuron_optimized] += increment
            target_hx = torch.tensor(target_hx, device=device)

            # Calculate the difference between the target and the pred
            diff = torch.abs(target_hx - preds)
            loss_sum = diff.mean() * target_func_loss_scale

            print("TargFunc loss: %f " % loss_sum)
            return loss_sum
        return hx_neuron_target_function

    def make_hx_direction_target_function(self, direction_id, timesteps):
        optimized_proxy_record = self.optimized_proxy_record
        directions_scale = self.directions_scale
        direction_id = int(direction_id)
        increment = self.increment
        device = self.device
        target_func_loss_scale = self.targ_func_loss_scale

        projector = HiddenStateDimensionalityReducer(self.hp, 'ica', self.num_episodes_precomputed)

        def hx_direction_target_function(preds_dict): # TODO individuate loss for batch eles
            preds = preds_dict['hx']
            preds = torch.stack(preds, dim=1).squeeze()
            preds = preds[:, timesteps]
            b, t, h = preds.shape
            preds = preds.reshape(b * t, h)
            pred_directions = projector.ica_transform(preds)
            # pred_magnitude = np.linalg.norm(preds[:, timesteps], axis=1)
            # directions_magnitude = np.linalg.norm(directions, axis=1)
            # direc_scales = pred_magnitude/directions_magnitude

            # Make a target that is more in the direction of the goal direction than
            # the current prediction.
            target_directions = pred_directions.clone().detach()
            target_directions[:, direction_id] += increment
            target_directions *= 0.999

            optimized_proxy = pred_directions[:, direction_id].mean().item()
            optimized_proxy_record.append(optimized_proxy)
            print("Opt quant: %f" % optimized_proxy)

            # Calculate the difference between the target and the pred
            diff = torch.abs(target_directions - pred_directions)
            loss_sum = diff.mean() * target_func_loss_scale

            print("TargFunc loss: %f " % loss_sum)
            return loss_sum
        return hx_direction_target_function

    def value_delta_incr_or_decr_target_function(self, preds_dict): # TODO individuate loss for batch eles
        preds = preds_dict['value']
        preds = torch.stack(preds, dim=1).squeeze()

        # Make a target that is simply slightly higher than
        # the current prediction.
        target_values = preds.clone().detach().cpu().numpy()
        print(target_values[:, :self.time_of_jump].mean())
        print(target_values[:, self.time_of_jump:].mean())
        optimized_proxy = target_values[:, self.time_of_jump:].mean() - \
                    target_values[:, :self.time_of_jump].mean()
        self.optimized_proxy_record.append(optimized_proxy)
        #base_increments = np.arange(start=-1, stop=1,
        #                            step=(2/target_values.shape[1]))
        #target_values += base_increments * self.increment
        target_values[:, :self.time_of_jump] -= self.increment
        target_values[:, self.time_of_jump:] += self.increment
        target_values = torch.tensor(target_values, device=self.device)

        # Calculate the difference between the target and the pred
        diff = torch.abs(target_values - preds)
        loss_sum = diff.mean() * self.targ_func_loss_scale

        print("TargFunc loss: %f " % loss_sum)

        return loss_sum

    def value_high_or_low_target_function(self, preds_dict): # TODO individuate loss for batch eles
        preds = preds_dict['value']
        preds = torch.stack(preds, dim=1).squeeze()

        # Make a target that is simply slightly higher than
        # the current prediction.
        target_values = preds.clone().detach().cpu().numpy()
        print(f"Target values: {target_values.mean()}")
        self.optimized_proxy_record.append(target_values.mean())
        target_values += self.increment
        target_values = torch.tensor(target_values, device=self.device)

        # Calculate the difference between the target and the pred
        diff = torch.abs(target_values - preds)
        loss_sum = diff.mean() * self.targ_func_loss_scale #+ terminals.sum() * 0.1

        print("TargFunc loss: %f " % loss_sum)

        return loss_sum

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
        optimized_proxy = loss_sum.item()#.clone().detach()
        print("Loss (distance): %f" % torch.sqrt(loss_sum))
        self.optimized_proxy_record.append(optimized_proxy)

        # Calculate the cumulative distribution of the samples' losses and
        # find the top quartile boundary
        diff_cum_df = torch.cumsum(diff.sum(dim=[1, 2]), dim=0)
        top_quart_ind = int(diff_cum_df.shape[0] * 0.75)
        loss_info_dict = {'top_quartile_loss': diff_cum_df[top_quart_ind]}

        print("TargFunc loss: %f " % loss_sum)

        return loss_sum, loss_info_dict



