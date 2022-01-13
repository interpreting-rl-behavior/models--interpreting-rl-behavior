import util.logger as logger  # from common.logger import Logger
import os
import torch
import numpy as np
from gen_model_experiment import GenerativeModelExperiment


class TrainingExperiment(GenerativeModelExperiment):
    """Inherits everything from GenerativeModelExperiment but adds a training
    method."""
    def __init__(self):
        super(TrainingExperiment, self).__init__()

    def run_training_loop(self):

        # Training cycle (Train, Save visualized random samples, Demonstrate
        #  reconstruction quality)
        for epoch in range(0, self.args.epochs + 1):
            self.train(epoch)

    def train(self, epoch):

        # Set up logging queue objects
        logger.info('Start training epoch {}'.format(epoch))

        # Prepare for training cycle
        self.gen_model.train()

        # Training cycle
        for batch_idx, data in enumerate(self.train_loader):

            # Make all data into floats and put on the right device
            data = {k: v.to(self.device).float() for k, v in data.items()}
            data = {k: torch.swapaxes(v, 0, 1) for k, v in data.items()}  # (B, T, :...) --> (T, B, :...)

            # Forward and backward pass and update generative model parameters
            self.optimizer.zero_grad()
            (   loss_dict_no_grad,
                loss_model,
                loss_bottleneck,
                loss_agent_aux_init,
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
                               use_true_agent_h0=True,
                               imagine=False,
                               calc_loss=True,
                               modal_sampling=False)

            loss_model_sum = torch.mean(torch.sum(loss_model, dim=0))  # sum over T, mean over B
            # loss_bottleneck has no mean because already summed over b
            loss_agent_aux_init = torch.mean(loss_agent_aux_init)  # mean over B
            loss = loss_model_sum + loss_bottleneck + loss_agent_aux_init
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), 100.)

            # Save detached loss tensors for logging and per-timestep inspection
            loss_dict_no_grad['loss_total'] = loss.clone().detach().cpu().numpy()
            loss_dict_no_grad['loss_model'] = loss_model.detach().cpu().numpy()
            loss_dict_no_grad = {k: v.clone().detach().cpu().numpy() \
                                 for k,v in loss_dict_no_grad.items() \
                                 if type(v) == torch.Tensor}

            # Freeze agent parameters but not model's.
            for p in self.gen_model.agent_env_stepper.agent.policy.parameters():
                if p.grad is not None:
                    p.grad.data = torch.zeros_like(p.grad.data)
            self.optimizer.step()

            # Logging and saving info
            if batch_idx % self.args.log_interval == 0:
                loss.item()
                logger.logkv('epoch', epoch)
                logger.logkv('batches', batch_idx)
                logger.logkv('loss_model', loss_model_sum.item())
                for k, v in loss_dict_no_grad.items():
                    print(v.shape)
                    l = np.mean(np.sum(v, axis=0)).item()
                    logger.logkv(k, l)
                logger.logkv('loss total=model+bneck+aux', loss.item())
                logger.dumpkvs()

            # Saving model
            if batch_idx % self.args.save_interval == 0:
                model_path = os.path.join(
                    self.sess_dir,
                    'model_epoch{}_batch{}.pt'.format(epoch, batch_idx))
                torch.save(
                    {'gen_model_state_dict': self.gen_model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict()},
                    model_path)
                logger.info('Generative model saved to {}'.format(model_path))

            B = data['ims'].shape[1]
            # self.save_preds( preds_dict, range(0,3), manual_action=None)

            # Demo recon quality without using true images
            if (epoch >= 1 and batch_idx % 20000 == 0) or (epoch < 1 and batch_idx % 5000 == 0):
                save_dir = self.sess_dir + '/recons_v_preds/'
                self.visualize(epoch, batch_idx=batch_idx, data=data, preds=preds_dict, use_true_actions=True, save_dir=save_dir, save_root='sample_true_ims_true_acts')
                self.visualize(epoch, batch_idx=batch_idx, data=None, preds=None, use_true_actions=True, save_dir=save_dir, save_root='sample_sim_ims_true_acts')
                self.visualize(epoch, batch_idx=batch_idx, data=None, preds=None, use_true_actions=False, save_dir=save_dir, save_root='sample_sim_ims_sim_acts')
                self.visualize_single(
                               epoch, batch_idx=batch_idx, data=None, preds=None, bottleneck_vec=None, use_true_actions=False, save_dir=save_dir, save_root='sample_from_rand_latent', batch_size=B)

if __name__ == "__main__":
    training_exp = TrainingExperiment()
    training_exp.run_training_loop()
