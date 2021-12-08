from common.env.procgen_wrappers import *
import util.logger as logger  # from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from train import create_venv
import os, yaml, argparse
import gym
import random
import torch
from generative.generative_models import AgentEnvironmentSimulator
from generative.procgen_dataset import ProcgenDataset
from gen_model_experiment import GenerativeModelExperiment
from collections import deque
import torchvision.io as tvio
from datetime import datetime


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
            (
                loss_model,
                loss_vae_kl,
                loss_agent_aux_init,
                priors,
                posts,
                samples,
                features,
                env_states,
                env_state,
                metrics_list,
                tensors_list,
                preds_dict
            ) = \
                self.gen_model(data=data, use_true_actions=True, imagine=False)
            # TODO check whether you can make losses from preds_dict alone and
            #  that they BP to the right nets
            loss = torch.mean(torch.sum(loss_model, dim=0))  # sum over T, mean over B
            loss += torch.mean(loss_vae_kl)  # mean over B
            loss += torch.mean(loss_agent_aux_init)  # mean over B
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), 100.)

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
                logger.logkv('loss total', loss.item())
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

            B = loss_vae_kl.shape[0]
            if (epoch >= 1 and batch_idx % 20000 == 0) or (epoch < 1 and batch_idx % 5000 == 0):
            # if batch_idx % 1 == 0 :
                self.visualize(epoch, batch_idx=batch_idx, data=data, preds=preds_dict,
                               use_true_actions=True, save_root='sample_true_ims_true_acts')

            # Demo recon quality without using true images
            if (epoch >= 1 and batch_idx % 20000 == 0) or (epoch < 1 and batch_idx % 5000 == 0):
                self.visualize(epoch, batch_idx=batch_idx, data=None, preds=None, use_true_actions=True, save_root='sample_sim_ims_true_acts')
                self.visualize(epoch, batch_idx=batch_idx, data=None, preds=None, use_true_actions=False, save_root='sample_sim_ims_sim_acts')
                self.visualize_single(
                               epoch, batch_idx=batch_idx, data=None, preds=None, latent_vec=None, use_true_actions=False, save_root='sample_from_rand_latent', batch_size=B)

if __name__ == "__main__":
    training_exp = TrainingExperiment()
    training_exp.run_training_loop()
