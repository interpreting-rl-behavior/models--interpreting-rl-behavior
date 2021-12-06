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
            (loss_model, priors, posts, samples, features, env_states,
            env_state, metrics_list, tensors_list, pred_actions_1hot, pred_agent_hs) = \
                self.gen_model(data=data, use_true_actions=True, imagine=False)

            loss = torch.mean(torch.sum(loss_model, dim=0))  # sum over T, mean over B
            # TODO confirm that masking of losses works as required
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), 100.)
            for p in self.gen_model.agent_env_stepper.agent.policy.parameters():
                if p.grad is not None:  # freeze agent parameters but not model's.
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

            # Visualize the predictions compared with the ground truth
            pred_images, pred_terminals, pred_rews = self.extract_preds_from_tensors(tensors_list)

            preds = {'ims': pred_images, 'actions': pred_actions_1hot, 'terminals': pred_terminals, 'rews': pred_rews}
            if (epoch >= 1 and batch_idx % 20000 == 0) or (epoch < 1 and batch_idx % 5000 == 0):
            # if batch_idx % 1 == 0 :
                self.visualize(epoch, batch_idx=batch_idx, data=data, preds=preds,
                               use_true_actions=True, save_root='sample')

            # Demo recon quality without using true images
            if (epoch >= 1 and batch_idx % 20000 == 0) or (epoch < 1 and batch_idx % 5000 == 0):
                self.visualize(epoch, batch_idx=batch_idx, data=None, preds=None,
                               use_true_actions=True, save_root='demo_true_acts')
                self.visualize(epoch, batch_idx=batch_idx, data=None, preds=None,
                               use_true_actions=False, save_root='demo_sim_acts')


if __name__ == "__main__":
    training_exp = TrainingExperiment()
    training_exp.run_training_loop()
