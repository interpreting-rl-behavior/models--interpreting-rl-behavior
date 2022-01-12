import util.logger as logger  # from common.logger import Logger
import os
import torch
from gen_model_experiment import GenerativeModelExperiment
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class LossOverTimeExperiment(GenerativeModelExperiment):
    """Inherits everything from GenerativeModelExperiment but adds a method that
    collects data for a 'loss over time' plot and a method to plot it."""
    def __init__(self):
        super(LossOverTimeExperiment, self).__init__()

        self.num_batches_collect = 16
        save_path = 'loss_over_time/'
        save_path = os.path.join(os.getcwd(), "analysis", save_path)
        loss_sess_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sess_path = os.path.join(save_path, loss_sess_name)
        os.makedirs(self.sess_path, exist_ok=True)

    def collect_losses(self):

        # Set up logging queue objects
        logger.info('Start collection of loss data')
        loss_dict_list = []

        # Prepare for training cycle
        self.gen_model.eval()

        # Collection cycle
        for batch_idx, data in enumerate(self.train_loader):

            print(f"Batch {batch_idx + 1}/{len(self.train_loader)}")
            if batch_idx == self.num_batches_collect - 1:
                break

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
                               imagine=True,
                               calc_loss=True,  # This is the only instance where imagine and calc_loss are both true
                               modal_sampling=False)

            loss_model = torch.mean(torch.sum(loss_model, dim=0))  # sum over T, mean over B
            # loss_bottleneck has no mean because already summed over b
            loss_agent_aux_init = torch.mean(loss_agent_aux_init)  # mean over B
            loss = loss_model + loss_bottleneck + loss_agent_aux_init

            # Save detached loss tensors for logging and per-timestep inspection
            loss_dict_no_grad['loss_model'] = loss_model.detach().cpu().numpy()
            loss_dict_no_grad = {k: v.clone().detach().cpu().numpy() \
                                 for k,v in loss_dict_no_grad.items() \
                                 if type(v) == torch.Tensor}

            # Dispose of losses that aren't per-timestep
            loss_dict_no_grad = {k: v for k, v in loss_dict_no_grad.items() if
                                 type(v) == np.ndarray}
            loss_dict_list.append(loss_dict_no_grad)

            # Logging and saving info
            if batch_idx % self.args.log_interval == 0:
                logger.logkv('batches', batch_idx)
                logger.logkv('loss_model', loss_model.item())
                for k, v in loss_dict_no_grad.items():
                    if type(v) == np.ndarray:
                        l = np.mean(np.sum(v, axis=0))
                    else:
                        l = v
                    logger.logkv(k, l)
                logger.logkv('loss total=model+bneck+aux', loss.item())
                logger.dumpkvs()

        # Restructure list of dicts --> dict of arrays
        loss_vectors = {}
        loss_keys = loss_dict_no_grad.keys()
        for k in loss_keys:
            loss_vectors[k] = np.array([l[k] for l in loss_dict_list])
            loss_vectors[k] = np.concatenate(loss_vectors[k], axis=-1) #stack on B dim
            filename = os.path.join(self.sess_path, k+'.npy')
            np.save(filename, loss_vectors[k])

    def plot_loss_over_time(self):
        loss_keys = ['kl_rssm', 'reconstr_ims', 'reconstr_agent_hs']

        data = {}
        for key in loss_keys:
            filename = os.path.join(self.sess_path, 'loss_'+key+'.npy')
            data[key] = np.load(filename)

        # Create plot and save to directory
        plt.style.use("ggplot")
        for key in loss_keys:
            x_vals = range(1, data[key].shape[0]+1)
            mean = np.mean(data[key], axis=1)
            higher = np.max(data[key], axis=1)
            lower = np.min(data[key], axis=1)
            std = np.std(data[key], axis=1)
            plt.plot(x_vals, mean, label=key)
            plt.fill_between(x_vals, mean - lower, mean + higher, alpha=0.3)
            plt.fill_between(x_vals, mean - std, mean + std, alpha=0.3)

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
        plt.xlabel("Simulation Step")
        plt.ylabel("MSE")

        # Get the stem of the input data filename
        plt.savefig(f"{self.sess_path}/loss_over_time_plot.png")
        plt.show()

if __name__ == "__main__":
    loss_experiment = LossOverTimeExperiment()
    loss_experiment.collect_losses()
    loss_experiment.plot_loss_over_time()
