import util.logger as logger  # from common.logger import Logger
import os
import torch
from gen_model_experiment import GenerativeModelExperiment
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from generative.rssm.functions import terminal_labels_to_mask
from dimred_projector import HiddenStateDimensionalityReducer



class PlotWeightMatricesExperiment(GenerativeModelExperiment):
    """Inherits everything from GenerativeModelExperiment but adds a method that
    collects data for a 'loss over time' plot and a method to plot it."""
    def __init__(self):
        super(PlotWeightMatricesExperiment, self).__init__()

        self.direction_type = self.hp.analysis.saliency.direction_type
        num_analysis_samples = self.hp.analysis.agent_h.num_episodes
        self.directions_transformer = \
            HiddenStateDimensionalityReducer(self.hp,
                                             self.direction_type,
                                             num_analysis_samples)
        self.save_plot_path = self.hp.analysis.plot_act_and_val_weights.save_dir
        self.save_plot_name = os.path.join(self.hp.analysis.plot_act_and_val_weights.save_dir, 'act_and_val_weights.png')

        self.coinrun_action_labels = {0: 'DownLeft', 1: 'Left', 2: 'JumpLeft',
                           3: 'Down', 4: 'Null', 5: 'JumpUp',
                           6: 'DownRight', 7: 'Right', 8: 'JumpRight',
                           9: 'Null', 10: 'Null', 11: 'Null',
                           12: 'Null', 13: 'Null', 14: 'Null'}

        if not (os.path.exists(self.save_plot_path)):
            os.makedirs(self.save_plot_path)

    def plot_action_and_value_heatmaps(self):
        params = [x for x in self.agent.policy.parameters()]
        w_dict = {"act": params[-8],
                  "val": params[-6]}
        for k, v in w_dict.items():
            w_dict[k] = self.directions_transformer.project_gradients_into_ica_space(v)
        b_dict = {"act": params[-7],
                  "val": params[-5]}

        # Convert weights from tensor to np array
        for wb_dict in [w_dict, b_dict]:
            for k, v in wb_dict.items():
                wb_dict[k] = v.cpu().detach().numpy()

        # Add dim to bias vectors
        for k, v in b_dict.items():
            b_dict[k] = np.expand_dims(v, axis=0).transpose(1,0)

        # Get max and min values for all the weights and biases to define colormap range
        w_max = max([max([v.max() for v in wb_dict.values()]) for wb_dict in [w_dict, b_dict]])
        w_min = min([min([v.min() for v in wb_dict.values()]) for wb_dict in [w_dict, b_dict]])
        w_abs_max = max([abs(w_max), abs(w_min)]) * 1.
        cmap = 'bwr_r'

        # create a figure
        fig = plt.figure()

        # # to change size of subplot's
        # # set height of each subplot as 8
        # fig.set_figheight(8)
        #
        # # set width of each subplot as 8
        # fig.set_figwidth(8)

        # create grid for different subplots
        spec = gridspec.GridSpec(ncols=2, nrows=2,
                                 width_ratios=[19, 5], wspace=0.0,
                                 hspace=0.1, height_ratios=[15, 1])


        # ax0 will take 0th position in
        # geometry(Grid we created for subplots),
        # as we defined the position as "spec[0]"
        ax0 = fig.add_subplot(spec[0], title="Action (top) and value (bottom) \nreadout matrices")
        ax0.imshow(w_dict['act'], cmap=cmap, interpolation='nearest', vmin=-w_abs_max, vmax=w_abs_max,)
        ax0.set_yticks(range(0,15), minor=False)
        ax0.set_yticklabels(self.coinrun_action_labels.values(), fontdict=None, minor=False)
        ax0.set_xticks([], minor=False)
        ax0.set_xticklabels([], fontdict=None, minor=False)


        # ax1 will take 0th position in
        # geometry(Grid we created for subplots),
        # as we defined the position as "spec[1]"
        ax1 = fig.add_subplot(spec[1], title="Biases")
        ax1.set_yticks(range(0,16), minor=False)
        ax1.set_yticklabels([], fontdict=None, minor=False)
        ax1.set_xticks([], minor=False)
        ax1.set_xticklabels([], fontdict=None, minor=False)
        im = ax1.imshow(b_dict['act'], cmap=cmap, interpolation='nearest', vmin=-w_abs_max, vmax=w_abs_max,)
        cb_ax = fig.add_axes([0.9, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)

        # ax2 will take 0th position in
        # geometry(Grid we created for subplots),
        # as we defined the position as "spec[2]"
        ax2 = fig.add_subplot(spec[2])
        ax2.set_yticks([0], minor=False)
        ax2.set_yticklabels(['Value out'], fontdict=None, minor=False)
        ax2.set_xticks(range(0, 16), minor=False)
        ax2.set_xticklabels(range(0, 16), fontdict=None, minor=False)
        ax2.set_xlabel("Independent component")
        ax2.imshow(w_dict['val'], cmap=cmap, interpolation='nearest', vmin=-w_abs_max, vmax=w_abs_max)

        # ax3 will take 0th position in
        # geometry(Grid we created for subplots),
        # as we defined the position as "spec[3]"
        ax3 = fig.add_subplot(spec[3])
        ax3.set_yticks([0], minor=False)
        ax3.set_yticklabels([], fontdict=None, minor=False)
        ax3.set_xticks([], minor=False)
        ax3.set_xticklabels([], fontdict=None, minor=False)
        ax3.imshow(b_dict['val'], cmap=cmap, interpolation='nearest', vmin=-w_abs_max, vmax=w_abs_max,)
        # display the plots
        plt.savefig(self.save_plot_name)

        print("boop")

if __name__ == "__main__":
    plotting_matrices_exp = PlotWeightMatricesExperiment()
    plotting_matrices_exp.plot_action_and_value_heatmaps()
