import os
import shutil
import argparse
import json

import einops
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml, munch
from dimred_projector import HiddenStateDimensionalityReducer



class DataImporter():
    def __init__(self):

        self.args = self.parse_args()

        self.sample_names = [f"sample_{i:05d}" for i in range(self.args.samples)]

        self.hx_analysis_dir = f"{self.args.input_directory}/analysis/hx_analysis_precomp"
        self.grad_analysis_dir = f"{self.args.input_directory}/analysis/jacob_analysis_precomp"
        hp_path = f"{self.args.input_directory}/hyperparams/interpreting_configs.yml"

        print('[Loading interpretation hyperparameters]')
        with open(hp_path, 'r') as f:
            hp = yaml.safe_load(f)[self.args.interpreting_params_name]
        for key, value in hp.items():
            print(key, ':', value)

        self.hp = munch.munchify(hp)

        self.n_suffix = self.hp.analysis.agent_h.num_episodes  # 2000#500#100#4000
        self.direction_type = self.hp.analysis.saliency.direction_type

        self.projector = HiddenStateDimensionalityReducer(self.hp,
                                                         self.direction_type,
                                                         self.n_suffix,
                                                         data_type=np.ndarray,
                                                         test_hx=False)

        # self.pca_components = np.load(
        #     f"{self.hx_analysis_dir}/pcomponents_{self.n_suffix}.npy")
        # self.all_hx_mu = np.load(f"{self.hx_analysis_dir}/hx_mu_{self.n_suffix}.npy")
        # self.all_hx_sigma = np.load(f"{self.hx_analysis_dir}/hx_std_{self.n_suffix}.npy")
        #
        self.min_pc_directions = int(self.hp.analysis.saliency.direction_ids[0])
        self.max_pc_directions = int(self.hp.analysis.saliency.direction_ids[2])
        #
        if self.direction_type == 'pca':
            # self.hx_to_loading_transform = self.pca_transform
            # self.project_gradients_transform = self.project_gradients_into_pc_space
            self.data_name_root = 'hx_pca_'
        elif self.direction_type == 'ica':
            # self.unmix_mat = np.load(f"{self.hx_analysis_dir}/ica_unmixing_matrix_hx_{self.n_suffix}.npy")
            # self.mix_mat = np.load(f"{self.hx_analysis_dir}/ica_mixing_matrix_hx_{self.n_suffix}.npy")
            # self.hx_to_loading_transform = self.ica_transform
            # self.project_gradients_transform = self.project_gradients_into_ica_space
            self.data_name_root = 'ica_source_signals_hx_'
            self.num_ica_components = self.hp.analysis.agent_h.n_components_ica

        self.cluster_ids = np.load(os.path.join(self.grad_analysis_dir, f"clusters_jacob_{self.n_suffix}.npy"))
        self.cluster_ids = self.cluster_ids
        self.cluster_dict = {}
        cluster_set = set(self.cluster_ids.tolist())
        for c in cluster_set:
            inds = (self.cluster_ids == c).nonzero()[0]
            samples = [f"sample_{i:05d}" for i in inds]
            self.cluster_dict[c] = samples

    def parse_args(self, ):
        parser = argparse.ArgumentParser(
            description='args for plotting')
        parser.add_argument(
            '--samples', type=int, default=10)
        parser.add_argument(
            '--input_directory', type=str, default=".")
        parser.add_argument(
            '--output_directory', type=str, default="../Brewing1.github.io/static/localData")  # change to static/data if you don't want local?
        parser.add_argument(
            '--interpreting_params_name', type=str, default="defaults")
        args = parser.parse_args()
        return args

    def find_extrema_values(self, hx):
        """
        For each ica/pca component, find high/medium/low threshold values based on the proportions
        given in self.hp.analysis.saliency.extrema_threshold.

        hx is a tensor of shape (total_examples, components).
        """
        # Sort the hxs for each component
        hx_sorted = np.sort(hx, axis=0)

        threshold = self.hp.analysis.saliency.extrema_threshold
        n = hx_sorted.shape[0]

        self.extrema_values = {
            "high": hx_sorted[n - int(n * threshold) - 1],
            "middle_upper": hx_sorted[int(n / 2 + (n * (threshold / 2))) - 1],
            "middle_lower": hx_sorted[int(n / 2 - (n * (threshold / 2))) - 1],
            "low": hx_sorted[int(n * threshold) - 1],
        }

    def get_extrema_samples(self, data):
        """
        Create a json object identifying which samples have activations that are low, middle or
        high for each pca/ica component. We store data corresponding to whether a sample has any
        activation in each group, as well as whether it has an activation in each group for the
        timestep in which the saliency is taken from.
        """
        extrema_list = {
            ext_type: {level: []
            for level in ["high", "middle", "low"]}
            for ext_type in ["any", "saliency_step"]
        }
        sample_names = list(data.keys())
        for sample_name in sample_names:
            sample_data = data[sample_name]
            hx_sample = np.array(sample_data["hx_loadings"])

            high_arr = hx_sample > self.extrema_values["high"]
            middle_arr = ((hx_sample < self.extrema_values["middle_upper"])
                          & (hx_sample > self.extrema_values["middle_lower"]))
            low_arr = hx_sample < self.extrema_values["low"]

            extrema_list["any"]["high"].append(np.any(high_arr, axis=0))
            extrema_list["any"]["middle"].append(np.any(middle_arr, axis=0))
            extrema_list["any"]["low"].append(np.any(low_arr, axis=0))

            extrema_list["saliency_step"]["high"].append(high_arr[sample_data["saliency_step"]])
            extrema_list["saliency_step"]["middle"].append(middle_arr[sample_data["saliency_step"]])
            extrema_list["saliency_step"]["low"].append(low_arr[sample_data["saliency_step"]])

        extrema_samples = {
            ext_type: {level: {comp: []
            for comp in range(hx_sample.shape[-1])}
            for level in ["high", "middle", "low"]}
            for ext_type in ["any", "saliency_step"]
        }
        for ext_type in ["any", "saliency_step"]:
            for level in ["high", "middle", "low"]:
                arr = np.array(extrema_list[ext_type][level])
                samples, components = arr.nonzero()
                for i in range(len(samples)):
                    extrema_samples[ext_type][level][components[i]].append(sample_names[samples[i]])

        return extrema_samples

    def sample_info_for_panel_data(self, sample_name):
        """
        Return the data formatted for inclusion in panel_data.json
        """
        sample_path = f"{self.args.input_directory}/generative/rec_gen_mod_data/informed_init/{sample_name}"
        hx = np.load(sample_path + '/agent_hs.npy')
        grad_hx_action = np.load(sample_path + '/grad_hx_action.npy')
        grad_hx_value = np.load(sample_path + '/grad_hx_value.npy')

        grad_hx_pcs = [
            np.load(sample_path + f'/grad_hx_hx_direction_%i_{self.direction_type}.npy' % idx)
            for idx in range(self.min_pc_directions, self.max_pc_directions)]

        hx_loadings = self.projector.transform(hx) #self.hx_to_loading_transform(hx).tolist()
        # Not entirely clear what the most principled choice is, especially on if we should scale by original hx_sigma.
        grad_hx_action_loadings = self.projector.project_gradients(grad_hx_action)
        grad_hx_value_loadings = self.projector.project_gradients(grad_hx_value)

        agent_logprobs = np.load(sample_path + '/agent_logprobs.npy')
        actions = agent_logprobs.argmax(axis=-1).tolist()


        # Want to know which timestep the saliency was taken from. Note that it would be more
        # principled to save this when running the saliency experiment as opposed to inferring it
        # by the first timestep in which the grads are all zero (as is done here).
        # saliency_step = np.where(
        #     (grad_hx_action_loadings == np.zeros(grad_hx_action_loadings.shape[1])).all(axis=1)
        # )[0][0]
        saliency_step = 4 # It's always going to be 4.
        loadings_dict = {
            "saliency_step": int(saliency_step),
            "actions": actions,
            "hx_loadings": hx_loadings.tolist(),
            "grad_hx_value_loadings": grad_hx_value_loadings.tolist(),
            "grad_hx_action_loadings": grad_hx_action_loadings.tolist(),
        }

        # Now do the same iteratively for the PC direction loadings
        grad_hx_direction_loadings_dict = {}
        for idx, grads_hx_pc_direction in zip(
                range(self.min_pc_directions, self.max_pc_directions), grad_hx_pcs):
            grad_hx_direction_loadings_dict.update(
                {'grad_hx_hx_direction_%i_loadings' % idx:
                     self.projector.project_gradients(grads_hx_pc_direction).tolist()})

        loadings_dict.update(grad_hx_direction_loadings_dict)

        return loadings_dict

    def make_img_set_from_arr(self, path, im_dict):
        for name, arr in im_dict.items():
            # Concatenate images horizontally. Needs einops package. #TODO(Lee): Surely we can do this operation without importing a package?
            comb_arr = einops.rearrange(arr, 'b h w c -> h (b w) c')
            im = Image.fromarray(comb_arr, mode='RGB')
            # im.save(f"{path}/all.png")
            im.save(f"{path}/{name}.png")

    def save_sample_images(self, sample_name):
        sample_in = f"{self.args.input_directory}/generative/rec_gen_mod_data/informed_init/{sample_name}"
        sample_out = f"{self.args.output_directory}/{sample_name}"
        os.mkdir(sample_out)

        # Store observations and saliencies to a dictionary
        im_dict = {}
        im_dict["obs"] = np.load(sample_in + '/ims.npy')
        im_dict["sal_action"] = np.load(sample_in + '/grad_processed_ims_action.npy')
        im_dict["sal_value"] = np.load(sample_in + '/grad_processed_ims_value.npy')
        # Note that we have a saliency array for each IC
        for idx in range(self.min_pc_directions, self.max_pc_directions):
            im_dict[f"sal_hx_direction_{idx}"] = np.load(
                sample_in + f'/grad_processed_ims_hx_direction_{idx}_{self.direction_type}.npy')

        # Create and save images from numpy arrays
        self.make_img_set_from_arr(sample_out, im_dict)

    def plot_hx_histograms(self, hx):
        outdir = f"{self.args.output_directory}/component_histograms"
        os.mkdir(outdir)
        n_samples = 4000
        # Reduce data for efficiency
        sub_hx = hx[:n_samples]

        for comp in range(hx.shape[1]):
            plt.figure()
            plt.hist(sub_hx[:,comp], bins=100)
            plt.title(f"Component {comp} - {n_samples} samples")
            plt.savefig(os.path.join(outdir, f"Component{comp}.png"))

        # Below is for creating a single plot for all components
        # nrows = 4
        # ncols = 4
        # fig, axs = plt.subplots(nrows, ncols, sharex=False)
        # fig.suptitle(f"Activation histograms per component - {n_samples} samples")
        # for comp in range(hx.shape[1]):
        #     x = comp // ncols
        #     y = comp % nrows
        #     axs[x, y].hist(sub_hx[:,comp], bins=30)
        # plt.tight_layout()
        # fig.savefig(os.path.join(outdir, "all.png"))

    def run(self, ):

        print(f"Output folder: {os.path.abspath(self.args.output_directory)}")
        print("This folder will be deleted and replaced with exported data.")

        confirm = input("Continue? y/[n]: ")
        if confirm.lower() in ["y", "yes"]:

            # Clear directory
            if os.path.exists(self.args.output_directory):
                shutil.rmtree(self.args.output_directory)
            os.mkdir(self.args.output_directory)

            # output panel_data.json
            print("Collecting sample data")
            data = {
                sample_name: self.sample_info_for_panel_data(sample_name)
                for sample_name in self.sample_names
            }
            hx_in_ica = np.concatenate([np.array(list(data.values())[i]['hx_loadings']) for i in range(len(data))], axis=0)
            print(
                "Making jsons")

            self.plot_hx_histograms(hx_in_ica)
            self.find_extrema_values(hx_in_ica)
            extrema = self.get_extrema_samples(data)

            with open(self.args.output_directory + "/extrema.json", 'w') as f:
                json.dump(extrema, f, indent=4)

            with open(self.args.output_directory + "/panel_data.json", 'w') as f:
                json.dump({
                    # Only store 3000 datapoints for each component (we couldn't show more on a
                    # plot easily anyway)
                    "base_hx_loadings": hx_in_ica[:3000].tolist(),
                    # was just 1000 instead of n_suffix
                    "samples": {
                        sample_name: self.sample_info_for_panel_data(
                            sample_name)
                        for sample_name in self.sample_names
                    },
                    "clusters": self.cluster_dict
                }, f)

            # make a folder for each sample for images
            for sample in self.sample_names:
                print(f"Importing sample {sample}")
                self.save_sample_images(sample)

            print("Done!")

        else:
            print("Process cancelled!")

    def compare_columnwise(self, array, vec, op=np.greater):
        num_rows, num_columns = array.shape
        comparisons = []
        for column_id in range(num_columns):
            scalar = vec[column_id]
            column = array[:, column_id]
            comparison = op(column, scalar)
            comparisons.append(comparison)
        comparisons = np.stack(comparisons, axis=-1)
        return comparisons


if __name__ == "__main__":
    importer = DataImporter()
    importer.run()