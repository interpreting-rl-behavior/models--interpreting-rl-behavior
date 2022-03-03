import os
import shutil
import argparse
import json

import numpy as np
from PIL import Image
import yaml, munch
from dimred_projector import HiddenStateDimensionalityReducer



class DataImporter():
    def __init__(self):

        self.args = self.parse_args()

        self.sample_names = [f"sample_{i:05d}" for i in range(self.args.samples)]

        self.hx_analysis_dir = f"{self.args.input_directory}/analysis/hx_analysis_precomp"
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

    def parse_args(self, ):
        parser = argparse.ArgumentParser(
            description='args for plotting')
        parser.add_argument(
            '--samples', type=int, default=10)
        parser.add_argument(
            '--input_directory', type=str, default=".")
        parser.add_argument(
            '--output_directory', type=str, default="../Brewing1.github.io/static/data")
        parser.add_argument(
            '--interpreting_params_name', type=str, default="defaults")
        args = parser.parse_args()
        return args

    def pca_transform(self, X):
        X_scaled = (X - self.all_hx_mu) / self.all_hx_sigma
        return X_scaled @ self.pca_components.T

    def project_gradients_into_pc_space(self, grad_data):
        sigma = np.diag(self.all_hx_sigma)
        grad_data = grad_data.T  # So each column is a grad vector for a hx
        scaled_pc_comps = self.pca_components @ sigma  # PCs calculated on X'=(X-mu)/sigma are scaled so it's like they were calculated on X
        projected_grads = scaled_pc_comps @ grad_data  # grads are projected onto the scaled PCs
        return projected_grads.T

    def ica_transform(self, X):
        X_scaled = (X - self.all_hx_mu) / self.all_hx_sigma
        pc_loadings = X_scaled @ self.pca_components.T
        pc_loadings = pc_loadings[:,:self.num_ica_components]
        source_signals = pc_loadings @ self.unmix_mat.T
        return source_signals

    def project_gradients_into_ica_space(self, grad_data): # TODO fix
        sigma = np.diag(self.all_hx_sigma)
        grad_data = grad_data.T  # So each column is a grad vector for a hx
        scaled_pc_comps = self.pca_components @ sigma  # PCs calculated on X'=(X-mu)/sigma are scaled so it's like they were calculated on X
        projected_grads_to_pc_space = scaled_pc_comps @ grad_data  # grads are projected onto the scaled PCs
        projected_grads_to_pc_space = projected_grads_to_pc_space[:self.num_ica_components, :]
        projected_grads_to_ic_space = projected_grads_to_pc_space.T @ self.mix_mat
        return projected_grads_to_ic_space

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
        saliency_step = np.where(
            (grad_hx_action_loadings == np.zeros(grad_hx_action_loadings.shape[1])).all(axis=1)
        )[0][0]
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

    def make_img_set_from_arr(self, path, arr):
        os.mkdir(path)
        for i in range(arr.shape[0]):
            im = Image.fromarray(arr[i], mode='RGB')
            im.save(f"{path}/{i}.png")
        # Concatenate images horizontally. Needs einops package.
        # comb_arr = einops.rearrange(arr, 'b h w c -> h (b w) c')
        # im = Image.fromarray(comb_arr, mode='RGB')
        # im.save(f"{path}/all.png")

    def save_sample_images(self, sample_name):
        sample_in = f"{self.args.input_directory}/generative/rec_gen_mod_data/informed_init/{sample_name}"
        sample_out = f"{self.args.output_directory}/{sample_name}"
        os.mkdir(sample_out)

        ims = np.load(sample_in + '/ims.npy')
        sal_action = np.load(sample_in + '/grad_processed_ims_action.npy')
        sal_value = np.load(sample_in + '/grad_processed_ims_value.npy')

        self.make_img_set_from_arr(f"{sample_out}/obs", ims)
        self.make_img_set_from_arr(f"{sample_out}/sal_action", sal_action)
        self.make_img_set_from_arr(f"{sample_out}/sal_value", sal_value)

        # Now do iteratively for PC direction saliency
        for idx in range(self.min_pc_directions, self.max_pc_directions):
            sal = np.load(
                sample_in + f'/grad_processed_ims_hx_direction_%i_{self.direction_type}.npy' % idx)

            # TODO maybe insert arrows here

            self.make_img_set_from_arr(f"{sample_out}/sal_hx_direction_%i" % idx,
                                  sal)

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
            print(
                "Making jsons")
            hx_in_pca = np.load(f"{self.hx_analysis_dir}/{self.data_name_root}{self.n_suffix}.npy")
            with open(self.args.output_directory + "/panel_data.json", 'w') as f:
                json.dump({
                    "base_hx_loadings": hx_in_pca[:3000, :20].tolist(),
                    # was just 1000 instead of n_suffix
                    "samples": {
                        sample_name: self.sample_info_for_panel_data(
                            sample_name)
                        for sample_name in self.sample_names
                    }
                }, f)

            # make a folder for each sample for images
            for sample in self.sample_names:
                print(f"Importing sample {sample}")
                self.save_sample_images(sample)

            print("Done!")

        else:
            print("Process cancelled!")

if __name__ == "__main__":
    importer = DataImporter()
    importer.run()