import os
import shutil
import argparse
import json

import numpy as np
from PIL import Image




def parse_args():
    parser = argparse.ArgumentParser(
        description='args for plotting')
    parser.add_argument(
        '--samples', type=int, default=10)
    parser.add_argument(
        '--input_directory', type=str, default="../raw-data")
    parser.add_argument(
        '--output_directory', type=str, default="../Brewing1.github.io/static/data")
    args = parser.parse_args()
    return args


def pca_transform(X, components, original_mu, original_sigma):
    X_scaled = (X - original_mu) / original_sigma
    return X_scaled @ components.T


def sample_info_for_panel_data(sample_name, pca_components, all_hx_mu, all_hx_sigma, args):
    """
    Return the data formatted for inclusion in panel_data.json
    """
    sample_path = f"{args.input_directory}/generative/recorded_informinit_gen_samples/{sample_name}"
    hx = np.load(sample_path + '/agent_hxs.npy') 

    hx_loadings = pca_transform(hx, pca_components, all_hx_mu, all_hx_sigma).tolist()
    return hx_loadings


def make_img_set_from_arr(path, arr):
    os.mkdir(path)
    for i in range(arr.shape[0]):
        im = Image.fromarray(arr[i], mode='RGB')
        im.save(f"{path}/{i}.png")


def save_sample_images(sample_name, args):
    sample_in = f"{args.input_directory}/generative/recorded_informinit_gen_samples/{sample_name}"
    sample_out = f"{args.output_directory}/{sample_name}"
    os.mkdir(sample_out)

    obs = np.load(sample_in + '/obs.npy')
    obs = np.moveaxis(obs, 1, 3)

    sal_action = np.load(sample_in + '/grad_processed_obs_action.npy')
    sal_value = np.load(sample_in + '/grad_processed_obs_value.npy')

    make_img_set_from_arr(f"{sample_out}/obs", obs)
    make_img_set_from_arr(f"{sample_out}/sal_action", sal_action)
    make_img_set_from_arr(f"{sample_out}/sal_value", sal_value)


def run():
    args = parse_args()

    sample_names = [f"sample_{i:05d}" for i in range(args.samples)]

    hx_analysis_dir = f"{args.input_directory}/analysis/hx_analysis_precomp"

    pca_components = np.load(f"{hx_analysis_dir}/pcomponents_1000.npy")
    all_hx_mu = np.load(f"{hx_analysis_dir}/hx_mu_1000.npy")
    all_hx_sigma = np.load(f"{hx_analysis_dir}/hx_std_1000.npy")

    print(f"Output folder: {os.path.abspath(args.output_directory)}");
    print("This folder will be deleted and replaced with exported data.")
    
    confirm = input("Continue? y/[n]: ")
    if confirm.lower() in ["y", "yes"]:

        # Clear directory
        if os.path.exists(args.output_directory):
            shutil.rmtree(args.output_directory)
        os.mkdir(args.output_directory)

        # output panel_data.json
        hx_in_pca = np.load(hx_analysis_dir + '/hx_pca_1000.npy')
        with open(args.output_directory+"/panel_data.json", 'w') as f:
            json.dump({
                "base_hx_loadings": hx_in_pca[:, :5].tolist(),
                "samples": {
                    sample_name: sample_info_for_panel_data(
                        sample_name,
                        pca_components,
                        all_hx_mu,
                        all_hx_sigma,
                        args)
                    for sample_name in sample_names
                }
            }, f)

        # make a folder for each sample for images 
        for sample in sample_names:
            save_sample_images(sample, args)

        print("Done!")

    else:
        print("Process cancelled!")




if __name__ == "__main__":
    run()

