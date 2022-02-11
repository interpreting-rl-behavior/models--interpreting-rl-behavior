from common.env.procgen_wrappers import *
import os
import torch


class HiddenStateDimensionalityReducer():
    def __init__(self, hp, type_of_dim_red, num_analysis_samples, data_type=torch.tensor, device='cuda'):
        """
        By default, the methods of this class take a NxD set of vectors,
        where N is the number of vectors and D is their dimension, and they
        output an NxD' set of vectors, where D' is the dimension of the
        transformed vectors.
        """
        super(HiddenStateDimensionalityReducer, self).__init__()
        self.hp = hp
        self.device = device
        self.type_of_dim_red = type_of_dim_red
        hx_analysis_dir = os.path.join(os.getcwd(), 'analysis',
                                       'hx_analysis_precomp')

        if data_type == torch.tensor:
            self.diag = torch.diag
        elif data_type == np.ndarray:
            self.diag = np.diag
        else:
            raise ValueError("Invalid data type. Must be one of torch.tensor or np.ndarray")


        if type_of_dim_red == 'pca' or type_of_dim_red == 'ica':
            self.transform = self.pca_transform
            self.project_gradients = self.project_gradients_into_pc_space

            directions_path = os.path.join(os.getcwd(), hx_analysis_dir,
                                           f'pcomponents_{num_analysis_samples}.npy')
            hx_mu_path = os.path.join(hx_analysis_dir, f'hx_mu_{num_analysis_samples}.npy')
            hx_std_path = os.path.join(hx_analysis_dir, f'hx_std_{num_analysis_samples}.npy')
            pc_variances_path = os.path.join(hx_analysis_dir, f'pc_loading_variances_{num_analysis_samples}.npy')

            self.pcs = np.load(directions_path)  # shape = (n_components, n_features)
            self.pcs = self.pcs.transpose(0, 1)  # shape = (n_features, n_components)
            self.hx_mu = np.load(hx_mu_path)
            self.hx_std = np.load(hx_std_path)
            self.pc_std = np.sqrt(np.load(pc_variances_path))

            if data_type == torch.tensor:
                self.pcs = torch.tensor(self.pcs).to(
                    device).requires_grad_()  # shape = (n_components, n_features)
                self.hx_mu = torch.tensor(self.hx_mu).to(
                    device).requires_grad_()
                self.hx_std = torch.tensor(self.hx_std).to(
                    device).requires_grad_()
                self.pc_std = torch.tensor(self.pc_std).to(
                    device).requires_grad_()

        if type_of_dim_red == 'ica':
            self.num_ica_components = self.hp.analysis.agent_h.n_components_ica
            self.transform = self.ica_transform
            self.project_gradients = self.project_gradients_into_ica_space


            ica_directions_path = os.path.join(os.getcwd(), hx_analysis_dir,
                f'ica_unmixing_matrix_hx_{num_analysis_samples}.npy')
            ica_mixmat_path = os.path.join(os.getcwd(), hx_analysis_dir,
                f'ica_mixing_matrix_hx_{num_analysis_samples}.npy')

            self.unmix_mat = np.load(ica_directions_path) # (n_components, n_features)
            self.unmix_mat = self.unmix_mat.transpose(0, 1)  # (n_features, n_components)
            self.mix_mat = np.load(ica_mixmat_path) # (n_features, n_components)
            #self.mix_mat = self.mix_mat.transpose(0, 1)  # (n_components, n_features)

            if data_type == torch.tensor:
                self.unmix_mat = torch.tensor(self.unmix_mat).to(
                    device).float().requires_grad_()  # (n_components, n_features)
                self.mix_mat = torch.tensor(self.mix_mat).to(
                    device).float().requires_grad_()  # (n_features, n_components)

    def pca_transform(self, hx):
        # Scale and project hx onto direction
        hx_z = (hx - self.hx_mu) / self.hx_std
        pc_loadings = hx_z @ self.pcs  # (n_datapoints, n_features) @ (n_features, n_components) --> (n_datapoints, n_components)
        return pc_loadings

    def ica_transform(self, hx):
        if len(hx.shape) > 2:
            hx = hx.squeeze()
        hx_z = (hx - self.hx_mu) / self.hx_std
        pc_loadings = hx_z @ self.pcs  # (n_datapoints, n_features) @ (n_features, n_components) --> (n_datapoints, n_components)
        pc_loadings = pc_loadings[:, :self.num_ica_components]  # (n_datapoints, n_components) --> (n_datapoints, n_components_ica)
        whitened_pc_loadings = pc_loadings / self.pc_std[:self.num_ica_components]
        source_signals = whitened_pc_loadings @ self.unmix_mat # Z'@W^T # (n_datapoints, n_components_ica) @ (n_components_ica, n_components_ica) --> (n_datapoints, n_components_ica)
        return source_signals

    def project_gradients_into_pc_space(self, grad_data):
        sigma = np.diag(self.hx_std)
        grad_data = grad_data.T  # So each column is a grad vector for a hx
        scaled_pc_comps = self.pcs.T @ sigma  # PCs calculated on X'=(X-mu)/sigma are scaled so it's like they were calculated on X
        projected_grads = scaled_pc_comps @ grad_data  # grads are projected onto the scaled PCs
        return projected_grads.T

    def project_gradients_into_ica_space(self, grad_data):
        pcs = self.pcs[:,:self.num_ica_components]
        pc_std = self.pc_std[:self.num_ica_components]

        Qt = self.diag(self.hx_std) @ pcs # (sigma_x @ C^T )
        Qt = Qt @ self.diag(pc_std)
        Qt = Qt @ self.mix_mat  # (cols are indep components)
        projected_grads = grad_data @ Qt
        # sigma = self.diag(self.hx_std)
        # grad_data = grad_data.T  # So each column is a grad vector for a hx i.e. (n_datapoints, n_features) --> (n_features, n_datapoints)
        # scaled_pc_comps = self.pcs.T @ sigma  # PCs calculated on X'=(X-mu)/sigma are scaled so it's like they were calculated on X
        # projected_grads_to_pc_space = scaled_pc_comps @ grad_data  # grads are projected onto the scaled PCs
        # projected_grads_to_pc_space = projected_grads_to_pc_space[:self.num_ica_components, :]
        # projected_grads_to_ic_space = projected_grads_to_pc_space.T @ self.mix_mat.T
        # return projected_grads_to_ic_space
        return projected_grads