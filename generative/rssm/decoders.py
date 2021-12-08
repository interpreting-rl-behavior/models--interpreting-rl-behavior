from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from .functions import *
from .common import *


class MultiDecoder(nn.Module):

    def __init__(self, features_dim, conf):
        super().__init__()
        self.image_weight = 1.
        self.vecobs_weight = 1.
        self.reward_weight = 1.
        self.terminal_weight = 1.

        self.image = ConvDecoder(in_dim=features_dim,
                                 out_channels=conf.image_channels,
                                 cnn_depth=conf.cnn_depth)

        self.reward = DenseNormalDecoder(in_dim=features_dim, hidden_layers=conf.reward_decoder_layers, layer_norm=conf.layer_norm)
        self.terminal = DenseBernoulliDecoder(in_dim=features_dim, hidden_layers=conf.terminal_decoder_layers, layer_norm=conf.layer_norm)
        self.vecobs = DenseNormalDecoder(in_dim=features_dim, out_dim=64, hidden_layers=4, layer_norm=conf.layer_norm)

    def training_step(self,
                      features: TensorTBIF,
                      obs: Dict[str, Tensor],
                      extra_metrics: bool = False
                      ) -> Tuple[TensorTBI, Dict[str, Tensor], Dict[str, Tensor]]:
        tensors = {}
        metrics = {}

        loss_image_tbi, loss_image, image_rec = self.image.training_step(features, obs['image'])
        metrics.update(loss_image=loss_image.detach().mean())
        tensors.update(loss_image=loss_image.detach(),
                       image_rec=image_rec.detach())

        loss_reward_tbi, loss_reward, reward_rec = self.reward.training_step(features, obs['reward'])
        metrics.update(loss_reward=loss_reward.detach().mean())
        tensors.update(loss_reward=loss_reward.detach(),
                       reward_rec=reward_rec.detach())

        loss_terminal_tbi, loss_terminal, terminal_rec = self.terminal.training_step(features, obs['terminal'])
        metrics.update(loss_terminal=loss_terminal.detach().mean())
        tensors.update(loss_terminal=loss_terminal.detach(),
                       terminal_rec=terminal_rec.detach())

        assert loss_image_tbi.shape == loss_reward_tbi.shape == loss_terminal_tbi.shape
        loss_reconstr = (self.image_weight * loss_image_tbi
                         + self.reward_weight * loss_reward_tbi
                         + self.terminal_weight * loss_terminal_tbi)

        if extra_metrics: # TODO consider masking the loss where terminal==1. I strongly suspect I will do this.
            mask_rewardp = obs['reward'] > 0  # mask where reward is positive
            loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
            metrics.update(loss_rewardp=nanmean(loss_rewardp))
            tensors.update(loss_rewardp=loss_rewardp)

            mask_rewardn = obs['reward'] < 0  # mask where reward is negative
            loss_rewardn = loss_reward * mask_rewardn / mask_rewardn  # set to nan where ~mask
            metrics.update(loss_rewardn=nanmean(loss_rewardn))
            tensors.update(loss_rewardn=loss_rewardn)

            mask_terminal1 = obs['terminal'] > 0  # mask where terminal is 1
            loss_terminal1 = loss_terminal * mask_terminal1 / mask_terminal1  # set to nan where ~mask
            metrics.update(loss_terminal1=nanmean(loss_terminal1))
            tensors.update(loss_terminal1=loss_terminal1)

        return loss_reconstr, metrics, tensors, image_rec, reward_rec, terminal_rec

    def inference_step(self,
                      features: TensorTBIF,
                      extra_metrics: bool = False
                      ) -> Tuple[TensorTBI, Dict[str, Tensor], Dict[str, Tensor]]:
        loss_reconstr = None
        tensors = {}
        metrics = {}
        image_rec = self.image.inference_step(features)
        reward_rec = self.reward.inference_step(features)
        terminal_rec = self.terminal.inference_step(features)

        return loss_reconstr, metrics, tensors, image_rec, reward_rec, terminal_rec


class ConvDecoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_channels=3,
                 cnn_depth=32,
                 mlp_layers=0,
                 layer_norm=True,
                 activation=nn.ELU
                 ):
        super().__init__()
        self.in_dim = in_dim
        kernels = (5, 5, 6, 6)
        stride = 2
        d = cnn_depth
        if mlp_layers == 0:
            layers = [
                nn.Linear(in_dim, d * 32),  # No activation here in DreamerV2
            ]
        else:
            hidden_dim = d * 32
            norm = nn.LayerNorm if layer_norm else NoNorm
            layers = [
                nn.Linear(in_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
            for _ in range(mlp_layers - 1):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    norm(hidden_dim, eps=1e-3),
                    activation()]

        self.model = nn.Sequential(
            # FC
            *layers,
            nn.Unflatten(-1, (d * 32, 1, 1)),  # type: ignore
            # Deconv
            nn.ConvTranspose2d(d * 32, d * 4, kernels[0], stride),
            activation(),
            nn.ConvTranspose2d(d * 4, d * 2, kernels[1], stride),
            activation(),
            nn.ConvTranspose2d(d * 2, d, kernels[2], stride),
            activation(),
            nn.ConvTranspose2d(d, out_channels, kernels[3], stride))

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        y += 0.5 # So that the predictions start closer to the mean and have
        # fewer negative values, which are disallowed in images
        return y

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 3)
        loss = 0.5 * torch.square(output - target).sum(dim=[-1, -2, -3])  # MSE
        return unflatten_batch(loss, bd)

    def training_step(self, features: TensorTBIF, target: TensorTBCHW) -> Tuple[TensorTBI, TensorTB, TensorTBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean(dim=2)  # TBICHW => TBCHW

        assert len(loss_tbi.shape) == 3 and len(decoded.shape) == 5
        return loss_tbi, loss_tb, decoded

    def inference_step(self, features: TensorTBIF) -> Tuple[TensorTBI, TensorTB, TensorTBCHW]:
        I = features.shape[2]
        decoded = self.forward(features)
        decoded = decoded.mean(dim=2)  # TBICHW => TBCHW
        return decoded


class DenseBernoulliDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.model = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Bernoulli(logits=y.float())
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        return -output.log_prob(target)

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_tbi, loss_tb, decoded

    def inference_step(self, features: TensorTBIF) -> Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        decoded = self.forward(features)
        decoded = decoded.mean.mean(dim=2)
        assert len(decoded.shape) == 2
        return decoded


class DenseNormalDecoder(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=400, hidden_layers=2, layer_norm=True, std=0.3989422804):
        super().__init__()
        self.model = MLP(in_dim, out_dim, hidden_dim, hidden_layers, layer_norm)
        self.std = std
        self.out_dim = out_dim

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Normal(loc=y, scale=torch.ones_like(y) * self.std)
        if self.out_dim > 1:
            p = D.independent.Independent(p, 1)  # Makes p.logprob() sum over last dim
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        var = self.std ** 2  # var cancels denominator, which makes loss = 0.5 (target-output)^2
        return -output.log_prob(target) * var

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, Tensor]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == (2 if self.out_dim == 1 else 3)
        return loss_tbi, loss_tb, decoded

    def inference_step(self, features: TensorTBIF) -> Tuple[TensorTBI, TensorTB, Tensor]:
        assert len(features.shape) == 4
        I = features.shape[2]
        decoded = self.forward(features)
        decoded = decoded.mean.mean(dim=2)
        assert len(decoded.shape) == (2 if self.out_dim == 1 else 3)
        return decoded
