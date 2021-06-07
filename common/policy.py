import torch.nn

from .misc_util import orthogonal_init
from .model import GRU
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class CategoricalPolicy(nn.Module):
    def __init__(self, 
                 embedder,
                 recurrent,
                 action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """ 
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        # small scale weight-initialization in policy enhances the stability        
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)
            self.init_hx = \
                torch.nn.Parameter(torch.randn(self.embedder.output_dim) * 0.01)

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx, masks):
        hidden = self.embedder(x)
        if self.recurrent:
            # Fill in init hx to get grads right (it's a hacky solution to use
            #  trainable initial hidden states, but it's hard to get it to work
            #  with this Kostrikov repo since it uses numpy so much).
            inithx_mask = [torch.all(hx[i] == self.init_hx) for i in
                           range(hx.shape[0])]
            hx[inithx_mask] = self.init_hx
            hidden, hx = self.gru(hidden, hx, masks)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hx
