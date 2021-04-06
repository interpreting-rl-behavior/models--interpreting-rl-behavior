# https://github.com/aserdega/convlstmgru/blob/master/convgru.py
# https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

class ResidualBlock(nn.Module):

    def __init__(self, channels,
                 actv=torch.selu,
                 kernel_sizes=[3, 3],
                 paddings=[1, 1]):
        super(ResidualBlock, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_sizes[0],
                               padding=paddings[0])
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_sizes[1],
                               padding=paddings[0])
        self.actv = actv

    def forward(self, x):
        inputs = x
        x = self.actv(x)
        x = self.conv0(x)
        x = self.actv(x)
        x = self.conv1(x)
        return x + inputs

class AssimilatorResidualBlock(nn.Module):
    """Assimilates a 1d tensor into a 3d tensor so it can be used by
    convolutional networks."""
    def __init__(self, channels,
                 vec_size,
                 actv=torch.selu,
                 kernel_sizes=[1, 3],
                 paddings=[0, 1]):
        super(AssimilatorResidualBlock, self).__init__()

        # Usually a 1x1 conv (see default args)
        self.conv0 = nn.Conv2d(in_channels=channels+vec_size,
                               out_channels=channels,
                               kernel_size=kernel_sizes[0],
                               padding=paddings[0])

        # Usually a normal 2d conv
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_sizes[1],
                               padding=paddings[1])
        self.actv = actv

    def forward(self, x, vec):
        inputs = x
        x = self.actv(x)

        # Concat vector along channel dim
        vec_block = torch.stack([vec]*x.shape[2]*x.shape[3], dim=2)
        vec_block = vec_block.view(vec.shape[0], vec.shape[-1], x.shape[2], x.shape[3])
        x = torch.cat([x, vec_block], dim=1)

        # Continue as in normal residual block
        x = self.conv0(x)
        x = self.actv(x)
        x = self.conv1(x)
        return x + inputs

class Attention(nn.Module):
    """Applies self attention

        Flattens a convolutional network output on the channel dimension
        then applies self-attention. This serves as a non-local layer.
    """
    def __init__(self, ch, which_conv=nn.Conv2d, name='attention'):
        super(Attention, self).__init__()

        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1,
                                     padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1,
                                   padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1,
                                 padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1,
                                 padding=0, bias=False)

        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)
    def forward(self, x):

        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])

        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)

        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)

        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2,
                                                          x.shape[2], x.shape[3]))
        return self.gamma * o + x


class ResOneByOne(nn.Module):
    """Connects two 3d tensors using a 1x1 convnet with residual connections

    Only input x has residual connections. Thanks to this residual connction,
    this module is essentially for modifying x using (x,y). Similar to the
    AssimilatorResidualBlock, it can be used to assimilate y into x, but
    in ResOneByOne both inputs are 3d tensors, whereas in
    AssimilatorResidualBlock only x is 3d tensors.

    """
    def __init__(self, in_channels, out_channels,
                 activation=nn.SELU()):
        super(ResOneByOne, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation = activation

        # Conv layers
        self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels,
                               kernel_size=1, padding=0)

    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)
        h = self.conv1x1(self.activation(h))
        return h + x

class ResBlockUp(nn.Module):
    """A Residual Block with optional upsampling.

    Adapted from [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py ) (Brock et al. 2018)

    """
    def __init__(self, in_channels, out_channels, hw,
                 which_conv=nn.Conv2d, which_norm=nn.LayerNorm,
                 activation=nn.SELU(), upsample=nn.UpsamplingNearest2d(scale_factor=2)):
        super(ResBlockUp, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_norm = which_conv, which_norm
        self.activation = activation
        self.upsample = upsample
        kern = 3
        stride = 1
        
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels,
                                     kernel_size=kern, stride=stride,
                                     padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels,
                                     kernel_size=kern, stride=stride,
                                     padding=1)

        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc: # learnable shortcut connection
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Normalization layers
        self.normalize1 = self.which_norm([self.in_channels, hw, hw])
        self.normalize2 = self.which_norm([self.out_channels, hw*2, hw*2])
        
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(self.normalize1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.normalize2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class ResBlockDown(nn.Module):
    """A Residual Block with optional downsampling.

    Adapted from [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py ) (Brock et al. 2018)

    """
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d,
                 wide=True,
                 preactivation=False, activation=nn.SELU(), downsample=None, ):
        super(ResBlockDown, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
        kern = 3
        stride = 1

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                     kernel_size=kern, stride=stride,
                                     padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels,
                                     kernel_size=kern, stride=stride,
                                     padding=1)
        self.learnable_sc = True if (
            in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.selu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


### Conv GRU after here:
class ConvGRUCell(nn.Module):
    """A Convolutional GRU network cell.

    Adapted from [aserdga/convlstmgru](https://github.com/aserdega/convlstmgru/blob/master/convgru.py)

    Parameters
    ----------
    input_size: (int, int)
        Height and width of input tensor as (height, width).
    input_dim: int
        Number of channels of input tensor.
    hidden_dim: int
        Number of channels of hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel.
    bias: bool
        Whether or not to add the bias.
    """
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, activation=torch.tanh, batchnorm=False, device='cuda:0'):
        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim          = input_dim
        self.hidden_dim         = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        self.activation  = activation
        self.batchnorm   = batchnorm

        self.device = device


        self.conv_zr = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=2 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv_h1 = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv_h2 = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.reset_parameters()

    def forward(self, input, h_prev):
        combined = torch.cat((input, h_prev), dim=1)  # concatenate along channel axis

        combined_conv = torch.sigmoid(self.conv_zr(combined))

        z, r = torch.split(combined_conv, self.hidden_dim, dim=1)

        h_ = self.activation(self.conv_h1(input) + r * self.conv_h2(h_prev))

        h_cur = (1 - z) * h_ + z * h_prev

        return h_cur

    def init_hidden(self, batch_size):
        state = torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        if "cuda" in str(self.device):
            state = state.cuda()
        return state

    def reset_parameters(self):
        #self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv_zr.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv_zr.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h1.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv_h1.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h2.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv_h2.bias.data.zero_()

        if self.batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()


class ConvGRU(nn.Module):
    """A Convolutional GRU network.

    Adapted from [aserdga/convlstmgru](https://github.com/aserdega/convlstmgru/blob/master/convgru.py)

    """
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, activation=torch.tanh, batchnorm=False, device='cuda:0'):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        activation  = self._extend_for_multilayer(activation, num_layers)

        if not len(kernel_size) == len(hidden_dim) == len(activation) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          activation=activation[i],
                                          batchnorm=batchnorm,
                                          device=device))

        self.cell_list = nn.ModuleList(cell_list)

        self.reset_parameters()

    def forward(self, input, hidden_state):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
        Returns
        -------
        last_state_list, layer_output
        """
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))

        if not hidden_state:
            hidden_state = self.get_init_states(cur_layer_input[0].size(0))

        seq_len = len(cur_layer_input)

        layer_output_list = []
        last_state_list   = []

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input=cur_layer_input[t],
                                                 h_prev=h)
                output_inner.append(h)

            cur_layer_input = output_inner
            last_state_list.append(h)

        layer_output = torch.stack(output_inner, dim=int(self.batch_first))

        return layer_output, last_state_list

    def reset_parameters(self):
        for c in self.cell_list:
            c.reset_parameters()

    def get_init_states(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list)
            and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param