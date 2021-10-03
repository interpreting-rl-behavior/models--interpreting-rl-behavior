# https://github.com/aserdega/convlstmgru/blob/master/convgru.py
# https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py

import torch
import torch.nn as nn

class LayeredConvNet(nn.Module):
    """Template class

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, stride=[2, 2, 2],
                 channels=[3, 64, 32, 32],
                 kernel_sizes=[6, 6, 6],
                 padding_hs=[1, 1, 1],
                 padding_ws=[1, 1, 1],
                 input_hw=64,
                 deconv=False,
                 layer_norm=False):
        super(LayeredConvNet, self).__init__()

        # Check all inps have right len
        if not len(channels)-1 == len(kernel_sizes) or \
           not len(kernel_sizes) == len(padding_hs) or \
           not len(padding_hs) == len(padding_ws) or \
           not len(padding_ws) == len(stride):
            raise ValueError("One of your conv lists has the wrong length.")

        self.input_hw = input_hw
        layer_in_hw = self.input_hw
        self.nets = nn.ModuleList([])
        self.deconv = deconv
        ch_in = channels[0]
        self.lns = [] # for debugging only
        for i, (ch_out, k, p_h, p_w, strd) in enumerate(zip(channels[1:],
                                                 kernel_sizes,
                                                 padding_hs,
                                                 padding_ws,
                                                 stride)):
            if deconv:
                net = nn.ConvTranspose2d(ch_in, ch_out,
                                         kernel_size=(k,k),
                                         stride=(strd,strd),
                                         padding=(p_h,p_w))
            else:
                net = nn.Conv2d(ch_in, ch_out,
                                kernel_size=(k,k),
                                stride=(strd,strd),
                                padding=(p_h,p_w))
            self.nets.append(net)
            if i < len(kernel_sizes) - 1: # Doesn't add actv or LN on last layer
                if layer_norm:
                    layer_in_hw = conv_output_size(layer_in_hw,
                                                   stride=strd,
                                                   padding=p_h,
                                                   kernel_size=k,
                                                   transposed=deconv)
                    ln = nn.LayerNorm([ch_out, layer_in_hw, layer_in_hw])
                    self.lns.append(ln)
                    self.nets.append(ln)
                # self.nets.append(nn.RReLU())
                self.nets.append(nn.LeakyReLU())
            ch_in = ch_out

    def forward(self, x):
        outs = []
        out = x
        for l in self.nets:
            out = l(out)
            outs.append(out)
        return out, outs

class LayeredResBlockUp(nn.Module):
    """
    A class that combines together several upsampling residual blocks.
    """
    def __init__(self,
                 input_hw=8,
                 input_ch=128,
                 hidden_ch=256,
                 output_hw=64,
                 output_ch=3):
        super(LayeredResBlockUp, self).__init__()

        self.input_hw = input_hw
        self.input_shape = (input_ch, input_hw, input_hw)
        self.input_size = int(torch.prod(torch.tensor(self.input_shape)).item())
        self.hidden_ch = hidden_ch
        self.output_ch = output_ch

        self.nets = nn.ModuleList([])

        # Recursively calculate hw
        image_out_size = 64
        doubles = 1
        while True:
            if image_out_size // (self.input_hw * 2**doubles ) == 1:
                break
            else:
                doubles += 1

        num_layers = doubles
        hw = input_hw
        for l in range(num_layers):
            if l == 0:  # Bottom
                in_ch = input_ch
                out_ch = self.hidden_ch
            elif l == num_layers - 1:  # Top
                in_ch = self.hidden_ch
                out_ch = self.output_ch
            else:  # Middle
                in_ch = self.hidden_ch
                out_ch = self.hidden_ch
            net = ResBlockUp(in_channels=in_ch,
                             out_channels=out_ch,
                             hw=hw)
            hw *= 2
            self.nets.append(net)
        self.output_shape = (self.output_ch, hw, hw)
        self.output_size = int(torch.prod(torch.tensor(self.output_shape)).item())

    def forward(self, x):
        outs = []
        out = x
        for l in self.nets:
            out = l(out)
            outs.append(out)
        return out, outs

class LayeredResBlockDown(nn.Module):
    """
    A class that combines together several downsampling residual blocks.
    """
    def __init__(self,
                 input_hw=64,
                 input_ch=3,
                 hidden_ch=256,
                 output_hw=8,
                 output_ch=128):
        super(LayeredResBlockDown, self).__init__()

        self.input_hw = input_hw
        self.input_shape = (input_ch, input_hw, input_hw)
        self.input_size = int(torch.prod(torch.tensor(self.input_shape)).item())
        self.hidden_ch = hidden_ch
        self.output_ch = output_ch

        self.nets = nn.ModuleList([])

        # Recursively calculate hw
        halvings = 1
        while True:
            if output_hw // (input_hw / 2**halvings) == 1:
                break
            else:
                halvings += 1

        num_layers = halvings
        hw = input_hw
        for l in range(num_layers):
            if l == 0:  # Bottom
                in_ch = input_ch
                out_ch = self.hidden_ch
            elif l == num_layers - 1:  # Top
                in_ch = self.hidden_ch
                out_ch = self.output_ch
            else:  # Middle
                in_ch = self.hidden_ch
                out_ch = self.hidden_ch
            net = ResBlockDown(in_channels=in_ch,
                               out_channels=out_ch,
                               hw=hw)
            hw /= 2
            self.nets.append(net)
        self.output_shape = (self.output_ch, hw, hw)
        self.output_size = int(torch.prod(torch.tensor(self.output_shape)).item())

    def forward(self, x):
        outs = []
        out = x
        for l in self.nets:
            out = l(out)
            outs.append(out)
        return out, outs

class NLayerPerceptron(nn.Module):
    """An N layer perceptron with a linear output.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, sizes=[256, 256], layer_norm=False):
        super(NLayerPerceptron, self).__init__()
        self.nets = nn.ModuleList([])
        for i in range(len(sizes)-1):
            net = nn.Linear(in_features=sizes[i],
                            out_features=sizes[i+1])
            self.nets.append(net)

            if i < len(sizes)-2:  # Doesn't add activation (or LN) to final layer
                if layer_norm:
                    self.nets.append(nn.LayerNorm(sizes[i+1]))
                self.nets.append(nn.LeakyReLU())

    def forward(self, x):
        outs = []
        out = x
        for l in self.nets:
            out = l(out)
            outs.append(out)
        return out, outs


class ResBlockUp(nn.Module):
    """A Residual Block with optional upsampling.

    Adapted from [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py ) (Brock et al. 2018)

    """

    def __init__(self, in_channels, out_channels, hw,
                 which_conv=nn.Conv2d, which_norm=nn.LayerNorm,
                 activation=nn.LeakyReLU(),
                 upsample=nn.UpsamplingNearest2d(scale_factor=2)):
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
        if self.learnable_sc:  # learnable shortcut connection
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Normalization layers
        self.normalize1 = self.which_norm([self.in_channels, hw, hw])
        self.normalize2 = self.which_norm([self.out_channels, hw * 2, hw * 2])

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
    def __init__(self, in_channels, out_channels, hw,
                 which_conv=nn.Conv2d,
                 which_norm = nn.LayerNorm,
                 activation=nn.LeakyReLU(), downsample=nn.MaxPool2d(2, 2)):
        super(ResBlockDown, self).__init__()
        hw = int(hw)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels
        self.which_conv = which_conv
        self.which_norm = which_norm
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        kern = 3
        stride = 1
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                     kernel_size=kern, stride=stride,
                                     padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels,
                                     kernel_size=kern, stride=stride,
                                     padding=1)
        # Normalization layers
        self.normalize1 = self.which_norm([self.in_channels, hw, hw])
        self.normalize2 = self.which_norm([self.out_channels, hw, hw])

        self.learnable_sc = True if (in_channels != out_channels) or \
                                    downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return x

    def forward(self, x):
        h = x
        h = self.conv1(self.normalize1(h))
        h = self.conv2(self.activation(self.normalize2(h)))
        if self.downsample:
            h = self.downsample(h)
        return h + self.shortcut(x)


def conv_output_size(input_hw, stride, padding, kernel_size, transposed=False):
    """assumes a square input"""
    if transposed:
        out_hw = stride * (input_hw - 1) + kernel_size - 2 * padding
    else:
        out_hw = ( (input_hw - kernel_size + ( 2 * padding)) / stride) + 1
    return int(out_hw)
