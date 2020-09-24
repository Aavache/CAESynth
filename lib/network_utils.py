import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Internal Libs
from . import spectral_norm as sn

class PixelWiseNormLayer(nn.Module):
    """
    Mentioned in '4.2 PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR'
    'Local response normalization'
    """

    def __init__(self):
        super(PixelWiseNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
    
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, device, shape, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)
        self.shape = shape

    def forward(self, ):
        scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
        sampled_noise = self.noise.repeat(*self.shape).normal_() * scale
        return sampled_noise

class ConvSN2D(nn.Module):
    """ Spetrally Normalized 2D Convolutional Neural Layer

    Parameters:
        input_ch (int): Number of input channels.
        output_ch (int): Number of output channels.
        ks (tuple, int): Kernel size of height, width, if integer, height and width will have same size.
        std (tuple, int): Stride size of height, width, if integer, height and width will have same size.
        pad (tuple, int): Padding size of height, width, if integer, height and width will have same size.
        use_bias (bool): Flag Using bias on the layer.
    """

    def __init__(self, input_ch, output_ch, ks, std, pad=0, use_bias= False):
        super(ConvSN2D, self).__init__()
        self.conv2d = sn.spectral_norm(nn.Conv2d(input_ch, output_ch, ks, std, pad, bias=use_bias))

    def forward(self, input):
        return self.conv2d(input)
    
class GANSynthBlock(nn.Module):

    def __init__(self, in_channel, out_channel, mode='enc'):
        super(GANSynthBlock, self).__init__()
        if mode == 'enc':
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (3,3), 1, padding=((3-1)//2,(3-1)//2)),
                    nn.Conv2d(out_channel, out_channel, (3,3), 1, padding=((3-1)//2,(3-1)//2)),
                    nn.LeakyReLU(0.2),
                    PixelWiseNormLayer(),
                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (3,3), 1, padding=((3-1)//2,(3-1)//2)),
                    nn.Conv2d(out_channel, out_channel, (3,3), 1, padding=((3-1)//2,(3-1)//2)),
                    nn.LeakyReLU(0.2),
                    PixelWiseNormLayer(),
                    nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, input):
        return self.conv(input)

class GANSynthBlock_2(nn.Module):

    def __init__(self, in_channel, out_channel, mode='enc'):
        super(GANSynthBlock_2, self).__init__()
        if mode == 'enc':
            # Spectrally Normalized Convolutional Layers
            self.conv = nn.Sequential(
                    ConvSN2D(in_channel, out_channel, (3,3), 1, pad=((3-1)//2,(3-1)//2)),
                    ConvSN2D(out_channel, out_channel, (3,3), 1, pad=((3-1)//2,(3-1)//2)),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))
        else: # de
            # Standard Convolutional Layers with Pixel Normalization
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (3,3), 1, padding=((3-1)//2,(3-1)//2)),
                    nn.Conv2d(out_channel, out_channel, (3,3), 1, padding=((3-1)//2,(3-1)//2)),
                    nn.LeakyReLU(0.2),
                    PixelWiseNormLayer(),
                    nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, input):
        return self.conv(input)

def create_gansynth_block(in_channel, out_channel, mode='enc'):
    block_ly = []
    block_ly.append(nn.Conv2d(in_channel, out_channel, (3,3), 1, padding=((3-1)//2,(3-1)//2)))
    block_ly.append(nn.Conv2d(out_channel, out_channel, (3,3), 1, padding=((3-1)//2,(3-1)//2)))
    block_ly.append(nn.LeakyReLU(0.2))
    block_ly.append(PixelWiseNormLayer())
    if mode == 'enc':
        block_ly.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))
    elif mode == 'dec':
        block_ly.append(nn.Upsample(scale_factor=2, mode='nearest'))
    else:
        raise NotImplementedError

    return block_ly

def skip_connection(forward_feat, skip_feat, skip_op):
    if skip_op == 'add':
        return forward_feat + skip_feat
    elif skip_op == 'concat':
        return torch.cat([forward_feat, skip_feat], dim=1)
    else:
        raise NotImplementedError

def down_sample2x2(tensor):
    return F.avg_pool2d(tensor, kernel_size = (2,2), stride=(2,2))

def up_sample2x2(tensor):
    return F.upsample(tensor, scale_factor=2, mode='nearest')

def var(x, dim=0):
    '''
    Calculates variance. [from https://github.com/DmitryUlyanov/AGE ]
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)

def populate_embed(z, noise='sphere'):
    '''
    Fills noise variable `z` with noise U(S^M) [from https://github.com/DmitryUlyanov/AGE ]
    '''
    #z.data.resize_(batch_size, nz) #, 1, 1)
    z.data.normal_(0, 1)
    if noise == 'sphere':
        normalize_(z.data)

def normalize_(x, dim=1):
    '''
    Projects points to a sphere inplace.
    '''
    zn = x.norm(2, dim=dim)
    zn = zn.unsqueeze(1)    
    x = x.div_(zn)
    x.expand_as(x)

def normalize(x, dim=1):
    '''
    Projects points to a sphere.
    '''
    zn = x.norm(2, dim=dim)
    zn = zn.unsqueeze(1)
    return x.div(zn).expand_as(x)  

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   - a list of networks
        requires_grad (bool)  - whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad