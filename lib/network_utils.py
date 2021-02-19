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

class NSynthBlock(nn.Module):

    def __init__(self, in_ch, out_ch, k_size=(4,4), stride=(2,2), pad=(1,1), mode='enc'):
        super(NSynthBlock, self).__init__()
        if mode == 'enc':
            self.conv = nn.Sequential(
                                nn.Conv2d(in_ch, out_ch, k_size, stride, padding=pad),
                                nn.LeakyReLU(0.2),
                                nn.BatchNorm2d(out_ch))
        else:
            self.conv = nn.Sequential(
                                nn.ConvTranspose2d(in_ch, out_ch, k_size, stride, padding=pad),
                                nn.LeakyReLU(0.2),
                                nn.BatchNorm2d(out_ch))

    def forward(self, input):
        return self.conv(input)
    
class GANSynthBlockWider(nn.Module):

    def __init__(self, in_channel, out_channel, mode='enc'):
        super(GANSynthBlockWider, self).__init__()
        if mode == 'enc':
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (3,9), 1, padding=((3-1)//2, (9-1)//2)),
                    nn.Conv2d(out_channel, out_channel, (3,9), 1, padding=((3-1)//2, (9-1)//2)),
                    nn.LeakyReLU(0.2),
                    PixelWiseNormLayer(),
                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (3,9), 1, padding=((3-1)//2,(9-1)//2)),
                    nn.Conv2d(out_channel, out_channel, (3,9), 1, padding=((3-1)//2,(9-1)//2)),
                    nn.LeakyReLU(0.2),
                    PixelWiseNormLayer(),
                    nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, input):
        return self.conv(input)

class DDGANSynthBlock(nn.Module):
    def __init__(self, in_channel, out_channel, class_size, device, mode='enc', prob=0.2):
        super(DDGANSynthBlock, self).__init__()
        self.cnn_1 = DeterministicDropoutCNN(in_channel, out_channel, class_size, device, (3,3), pad=((3-1)//2,(3-1)//2), prob=prob)
        self.cnn_2 = DeterministicDropoutCNN(out_channel, out_channel, class_size, device, (3,3), pad=((3-1)//2,(3-1)//2), prob=prob)
        if mode == 'enc':
            self.sampling_layer = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)
        else:
            self.sampling_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.pixel_norm = PixelWiseNormLayer()

    def restore_weights(self, class_id):
        self.cnn_1.restore_weights(class_id)
        self.cnn_2.restore_weights(class_id)

    def forward(self, input, class_id):
        h = self.cnn_1(input, class_id)
        h = F.leaky_relu(self.cnn_2(h, class_id), 0.2)
        h = self.pixel_norm(h)
        out = self.sampling_layer(h)

        return out

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

class DeterministicDropoutLN(nn.Module):
    '''
    '''
    def __init__(self, input_size, out_size, class_size, device, prob=0.2):
        super(DeterministicDropoutLN, self).__init__()
        self.ln = torch.nn.Linear(input_size, out_size)
        _dropout_masks = build_dropout_mask(class_size, input_size, prob)
        _dropout_masks = nn.Parameter(_dropout_masks.type(torch.FloatTensor), requires_grad=False)
        self.register_buffer('dropout_masks', _dropout_masks)
        self.dropout_masks = self.dropout_masks.to(device)
        self.weight_backup = self.ln.weight.clone().detach().data

    def mask_weights(self, class_id):
        '''
        This method first backup the parameters and then mask the parameters of the linear layer
        according to the class mask.
        Args:
            class_id(tensor) - identification of the class in order to get respective mask
        '''
        self.weight_backup = self.ln.weight.clone().detach().data
        self.ln.weight.data = self.ln.weight * self.dropout_masks[class_id:class_id+1,:].expand(*self.ln.weight.size())

    def restore_weights(self, class_id):
        '''
        This method merges the new updated parameters with the dropped parameters 
        located in the backup according to its class
        Args:
            class_id(tensor) - identification of the class in order to get respective mask
        '''
        mask_condition = self.dropout_masks[class_id:class_id+1,:].expand(*self.ln.weight.size()) > 0
        self.ln.weight.data = torch.where(mask_condition, self.ln.weight, self.weight_backup)

    def forward(self, input, class_id):
        self.mask_weights(class_id)
        out = self.ln(input)
        return out

class DeterministicDropoutCNN(nn.Module):
    '''
    '''
    def __init__(self, input_ch, out_ch, class_size, device, kr_size=(3,3), stride=(1,1), pad=(0,0) ,prob=0.2):
        super(DeterministicDropoutCNN, self).__init__()
        self.cnn = torch.nn.Conv2d(input_ch, out_ch, kr_size, stride, pad)

        _dropout_masks = build_dropout_mask(class_size, out_ch, prob)
        _dropout_masks = nn.Parameter(_dropout_masks.type(torch.FloatTensor), requires_grad=False)
        self.register_buffer('dropout_masks', _dropout_masks)
        self.dropout_masks = self.dropout_masks.to(device)
        
        self.weight_backup = self.cnn.weight.clone().detach().data

    def mask_weights(self, class_id):
        '''
        This method first backup the parameters and then mask the parameters of the linear layer
        according to the class mask.
        Args:
            class_id(tensor) - identification of the class in order to get respective mask
        '''
        self.weight_backup = self.cnn.weight.clone().detach().data
        # TODO: Make it more readable
        dropout_mask = self.dropout_masks[class_id,:].squeeze(0).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(*self.cnn.weight.size())
        self.cnn.weight.data = self.cnn.weight * dropout_mask

    def restore_weights(self, class_id):
        '''
        This method merges the new updated parameters with the dropped parameters 
        located in the backup according to its class
        Args:
            class_id(tensor) - identification of the class in order to get respective mask
        '''
        dropout_mask = self.dropout_masks[class_id,:].squeeze(0).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(*self.cnn.weight.size())
        mask_condition = dropout_mask > 0
        self.cnn.weight.data = torch.where(mask_condition, self.cnn.weight, self.weight_backup)

    def forward(self, input, class_id):
        self.mask_weights(class_id)
        out = self.cnn(input)
        return out

def build_dropout_mask(class_size, mask_size, prob):
    '''
    This methods builds a list of dropout masks
    Args:
        class_size: The number of classes or masks.
        mask_size: The size of each mask.
        prob: a probability of dropping [0-1]
    Returns:
        tensor of masks [mask_len, mask_size]
    ''' 
    mask = np.random.binomial(1, (1-prob), (class_size, mask_size))
    return torch.from_numpy(mask)

def create_gansynth_block(in_ch, out_ch, mode='enc'):
    block_ly = []
    block_ly = []
    block_ly.append(nn.Conv2d(in_ch, out_ch, (3,3), 1, padding=((3-1)//2,(3-1)//2)))
    block_ly.append(nn.Conv2d(out_ch, out_ch, (3,3), 1, padding=((3-1)//2,(3-1)//2)))
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
    if tensor.size(2)!= 1 and tensor.size(3)!= 1:
        return F.avg_pool2d(tensor, kernel_size=(2,2), stride=(2,2))
    elif tensor.size(2)!= 1 and tensor.size(3) == 1:
        return F.avg_pool2d(tensor, kernel_size=(2,1), stride=(2,1))
    elif tensor.size(2)== 1 and tensor.size(3) != 1:
        return F.avg_pool2d(tensor, kernel_size=(1,2), stride=(1,2))
    else:
        return tensor

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

def log_gauss(z, mu, logvar):
    llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
    return torch.sum(llh, dim=1)    

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu

def build_mu_emb(shape):
    mu_emb = nn.Embedding(*shape)
    nn.init.xavier_uniform_(mu_emb.weight)
    mu_emb.weight.requires_grad = True
    return mu_emb

def build_logvar_emb(shape, pow_exp=0, is_trainable=False):
    logvar_emb = nn.Embedding(*shape)
    init_sigma = np.exp(pow_exp)
    init_logvar = np.log(init_sigma ** 2)
    nn.init.constant_(logvar_emb.weight, init_logvar)
    logvar_emb.weight.requires_grad = is_trainable
    return logvar_emb

def infer_class(z, mu, logvar, n_class, device):
    batch_size = z.shape[0]
    log_q_y_logit = torch.zeros(batch_size, n_class).type(z.type())

    for k_i in torch.arange(0, n_class).to(device):
        mu_k, logvar_k = mu(k_i), logvar(k_i)
        log_q_y_logit[:, k_i] = log_gauss(z, mu_k, logvar_k) + np.log(1 / n_class)

    q_y = torch.nn.functional.softmax(log_q_y_logit, dim=1)
    return log_q_y_logit, q_y

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