# External libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Internal libs
from . import network_utils as net_utils
from . import aspp
segment_feature_ratio = {64000:128}

class SkipGanSynthAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, skip_op = 'add'):
        super(SkipGanSynthAE, self).__init__()
        kw = feat_width//64
        self.skip_op = skip_op     
        dec_ch_scale = 2 if self.skip_op == 'concat' else 1
        self.enc_1 = nn.Conv2d(in_channels, 32, (1,1), 1)
        self.enc_2 = nn.Sequential(*net_utils.create_gansynth_block(32, 32, 'enc'))
        self.enc_3 = nn.Sequential(*net_utils.create_gansynth_block(32, 64, 'enc'))
        self.enc_4 = nn.Sequential(*net_utils.create_gansynth_block(64, 128, 'enc'))
        self.enc_5 = nn.Sequential(*net_utils.create_gansynth_block(128, 256, 'enc'))
        self.enc_6 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'enc'))
        self.enc_7 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'enc'))
        self.enc_8 = nn.Conv2d(256, latent_size, (16,kw), 1)
        self.enc_net_params = list(self.enc_1.parameters()) + list(self.enc_2.parameters()) + \
                         list(self.enc_3.parameters()) + list(self.enc_4.parameters()) + \
                         list(self.enc_5.parameters()) + list(self.enc_6.parameters()) + \
                         list(self.enc_7.parameters()) + list(self.enc_8.parameters())

        self.dec_1 = nn.Conv2d(latent_size, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)
        self.dec_2 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_3 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_4 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 128, 'dec'))
        self.dec_5 = nn.Sequential(*net_utils.create_gansynth_block(128*dec_ch_scale, 64, 'dec'))
        self.dec_6 = nn.Sequential(*net_utils.create_gansynth_block(64*dec_ch_scale, 32, 'dec'))
        self.dec_7 = nn.Sequential(*net_utils.create_gansynth_block(32*dec_ch_scale, 32, 'dec'))
        self.dec_8 = nn.Conv2d(32*dec_ch_scale, in_channels, kernel_size=(1,1), stride=1)
    
    def encode(self, input):
        he_1 = self.enc_1(input)
        he_2 = self.enc_2(he_1)
        he_3 = self.enc_3(he_2)
        he_4 = self.enc_4(he_3)
        he_5 = self.enc_5(he_4)
        he_6 = self.enc_6(he_5)
        he_7 = self.enc_7(he_6)
        latent = self.enc_8(he_7)
        return latent, [he_1,he_2,he_3,he_4,he_5,he_6,he_7]

    def decode(self, latent, skip, cond=None):
        hd_1 = self.dec_1(latent)
        hd_2 = self.dec_2(net_utils.skip_connection(hd_1, skip[6], self.skip_op))
        hd_3 = self.dec_3(net_utils.skip_connection(hd_2, skip[5], self.skip_op))
        hd_4 = self.dec_4(net_utils.skip_connection(hd_3, skip[4], self.skip_op))
        hd_5 = self.dec_5(net_utils.skip_connection(hd_4, skip[3], self.skip_op))
        hd_6 = self.dec_6(net_utils.skip_connection(hd_5, skip[2], self.skip_op))
        hd_7 = self.dec_7(net_utils.skip_connection(hd_6, skip[1], self.skip_op))
        out = self.dec_8(net_utils.skip_connection(hd_7, skip[0], self.skip_op))

        return out

    def forward(self, input, cond=None):
        latent, skip = self.encode(input)
        out = self.decode(latent, skip)

        return out, latent

class SkipASPPGanSynthAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, skip_op = 'add'):
        super(SkipASPPGanSynthAE, self).__init__()
        kw = feat_width//64
        self.skip_op = skip_op     
        dec_ch_scale = 2 if self.skip_op == 'concat' else 1
        self.enc_1 = nn.Conv2d(in_channels, 32, (1,1), 1)
        self.aspp_1 = aspp.ASPP(32, 32)
        self.enc_2 = nn.Sequential(*net_utils.create_gansynth_block(32, 32, 'enc'))
        self.aspp_2 = aspp.ASPP(32, 32)
        self.enc_3 = nn.Sequential(*net_utils.create_gansynth_block(32, 64, 'enc'))
        self.aspp_3 = aspp.ASPP(64, 64)
        self.enc_4 = nn.Sequential(*net_utils.create_gansynth_block(64, 128, 'enc'))
        self.aspp_4 = aspp.ASPP(128, 128)
        self.enc_5 = nn.Sequential(*net_utils.create_gansynth_block(128, 256, 'enc'))
        self.aspp_5 = aspp.ASPP(256, 256)
        self.enc_6 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'enc'))
        self.aspp_6 = aspp.ASPP(256, 256)
        self.enc_7 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'enc'))
        self.aspp_7 = aspp.ASPP(256, 256)
        self.enc_8 = nn.Conv2d(256, latent_size, (16,kw), 1)
        self.enc_net_params = list(self.enc_1.parameters()) + list(self.enc_2.parameters()) + \
                            list(self.enc_3.parameters()) + list(self.enc_4.parameters()) + \
                            list(self.enc_5.parameters()) + list(self.enc_6.parameters()) + \
                            list(self.enc_7.parameters()) + list(self.enc_8.parameters()) + \
                            list(self.aspp_1.parameters()) + list(self.aspp_2.parameters()) + \
                            list(self.aspp_3.parameters()) + list(self.aspp_4.parameters()) + \
                            list(self.aspp_5.parameters()) + list(self.aspp_6.parameters()) + \
                            list(self.aspp_7.parameters())

        self.dec_1 = nn.Conv2d(latent_size, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)
        self.dec_2 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_3 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_4 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 128, 'dec'))
        self.dec_5 = nn.Sequential(*net_utils.create_gansynth_block(128*dec_ch_scale, 64, 'dec'))
        self.dec_6 = nn.Sequential(*net_utils.create_gansynth_block(64*dec_ch_scale, 32, 'dec'))
        self.dec_7 = nn.Sequential(*net_utils.create_gansynth_block(32*dec_ch_scale, 32, 'dec'))
        self.dec_8 = nn.Conv2d(32*dec_ch_scale, in_channels, kernel_size=(1,1), stride=1)
    
    def encode(self, input):
        he_1 = self.enc_1(input)
        skip_1 = self.aspp_1(he_1)
        he_2 = self.enc_2(he_1)
        skip_2 = self.aspp_2(he_2)
        he_3 = self.enc_3(he_2)
        skip_3 = self.aspp_3(he_3)
        he_4 = self.enc_4(he_3)
        skip_4 = self.aspp_4(he_4)
        he_5 = self.enc_5(he_4)
        skip_5 = self.aspp_5(he_5)
        he_6 = self.enc_6(he_5)
        skip_6 = self.aspp_6(he_6)
        he_7 = self.enc_7(he_6)
        skip_7 = self.aspp_7(he_7)
        latent = self.enc_8(he_7)
        return latent, [skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7]

    def decode(self, latent, skip, cond=None):
        hd_1 = self.dec_1(latent)
        hd_2 = self.dec_2(net_utils.skip_connection(hd_1, skip[6], self.skip_op))
        hd_3 = self.dec_3(net_utils.skip_connection(hd_2, skip[5], self.skip_op))
        hd_4 = self.dec_4(net_utils.skip_connection(hd_3, skip[4], self.skip_op))
        hd_5 = self.dec_5(net_utils.skip_connection(hd_4, skip[3], self.skip_op))
        hd_6 = self.dec_6(net_utils.skip_connection(hd_5, skip[2], self.skip_op))
        hd_7 = self.dec_7(net_utils.skip_connection(hd_6, skip[1], self.skip_op))
        out = self.dec_8(net_utils.skip_connection(hd_7, skip[0], self.skip_op))

        return out

    def forward(self, input, cond=None):
        latent, skip = self.encode(input)
        out = self.decode(latent, skip)

        return out, latent

class SkipGanSynthAE_v2(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, skip_op = 'add'):
        super(SkipGanSynthAE_v2, self).__init__()
        kw = feat_width//64
        self.skip_op = skip_op     
        dec_ch_scale = 2 if self.skip_op == 'concat' else 1
        self.enc_1 = nn.Conv2d(in_channels, 32, (1,1), 1)
        self.enc_2 = nn.Sequential(*net_utils.create_gansynth_block(32, 32, 'enc'))
        self.enc_3 = nn.Sequential(*net_utils.create_gansynth_block(32, 64, 'enc'))
        self.enc_4 = nn.Sequential(*net_utils.create_gansynth_block(64, 128, 'enc'))
        self.enc_5 = nn.Sequential(*net_utils.create_gansynth_block(128, 256, 'enc'))
        self.enc_6 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'enc'))
        self.enc_7 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'enc'))
        self.enc_8 = nn.Conv2d(256, latent_size, (16,kw), 1)
        self.enc_net_params = list(self.enc_1.parameters()) + list(self.enc_2.parameters()) + \
                         list(self.enc_3.parameters()) + list(self.enc_4.parameters()) + \
                         list(self.enc_5.parameters()) + list(self.enc_6.parameters()) + \
                         list(self.enc_7.parameters()) + list(self.enc_8.parameters())

        self.dec_1 = nn.Conv2d(latent_size, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)
        self.dec_2 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'dec'))
        self.dec_3 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_4 = nn.Sequential(*net_utils.create_gansynth_block(256, 128, 'dec'))
        self.dec_5 = nn.Sequential(*net_utils.create_gansynth_block(128*dec_ch_scale, 64, 'dec'))
        self.dec_6 = nn.Sequential(*net_utils.create_gansynth_block(64, 32, 'dec'))
        self.dec_7 = nn.Sequential(*net_utils.create_gansynth_block(32*dec_ch_scale, 32, 'dec'))
        self.dec_8 = nn.Conv2d(32, in_channels, kernel_size=(1,1), stride=1)
    
    def encode(self, input):
        he_1 = self.enc_1(input)
        he_2 = self.enc_2(he_1)
        he_3 = self.enc_3(he_2)
        he_4 = self.enc_4(he_3)
        he_5 = self.enc_5(he_4)
        he_6 = self.enc_6(he_5)
        he_7 = self.enc_7(he_6)
        latent = self.enc_8(he_7)
        return latent, [he_2,he_4,he_6]

    def decode(self, latent, skip, cond=None):
        hd_1 = self.dec_1(latent)
        hd_2 = self.dec_2(hd_1)
        hd_3 = self.dec_3(net_utils.skip_connection(hd_2, skip[2], self.skip_op))
        hd_4 = self.dec_4(hd_3)
        hd_5 = self.dec_5(net_utils.skip_connection(hd_4, skip[1], self.skip_op))
        hd_6 = self.dec_6(hd_5)
        hd_7 = self.dec_7(net_utils.skip_connection(hd_6, skip[0], self.skip_op))
        out = self.dec_8(hd_7)

        return out

    def forward(self, input, cond=None):
        latent, skip = self.encode(input)
        out = self.decode(latent, skip)

        return out, latent

class GanSynthAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128):
        super(GanSynthAE, self).__init__()
        kw = feat_width//64
        enc_layers = []
        enc_layers += [nn.Conv2d(in_channels, 32, (1,1), 1)]
        enc_layers += net_utils.create_gansynth_block(32, 32, 'enc')
        enc_layers += net_utils.create_gansynth_block(32, 64, 'enc')
        enc_layers += net_utils.create_gansynth_block(64, 128, 'enc')
        enc_layers += net_utils.create_gansynth_block(128, 256, 'enc')
        enc_layers += net_utils.create_gansynth_block(256, 256, 'enc')
        enc_layers += net_utils.create_gansynth_block(256, 256, 'enc')
        enc_layers += [nn.Conv2d(256, latent_size, (16,kw), 1)]
        self.enc_net = nn.Sequential(*enc_layers)
        self.enc_net_params = self.enc_net.parameters()

        dec_layers = []
        dec_layers += [nn.Conv2d(latent_size, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)]
        dec_layers += net_utils.create_gansynth_block(256, 256, 'dec')
        dec_layers += net_utils.create_gansynth_block(256, 256, 'dec')
        dec_layers += net_utils.create_gansynth_block(256, 128, 'dec')
        dec_layers += net_utils.create_gansynth_block(128, 64, 'dec')
        dec_layers += net_utils.create_gansynth_block(64, 32, 'dec')
        dec_layers += net_utils.create_gansynth_block(32, 32, 'dec')
        dec_layers += [nn.Conv2d(32, in_channels, kernel_size=(1,1), stride=1)]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encode(self, input):
        return self.enc_net(input)

    def decode(self, latent, cond=None):
        return self.dec_net(latent)

    def forward(self, input, cond=None):
        latent = self.enc_net(input)
        out = self.dec_net(latent)
        
        return out, latent

class GanSynthDisc(nn.Module):
    ''' 
    '''
    def __init__(self, in_channels = 2, feat_width = 128):
        super(GanSynthDisc, self).__init__()
        kw = feat_width//64
        layers = []
        layers += [nn.Conv2d(in_channels, 32, (1,1), 1)]
        layers += net_utils.create_gansynth_block(32, 32)
        layers += net_utils.create_gansynth_block(32, 64)
        layers += net_utils.create_gansynth_block(64, 128)
        layers += net_utils.create_gansynth_block(128, 256)
        layers += net_utils.create_gansynth_block(256, 256)
        layers += net_utils.create_gansynth_block(256, 128)
        self.net = nn.Sequential(*layers)
        self.linear = nn.Linear(4096, 1)

    def forward(self, x):
        h = self.net(x)
        out = self.linear(h.view(h.size(0), -1))
        return out

class GrowingGanSynthEnc(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, config = None):
        super(GrowingGanSynthEnc, self).__init__()
        kw = feat_width//64
        self.config = config # Updated overtime
        self.enc_net = nn.ModuleList([nn.Conv2d(in_channels, 32, (1,1), 1),
                                    net_utils.GANSynthBlock(32, 32, 'enc'),
                                    net_utils.GANSynthBlock(32, 64, 'enc'),
                                    net_utils.GANSynthBlock(64, 128, 'enc'),
                                    net_utils.GANSynthBlock(128, 256, 'enc'),
                                    net_utils.GANSynthBlock(256, 256, 'enc'),
                                    net_utils.GANSynthBlock(256, 256, 'enc'),
                                    nn.Conv2d(256, latent_size, (16,kw), 1)])
        self.enc_net_params = self.enc_net.parameters()

        self.from_rgb = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                       nn.Conv2d(in_channels, 32, 1),
                                       nn.Conv2d(in_channels, 32, 1),
                                       nn.Conv2d(in_channels, 64, 1),
                                       nn.Conv2d(in_channels, 128, 1),
                                       nn.Conv2d(in_channels, 256, 1),
                                       nn.Conv2d(in_channels, 256, 1),
                                       nn.Conv2d(in_channels, 256, 1)])

    def forward(self, input):
        for i, (conv, from_rgb) in enumerate(zip(self.enc_net, self.from_rgb)):
            phase_idx = len(self.enc_net) - self.config['phase'] - 1
            if i == phase_idx:
                if i !=  len(self.enc_net)- 1 and self.config['status'] == 'fadein':
                    out = from_rgb(input)
                    out = conv(out)
                    # The first layer has no upsampling, therefore we do not need to upsample the skip connection
                    input_down = net_utils.down_sample2x2(input) if i != 0 else input
                    skip_rgb = self.from_rgb[i + 1](input_down)
                    out = self.config['alpha'] * skip_rgb + (1 - self.config['alpha']) * out # alpha descreses from 0 to 1
                else: # Stable
                    if phase_idx == 0:
                        out = conv(input)
                    else:
                        out = from_rgb(input)
                        out = conv(out)
            elif i > phase_idx:
                out = conv(out)
        return out
    
class GrowingGanSynthDec(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, config = None):
        super(GrowingGanSynthDec, self).__init__()
        kw = feat_width//64
        self.config = config
        self.dec_net = nn.ModuleList([nn.Conv2d(latent_size, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False),
                                    net_utils.GANSynthBlock(256, 256, 'dec'),
                                    net_utils.GANSynthBlock(256, 256, 'dec'),
                                    net_utils.GANSynthBlock(256, 128, 'dec'),
                                    net_utils.GANSynthBlock(128, 64, 'dec'),
                                    net_utils.GANSynthBlock(64, 32, 'dec'),
                                    net_utils.GANSynthBlock(32, 32, 'dec'),
                                    nn.Conv2d(32, in_channels, kernel_size=(1,1), stride=1)])

        self.to_rgb = nn.ModuleList([nn.Conv2d(256, in_channels, 1), #Each has 3 out channels and kernel size 1x1!
                                     nn.Conv2d(256, in_channels, 1),
                                     nn.Conv2d(256, in_channels, 1),
                                     nn.Conv2d(128, in_channels, 1),
                                     nn.Conv2d(64, in_channels, 1),
                                     nn.Conv2d(32, in_channels, 1),
                                     nn.Conv2d(32, in_channels, 1),
                                     nn.Conv2d(in_channels, in_channels, 1)])

    def forward(self, input):
        for i, (conv, to_rgb) in enumerate(zip(self.dec_net, self.to_rgb)):
            if i == 0:
                out = conv(input)
            else:
                out = conv(out)
            if i == self.config['phase']:
                if i > 0 and self.config['status'] == 'fadein':
                    skip_rgb = to_rgb(out)
                    out = to_rgb(out)
                    out = self.config['alpha'] * skip_rgb + (1 - self.config['alpha']) * out # alpha descreses from 0 to 1
                elif i != len(self.dec_net)-1: # Stable and not the last layer.
                    out = to_rgb(out)
                break
        return out


def instantiate_autoencoder(opt):
    """ Given the options file, it instantiate a Decoder model.

    Parameters:s
        opt (Dictionary): Options.
    Returns:
        Decoder instance.
    """ 
    ref_segm_size = int(list(segment_feature_ratio.keys())[0])
    feat_width = int((opt['data']['segment_size']/ref_segm_size) * segment_feature_ratio[ref_segm_size])
    if opt['model']['ae'] == 'skipgansynth':
        return SkipGanSynthAE(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)
    elif opt['model']['ae'] == 'skipcatgansynth':
        return SkipGanSynthAE(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width,
                        skip_op='concat')
    elif opt['model']['ae'] == 'skipasppcatgansynth':
        return SkipASPPGanSynthAE(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width,
                        skip_op='concat')
    elif opt['model']['ae'] == 'gansynth':
        return GanSynthAE(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)
    #elif opt['model']['ae'] == 'shallowgansynth':
    #    return ShallowGanSynthAE(opt['model']['timbre_latent_size'],
    #                    opt['model']['in_ch'],
    #                    feat_width= feat_width)
    elif opt['model']['ae'] == 'skipcatgansynth_v2':
        return SkipGanSynthAE_v2(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width,
                        skip_op='concat')
    else:
        raise NotImplementedError

def instantiate_encoder(opt):
    ref_segm_size = int(list(segment_feature_ratio.keys())[0])
    feat_width = int((opt['data']['segment_size']/ref_segm_size) * segment_feature_ratio[ref_segm_size])
    if opt['model']['ae'] == 'growgansynth':
        return GrowingGanSynthEnc(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)

def instantiate_decoder(opt):
    ref_segm_size = int(list(segment_feature_ratio.keys())[0])
    feat_width = int((opt['data']['segment_size']/ref_segm_size) * segment_feature_ratio[ref_segm_size])
    if opt['model']['ae'] == 'growgansynth':
        return GrowingGanSynthDec(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)

def instantiate_discriminator(opt):
    """ Given the options file, it instantiate a discriminator model.
    Parameters:s
        opt (Dictionary): Options.
    Returns:
        classifier instance.
    """ 
    if opt['model']['disc'] == 'gansynth':
        return GanSynthDisc(opt['model']['in_ch'])
    else:
        raise NotImplementedError

