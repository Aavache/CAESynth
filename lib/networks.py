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

class SkipPostGanSynthAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, device, in_channels = 2, feat_width = 256, skip_op = 'add',
                    noise_mean=0 , noise_std=1):
        super(SkipPostGanSynthAE, self).__init__()
        kw = feat_width//64
        self.skip_op = skip_op
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.device = device
        dec_ch_scale = 2 if self.skip_op == 'concat' else 1
        self.enc_1 = nn.Conv2d(in_channels, 32, (1,1), 1)
        self.enc_2 = nn.Sequential(*net_utils.create_gansynth_block(32, 32, 'enc'))
        self.enc_3 = nn.Sequential(*net_utils.create_gansynth_block(32, 64, 'enc'))
        self.enc_4 = nn.Sequential(*net_utils.create_gansynth_block(64, 128, 'enc'))
        self.enc_5 = nn.Sequential(*net_utils.create_gansynth_block(128, 256, 'enc'))
        self.enc_6 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'enc'))
        self.enc_7 = nn.Sequential(*net_utils.create_gansynth_block(256, 256, 'enc'))
        self.enc_8 = nn.Conv2d(256, latent_size, (kw,kw), 1)
        self.enc_net_params = list(self.enc_1.parameters()) + list(self.enc_2.parameters()) + \
                         list(self.enc_3.parameters()) + list(self.enc_4.parameters()) + \
                         list(self.enc_5.parameters()) + list(self.enc_6.parameters()) + \
                         list(self.enc_7.parameters()) + list(self.enc_8.parameters())

        self.dec_1 = nn.Conv2d(2*latent_size, 256, kernel_size=(kw,kw), stride=1, padding=(kw-1,(kw-1)), bias=False)
        self.dec_2 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_3 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_4 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 128, 'dec'))
        self.dec_5 = nn.Sequential(*net_utils.create_gansynth_block(128*dec_ch_scale, 64, 'dec'))
        self.dec_6 = nn.Sequential(*net_utils.create_gansynth_block(64*dec_ch_scale, 32, 'dec'))
        self.dec_7 = nn.Sequential(*net_utils.create_gansynth_block(32*dec_ch_scale, 32, 'dec'))
        self.dec_8 = nn.Conv2d(32*dec_ch_scale, in_channels, kernel_size=(1,1), stride=1)
        #self.dec_9 = nn.Conv2d(2*in_channels, in_channels, kernel_size=(1,1), stride=1)
    
    def encode(self, input):
        he_1 = self.enc_1(input)
        he_2 = self.enc_2(he_1)
        he_3 = self.enc_3(he_2)
        he_4 = self.enc_4(he_3)
        he_5 = self.enc_5(he_4)
        he_6 = self.enc_6(he_5)
        he_7 = self.enc_7(he_6)
        latent = self.enc_8(he_7)
        return latent, [input, he_1,he_2,he_3,he_4,he_5,he_6,he_7]

    def decode(self, latent, skip, cond=None):
        hd_1 = self.dec_1(latent)
        hd_2 = self.dec_2(net_utils.skip_connection(hd_1, skip[7], self.skip_op))
        hd_3 = self.dec_3(net_utils.skip_connection(hd_2, skip[6], self.skip_op))
        hd_4 = self.dec_4(net_utils.skip_connection(hd_3, skip[5], self.skip_op))
        hd_5 = self.dec_5(net_utils.skip_connection(hd_4, skip[4], self.skip_op))
        hd_6 = self.dec_6(net_utils.skip_connection(hd_5, skip[3], self.skip_op))
        hd_7 = self.dec_7(net_utils.skip_connection(hd_6, skip[2], self.skip_op))
        hd_8 = self.dec_8(net_utils.skip_connection(hd_7, skip[1], self.skip_op))
        #out = self.dec_9(net_utils.skip_connection(hd_8, skip[0], 'add'))
        out = net_utils.skip_connection(hd_8, skip[0], 'add')

        return out

    def forward(self, input, cond=None):
        latent, skip = self.encode(input)
        noise = torch.FloatTensor(latent.size()).normal_(self.noise_mean, self.noise_std).to(self.device)
        latent = torch.cat([latent, noise], dim=1)
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
        out = torch.tanh(self.dec_net(latent))
        
        return out, latent

class GanSynthAE_2(nn.Module):
    ''' 
    Using 256X256 features
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 256):
        super(GanSynthAE_2, self).__init__()
        kw = feat_width//64
        enc_layers = []
        enc_layers += [nn.Conv2d(in_channels, 32, (1,1), 1)]
        enc_layers += net_utils.create_gansynth_block(32, 32, 'enc')
        enc_layers += net_utils.create_gansynth_block(32, 64, 'enc')
        enc_layers += net_utils.create_gansynth_block(64, 128, 'enc')
        enc_layers += net_utils.create_gansynth_block(128, 256, 'enc')
        enc_layers += net_utils.create_gansynth_block(256, 256, 'enc')
        enc_layers += net_utils.create_gansynth_block(256, 256, 'enc')
        enc_layers += [nn.Conv2d(256, latent_size, (kw,kw), 1)]
        self.enc_net = nn.Sequential(*enc_layers)
        self.enc_net_params = self.enc_net.parameters()

        dec_layers = []
        dec_layers += [nn.Conv2d(latent_size, 256, kernel_size=(kw,kw), stride=1, padding=(kw-1,kw-1), bias=False)]
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
        out = torch.tanh(self.dec_net(latent))
        
        return out, latent

class GanSynthAE_3(nn.Module):
    ''' 
    Using 1024X128 features
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128):
        super(GanSynthAE_3, self).__init__()
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

        self.linear = nn.Linear(latent_size, latent_size)
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

    def decode(self, latent):
        h = latent.squeeze(2).squeeze(2)
        h = self.linear(h).unsqueeze(2).unsqueeze(2)
        return self.dec_net(h)

    def forward(self, input):
        latent = self.encode(input)
        out = torch.tanh(self.decode(latent))
        
        return out, latent

class FC1(nn.Module):
    def __init__(self, input_size=1024, output_size=1024):
        super(FC1, self).__init__()
        net = [] 
        net += [nn.Linear(input_size,1024)]
        net += [nn.Tanh()]
        net += [nn.Linear(1024,1024)]
        net += [nn.Tanh()]
        net += [nn.Linear(1024,1024)]
        net += [nn.Tanh()]
        net += [nn.Linear(1024, output_size)]
        self.m2p_net = nn.Sequential(*net)
    
    def forward(self, input):
        return self.m2p_net(input)

class FC2(nn.Module):
    def __init__(self, input_size=1024, output_size=1024):
        super(FC2, self).__init__()
        net = [] 
        net += [nn.Linear(input_size,1024)]
        net += [nn.Tanh()]
        net += [nn.Linear(1024,1024)]
        net += [nn.Tanh()]
        net += [nn.Linear(1024,1024)]
        net += [nn.Tanh()]
        net += [nn.Linear(1024, output_size)]
        net += [nn.Tanh()]
        self.m2p_net = nn.Sequential(*net)
    
    def forward(self, input):
        return self.m2p_net(input)

class ResConv1(nn.Module):
    def __init__(self):
        super(ResConv1, self).__init__()
        self.conv1x1_start = nn.Conv1d(1,32,1)
        self.b1_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b1_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b2_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b2_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b3_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b3_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b4_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b4_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b5_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b5_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b6_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b6_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.conv1x1_end = nn.Conv1d(32,1,1)
    
    def forward(self, input):
        x = torch.unsqueeze(input, dim=1) 

        h_start = self.conv1x1_start(x)

        h_b1 = torch.tanh(self.b1_ly1(h_start))
        h_b1 = self.b1_ly2(h_b1) + h_start

        h_b2 = torch.tanh(self.b2_ly1(h_b1))
        h_b2 = self.b2_ly2(h_b2) + h_b1

        h_b3 = torch.tanh(self.b3_ly1(h_b2))
        h_b3 = self.b3_ly2(h_b3) + h_b2

        h_b4 = torch.tanh(self.b4_ly1(h_b3))
        h_b4 = self.b4_ly2(h_b4) + h_b3

        h_b5 = torch.tanh(self.b5_ly1(h_b4))
        h_b5 = self.b5_ly2(h_b5) + h_b4

        h_b6 = torch.tanh(self.b6_ly1(h_b5))
        h_b6 = self.b6_ly2(h_b6) + h_b5

        out = self.conv1x1_end(h_b6)

        return torch.squeeze(out, dim=1)

class DisConv1(nn.Module):
    def __init__(self, in_size):
        super(DisConv1, self).__init__()
        self.conv1x1_start = nn.Conv1d(1,32,1)
        self.b1_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b1_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b2_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b2_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b3_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b3_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b4_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b4_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b5_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b5_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.b6_ly1 = nn.Conv1d(32,32, 5,1,2)
        self.b6_ly2 = nn.Conv1d(32,32, 5,1,2)
        self.conv1x1_end = nn.Conv1d(32,1,1)

        self.linear_1 = nn.Linear(in_size,64)
        self.linear_2 = nn.Linear(64,1)
    
    def forward(self, input):
        x = torch.unsqueeze(input, dim=1) 

        h_start = self.conv1x1_start(x)

        h_b1 = F.leaky_relu(self.b1_ly1(h_start))
        h_b1 = self.b1_ly2(h_b1) + h_start

        h_b2 = F.leaky_relu(self.b2_ly1(h_b1))
        h_b2 = self.b2_ly2(h_b2) + h_b1

        h_b3 = F.leaky_relu(self.b3_ly1(h_b2))
        h_b3 = self.b3_ly2(h_b3) + h_b2

        h_b4 = F.leaky_relu(self.b4_ly1(h_b3))
        h_b4 = self.b4_ly2(h_b4) + h_b3

        h_b5 = F.leaky_relu(self.b5_ly1(h_b4))
        h_b5 = self.b5_ly2(h_b5) + h_b4

        h_b6 = F.leaky_relu(self.b6_ly1(h_b5))
        h_b6 = self.b6_ly2(h_b6) + h_b5

        h = torch.squeeze(self.conv1x1_end(h_b6), dim=1)

        out = F.leaky_relu(self.linear_1(h))
        return self.linear_2(out)

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
        self.linear = nn.Linear(2048, 1)

    def forward(self, x):
        h = self.net(x)
        out = self.linear(h.view(h.size(0), -1))
        return out

class SkipGanSynthAE_2(nn.Module):
    ''' 
    Using 256X256 features and used for Mag2Phase
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 256, skip_op = 'add'):
        super(SkipGanSynthAE_2, self).__init__()
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
        self.enc_8 = nn.Conv2d(256, latent_size, (kw,kw), 1)
        self.enc_net_params = list(self.enc_1.parameters()) + list(self.enc_2.parameters()) + \
                         list(self.enc_3.parameters()) + list(self.enc_4.parameters()) + \
                         list(self.enc_5.parameters()) + list(self.enc_6.parameters()) + \
                         list(self.enc_7.parameters()) + list(self.enc_8.parameters())

        self.dec_1 = nn.Conv2d(latent_size, 256, kernel_size=(kw,kw), stride=1, padding=(kw-1,(kw-1)), bias=False)
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
    
class GrowingGanSynthEnc_2(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, config = None):
        super(GrowingGanSynthEnc_2, self).__init__()
        kw = feat_width//64
        self.config = config # Updated overtime
        self.enc_net = nn.ModuleList([net_utils.ConvSN2D(in_channels, 32, (1,1), 1),
                                    net_utils.GANSynthBlock_2(32, 32, 'enc'),
                                    net_utils.GANSynthBlock_2(32, 64, 'enc'),
                                    net_utils.GANSynthBlock_2(64, 128, 'enc'),
                                    net_utils.GANSynthBlock_2(128, 256, 'enc'),
                                    net_utils.GANSynthBlock_2(256, 256, 'enc'),
                                    net_utils.GANSynthBlock_2(256, 256, 'enc'),
                                    net_utils.ConvSN2D(256, latent_size, (16,kw), 1)])
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
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, config = None, out_act=True):
        super(GrowingGanSynthDec, self).__init__()
        kw = feat_width//64
        self.config = config
        self.out_act = out_act
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
                    out = self.config['alpha'] * skip_rgb + (1 - self.config['alpha']) * out # alpha descreses from 1 to 0 
                elif i != len(self.dec_net)-1: # Stable and not the last layer.
                    out = to_rgb(out)
                break
        # Applying last activation is optional
        if self.out_act:
            return torch.tanh(out)
        else:
            return out

def instantiate_autoencoder(opt, device=None):
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
    elif opt['model']['ae'] == 'gansynth_3':
        return GanSynthAE_3(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'])
                        #feat_width= feat_width)
    elif opt['model']['ae'] == 'skipcatgansynth_v2':
        return SkipGanSynthAE_v2(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width,
                        skip_op='concat')
    elif opt['model']['ae'] == 'skippostcatgansynth':
        return SkipPostGanSynthAE(opt['model']['latent_size'],
                        device,
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
    elif opt['model']['ae'] == 'growgansynth2':
        return GrowingGanSynthEnc_2(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)

def instantiate_decoder(opt):
    ref_segm_size = int(list(segment_feature_ratio.keys())[0])
    feat_width = int((opt['data']['segment_size']/ref_segm_size) * segment_feature_ratio[ref_segm_size])
    if opt['model']['ae'] == 'growgansynth':
        return GrowingGanSynthDec(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width,
                        out_act= False)
    elif opt['model']['ae'] == 'growgansynth2':
        return GrowingGanSynthDec(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width,
                        out_act= True)

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

def instantiate_mag2phase(opt):
    """ Given the options file, it instantiate a magnitude to phase model.
    Parameters:s
        opt (Dictionary): Options.
    Returns:
        m2p instance.
    """ 
    if opt['model']['m2p'] == 'fc1':
        return FC1(opt['model']['in_size'],
                opt['model']['out_size'])
    if opt['model']['m2p'] == 'fc2':
        return FC2(opt['model']['in_size'],
                opt['model']['out_size'])
    elif opt['model']['m2p'] == 'rescv1':
        return ResConv1()
    elif opt['model']['m2p'] == 'gansynth':
        return GanSynthAE_2(512, 1, feat_width= 256)
    elif opt['model']['m2p'] == 'skipgansynth':
        return SkipGanSynthAE_2(512, 1, feat_width= 256, skip_op ='concat')
    else:
        raise NotImplementedError

def instantiate_m2p_disc(opt):
    """ Given the options file, it instantiate a magnitude to phase model.
    Parameters:s
        opt (Dictionary): Options.
    Returns:
        m2p instance.
    """ 
    if opt['model']['m2pdisc'] == 'discv1':
        return DisConv1(opt['model']['in_size'])
    elif opt['model']['m2pdisc'] == 'fc1':
        return FC1(opt['model']['in_size'],
                output_size=1)
    elif opt['model']['m2pdisc'] == 'gansynth':
        return GanSynthDisc(opt['model']['in_ch'])
    else:
        raise NotImplementedError

def instantiate_vocoder(opt):
    """ Given the options file, it instantiate a magnitude to phase model.
    Parameters:s
        opt (Dictionary): Options.
    Returns:
        m2p instance.
    """ 
    if opt['model']['voc'] == 'waveglow':
        return DisConv1(opt['model']['in_size'])
    else:
        raise NotImplementedError


