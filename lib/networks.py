# External libs
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Internal libs
from . import network_utils as net_utils

class GrowNSynthEnc(nn.Module):
    ''' 
    '''
    def __init__(self, in_ch=1, latent_size=512, config=None):
        super(GrowNSynthEnc, self).__init__()

        self.config = config # Updated overtime
        self.enc_net = nn.ModuleList([nn.Conv2d(in_ch, 128, (1,1), 1), # 9 [1024,64]
                                    net_utils.NSynthBlock(128, 128, mode='enc'), # 8 [512,32]
                                    net_utils.NSynthBlock(128, 256, mode='enc'), # 7 [256,16]
                                    net_utils.NSynthBlock(256, 256, mode='enc'), # 6 [128,8]
                                    net_utils.NSynthBlock(256, 256, mode='enc'), # 5 [64,4]
                                    net_utils.NSynthBlock(256, 256, k_size=(4,2), stride=(2,2),pad=(1,0), mode='enc'), # 4 [32,2]
                                    net_utils.NSynthBlock(256, 512, k_size=(4,2), stride=(2,1),pad=(1,0), mode='enc'), # 3 [16,1]
                                    net_utils.NSynthBlock(512, 512, k_size=(4,1), stride=(2,1),pad=(1,0), mode='enc'), # 2 [8,1]
                                    net_utils.NSynthBlock(512, 512, k_size=(4,1), stride=(2,1),pad=(1,0), mode='enc'), # 1 [4,1]
                                    nn.Conv2d(512, latent_size, (4,1), 1)])

        self.from_rgb = nn.ModuleList([nn.Conv2d(in_ch, in_ch, 1), # Input channels
                                       nn.Conv2d(in_ch, 128, 1),
                                       nn.Conv2d(in_ch, 128, 1),
                                       nn.Conv2d(in_ch, 256, 1),
                                       nn.Conv2d(in_ch, 256, 1),
                                       nn.Conv2d(in_ch, 256, 1),
                                       nn.Conv2d(in_ch, 256, 1),
                                       nn.Conv2d(in_ch, 512, 1),
                                       nn.Conv2d(in_ch, 512, 1),
                                       nn.Conv2d(in_ch, 512, 1)])

    def forward(self, input):
        phase_idx = len(self.enc_net) - self.config['phase'] - 1
        for i, (conv, from_rgb) in enumerate(zip(self.enc_net, self.from_rgb)):
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

class GrowNSynthDec(nn.Module):
    def __init__(self, in_ch=1, latent_size=1024, config=None):
        super(GrowNSynthDec, self).__init__()
        self.config = config
        self.dec_net = nn.ModuleList([nn.ConvTranspose2d(latent_size, 1024, kernel_size=(4,1), stride=1),
                                    net_utils.NSynthBlock(1024, 512, k_size=(4,1), stride=(2,1), pad=(1,0), mode='dec'),
                                    net_utils.NSynthBlock(512, 512, k_size=(4,1), stride=(2,1), pad=(1,0), mode='dec'),
                                    net_utils.NSynthBlock(512, 256, k_size=(4,2), stride=(2,1),pad=(1,0), mode='dec'),
                                    net_utils.NSynthBlock(256, 256, k_size=(4,2), stride=(2,2),pad=(1,0), mode='dec'),
                                    net_utils.NSynthBlock(256, 256, mode='dec'),
                                    net_utils.NSynthBlock(256, 256, mode='dec'),
                                    net_utils.NSynthBlock(256, 128, mode='dec'),
                                    net_utils.NSynthBlock(128, 128, mode='dec'),
                                    nn.ConvTranspose2d(128, in_ch, kernel_size=(1,1), stride=1)])

        self.to_rgb = nn.ModuleList([nn.Conv2d(1024, in_ch, 1), # output channels
                                     nn.Conv2d(512, in_ch, 1),
                                     nn.Conv2d(512, in_ch, 1),
                                     nn.Conv2d(256, in_ch, 1),
                                     nn.Conv2d(256, in_ch, 1),
                                     nn.Conv2d(256, in_ch, 1),
                                     nn.Conv2d(256, in_ch, 1),
                                     nn.Conv2d(128, in_ch, 1),
                                     nn.Conv2d(128, in_ch, 1),
                                     nn.Conv2d(in_ch, in_ch, 1)])

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
        return out

class NSynthEnc(nn.Module):
    ''' 
    Classifier that operates with [1,1024,64]
    '''
    def __init__(self, in_ch = 1):
        super(NSynthEnc, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_ch, 128, (1,1), (1,1)),
                 nn.Conv2d(128, 128, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(128),
                 nn.Conv2d(128, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,1), (1,0)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (1,1))]
        self.net = nn.Sequential(*layers)
        self.c= nn.Conv2d(1, 1, (4,4), (2,2), (1,1))
        self.c2= nn.Conv2d(1, 1, (4,1), (2,1))
        self.c3= nn.Conv2d(1, 1, (5,5), (5,5))

    def forward(self, input):
        return self.net(input)

class NSynthDec(nn.Module):
    ''' 
    Classifier that operates with [1,1024,64]
    '''
    def __init__(self, in_ch = 1):
        super(NSynthDec, self).__init__()
        layers = []
        layers += [nn.ConvTranspose2d(1024, 1024, (4,1), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(1024),
                 nn.ConvTranspose2d(1024, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(512),
                 nn.ConvTranspose2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(512),
                 nn.ConvTranspose2d(512, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 128, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(0.2),
                 nn.BatchNorm2d(128),
                 nn.ConvTranspose2d(128, 128, (4,4), (2,2), (1,1)),
                 nn.ConvTranspose2d(128, in_ch, (1,1), (1,1))]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class DualNSynthAEWithPTClass(nn.Module):
    ''' 
    Autoncoder that operates with [B,1024,64]
    '''
    def __init__(self, in_ch = 1024, pitch_class=84, timbre_class=28):
        super(DualNSynthAEWithPTClass, self).__init__()
        
        self.enc_t = NSynthEnc(in_ch)
        self.enc_p = NSynthEnc(in_ch)
        self.dec = NSynthDec(in_ch)

        self.class_t = Classifier(512, timbre_class, 512)
        self.class_p = Classifier(512, pitch_class, 512)
    
    def encode(self, input):
        return self.enc_t(input), self.enc_p(input) #self.dual_ae.encode(input) # Returns a tuple with two embeddings

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_t(h_t), self.class_p(h_p)

class GrowNSynthAEWithPTClass(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, in_ch=1, pitch_class=84, timbre_class=28):
        super(GrowNSynthAEWithPTClass, self).__init__()
        self.enc_t = GrowNSynthEnc(in_ch)
        self.enc_p = GrowNSynthEnc(in_ch)
        self.dec = GrowNSynthDec(in_ch)

        self.class_t = Classifier(512, timbre_class, 512)
        self.class_p = Classifier(512, pitch_class, 512)
    
    def encode(self, input):
        return self.enc_t(input), self.enc_p(input)
    
    def update_config(self, config):
        self.enc_t.config = config
        self.enc_p.config = config
        self.dec.config = config

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_t(h_t), self.class_p(h_p)

class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=84, mid_size=512):
        super(Classifier, self).__init__()
        net_ly = [] 
        net_ly += [nn.Linear(input_size, mid_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(mid_size, mid_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(mid_size, mid_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(mid_size, output_size)]
        self.net = nn.Sequential(*net_ly)
    
    def forward(self, input):
        while len(input.shape)>2:
            input = input.squeeze(-1)
        return self.net(input)

def instantiate_net(opt):
    current_module = sys.modules[__name__]
    class_ = getattr(current_module, opt['name'])
    return class_(**opt['params'])