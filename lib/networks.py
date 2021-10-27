# External libs
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Internal libs
#from . import network_utils as net_utils

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

class GrowEncoder(nn.Module):
    ''' 
    '''
    def __init__(self, in_ch=1, latent_size=512, config=None):
        super(GrowEncoder, self).__init__()

        self.config = config # Updated overtime
        self.enc_net = nn.ModuleList([nn.Conv2d(in_ch, 128, (1,1), 1), # 9 [1024,64]
                                    NSynthBlock(128, 128, mode='enc'), # 8 [512,32]
                                    NSynthBlock(128, 256, mode='enc'), # 7 [256,16]
                                    NSynthBlock(256, 256, mode='enc'), # 6 [128,8]
                                    NSynthBlock(256, 256, mode='enc'), # 5 [64,4]
                                    NSynthBlock(256, 256, k_size=(4,2), stride=(2,2),pad=(1,0), mode='enc'), # 4 [32,2]
                                    NSynthBlock(256, 512, k_size=(4,2), stride=(2,1),pad=(1,0), mode='enc'), # 3 [16,1]
                                    NSynthBlock(512, 512, k_size=(4,1), stride=(2,1),pad=(1,0), mode='enc'), # 2 [8,1]
                                    NSynthBlock(512, 512, k_size=(4,1), stride=(2,1),pad=(1,0), mode='enc'), # 1 [4,1]
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

class GrowDecoder(nn.Module):
    def __init__(self, in_ch=1, latent_size=1024, config=None):
        super(GrowDecoder, self).__init__()
        self.config = config
        self.dec_net = nn.ModuleList([nn.ConvTranspose2d(latent_size, 1024, kernel_size=(4,1), stride=1),
                                    NSynthBlock(1024, 512, k_size=(4,1), stride=(2,1), pad=(1,0), mode='dec'),
                                    NSynthBlock(512, 512, k_size=(4,1), stride=(2,1), pad=(1,0), mode='dec'),
                                    NSynthBlock(512, 256, k_size=(4,2), stride=(2,1),pad=(1,0), mode='dec'),
                                    NSynthBlock(256, 256, k_size=(4,2), stride=(2,2),pad=(1,0), mode='dec'),
                                    NSynthBlock(256, 256, mode='dec'),
                                    NSynthBlock(256, 256, mode='dec'),
                                    NSynthBlock(256, 128, mode='dec'),
                                    NSynthBlock(128, 128, mode='dec'),
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

class MFEncoder(nn.Module):
    ''' 
     Input shape [1,1024,64]
    '''
    def __init__(self, in_ch = 1, latent_size=512, alpha=0.2):
        super(MFEncoder, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_ch, 128, (1,1), (1,1)),
                 nn.Conv2d(128, 128, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(128),
                 nn.Conv2d(128, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, latent_size, (4,1), (1,1))]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class MFEncoder2(nn.Module):
    ''' 
     Input shape [1,1024,64]
    '''
    def __init__(self, in_ch = 1, latent_size=512):
        super(MFEncoder2, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_ch, 128, (1,1), (1,1)),
                 nn.Conv2d(128, 128, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(128),
                 nn.Conv2d(128, 256, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,1), (1,0)),
                 nn.Tanh(),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 512, (4,1), (2,1), (1,0)),
                 nn.Tanh(),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.Tanh(),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.Tanh(),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, latent_size, (4,1), (1,1))]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class SFEncoder(nn.Module):
    ''' 
     Input shape [B, 1024]
    '''
    def __init__(self, in_size = 1024, mid_size = 512, latent_size=64):
        super(SFEncoder, self).__init__()
        layers = []
        layers = [nn.Linear(in_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, latent_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class MFDecoder2(nn.Module):
    ''' 
        Input tensor shape [1,1024,64]
    '''
    def __init__(self, in_ch = 1, latent_size=1024):
        super(MFDecoder2, self).__init__()
        layers = []
        layers += [nn.ConvTranspose2d(latent_size, 1024, (4,1), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(1024),
                 nn.ConvTranspose2d(1024, 512, (4,1), (2,1), (1,0)),
                 nn.Tanh(),
                 nn.BatchNorm2d(512),
                 nn.ConvTranspose2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.Tanh(),
                 nn.BatchNorm2d(512),
                 nn.ConvTranspose2d(512, 256, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 128, (4,4), (2,2), (1,1)),
                 nn.Tanh(),
                 nn.BatchNorm2d(128),
                 nn.ConvTranspose2d(128, 128, (4,4), (2,2), (1,1)),
                 nn.ConvTranspose2d(128, in_ch, (1,1), (1,1))]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class MFDecoder(nn.Module):
    ''' 
        Input tensor shape [1,1024,64]
    '''
    def __init__(self, in_ch = 1, latent_size=1024, alpha=0.2):
        super(MFDecoder, self).__init__()
        layers = []
        layers += [nn.ConvTranspose2d(latent_size, 1024, (4,1), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(1024),
                 nn.ConvTranspose2d(1024, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.ConvTranspose2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.ConvTranspose2d(512, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 128, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(128),
                 nn.ConvTranspose2d(128, 128, (4,4), (2,2), (1,1)),
                 nn.ConvTranspose2d(128, in_ch, (1,1), (1,1))]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class SFDecoder(nn.Module):
    ''' 
     Input shape [B, 1024]
    '''
    def __init__(self, in_size = 1024, mid_size = 512, latent_size=64):
        super(SFDecoder, self).__init__()
        layers = []
        layers = [nn.Linear(latent_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, in_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class MFCAESynth(nn.Module):
    ''' 
    NSynthAEWithTClass
    Autoncoder that operates with [B,1024,64]
    '''
    def __init__(self, in_ch = 1024, pitch_class=84, timbre_class=28):
        super(MFCAESynth, self).__init__()
        
        self.enc_t = MFEncoder(in_ch, 512)
        self.dec = MFDecoder(in_ch, pitch_class + 512)

        self.class_t = Classifier(512, timbre_class, 512)
    
    def encode(self, input):
        return self.enc_t(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input, pitch):
        h_t = self.encode(input)
        
        return self.decode(torch.cat([h_t, pitch], dim=1)), h_t, self.class_t(h_t)

class MFCAESynth2(nn.Module):
    ''' 
    NSynthAEWithTClass
    Autoncoder that operates with [B,1024,64]
    '''
    def __init__(self, in_ch = 1024, pitch_class=84, timbre_class=28):
        super(MFCAESynth2, self).__init__()
        
        self.enc_t = MFEncoder2(in_ch, 512)
        self.dec = MFDecoder2(in_ch, pitch_class + 512)

        self.class_t = Classifier(512, timbre_class, 512)
    
    def encode(self, input):
        return self.enc_t(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input, pitch):
        h_t = self.encode(input)
        return self.decode(torch.cat([h_t, pitch], dim=1)), h_t, self.class_t(h_t)

class SFCAESynth(nn.Module):
    ''' 
    DualFCAEUnified
        Single-Frame CAESynth network with FC layers and timbre classifier.
    '''
    def __init__(self, in_size = 1024, timbre_size=512, pitch_class=84, timbre_class=28, mid_size=512):
        super(SFCAESynth, self).__init__()

        self.enc = SFEncoder(in_size, mid_size, timbre_size)
        #self.fc = Classifier(timbre_size+pitch_class, timbre_size, mid_size)
        self.dec = SFDecoder(in_size, mid_size, timbre_size+pitch_class)
        self.timbre_class = Classifier(timbre_size, timbre_class, mid_size)
    
    def encode(self, input):
        in_data = input.squeeze(0).squeeze(0)
        in_data = torch.transpose(in_data, 0, 1)
        timbre_latent = self.enc(in_data)
        return timbre_latent

    def decode(self, latent):
        out = self.dec(latent)
        return torch.transpose(out, 0, 1).unsqueeze(0).unsqueeze(0)

    def forward(self, input, pitch):
        timbre_latent = self.encode(input)
        out = self.decode(torch.cat([timbre_latent, pitch],dim=1))
        return out, timbre_latent, self.timbre_class(timbre_latent)

class MFBaseline(nn.Module):
    ''' 
        Based on https://arxiv.org/pdf/1704.01279.pdf baseline which concatenates the pitch to the timbre latent code.
    '''
    def __init__(self, in_ch = 1024, pitch_class=84, timbre_class=28):
        super(MFBaseline, self).__init__()
        self.enc_t = MFEncoder(in_ch, 512)
        self.dec = MFDecoder(in_ch, pitch_class + 512)
    
    def encode(self, input):
        return self.enc_t(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input, pitch):
        h_t = self.encode(input)
        return self.decode(torch.cat([h_t, pitch], dim=1)), h_t, None

class SFBaseline(nn.Module):
    ''' 
        Based on https://arxiv.org/pdf/2001.11296.pdf which concatenates the pitch to the timbre latent code.
    '''
    def __init__(self, in_size = 1024, timbre_size=512, pitch_class=84, timbre_class=28, mid_size=512):
        super(SFBaseline, self).__init__()

        self.enc_t = SFEncoder(in_size + pitch_class, mid_size, timbre_size)
        self.dec = SFDecoder(in_size, mid_size, pitch_class + timbre_size)

    def encode(self, input, pitch):
        in_data = input.squeeze(0).squeeze(0)
        in_data = torch.transpose(in_data, 0, 1)
        in_data = torch.cat([in_data, pitch], dim=1)
        return self.enc_t(in_data)

    def decode(self, latent):
        out = self.dec(latent)
        return torch.transpose(out, 0, 1).unsqueeze(0).unsqueeze(0)

    def forward(self, input, pitch):
        h_t = self.encode(input, pitch)
        return self.decode(torch.cat([h_t, pitch], dim=1)), h_t, None

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
        #in_data = input.squeeze(0).squeeze(0)
        while len(input.shape)>2:
            input = input.squeeze(-1)
        return self.net(input)

def instantiate_net(opt):
    current_module = sys.modules[__name__]
    class_ = getattr(current_module, opt['name'])
    return class_(**opt['params'])