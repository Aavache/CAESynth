# External libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Internal libs
from . import network_utils as net_utils
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

        self.dec_1 = nn.Conv2d(latent_size, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)
        self.dec_2 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_3 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 256, 'dec'))
        self.dec_4 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 128, 'dec'))
        self.dec_5 = nn.Sequential(*net_utils.create_gansynth_block(128*dec_ch_scale, 64, 'dec'))
        self.dec_6 = nn.Sequential(*net_utils.create_gansynth_block(64*dec_ch_scale, 32, 'dec'))
        self.dec_7 = nn.Sequential(*net_utils.create_gansynth_block(32*dec_ch_scale, 32, 'dec'))
        self.dec_8 = nn.Conv2d(32*dec_ch_scale, in_channels, kernel_size=(1,1), stride=1)
        self.act = nn.Tanh()
    
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
        hd_8 = self.dec_8(net_utils.skip_connection(hd_7, skip[0], self.skip_op))
        out = self.act(hd_8)

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

        dec_layers = []
        dec_layers += [nn.Conv2d(latent_size, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)]
        dec_layers += net_utils.create_gansynth_block(256, 256, 'dec')
        dec_layers += net_utils.create_gansynth_block(256, 256, 'dec')
        dec_layers += net_utils.create_gansynth_block(256, 128, 'dec')
        dec_layers += net_utils.create_gansynth_block(128, 64, 'dec')
        dec_layers += net_utils.create_gansynth_block(64, 32, 'dec')
        dec_layers += net_utils.create_gansynth_block(32, 32, 'dec')
        dec_layers += [nn.Conv2d(32, in_channels, kernel_size=(1,1), stride=1)]
        dec_layers += [nn.Tanh()]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encoder(self, input):
        return self.enc_net(input)

    def decode(self, latent, cond=None):
        return self.dec_net(latent)

    def forward(self, input, cond=None):
        latent = self.enc_net(input)
        out = self.dec_net(latent)
        
        return out, latent

class GanSynthDec(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, out_channels = 2, feat_width=128):
        super(GanSynthDec, self).__init__()
        layers = []
        kw = feat_width//64
        layers.append(nn.Conv2d(latent_size, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False))
        layers += net_utils.create_gansynth_block(256, 256)
        layers += net_utils.create_gansynth_block(256, 256)
        layers += net_utils.create_gansynth_block(256, 256)
        layers += net_utils.create_gansynth_block(256, 128)
        layers += net_utils.create_gansynth_block(128, 64)
        layers += net_utils.create_gansynth_block(64, 32)
        layers.append(nn.Conv2d(32, out_channels, kernel_size=(1,1), stride=1))
        layers += [nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z_timbre, z_pitch=None):
        if z_pitch is not None:
            while (len(z_pitch.size()) != 4):
                z_pitch = z_pitch.unsqueeze(-1)
            z = torch.cat([z_timbre, z_pitch], dim = 1)
        else:
            z = z_timbre
        x_recon = self.net(z)

        return x_recon

class GanSynthEnc(nn.Module):
    ''' 
    '''
    def __init__(self, in_channels = 2, feat_width = 128, latent_pitch_size=None, latent_timbre_size=None, 
            pitch_enc = True, timbre_enc=True):
        super(GanSynthEnc, self).__init__()
        self.pitch_enc = pitch_enc
        self.timbre_enc = timbre_enc
        kw = feat_width//64
        layers = []
        layers += [nn.Conv2d(in_channels, 32, (1,1), 1)]
        layers += net_utils.create_gansynth_block(32, 32)
        layers += net_utils.create_gansynth_block(32, 64)
        layers += net_utils.create_gansynth_block(64, 128)
        layers += net_utils.create_gansynth_block(128, 256)
        layers += net_utils.create_gansynth_block(256, 256)
        layers += net_utils.create_gansynth_block(256, 256)
        self.net = nn.Sequential(*layers)

        if pitch_enc:
            if timbre_enc:
                self.timbre_ly = nn.Conv2d(128, latent_timbre_size, (16,kw), 1)
                self.pitch_class = nn.Linear(4096, latent_pitch_size)
            else:
                self.pitch_class = nn.Linear(8192, latent_pitch_size)

        else:
            self.timbre_ly = nn.Conv2d(256, latent_timbre_size, (16,kw), 1)

    def forward(self, x):
        latent_x = self.net(x)
        if self.pitch_enc:
            if self.timbre_enc:
                pitch_chunk, timbre_chunk = torch.chunk(latent_x, chunks = 2, dim=1)
                pitch = F.softmax(self.pitch_class(pitch_chunk.view(pitch_chunk.size(0), -1)))
                timbre = self.timbre_ly(timbre_chunk)
                return timbre, pitch
            else:
                pitch = F.softmax(self.pitch_class(latent_x.view(latent_x.size(0), -1)))
                return pitch
        else:
            timbre = self.timbre_ly(latent_x)
            return timbre

def instantiate_encoder(opt):
    """ Given the options file, it instantiate a encoder model

    Parameters:
        opt (Dictionary): Options.
    Returns:
        encoder instance.
    """ 
    ref_segm_size = int(list(segment_feature_ratio.keys())[0])
    feat_width = int((opt['data']['segment_size']/ref_segm_size) * segment_feature_ratio[ref_segm_size])
    if opt['model']['enc'] == 'gansynth_p':
        return GanSynthEnc(
                opt['model']['in_ch'],
                feat_width=feat_width,
                latent_pitch_size=opt['data']['pitch_size'],
                latent_timbre_size = opt['model']['timbre_latent_size'],
                pitch_enc=True)
    elif opt['model']['enc'] == 'gansynth':
        return GanSynthEnc(opt['model']['in_ch'],
                feat_width=feat_width,
                latent_timbre_size=opt['model']['timbre_latent_size'], 
                pitch_enc=False)
    else:
        raise NotImplementedError

def instantiate_decoder(opt):
    """ Given the options file, it instantiate a Decoder model.

    Parameters:s
        opt (Dictionary): Options.
    Returns:
        Decoder instance.
    """ 
    ref_segm_size = int(list(segment_feature_ratio.keys())[0])
    feat_width = int((opt['data']['segment_size']/ref_segm_size) * segment_feature_ratio[ref_segm_size])
    if opt['model']['dec'] == 'gansynth':
        return GanSynthDec(opt['model']['pitch_latent_size'] + opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)
    else:
        raise NotImplementedError

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
    elif opt['model']['ae'] == 'gansynth':
        return GanSynthAE(opt['model']['timbre_latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)
    else:
        raise NotImplementedError

def instantiate_classifier(opt):
    """ Given the options file, it instantiate a classifier model.

    Parameters:s
        opt (Dictionary): Options.
    Returns:
        classifier instance.
    """ 
    if opt['model']['class'] == 'gansynth':
        return GanSynthEnc(
                opt['model']['in_ch'],
                latent_pitch_size=opt['data']['pitch_size'],
                pitch_enc=True,
                timbre_enc=False)
    else:
        raise NotImplementedError

