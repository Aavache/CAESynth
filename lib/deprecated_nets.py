# External libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Internal libs
from . import network_utils as net_utils


class CasAE(nn.Module):
    def __init__(self, latent_size, in_channels = 2, feat_width = 128):
        super(CasAE, self).__init__()
        kw = feat_width//64
        timbre_latent_size = latent_size//2
        pitch_latent_size = timbre_latent_size
        
        # Timbre encoder
        t_enc_layers = []
        t_enc_layers += [nn.Conv2d(in_channels, 32, (1,1), 1)]
        t_enc_layers += net_utils.create_gansynth_block(32, 32, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(32, 64, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(64, 128, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(128, 256, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(256, 256, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(256, 256, 'enc')
        t_enc_layers += [nn.Conv2d(256, timbre_latent_size, (16,kw), 1)]
        self.t_enc_net = nn.Sequential(*t_enc_layers)
        self.t_enc_net_params = self.t_enc_net.parameters()

        # Pitch encoder
        self.p_enc_1 = nn.Conv2d(in_channels, 32, (1,1), 1)
        self.p_1x1_1 = nn.Conv2d(timbre_latent_size, 32, (1,1), 1)
        self.p_enc_2 = net_utils.GANSynthBlock(32, 32, 'enc')
        self.p_1x1_2 = nn.Conv2d(timbre_latent_size, 32, (1,1), 1)
        self.p_enc_3 = net_utils.GANSynthBlock(32, 64, 'enc')
        self.p_1x1_3 = nn.Conv2d(timbre_latent_size, 64, (1,1), 1)
        self.p_enc_4 = net_utils.GANSynthBlock(64, 128, 'enc')
        self.p_1x1_4 = nn.Conv2d(timbre_latent_size, 128, (1,1), 1)
        self.p_enc_5 = net_utils.GANSynthBlock(128, 256, 'enc')
        self.p_1x1_5 = nn.Conv2d(timbre_latent_size, 256, (1,1), 1)
        self.p_enc_6 = net_utils.GANSynthBlock(256, 256, 'enc')
        self.p_1x1_6 = nn.Conv2d(timbre_latent_size, 256, (1,1), 1)
        self.p_enc_7 = net_utils.GANSynthBlock(256, 256, 'enc')
        self.p_1x1_7 = nn.Conv2d(timbre_latent_size, 256, (1,1), 1)
        self.p_enc_8 = nn.Conv2d(256, pitch_latent_size, (16,kw), 1)
        self.p_enc_net_params = list(self.p_1x1_1.parameters()) + list(self.p_enc_1.parameters()) + \
                    list(self.p_1x1_2.parameters()) + list(self.p_enc_2.parameters()) + \
                    list(self.p_1x1_3.parameters()) + list(self.p_enc_3.parameters()) + \
                    list(self.p_1x1_4.parameters()) + list(self.p_enc_4.parameters()) + \
                    list(self.p_1x1_5.parameters()) + list(self.p_enc_5.parameters()) + \
                    list(self.p_1x1_6.parameters()) + list(self.p_enc_6.parameters()) + \
                    list(self.p_1x1_7.parameters()) + list(self.p_enc_7.parameters()) + \
                    list(self.p_enc_8.parameters())

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
        h_timbre = self.t_enc_net(input)

        h_timbre_cond = h_timbre.detach()
        h_p = self.p_enc_1(input) 
        h_p = h_p + self.p_1x1_1(h_timbre_cond).expand(*h_p.size())
        h_p = self.p_enc_2(h_p)
        h_p = h_p + self.p_1x1_2(h_timbre_cond).expand(*h_p.size())
        h_p = self.p_enc_3(h_p)
        h_p = h_p + self.p_1x1_3(h_timbre_cond).expand(*h_p.size())
        h_p = self.p_enc_4(h_p)
        h_p = h_p + self.p_1x1_4(h_timbre_cond).expand(*h_p.size())
        h_p = self.p_enc_5(h_p)
        h_p = h_p + self.p_1x1_5(h_timbre_cond).expand(*h_p.size())
        h_p = self.p_enc_6(h_p)
        h_p = h_p + self.p_1x1_6(h_timbre_cond).expand(*h_p.size())
        h_p = self.p_enc_7(h_p)
        h_p = h_p + self.p_1x1_7(h_timbre_cond).expand(*h_p.size())
        h_pitch = self.p_enc_8(h_p)

        return h_timbre, h_pitch

    def decode(self, latent):
        return self.dec_net(latent)

    def forward(self, input):
        h_timbre, h_pitch = self.encode(input)
        latent = torch.cat([h_timbre, h_pitch], dim=1)        
        out = self.decode(latent)

        return out, h_timbre, h_pitch


class GMVAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 256, feat_width = 64, mid_ch = 512,
                        pitch_size= 83, timbre_size=5, is_train=True, device=None):
        super(GMVAE, self).__init__()
        self.mid_ch = mid_ch
        self.flat_size = mid_ch*feat_width
        self.is_train = is_train
        self.pitch_size = pitch_size
        self.timbre_size = timbre_size
        self.device = device

        self.pitch_mu_emb = net_utils.build_mu_emb([pitch_size, latent_size])
        self.pitch_logvar_emb = net_utils.build_mu_emb([pitch_size, latent_size])
        self.timbre_mu_emb = net_utils.build_mu_emb([timbre_size, latent_size])
        self.timbre_logvar_emb = net_utils.build_mu_emb([timbre_size, latent_size])

        enc_p_layers = []
        enc_p_layers += [nn.Conv1d(in_channels, mid_ch, 1, 1)]
        enc_p_layers += [nn.ReLU()]
        enc_p_layers += [nn.BatchNorm1d(mid_ch)]
        enc_p_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        enc_p_layers += [nn.ReLU()]
        enc_p_layers += [nn.BatchNorm1d(mid_ch)]
        enc_p_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        enc_p_layers += [nn.ReLU()]
        enc_p_layers += [nn.BatchNorm1d(mid_ch)]
        enc_p_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        self.enc_p_net = nn.Sequential(*enc_p_layers)
        self.enc_p_ln = nn.Linear(self.flat_size, 512)
        self.enc_mu_p_ln = nn.Linear(512, latent_size)
        self.enc_logvar_p_ln = nn.Linear(512, latent_size)

        enc_t_layers = []
        enc_t_layers += [nn.Conv1d(in_channels, mid_ch, 1, 1)]
        enc_t_layers += [nn.ReLU()]
        enc_t_layers += [nn.BatchNorm1d(mid_ch)]
        enc_t_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        enc_t_layers += [nn.ReLU()]
        enc_t_layers += [nn.BatchNorm1d(mid_ch)]
        enc_t_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        enc_t_layers += [nn.ReLU()]
        enc_t_layers += [nn.BatchNorm1d(mid_ch)]
        enc_t_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        self.enc_t_net = nn.Sequential(*enc_t_layers)
        self.enc_t_ln = nn.Linear(self.flat_size, 512)
        self.enc_mu_t_ln = nn.Linear(512, latent_size)
        self.enc_logvar_t_ln = nn.Linear(512, latent_size)

        self.dec_ln = nn.Linear(2*latent_size, self.flat_size)
        dec_layers = []
        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        dec_layers += [nn.ReLU()]
        dec_layers += [nn.BatchNorm1d(mid_ch)]
        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        dec_layers += [nn.ReLU()]
        dec_layers += [nn.BatchNorm1d(mid_ch)]
        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        dec_layers += [nn.ReLU()]
        dec_layers += [nn.BatchNorm1d(mid_ch)]
        dec_layers += [nn.Conv1d(mid_ch, in_channels, 1, 1)]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encode(self, input):
        # The input has 4 axis
        input = input.squeeze(1)

        # Encoding Timbre
        h_t = self.enc_t_net(input)
        h_t = h_t.view(h_t.size(0),-1)
        h_t = torch.relu(self.enc_t_ln(h_t))
        mu_t = self.enc_mu_t_ln(h_t)
        logvar_t = self.enc_logvar_t_ln(h_t)
        z_t = net_utils.reparameterize(mu_t, logvar_t)
        log_q_y_logit, q_y = net_utils.infer_class(z_t, self.timbre_mu_emb, self.timbre_logvar_emb, self.timbre_size, self.device)

        # Encoding Pitch
        h_p = self.enc_p_net(input)
        h_p = h_p.view(h_p.size(0),-1)
        h_p = torch.relu(self.enc_p_ln(h_p))
        mu_p = self.enc_mu_p_ln(h_p)
        logvar_p = self.enc_logvar_p_ln(h_p)
        z_p = net_utils.reparameterize(mu_p, logvar_p)

        return z_t, mu_t, logvar_t, log_q_y_logit, q_y, z_p, mu_p, logvar_p

    def decode(self, latent):
        h = torch.relu(self.dec_ln(latent))
        h = h.unsqueeze(2)
        h = h.view(h.size(0), self.mid_ch, -1)
        return self.dec_net(h)
    
    def infer(self, input):
        # The input has 4 axis
        input = input.squeeze(1)

        # Encoding Timbre
        h_t = self.enc_t_net(input)
        h_t = h_t.view(h_t.size(0),-1)
        h_t = torch.relu(self.enc_t_ln(h_t))
        mu_t = self.enc_mu_t_ln(h_t)        

        # Encoding Pitch
        h_p = self.enc_p_net(input)
        h_p = h_p.view(h_p.size(0),-1)
        h_p = torch.relu(self.enc_p_ln(h_p))
        mu_p = self.enc_mu_p_ln(h_p)

        recon = self.decode(torch.cat([mu_t, mu_p], dim=1))

        return recon, mu_t, mu_p

    def forward(self, input):
        z_t, mu_t, logvar_t, log_q_y_logit, q_y, z_p, mu_p, logvar_p = self.encode(input)
        recon = self.decode(torch.cat([z_t, z_p], dim=1))
        return recon, z_t, mu_t, logvar_t, log_q_y_logit, q_y, z_p, mu_p, logvar_p

class VAEGanSynthAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128):
        super(VAEGanSynthAE, self).__init__()
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
        self.mu_ln = nn.Linear(latent_size, latent_size)
        self.log_var_ln = nn.Linear(latent_size, latent_size)

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
        h = self.enc_net(input)
        h = h.squeeze(2).squeeze(2)
        mu = self.mu_ln(h)
        log_var = self.log_var_ln(h)
        return mu, log_var

    def decode(self, latent, cond=None):
        return self.dec_net(latent)
    
    def forward(self, input, cond=None):
        mu, log_var = self.encode(input)
        z = net_utils.reparameterize(mu, log_var)
        z = z.unsqueeze(2).unsqueeze(2)
        out = self.decode(z)
        return out, z, mu, log_var

class LightVAEGanSynthAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, mid_ch=64):
        super(LightVAEGanSynthAE, self).__init__()
        kw = feat_width//64
        enc_layers = []
        enc_layers += [nn.Conv2d(in_channels, mid_ch, (1,1), 1)]
        enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        enc_layers += [nn.Conv2d(mid_ch, mid_ch, (16,kw), 1)]
        enc_layers += [nn.Conv2d(mid_ch, latent_size, (1,1), 1)]
        self.enc_net = nn.Sequential(*enc_layers)
        self.mu_ln = nn.Linear(latent_size, latent_size)
        self.log_var_ln = nn.Linear(latent_size, latent_size)

        dec_layers = []
        dec_layers += [nn.Conv2d(latent_size, mid_ch, (1,1), 1)]
        dec_layers += [nn.Conv2d(mid_ch, mid_ch, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)]
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += [nn.Conv2d(mid_ch, in_channels, kernel_size=(1,1), stride=1)]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encode(self, input):
        h = self.enc_net(input)
        h = h.squeeze(2).squeeze(2)
        mu = self.mu_ln(h)
        log_var = self.log_var_ln(h)
        return mu, log_var

    def decode(self, latent, cond=None):
        return self.dec_net(latent)

    def forward(self, input, cond=None):
        mu, log_var = self.encode(input)
        z = net_utils.reparameterize(mu, log_var)
        z = z.unsqueeze(2).unsqueeze(2)
        out = self.decode(z)
        return out, z, mu, log_var
    
class VAE1D(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, mid_ch = 64):
        super(VAE1D, self).__init__()
        self.mid_ch = mid_ch
        self.flat_size = mid_ch*feat_width
        enc_layers = []
        enc_layers += [nn.Conv1d(in_channels, mid_ch, 1, 1)]
        # enc_layers += [nn.ReLU()]
        enc_layers += [nn.BatchNorm1d(mid_ch)]
        enc_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        # enc_layers += [nn.ReLU()]
        enc_layers += [nn.BatchNorm1d(mid_ch)]
        enc_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        # enc_layers += [nn.ReLU()]
        enc_layers += [nn.BatchNorm1d(mid_ch)]
        enc_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        self.enc_net = nn.Sequential(*enc_layers)
        self.enc_ln = nn.Linear(self.flat_size, latent_size)

        self.mu_ln = nn.Linear(latent_size, latent_size)
        self.log_var_ln = nn.Linear(latent_size, latent_size)

        self.dec_ln = nn.Linear(latent_size, self.flat_size)
        dec_layers = []
        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        # dec_layers += [nn.ReLU()]
        dec_layers += [nn.BatchNorm1d(mid_ch)]
        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        # dec_layers += [nn.ReLU()]
        dec_layers += [nn.BatchNorm1d(mid_ch)]
        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, 1)]
        # dec_layers += [nn.ReLU()]
        dec_layers += [nn.BatchNorm1d(mid_ch)]
        dec_layers += [nn.Conv1d(mid_ch, in_channels, 1, 1)]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encode(self, input):
        # The input has 4 axis
        input = input.squeeze(1)

        h = self.enc_net(input)
        h = h.view(h.size(0),-1)
        h = torch.relu(self.enc_ln(h))
        mu = self.mu_ln(h)
        log_var = self.log_var_ln(h)
        return mu, log_var

    def decode(self, latent):
        h = torch.relu(self.dec_ln(latent))
        h = h.unsqueeze(2)
        h = h.view(h.size(0), self.mid_ch, -1)
        return self.dec_net(h)

    def forward(self, input, cond=None):
        in_shape = input.size()
        mu, log_var = self.encode(input)
        z = net_utils.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, z, mu, log_var

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


class ShallowGanSynthAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128):
        super(ShallowGanSynthAE, self).__init__()
        kw = feat_width//8
        enc_layers = []
        enc_layers += [nn.Conv2d(in_channels, 32, (1,1), 1)]
        enc_layers += net_utils.create_gansynth_block(32, 64, 'enc')
        enc_layers += net_utils.create_gansynth_block(64, 128, 'enc')
        enc_layers += net_utils.create_gansynth_block(128, 256, 'enc')
        enc_layers += [nn.Conv2d(256, latent_size, (128,kw), 1)]  
        self.enc_net = nn.Sequential(*enc_layers)
        self.enc_net_params = self.enc_net.parameters()

        dec_layers = []
        dec_layers += [nn.Conv2d(latent_size, 256, kernel_size=(128,kw), stride=1, padding=(127,(kw-1)), bias=False)]
        dec_layers += net_utils.create_gansynth_block(256, 128, 'dec')
        dec_layers += net_utils.create_gansynth_block(128, 64, 'dec')
        dec_layers += net_utils.create_gansynth_block(64, 32, 'dec')
        dec_layers += [nn.Conv2d(32, in_channels, kernel_size=(1,1), stride=1)]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encode(self, input):
        latent = self.enc_net(input)
        return latent

    def decode(self, latent, cond=None):
        out = self.dec_net(latent)
        return out

    def forward(self, input, cond=None):
        latent = self.encode(input)
        out = self.decode(latent)

        return out, latent

class SkipShallowGanSynthAE(nn.Module):
    ''' 
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128, skip_op = 'add'):
        super(SkipShallowGanSynthAE, self).__init__()
        kw = feat_width//8
        self.skip_op = skip_op     
        dec_ch_scale = 2 if self.skip_op == 'concat' else 1
        self.enc_1 = nn.Conv2d(in_channels, 32, (1,1), 1)
        self.enc_2 = nn.Sequential(*net_utils.create_gansynth_block(32, 64, 'enc'))
        self.enc_3 = nn.Sequential(*net_utils.create_gansynth_block(64, 128, 'enc'))
        self.enc_4 = nn.Sequential(*net_utils.create_gansynth_block(128, 256, 'enc'))
        self.enc_5 = nn.Conv2d(256, latent_size, (128,kw), 1)
        self.enc_net_params = list(self.enc_1.parameters()) + list(self.enc_2.parameters()) + \
                         list(self.enc_3.parameters()) + list(self.enc_4.parameters()) + \
                         list(self.enc_5.parameters())

        self.dec_1 = nn.Conv2d(latent_size, 256, kernel_size=(128,kw), stride=1, padding=(127,(kw//2)), bias=False)
        self.dec_2 = nn.Sequential(*net_utils.create_gansynth_block(256*dec_ch_scale, 128, 'dec'))
        self.dec_3 = nn.Sequential(*net_utils.create_gansynth_block(128*dec_ch_scale, 64, 'dec'))
        self.dec_4 = nn.Sequential(*net_utils.create_gansynth_block(64*dec_ch_scale, 32, 'dec'))
        self.dec_5 = nn.Conv2d(32*dec_ch_scale, in_channels, kernel_size=(1,1), stride=1)
    
    def encode(self, input):
        he_1 = self.enc_1(input)
        he_2 = self.enc_2(he_1)
        he_3 = self.enc_3(he_2)
        he_4 = self.enc_4(he_3)
        latent = self.enc_5(he_4)
        return latent, [he_1,he_2,he_3,he_4]

    def decode(self, latent, skip, cond=None):
        hd_1 = self.dec_1(latent)
        hd_2 = self.dec_2(net_utils.skip_connection(hd_1, skip[3], self.skip_op))
        hd_3 = self.dec_3(net_utils.skip_connection(hd_2, skip[2], self.skip_op))
        hd_4 = self.dec_4(net_utils.skip_connection(hd_3, skip[1], self.skip_op))
        out = self.dec_5(net_utils.skip_connection(hd_4, skip[0], self.skip_op))

        return out

    def forward(self, input, cond=None):
        latent, skip = self.encode(input)
        out = self.decode(latent, skip)

        return out, latent

class ConditionalSmallGanSynthClass(nn.Module):
    ''' 
    '''
    def __init__(self, cond_size, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(ConditionalSmallGanSynthClass, self).__init__()
        feat_size = (16,4) 

        self.enc_1 = nn.Conv2d(in_ch + cond_size, mid_ch, (1,1), 1)
        self.enc_2 = net_utils.GANSynthBlock(mid_ch + cond_size, mid_ch, 'enc') # 128x32
        self.enc_3 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 64x16
        self.enc_4 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 32x8
        self.enc_5 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 16x4

        self.ln = nn.Linear(mid_ch * feat_size[0]* feat_size[1], latent_size)
        self.classifier_ln = Classifier(latent_size, out_size)

    def forward(self, x, cond):

        cond_exp = cond.unsqueeze(2).unsqueeze(2)
        shape = (cond_exp.size(0), cond_exp.size(1), x.size(2), x.size(3))
        h = self.enc_1(torch.cat([x, cond_exp.expand(*shape)], dim=1))
        shape = (cond_exp.size(0), cond_exp.size(1), h.size(2), h.size(3))
        h = self.enc_2(torch.cat([h, cond_exp.expand(*shape)], dim=1))
        h = self.enc_3(h)
        h = self.enc_4(h)
        h = self.enc_5(h)

        out = self.ln(h.view(h.size(0), -1))
        out = self.classifier_ln(out)
        return out
        
class InputCondSmallGanSynthClass(nn.Module):
    ''' 
    '''
    def __init__(self, cond_size, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(InputCondSmallGanSynthClass, self).__init__()
        feat_size = (16,4) 

        self.enc_1 = nn.Conv2d(in_ch + cond_size, mid_ch, (1,1), 1)
        self.enc_2 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 128x32
        self.enc_3 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 64x16
        self.enc_4 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 32x8
        self.enc_5 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 16x4

        self.ln = nn.Linear(mid_ch * feat_size[0]* feat_size[1], latent_size)
        self.classifier_ln = Classifier(latent_size, out_size)

    def forward(self, x, cond):

        cond_exp = cond.unsqueeze(2).unsqueeze(2)
        shape = (cond_exp.size(0), cond_exp.size(1), x.size(2), x.size(3))
        h = self.enc_1(torch.cat([x, cond_exp.expand(*shape)], dim=1))
        h = self.enc_2(h)
        h = self.enc_3(h)
        h = self.enc_4(h)
        h = self.enc_5(h)

        out = self.ln(h.view(h.size(0), -1))
        out = self.classifier_ln(out)
        return out

class LatentCondSmallGanSynthClass(nn.Module):
    ''' 
    '''
    def __init__(self, cond_size, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(LatentCondSmallGanSynthClass, self).__init__()
        feat_size = (16,4) 

        self.enc_1 = nn.Conv2d(in_ch, mid_ch, (1,1), 1)
        self.enc_2 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 128x32
        self.enc_3 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 64x16
        self.enc_4 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 32x8
        self.enc_5 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 16x4

        self.ln = nn.Linear(mid_ch * feat_size[0]* feat_size[1], latent_size)
        self.classifier_ln = Classifier(latent_size + cond_size, out_size)

    def forward(self, x, cond):
        h = self.enc_1(x)
        h = self.enc_2(h)
        h = self.enc_3(h)
        h = self.enc_4(h)
        h = self.enc_5(h)

        out = self.ln(h.view(h.size(0), -1))
        out = self.classifier_ln(torch.cat([out,cond],dim=1))
        return out

class DualSGanSynthAEWithPClass(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_channels = 2, mid_ch=64, pitch_class=84):
        super(DualSGanSynthAEWithPClass, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        enc_t_layers = []
        enc_t_layers += [nn.Conv2d(in_channels, mid_ch, (1,1), 1)]
        enc_t_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        enc_t_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        enc_t_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        enc_t_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.enc_t_net = nn.Sequential(*enc_t_layers)
        self.enc_t_ln = nn.Linear(mid_ch * self.feat_size[0]* self.feat_size[1], timbre_size)

        enc_p_layers = []
        enc_p_layers += [nn.Conv2d(in_channels, mid_ch, (1,1), 1)]
        enc_p_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        enc_p_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        enc_p_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        enc_p_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.enc_p_net = nn.Sequential(*enc_p_layers)
        self.enc_p_ln = nn.Linear(mid_ch * self.feat_size[0]* self.feat_size[1], pitch_size)

        self.class_p_net = Classifier(pitch_size, pitch_class,512)

        self.dec_ln = nn.Linear(pitch_size + timbre_size, mid_ch * self.feat_size[0]* self.feat_size[1])
        dec_layers = []
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += [nn.Conv2d(mid_ch, in_channels, kernel_size=(1,1), stride=1)]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encode(self, input):
        batch_size = input.size(0)

        h_t = self.enc_t_net(input)
        h_t = h_t.view([batch_size, -1]) 
        h_t = self.enc_t_ln(h_t)

        h_p = self.enc_p_net(input)
        h_p = h_p.view([batch_size, -1]) 
        h_p = self.enc_p_ln(h_p) 

        return h_t, h_p

    def decode(self, latent):
        latent = self.dec_ln(latent)
        latent = latent.view([latent.size(0), self.mid_ch, self.feat_size[0], self.feat_size[1]])
        return self.dec_net(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_p_net(h_p)

class SmallGMVAESynth(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, latent_size, in_channels = 2, mid_ch=64,
                pitch_size= 83, timbre_size=5, is_train=True, device=None):
        super(SmallGMVAESynth, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        self.pitch_size = pitch_size
        self.timbre_size = timbre_size
        self.device = device

        self.pitch_mu_emb = net_utils.build_mu_emb([pitch_size, latent_size])
        self.pitch_logvar_emb = net_utils.build_mu_emb([pitch_size, latent_size])
        self.timbre_mu_emb = net_utils.build_mu_emb([timbre_size, latent_size])
        self.timbre_logvar_emb = net_utils.build_mu_emb([timbre_size, latent_size])

        enc_p_layers = []
        enc_p_layers += [nn.Conv2d(in_channels, mid_ch, (1,1), 1)]
        enc_p_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        enc_p_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        enc_p_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        enc_p_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.enc_p_net = nn.Sequential(*enc_p_layers)
        self.enc_p_ln = nn.Linear(mid_ch * self.feat_size[0]* self.feat_size[1], latent_size)
        self.enc_mu_p_ln = nn.Linear(latent_size, latent_size)
        self.enc_logvar_p_ln = nn.Linear(latent_size, latent_size)

        enc_t_layers = []
        enc_t_layers += [nn.Conv2d(in_channels, mid_ch, (1,1), 1)]
        enc_t_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        enc_t_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        enc_t_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        enc_t_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.enc_t_net = nn.Sequential(*enc_t_layers)
        self.enc_t_ln = nn.Linear(mid_ch * self.feat_size[0]* self.feat_size[1], latent_size)
        self.enc_mu_t_ln = nn.Linear(latent_size, latent_size)
        self.enc_logvar_t_ln = nn.Linear(latent_size, latent_size)

        self.dec_ln = nn.Linear(latent_size*2, mid_ch * self.feat_size[0]* self.feat_size[1])
        dec_layers = []
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += [nn.Conv2d(mid_ch, in_channels, kernel_size=(1,1), stride=1)]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encode(self, input):
        # Encoding Timbre
        h_t = self.enc_t_net(input)
        h_t = h_t.view(h_t.size(0),-1)
        h_t = torch.relu(self.enc_t_ln(h_t))
        mu_t = self.enc_mu_t_ln(h_t)
        logvar_t = self.enc_logvar_t_ln(h_t)
        z_t = net_utils.reparameterize(mu_t, logvar_t)
        log_q_y_logit, q_y = net_utils.infer_class(z_t, self.timbre_mu_emb, self.timbre_logvar_emb, self.timbre_size, self.device)

        # Encoding Pitch
        h_p = self.enc_p_net(input)
        h_p = h_p.view(h_p.size(0),-1)
        h_p = torch.relu(self.enc_p_ln(h_p))
        mu_p = self.enc_mu_p_ln(h_p)
        logvar_p = self.enc_logvar_p_ln(h_p)
        z_p = net_utils.reparameterize(mu_p, logvar_p)

        return z_t, mu_t, logvar_t, log_q_y_logit, q_y, z_p, mu_p, logvar_p

    def infer(self, input):
        # Encoding Timbre
        h_t = self.enc_t_net(input)
        h_t = h_t.view(h_t.size(0),-1)
        h_t = torch.relu(self.enc_t_ln(h_t))
        mu_t = self.enc_mu_t_ln(h_t)        

        # Encoding Pitch
        h_p = self.enc_p_net(input)
        h_p = h_p.view(h_p.size(0),-1)
        h_p = torch.relu(self.enc_p_ln(h_p))
        mu_p = self.enc_mu_p_ln(h_p)
        
        recon = self.decode(torch.cat([mu_t, mu_p], dim=1))

        return recon, mu_t, mu_p

    def decode(self, latent):
        latent = self.dec_ln(latent)
        latent = latent.view([latent.size(0), self.mid_ch, self.feat_size[0], self.feat_size[1]])
        return self.dec_net(latent)

    def forward(self, input):
        z_t, mu_t, logvar_t, log_q_y_logit, q_y, z_p, mu_p, logvar_p = self.encode(input)
        recon = self.decode(torch.cat([z_t, z_p], dim=1))
        return recon, z_t, mu_t, logvar_t, log_q_y_logit, q_y, z_p, mu_p, logvar_p


# TODO: to erase
class SmallGanSynthClassTMP(nn.Module):
    ''' 
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(SmallGanSynthClassTMP, self).__init__()
        feat_size = (16,4) 
        layers = []
        layers += [nn.Conv2d(in_ch, mid_ch, (1,1), 1)]
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.net = nn.Sequential(*layers)
        self.ln = nn.Linear(mid_ch * feat_size[0]* feat_size[1], latent_size)
        self.classifier_ln = Classifier(latent_size, out_size)

    def forward(self, x):
        h = self.net(x)
        latent = self.ln(h.view(h.size(0), -1))
        class_ = self.classifier_ln(latent)
        return class_, latent

class SmallGanSynthMultiClass(nn.Module):
    ''' 
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(SmallGanSynthMultiClass, self).__init__()
        feat_size = (16,4) 
        layers = []
        layers += [nn.Conv2d(in_ch, mid_ch, (1,1), 1)]
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.net = nn.Sequential(*layers)
        self.ln = nn.Linear(mid_ch * feat_size[0]* feat_size[1], latent_size)

        self.classifier_ln20 = Classifier(latent_size, out_size) # Pitch[20-29]
        self.classifier_ln30 = Classifier(latent_size, out_size) # Pitch[30-39]
        self.classifier_ln40 = Classifier(latent_size, out_size) # Pitch[40-49]
        self.classifier_ln50 = Classifier(latent_size, out_size) # Pitch[50-59]
        self.classifier_ln60 = Classifier(latent_size, out_size) # Pitch[60-69]
        self.classifier_ln70 = Classifier(latent_size, out_size) # Pitch[70-82]
    
    def encode(self, x, pitch=None):
        h = self.net(x)
        return self.ln(h.view(h.size(0), -1))       

    def forward(self, x, pitch):
        latent = self.encode(x, None)
        
        if pitch.data < 30: 
            class_ = self.classifier_ln20(latent)
        elif pitch.data >=30 and pitch.data < 40: 
            class_ = self.classifier_ln30(latent)
        elif pitch.data >=40 and pitch.data < 50: 
            class_ = self.classifier_ln40(latent)
        elif pitch.data >=50 and pitch.data < 60: 
            class_ = self.classifier_ln50(latent)
        elif pitch.data >=60 and pitch.data < 70: 
            class_ = self.classifier_ln60(latent)
        else: # pitch.data >=70:
            class_ = self.classifier_ln70(latent)

        return class_, latent

class MultiSmallGanSynthMultiClass(nn.Module):
    ''' 
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(MultiSmallGanSynthMultiClass, self).__init__()
        self.enc20 = SmallGanSynthClass2(in_ch = 1, mid_ch = 64, out_size = 64) # Pitch[20-29]
        self.enc30 = SmallGanSynthClass2(in_ch = 1, mid_ch = 64, out_size = 64) # Pitch[30-39]
        self.enc40 = SmallGanSynthClass2(in_ch = 1, mid_ch = 64, out_size = 64) # Pitch[40-49]
        self.enc50 = SmallGanSynthClass2(in_ch = 1, mid_ch = 64, out_size = 64) # Pitch[50-59]
        self.enc60 = SmallGanSynthClass2(in_ch = 1, mid_ch = 64, out_size = 64) # Pitch[60-69]
        self.enc70 = SmallGanSynthClass2(in_ch = 1, mid_ch = 64, out_size = 64) # Pitch[70-82]

        self.classifier_ln20 = Classifier(latent_size, out_size) # Pitch[20-29]
        self.classifier_ln30 = Classifier(latent_size, out_size) # Pitch[30-39]
        self.classifier_ln40 = Classifier(latent_size, out_size) # Pitch[40-49]
        self.classifier_ln50 = Classifier(latent_size, out_size) # Pitch[50-59]
        self.classifier_ln60 = Classifier(latent_size, out_size) # Pitch[60-69]
        self.classifier_ln70 = Classifier(latent_size, out_size) # Pitch[70-82]
    
    def encode(self, x, pitch):
        if pitch.data < 30: 
            latent = self.enc20(x)
        elif pitch.data >=30 and pitch.data < 40: 
            latent = self.enc30(x)
        elif pitch.data >=40 and pitch.data < 50: 
            latent = self.enc40(x)
        elif pitch.data >=50 and pitch.data < 60: 
            latent = self.enc50(x)
        elif pitch.data >=60 and pitch.data < 70: 
            latent = self.enc60(x)
        else: # pitch.data >=70:
            latent = self.enc70(x)
        return latent

    def forward(self, x, pitch):
        if pitch.data < 30: 
            latent = self.enc20(x)
            class_ = self.classifier_ln20(latent)
        elif pitch.data >=30 and pitch.data < 40: 
            latent = self.enc30(x)
            class_ = self.classifier_ln30(latent)
        elif pitch.data >=40 and pitch.data < 50: 
            latent = self.enc40(x)
            class_ = self.classifier_ln40(latent)
        elif pitch.data >=50 and pitch.data < 60: 
            latent = self.enc50(x)
            class_ = self.classifier_ln50(latent)
        elif pitch.data >=60 and pitch.data < 70: 
            latent = self.enc60(x)
            class_ = self.classifier_ln60(latent)
        else: # pitch.data >=70:
            latent = self.enc70(x)
            class_ = self.classifier_ln70(latent)
        return class_, latent

class SmallDDGanSynthMultiClass(nn.Module):
    ''' 
    '''
    def __init__(self, device, cond_size, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352, prob=0.2):
        super(SmallDDGanSynthMultiClass, self).__init__()

        self.enc = SmallDDGanSynth2(device ,cond_size, in_ch = 1, mid_ch = 64, out_size = 64, prob=prob) # Pitch[20-29]

        self.classifier_ln20 = Classifier(latent_size, out_size) # Pitch[20-29]
        self.classifier_ln30 = Classifier(latent_size, out_size) # Pitch[30-39]
        self.classifier_ln40 = Classifier(latent_size, out_size) # Pitch[40-49]
        self.classifier_ln50 = Classifier(latent_size, out_size) # Pitch[50-59]
        self.classifier_ln60 = Classifier(latent_size, out_size) # Pitch[60-69]
        self.classifier_ln70 = Classifier(latent_size, out_size) # Pitch[70-82]

    def restore_weights(self, pitch):
        self.enc.restore_weights(pitch)
    
    def encode(self, x, pitch):
        latent = self.enc(x, pitch)
        return latent

    def forward(self, x, pitch):
        latent = self.encode(x, pitch)
        if pitch.data < 30: 
            class_ = self.classifier_ln20(latent)
        elif pitch.data >=30 and pitch.data < 40: 
            class_ = self.classifier_ln30(latent)
        elif pitch.data >=40 and pitch.data < 50: 
            class_ = self.classifier_ln40(latent)
        elif pitch.data >=50 and pitch.data < 60: 
            class_ = self.classifier_ln50(latent)
        elif pitch.data >=60 and pitch.data < 70: 
            class_ = self.classifier_ln60(latent)
        else: # pitch.data >=70:
            class_ = self.classifier_ln70(latent)
        return class_, latent

class SmallGanSynthClass2(nn.Module):
    ''' 
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(SmallGanSynthClass2, self).__init__()
        feat_size = (16,4) 
        layers = []
        layers += [nn.Conv2d(in_ch, mid_ch, (1,1), 1)]
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.net = nn.Sequential(*layers)
        self.ln = nn.Linear(mid_ch * feat_size[0]* feat_size[1], out_size)
        #self.classifier_ln = Classifier(latent_size, out_size)

    def encode(self, x):
        return self.forward(x)

    def forward(self, x):
        h = self.net(x)
        out = self.ln(h.view(h.size(0), -1))
        #out = self.classifier_ln(out)
        return out


class SmallGanSynthDDClass(nn.Module):
    ''' 
    Deterministic Dropout is only used in Linear Layers
    '''
    def __init__(self, device, cond_size, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352, prob=0.2):
        super(SmallGanSynthDDClass, self).__init__()
        feat_size = (16,4) 

        self.enc_1 = nn.Conv2d(in_ch, mid_ch, (1,1), 1)
        self.enc_2 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 128x32
        self.enc_3 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 64x16
        self.enc_4 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 32x8
        self.enc_5 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 16x4

        self.ln = nn.Linear(mid_ch * feat_size[0]* feat_size[1], latent_size)
        self.class_ddln = DDClassifier(latent_size, out_size, cond_size, device=device, prob=prob)

    def restore_weights(self, class_id):
        self.class_ddln.restore_weights(class_id)

    def encode(self, x):
        h = self.enc_1(x)
        h = self.enc_2(h)
        h = self.enc_3(h)
        h = self.enc_4(h)
        h = self.enc_5(h)
        return self.ln(h.view(h.size(0), -1))

    def classify(self, latent, class_id):
        return self.class_ddln(latent, class_id)

    def forward(self, x, class_id):
        latent = self.encode(x)
        out = self.classify(latent, class_id)
        return out, latent

class SmallDDGanSynthDDClass(nn.Module):
    ''' 
    Deterministic Dropout is used in CNNs and also Linear Layers
    '''
    def __init__(self, device, cond_size, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352, prob=0.2):
        super(SmallDDGanSynthDDClass, self).__init__()
        feat_size = (16,4) 

        self.enc_1 = net_utils.DeterministicDropoutCNN(in_ch, mid_ch, cond_size, 
                            device,kr_size=(1,1), prob=prob)
        self.enc_2 = net_utils.DDGANSynthBlock(mid_ch, mid_ch, cond_size, device, 'enc', prob=prob) # 128x32
        self.enc_3 = net_utils.DDGANSynthBlock(mid_ch, mid_ch, cond_size, device, 'enc', prob=prob) # 64x16
        self.enc_4 = net_utils.DDGANSynthBlock(mid_ch, mid_ch, cond_size, device, 'enc', prob=prob) # 32x8
        self.enc_5 = net_utils.DDGANSynthBlock(mid_ch, mid_ch, cond_size, device, 'enc', prob=prob) # 16x4

        self.ddln = net_utils.DeterministicDropoutLN(mid_ch * feat_size[0]* feat_size[1], 
                            latent_size, cond_size, device, prob)
        self.class_ddln = DDClassifier(latent_size, out_size, cond_size, 
                            device=device, prob=prob)

    def restore_weights(self, class_id):
        self.enc_1.restore_weights(class_id)
        self.enc_2.restore_weights(class_id)
        self.enc_3.restore_weights(class_id)
        self.enc_3.restore_weights(class_id)
        self.enc_5.restore_weights(class_id)
        self.ddln.restore_weights(class_id)
        self.class_ddln.restore_weights(class_id)

    def forward(self, x, class_id):
        h = self.enc_1(x, class_id)
        h = self.enc_2(h, class_id)
        h = self.enc_3(h, class_id)
        h = self.enc_4(h, class_id)
        h = self.enc_5(h, class_id)
        h = self.ddln(h.view(h.size(0), -1), class_id)
        out = self.class_ddln(h, class_id)
        return out

class SmallDDGanSynth2(nn.Module):
    ''' 
    Deterministic Dropout is used in CNNs and also Linear Layers without Classifier
    '''
    def __init__(self, device, cond_size, in_ch = 1, mid_ch = 64, out_size = 352, prob=0.2):
        super(SmallDDGanSynth2, self).__init__()
        feat_size = (16,4) 

        self.enc_1 = net_utils.DeterministicDropoutCNN(in_ch, mid_ch, cond_size, 
                            device,kr_size=(1,1), prob=prob)
        self.enc_2 = net_utils.DDGANSynthBlock(mid_ch, mid_ch, cond_size, device, 'enc', prob=prob) # 128x32
        self.enc_3 = net_utils.DDGANSynthBlock(mid_ch, mid_ch, cond_size, device, 'enc', prob=prob) # 64x16
        self.enc_4 = net_utils.DDGANSynthBlock(mid_ch, mid_ch, cond_size, device, 'enc', prob=prob) # 32x8
        self.enc_5 = net_utils.DDGANSynthBlock(mid_ch, mid_ch, cond_size, device, 'enc', prob=prob) # 16x4

        self.ddln = net_utils.DeterministicDropoutLN(mid_ch * feat_size[0]* feat_size[1], 
                            out_size, cond_size, device, prob)

    def restore_weights(self, class_id):
        self.enc_1.restore_weights(class_id)
        self.enc_2.restore_weights(class_id)
        self.enc_3.restore_weights(class_id)
        self.enc_3.restore_weights(class_id)
        self.enc_5.restore_weights(class_id)
        self.ddln.restore_weights(class_id)

    def forward(self, x, class_id):
        h = self.enc_1(x, class_id)
        h = self.enc_2(h, class_id)
        h = self.enc_3(h, class_id)
        h = self.enc_4(h, class_id)
        h = self.enc_5(h, class_id)
        out = self.ddln(h.view(h.size(0), -1), class_id)
        return out


class DDClassifier(nn.Module):
    '''
    Deterministic Dropout Linear Classifier
    '''
    def __init__(self, input_size, output_size, class_size, device, mid_size=512, prob=0.2):
        super(DDClassifier, self).__init__()
        self.linears = nn.ModuleList([net_utils.DeterministicDropoutLN(input_size, mid_size, class_size, device, prob),
        #self.AE = networks.SmallDDGanSynthDDClass(self.device, 84 ,1, 64, 64, 352, prob=opt['model']['drop_prob'])
                        net_utils.DeterministicDropoutLN(mid_size, mid_size, class_size, device, prob),
                        net_utils.DeterministicDropoutLN(mid_size, mid_size, class_size, device, prob),
                        net_utils.DeterministicDropoutLN(mid_size, output_size, class_size, device, prob)])
    
    def restore_weights(self, class_id):
        for layer in self.linears:
            layer.restore_weights(class_id)

    def forward(self, input, class_id):
        for idx, layer in enumerate(self.linears):
            if idx == 0:
                h = F.relu(layer(input, class_id))
            elif idx != len(self.linears)-1:
                h = F.relu(layer(h, class_id))
            else:
                out = layer(h, class_id)
        return out


class CasAE_2(nn.Module):
    ''' Similar architecture to CasAE but this architecture's varies the pitch encoder in 
    that the timbre embedding is concatenated without channel reduction.
    '''
    def __init__(self, latent_size, in_channels = 2, feat_width = 128):
        super(CasAE_2, self).__init__()
        kw = feat_width//64
        timbre_latent_size = latent_size//2
        pitch_latent_size = timbre_latent_size
        
        # Timbre encoder
        t_enc_layers = []
        t_enc_layers += [nn.Conv2d(in_channels, 32, (1,1), 1)]
        t_enc_layers += net_utils.create_gansynth_block(32, 32, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(32, 64, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(64, 128, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(128, 256, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(256, 256, 'enc')
        t_enc_layers += net_utils.create_gansynth_block(256, 256, 'enc')
        t_enc_layers += [nn.Conv2d(256, timbre_latent_size, (16,kw), 1)]
        self.t_enc_net = nn.Sequential(*t_enc_layers)
        self.t_enc_net_params = self.t_enc_net.parameters()

        # Pitch encoder
        self.p_enc_1 = nn.Conv2d(in_channels + timbre_latent_size, 32, (1,1), 1)
        self.p_enc_2 = net_utils.GANSynthBlock(32 + timbre_latent_size, 32, 'enc')
        self.p_enc_3 = net_utils.GANSynthBlock(32 + timbre_latent_size, 64, 'enc')
        self.p_enc_4 = net_utils.GANSynthBlock(64, 128, 'enc')
        self.p_enc_5 = net_utils.GANSynthBlock(128, 256, 'enc')
        self.p_enc_6 = net_utils.GANSynthBlock(256, 256, 'enc')
        self.p_enc_7 = net_utils.GANSynthBlock(256, 256, 'enc')
        self.p_enc_8 = nn.Conv2d(256, pitch_latent_size, (16,kw), 1)
        self.p_enc_net_params = list(self.p_enc_1.parameters()) + list(self.p_enc_2.parameters()) + \
                    list(self.p_enc_3.parameters()) + list(self.p_enc_4.parameters()) + \
                    list(self.p_enc_5.parameters()) + list(self.p_enc_6.parameters()) + \
                    list(self.p_enc_7.parameters()) + list(self.p_enc_8.parameters())                    
        
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
        h_timbre = self.t_enc_net(input)

        h_timbre_cond = h_timbre.detach()
        shape = (h_timbre_cond.size(0), h_timbre_cond.size(1), input.size(2), input.size(3))
        h_p = self.p_enc_1(torch.cat([input ,h_timbre_cond.expand(*shape)], dim=1)) 
        shape = (h_timbre_cond.size(0), h_timbre_cond.size(1), h_p.size(2), h_p.size(3))
        h_p = self.p_enc_2(torch.cat([h_p ,h_timbre_cond.expand(*shape)], dim=1))
        shape = (h_timbre_cond.size(0), h_timbre_cond.size(1), h_p.size(2), h_p.size(3))
        h_p = self.p_enc_3(torch.cat([h_p ,h_timbre_cond.expand(*shape)], dim=1))
        h_p = self.p_enc_4(h_p)
        h_p = self.p_enc_5(h_p)
        h_p = self.p_enc_6(h_p)
        h_p = self.p_enc_7(h_p)
        h_pitch = self.p_enc_8(h_p)

        return h_timbre, h_pitch

    def decode(self, latent):
        return self.dec_net(latent)

    def forward(self, input):
        h_timbre, h_pitch = self.encode(input)
        latent = torch.cat([h_timbre, h_pitch], dim=1)      
        out = self.decode(latent)

        return out, h_timbre, h_pitch


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

class SmallCasAE_3(nn.Module):
    ''' Small feature input [256,64] with cascade encoders. Also includes input pitch and timbre classification
    '''
    def __init__(self, timbre_size, pitch_size, in_channels = 2, mid_ch=64, pitch_class=84, timbre_class=352):
        super(SmallCasAE_3, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        # Timbre encoder
        t_enc_layers = []
        t_enc_layers += [nn.Conv2d(in_channels, mid_ch, (1,1), 1)]
        t_enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        t_enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        t_enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        t_enc_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.t_enc_net = nn.Sequential(*t_enc_layers)
        self.t_enc_net_params = self.t_enc_net.parameters()
        self.t_enc_ln = nn.Linear(mid_ch * self.feat_size[0]* self.feat_size[1], timbre_size)

        self.class_t_net = Classifier(timbre_size, timbre_class,512)

        # Pitch encoder
        self.p_enc_1 = nn.Conv2d(in_channels + timbre_size, mid_ch, (1,1), 1)
        self.p_enc_2 = net_utils.GANSynthBlock(mid_ch + timbre_size, mid_ch, 'enc') # 128x32
        self.p_enc_3 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 64x16
        self.p_enc_4 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 32x8
        self.p_enc_5 = net_utils.GANSynthBlock(mid_ch, mid_ch, 'enc') # 16x4
        self.p_enc_ln = nn.Linear(mid_ch * self.feat_size[0]* self.feat_size[1], pitch_size) 

        self.class_p_net = Classifier(pitch_size, pitch_class, 512)

        self.dec_ln = nn.Linear(pitch_size + timbre_size, mid_ch * self.feat_size[0]* self.feat_size[1])
        dec_layers = []
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        dec_layers += [nn.Conv2d(mid_ch, in_channels, kernel_size=(1,1), stride=1)]
        self.dec_net = nn.Sequential(*dec_layers)
    
    def encode(self, input):
        batch_size = input.size(0)

        h_t = self.t_enc_net(input)
        h_t = h_t.view([batch_size, -1]) 
        h_t = self.t_enc_ln(h_t)

        h_timbre_cond = h_t.unsqueeze(2).unsqueeze(2)
        shape = (h_timbre_cond.size(0), h_timbre_cond.size(1), input.size(2), input.size(3))
        h_p = self.p_enc_1(torch.cat([input, h_timbre_cond.expand(*shape)], dim=1))
        shape = (h_timbre_cond.size(0), h_timbre_cond.size(1), h_p.size(2), h_p.size(3))
        h_p = self.p_enc_2(torch.cat([h_p, h_timbre_cond.expand(*shape)], dim=1))
        h_p = self.p_enc_3(h_p)
        h_p = self.p_enc_4(h_p)
        h_p = self.p_enc_5(h_p)
        h_p = h_p.view([batch_size, -1]) 
        h_p = self.p_enc_ln(h_p) 

        return h_t, h_p

    def decode(self, latent):
        latent = self.dec_ln(latent)
        latent = latent.view([latent.size(0), self.mid_ch, self.feat_size[0], self.feat_size[1]])
        return self.dec_net(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_t_net(h_t), self.class_p_net(h_p)


class GanSynthEnc(nn.Module):
    ''' 
    Classifier that operates with [1,1024,64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64):
        super(GanSynthEnc, self).__init__()
        kw = 1
        layers = []
        layers += [nn.Conv2d(in_ch, mid_ch, (1,1), 1)]
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc')
        layers += [nn.Conv2d(mid_ch, latent_size, (16,kw), 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class GanSynthEnc2(nn.Module):
    ''' 
    Classifier that operates with [1,1024,64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64):
        super(GanSynthEnc2, self).__init__()
        kw = 1
        layers = []
        layers += [nn.Conv2d(in_ch, 32, (1,1), 1)]
        layers += net_utils.create_gansynth_block(32, 32, 'enc')
        layers += net_utils.create_gansynth_block(32, 64, 'enc')
        layers += net_utils.create_gansynth_block(64, 64, 'enc')
        layers += net_utils.create_gansynth_block(64, 128, 'enc')
        layers += net_utils.create_gansynth_block(128, 128, 'enc')
        layers += net_utils.create_gansynth_block(128, 256, 'enc')
        layers += [nn.Conv2d(256, 256, (16,kw), 1)]
        layers += [nn.Conv2d(256, latent_size, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class SmallGanSynthEnc(nn.Module):
    ''' 
    Classifier that operates with [1,256, 64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64):
        super(SmallGanSynthEnc, self).__init__()
        self.feat_size = [16,4]
        
        layers = []
        layers += [nn.Conv2d(in_ch, mid_ch, (1,1), 1)]
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 128x32
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 64x16
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 32x8
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'enc') # 16x4
        self.net = nn.Sequential(*layers)
        self.ln = nn.Linear(mid_ch * self.feat_size[0]* self.feat_size[1], latent_size)

    def forward(self, input):
        latent = self.net(input)
        latent = latent.view([latent.size(0), -1]) 
        return self.ln(latent)

class SmallGanSynthEnc2(nn.Module):
    ''' 
    Classifier that operates with [1,256, 64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64):
        super(SmallGanSynthEnc2, self).__init__()
        self.feat_size = [16,4]
        
        layers = []
        layers += [nn.Conv2d(in_ch, 32, (1,1), 1)]
        layers += net_utils.create_gansynth_block(32, 64, 'enc') # 128x32
        layers += net_utils.create_gansynth_block(64, 128, 'enc') # 64x16
        layers += net_utils.create_gansynth_block(128, 256, 'enc') # 32x8
        layers += net_utils.create_gansynth_block(256, 256, 'enc') # 16x4
        self.net = nn.Sequential(*layers)
        self.ln = nn.Linear(256 * self.feat_size[0]* self.feat_size[1], latent_size)

    def forward(self, input):
        latent = self.net(input)
        latent = latent.view([latent.size(0), -1]) 
        return self.ln(latent)

class GanSynthDec(nn.Module):
    ''' 
    Classifier that operates with [1,1024,64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64):
        super(GanSynthDec, self).__init__()
        kw = 1
        layers = []
        layers += [nn.Conv2d(latent_size, mid_ch, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)]
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += [nn.Conv2d(mid_ch, in_ch, kernel_size=(1,1), stride=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)

class GanSynthDec2(nn.Module):
    ''' 
    Classifier that operates with [1,1024,64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64):
        super(GanSynthDec2, self).__init__()
        kw = 1
        layers = []
        layers += [nn.Conv2d(latent_size, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=(16,kw), stride=1, padding=(15,(kw//2)), bias=False)]
        layers += net_utils.create_gansynth_block(256, 128, 'dec')
        layers += net_utils.create_gansynth_block(128, 128, 'dec')
        layers += net_utils.create_gansynth_block(128, 64, 'dec')
        layers += net_utils.create_gansynth_block(64, 64, 'dec')
        layers += net_utils.create_gansynth_block(64, 32, 'dec')
        layers += net_utils.create_gansynth_block(32, 32, 'dec')
        layers += [nn.Conv2d(32, in_ch, kernel_size=(1,1), stride=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)

class SmallGanSynthDec(nn.Module):
    ''' 
    Classifier that operates with [1,256, 64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64):
        super(SmallGanSynthDec, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch

        self.ln = nn.Linear(latent_size, mid_ch * self.feat_size[0]* self.feat_size[1])
        layers = []
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += net_utils.create_gansynth_block(mid_ch, mid_ch, 'dec')
        layers += [nn.Conv2d(mid_ch, in_ch, kernel_size=(1,1), stride=1)]
        self.dec_net = nn.Sequential(*layers)

    def forward(self, latent):
        latent_ln = self.ln(latent)
        latent_ln = latent_ln.view([latent_ln.size(0), self.mid_ch, self.feat_size[0], self.feat_size[1]])
        return self.dec_net(latent_ln)

class SmallGanSynthDec2(nn.Module):
    ''' 
    Classifier that operates with [1,256, 64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64):
        super(SmallGanSynthDec2, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = 256

        self.ln = nn.Linear(latent_size, 256 * self.feat_size[0]* self.feat_size[1])
        layers = []
        layers += net_utils.create_gansynth_block(256, 256, 'dec')
        layers += net_utils.create_gansynth_block(256, 128, 'dec')
        layers += net_utils.create_gansynth_block(128, 64, 'dec')
        layers += net_utils.create_gansynth_block(64, 32, 'dec')
        layers += [nn.Conv2d(32, in_ch, kernel_size=(1,1), stride=1)]
        self.dec_net = nn.Sequential(*layers)

    def forward(self, latent):
        latent_ln = self.ln(latent)
        latent_ln = latent_ln.view([latent_ln.size(0), self.mid_ch, self.feat_size[0], self.feat_size[1]])
        return self.dec_net(latent_ln)

class GanSynthAE(nn.Module):
    ''' 
    Autoncoder that operates with [1,1024,128]
    '''
    def __init__(self, in_channels, mid_ch, latent_size):
        super(GanSynthAE, self).__init__()

        self.enc = GanSynthEnc(in_channels, mid_ch, latent_size)
        self.dec = GanSynthDec(in_channels, mid_ch, latent_size)

    def encode(self, input):
        return self.enc(input)

    def decode(self, latent, cond=None):
        return self.dec(latent)

    def forward(self, input, cond=None):
        latent = self.enc_net(input)
        out = self.dec_net(latent)
        
        return out, latent

class GanSynthClass(nn.Module):
    ''' 
    Classifier that operates with [1,1024,64]
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(GanSynthClass, self).__init__()
        self.enc_net = GanSynthEnc(in_ch, mid_ch, latent_size)
        self.classifier_ln = Classifier(latent_size, out_size)
    
    def forward(self, input):
        latent = self.enc_net(input)
        latent = latent.squeeze(2).squeeze(2)
        out = self.classifier_ln(latent)

        return out, latent

class GanSynthEnc1D(nn.Module):
    ''' 
    1D Based Encoder [B,1024,64]
    '''
    def __init__(self, in_ch = 1024, mid_ch = 512, latent_size= 64):
        super(GanSynthEnc1D, self).__init__()
        enc_layers = []

        enc_layers += [nn.Conv1d(in_ch, mid_ch, 1, 1),
                    nn.Conv1d(mid_ch, mid_ch, 5, 1, padding=(5-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]

        enc_layers += [nn.Conv1d(mid_ch, mid_ch, 5, 1, padding=(5-1)//2),
                    nn.Conv1d(mid_ch, mid_ch, 5, 1, padding=(5-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]

        enc_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, padding=(3-1)//2),
                    nn.Conv1d(mid_ch, mid_ch, 3, 1, padding=(3-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]

        enc_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, padding=(3-1)//2),
                    nn.Conv1d(mid_ch, mid_ch, 3, 1, padding=(3-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]

        enc_layers += [nn.Conv1d(mid_ch, mid_ch, 4, 1),
                    nn.Conv1d(mid_ch, latent_size, 1, 1),
                    nn.LeakyReLU(0.2)]

        self.enc_net = nn.Sequential(*enc_layers)
    
    def forward(self, input):
        input_1d = input.squeeze(1)
        return self.enc_net(input_1d)

class GanSynthDec1D(nn.Module):
    ''' 
    1D Based Decoder [B,1024,64]
    '''
    def __init__(self, in_ch = 1024, mid_ch = 512, latent_size= 64):
        super(GanSynthDec1D, self).__init__()
        dec_layers = []

        dec_layers += [nn.Conv1d(latent_size, mid_ch, 1, 1),
                    nn.Conv1d(mid_ch, mid_ch, 4, 1, padding=3),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest')]

        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 3, 1, padding=(3-1)//2),
                    nn.Conv1d(mid_ch, mid_ch, 3, 1, padding=(3-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest')]

        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 5, 1, padding=(5-1)//2),
                    nn.Conv1d(mid_ch, mid_ch, 5, 1, padding=(5-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest')]

        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 5, 1, padding=(5-1)//2),
                    nn.Conv1d(mid_ch, mid_ch, 5, 1, padding=(5-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest')]

        dec_layers += [nn.Conv1d(mid_ch, mid_ch, 5, 1, padding=(5-1)//2),
                    nn.Conv1d(mid_ch, in_ch, 1, 1),
                    nn.Tanh()]

        self.dec_net = nn.Sequential(*dec_layers)
    
    def forward(self, latent):
        while len(latent.shape)<3:
            latent = latent.unsqueeze(-1)
        return self.dec_net(latent)

class GanSynthAE1D(nn.Module):
    ''' 
    1D Based AE [B,1024,64]
    '''
    def __init__(self, in_ch = 1024, mid_ch = 512, latent_size= 64):
        super(GanSynthAE1D, self).__init__()
        self.encoder = GanSynthEnc1D(in_ch, mid_ch, latent_size)
        self.decoder = GanSynthDec1D(in_ch, mid_ch, latent_size)
        
    def encode(self, input):
        return self.encoder(input)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, input):
        return self.decode(self.encode(input))

class DualGanSynthAE1D(nn.Module):
    ''' 
    1D Based AE [B,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 1024, mid_ch=64):
        super(DualGanSynthAE1D, self).__init__()
        self.enc_p = GanSynthEnc1D(in_ch, mid_ch, pitch_size)
        self.enc_t = GanSynthEnc1D(in_ch, mid_ch, timbre_size)
        self.dec = GanSynthDec1D(in_ch, mid_ch, pitch_size+timbre_size)
    
    def encode(self, input):
        return self.enc_t(input), self.enc_p(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p

class DualGanSynthAE1DWithPTClass(nn.Module):
    ''' 
    Autoncoder that operates with [B,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 1024, mid_ch=64, pitch_class=84, timbre_class=84):
        super(DualGanSynthAE1DWithPTClass, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.dual_ae = DualGanSynthAE1D(timbre_size, pitch_size, in_ch, mid_ch)
        self.class_t = Classifier(timbre_size, timbre_class, timbre_size)
        self.class_p = Classifier(pitch_size, pitch_class, pitch_size)
    
    def encode(self, input):
        return self.dual_ae.encode(input) # Returns a tuple with two embeddings

    def decode(self, latent):
        return self.dual_ae.decode(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_t(h_t), self.class_p(h_p)

class SmallGanSynthAE(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, latent_size, in_ch = 2, mid_ch=64):
        super(SmallGanSynthAE, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.enc = SmallGanSynthEnc(in_ch, mid_ch, latent_size)
        self.dec = SmallGanSynthDec(in_ch, mid_ch, latent_size)
    
    def encode(self, input):
        return self.enc(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        latent = self.encode(input)
        return self.decode(latent), latent

class SmallGanSynthAE2(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, latent_size, in_ch = 2, mid_ch=64):
        super(SmallGanSynthAE2, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.enc = SmallGanSynthEnc2(in_ch, mid_ch, latent_size)
        self.dec = SmallGanSynthDec2(in_ch, mid_ch, latent_size)
    
    def encode(self, input):
        return self.enc(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        latent = self.encode(input)
        return self.decode(latent), latent

class SmallDualGanSynthAE(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64):
        super(SmallDualGanSynthAE, self).__init__()
        self.enc_t = SmallGanSynthEnc(in_ch, mid_ch, timbre_size)
        self.enc_p = SmallGanSynthEnc(in_ch, mid_ch, pitch_size)
        self.dec = SmallGanSynthDec(in_ch, mid_ch, timbre_size + pitch_size)
    
    def encode(self, input):
        return self.enc_t(input), self.enc_p(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p

class GrowSmallDualGanSynthAE2(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64):
        super(GrowSmallDualGanSynthAE2, self).__init__()
        self.enc_t = GrowSmallGanSynthEnc(in_ch, mid_ch, timbre_size)
        self.enc_p = GrowSmallGanSynthEnc(in_ch, mid_ch, pitch_size)
        self.dec = GrowSmallGanSynthDec(in_ch, mid_ch, timbre_size + pitch_size)
    
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
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p

class SmallDualGanSynthAE2(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64):
        super(SmallDualGanSynthAE2, self).__init__()
        self.enc_t = SmallGanSynthEnc2(in_ch, mid_ch, timbre_size)
        self.enc_p = SmallGanSynthEnc2(in_ch, mid_ch, pitch_size)
        self.dec = SmallGanSynthDec2(in_ch, mid_ch, timbre_size + pitch_size)
    
    def encode(self, input):
        return self.enc_t(input), self.enc_p(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p

class DualGanSynthAE(nn.Module):
    ''' 
    Autoncoder that operates with [1,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64, pitch_class=84, timbre_class=84):
        super(DualGanSynthAE, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.enc_t = GanSynthEnc(in_ch, mid_ch, timbre_size)
        self.enc_p = GanSynthEnc(in_ch, mid_ch, pitch_size)
        self.dec = GanSynthDec(in_ch, mid_ch, timbre_size + pitch_size)
    
    def encode(self, input):
        return self.enc_t(input), self.enc_p(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p

class DualFCAE(nn.Module):
    ''' 
    Autoncoder that operates with [1,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64):
        super(DualFCAE, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        enc_t = [nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, timbre_size)]
        self.enc_t = nn.Sequential(*enc_t)

        enc_p = [nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, pitch_size)]
        self.enc_p = nn.Sequential(*enc_p)

        dec = [nn.Linear(timbre_size+pitch_size, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(), # Removing, the data is between [-1,1]
                nn.Linear(1024, 256)]
        self.dec = nn.Sequential(*dec)

    def encode(self, input):
        return self.enc_t(input), self.enc_p(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        in_data = input.squeeze(0).squeeze(0)
        in_data = torch.transpose(in_data, 0, 1)
        h_t, h_p = self.encode(in_data)
        out = self.decode(torch.cat([h_t, h_p], dim=1))
        out = torch.transpose(out, 0, 1).unsqueeze(0).unsqueeze(0)
        return out, h_t, h_p

class DualFCAE2(nn.Module):
    ''' 
    Autoncoder that operates with [1,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64):
        super(DualFCAE2, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        enc = [nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU()]
        self.enc = nn.Sequential(*enc)

        self.enc_t = nn.Linear(1024, timbre_size)
        self.enc_p = nn.Linear(1024, pitch_size)

        dec = [nn.Linear(timbre_size+pitch_size, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 256)]
        self.dec = nn.Sequential(*dec)

    def encode(self, input):
        h = self.enc(input)
        return self.enc_t(h), self.enc_p(h)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        in_data = input.squeeze(0).squeeze(0)
        in_data = torch.transpose(in_data, 0, 1)
        h_t, h_p = self.encode(in_data)
        out = self.decode(torch.cat([h_t, h_p], dim=1))
        out = torch.transpose(out, 0, 1).unsqueeze(0).unsqueeze(0)
        return out, h_t, h_p

class DualFCAEWithPTClass(nn.Module):
    ''' 
    Autoncoder that operates with [1,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64, pitch_class=84, timbre_class=84):
        super(DualFCAEWithPTClass, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        enc_t = [nn.Linear(256, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, timbre_size)]
        self.enc_t = nn.Sequential(*enc_t)

        enc_p = [nn.Linear(256, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, pitch_size)]
        self.enc_p = nn.Sequential(*enc_p)

        dec = [nn.Linear(timbre_size+pitch_size, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, 256)]
        self.dec = nn.Sequential(*dec)

        self.class_t = Classifier(timbre_size, timbre_class, 512)
        self.class_p = Classifier(pitch_size, pitch_class, 512)

    def encode(self, input):
        return self.enc_t(input), self.enc_p(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        in_data = input.squeeze(0).squeeze(0)
        in_data = torch.transpose(in_data, 0, 1)
        h_t, h_p = self.encode(in_data)
        out = self.decode(torch.cat([h_t, h_p], dim=1))
        out = torch.transpose(out, 0, 1).unsqueeze(0).unsqueeze(0)
        return out, h_t, h_p, self.class_t(h_t), self.class_p(h_p)


class SmallDualGanSynthAEWithPTClass(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64, pitch_class=84, timbre_class=84):
        super(SmallDualGanSynthAEWithPTClass, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.dual_ae = SmallDualGanSynthAE(timbre_size, pitch_size, in_ch, mid_ch)
        self.class_t = Classifier(timbre_size, timbre_class, 512)
        self.class_p = Classifier(pitch_size, pitch_class, 512)
    
    def encode(self, input):
        return self.dual_ae.encode(input) # Returns a tuple with two embeddings

    def decode(self, latent):
        return self.dual_ae.decode(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_t(h_t), self.class_p(h_p)

class SmallDualGanSynthAE2WithPTClass(nn.Module):
    ''' 
    Autoncoder that operates with [1,256,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64, pitch_class=84, timbre_class=84):
        super(SmallDualGanSynthAE2WithPTClass, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.dual_ae = SmallDualGanSynthAE2(timbre_size, pitch_size, in_ch, mid_ch)
        self.class_t = Classifier(timbre_size, timbre_class, 512)
        self.class_p = Classifier(pitch_size, pitch_class, 512)
    
    def encode(self, input):
        return self.dual_ae.encode(input) # Returns a tuple with two embeddings

    def decode(self, latent):
        return self.dual_ae.decode(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_t(h_t), self.class_p(h_p)

class DualGanSynthAEWithPTClass(nn.Module):
    ''' 
    Autoncoder that operates with [1,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64, pitch_class=84, timbre_class=84):
        super(DualGanSynthAEWithPTClass, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.dual_ae = DualGanSynthAE(timbre_size, pitch_size, in_ch, mid_ch)
        self.class_t = Classifier(timbre_size, timbre_class, 512)
        self.class_p = Classifier(pitch_size, pitch_class, 512)
    
    def encode(self, input):
        return self.dual_ae.encode(input) # Returns a tuple with two embeddings

    def decode(self, latent):
        return self.dual_ae.decode(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_t(h_t), self.class_p(h_p)

class FC1(nn.Module):
    def __init__(self, input_size=1024, mid_size=1024, output_size=1024):
        super(FC2, self).__init__()
        net = [] 
        net += [nn.Linear(input_size,mid_size)]
        net += [nn.Tanh()]
        net += [nn.Linear(mid_size,mid_size)]
        net += [nn.Tanh()]
        net += [nn.Linear(mid_size,mid_size)]
        net += [nn.Tanh()]
        net += [nn.Linear(mid_size, output_size)]
        net += [nn.Tanh()]
        self.m2p_net = nn.Sequential(*net)
    
    def forward(self, input):
        if len(input.size()) > 2:
            input = input.squeeze(2).squeeze(2)
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

class SmallGanSynthClass(nn.Module):
    ''' 
    '''
    def __init__(self, in_ch = 1, mid_ch = 64, latent_size= 64, out_size = 352):
        super(SmallGanSynthClass, self).__init__()
        self.enc = SmallGanSynthEnc(in_ch, mid_ch, latent_size)
        self.classifier_ln = Classifier(latent_size, out_size)

    def forward(self, input):
        h = self.enc(input)
        out = self.classifier_ln(h)
        return out, h

class DualGanSynthAE2WithPTClass(nn.Module):
    ''' 
    Autoncoder that operates with [1,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64, pitch_class=84, timbre_class=84):
        super(DualGanSynthAE2WithPTClass, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.dual_ae = DualGanSynthAE2(timbre_size, pitch_size, in_ch, mid_ch)
        self.class_t = Classifier(timbre_size, timbre_class, 512)
        self.class_p = Classifier(pitch_size, pitch_class, 512)
    
    def encode(self, input):
        return self.dual_ae.encode(input) # Returns a tuple with two embeddings

    def decode(self, latent):
        return self.dual_ae.decode(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p, self.class_t(h_t), self.class_p(h_p)



class DualGanSynthAE2(nn.Module):
    ''' 
    Autoncoder that operates with [1,1024,64]
    '''
    def __init__(self, timbre_size, pitch_size, in_ch = 2, mid_ch=64, pitch_class=84, timbre_class=84):
        super(DualGanSynthAE2, self).__init__()
        self.feat_size = [16,4]
        self.mid_ch = mid_ch
        
        self.enc_t = GanSynthEnc2(in_ch, mid_ch, timbre_size)
        self.enc_p = GanSynthEnc(in_ch, mid_ch, pitch_size)
        self.dec = GanSynthDec2(in_ch, mid_ch, timbre_size + pitch_size)
    
    def encode(self, input):
        return self.enc_t(input), self.enc_p(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input):
        h_t, h_p = self.encode(input)
        return self.decode(torch.cat([h_t, h_p], dim=1)), h_t, h_p


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
    elif opt['model']['ae'] == 'casae':
        return CasAE(opt['model']['latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)
    elif opt['model']['ae'] == 'casae2':
        return CasAE_2(opt['model']['latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)
    elif opt['model']['ae'] == 'vaegansynth':
        return VAEGanSynthAE(opt['model']['latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width)
    elif opt['model']['ae'] == 'lightvaegansynth':
        return LightVAEGanSynthAE(opt['model']['latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width,
                        mid_ch=opt['model']['mid_ch'])
    elif opt['model']['ae'] == 'smallgansynth':
        return SmallGanSynthAE(opt['model']['latent_size'],
                        opt['model']['in_ch'],
                        mid_ch=opt['model']['mid_ch'])
    elif opt['model']['ae'] == 'vae1d':
        return VAE1D(opt['model']['latent_size'],
                        opt['model']['in_ch'],
                        feat_width= feat_width,
                        mid_ch=opt['model']['mid_ch'])
    elif opt['model']['ae'] == 'gmvae':
        return GMVAE(opt['model']['latent_size'],
                        opt['model']['in_ch'],
                        feat_width= 64,
                        mid_ch=opt['model']['mid_ch'],
                        pitch_size=opt['data']['pitch_size'], 
                        timbre_size=opt['data']['timbre_size'],
                        device= device)
    elif opt['model']['ae'] == 'smallgmvaesynth':
        return SmallGMVAESynth(opt['model']['latent_size'],
                        opt['model']['in_ch'],
                        mid_ch=opt['model']['mid_ch'],
                        pitch_size=opt['data']['pitch_size'], 
                        timbre_size=opt['data']['timbre_size'],
                        device= device)
    else:
        raise NotImplementedError

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
